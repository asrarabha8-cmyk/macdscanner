"""Gap Fade Watch — ماسح تلاشي الفجوات (متابعة ورقية فقط، بدون صفقات).

بعد إغلاق كل يوم تداول:
1. يلتقط فجوات الصعود القوية: افتتاح أعلى من إغلاق أمس بـ10% أو أكثر، حجم 3 أضعاف متوسط
   50 يوماً، إغلاق فوق الافتتاح، سعر 10$ أو أكثر، وسيولة يومية (متوسط 30 يوماً × السعر)
   أكثر من 20 مليون دولار (متوسطة أو كبيرة).
2. يسجلها في logs/gap_fade/events.csv.
3. يحدّث نتائج الأحداث السابقة: التغير من افتتاح اليوم التالي بعد 5 و10 و20 يوم تداول
   (سالب = السهم تراجع = في صالح البوت)، وهل ارتفع +15% قبل أن ينزل -10% خلال 20 يوماً.
4. يرسل ملخصاً على تيليجرام.
"""
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "gap_fade" / "events.csv"
BASE = "https://api.polygon.io"
GAP, VOLX, MIN_PRICE, MIN_DV = 0.10, 3.0, 10.0, 2e7
HORIZONS = (5, 10, 20)
COLS = ["date", "sym", "gap_pct", "close", "dollar_vol_m", "tier", "entry_open",
        "chg5", "chg10", "chg20", "race", "done"]


def get(url, params):
    q = urllib.parse.urlencode(params)
    for _ in range(5):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(10)
    return {}


def daily_panel(key, days=110):
    frames, d = [], date.today()
    end = d
    d = d - timedelta(days=days)
    while d <= end:
        if d.weekday() < 5:
            j = get(f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d}", {"adjusted": "true", "apiKey": key})
            rows = j.get("results") or []
            if rows:
                f = pd.DataFrame(rows)[["T", "o", "h", "l", "c", "v"]]
                f["date"] = pd.Timestamp(d)
                frames.append(f)
        d += timedelta(days=1)
    df = pd.concat(frames, ignore_index=True).rename(columns={"T": "sym"})
    df = df[df["sym"].str.fullmatch(r"[A-Z]{1,5}")]
    return df.sort_values(["sym", "date"]).reset_index(drop=True)


def find_events(df):
    g = df.groupby("sym", sort=False)
    df["pc"] = g["c"].shift(1)
    df["av50p"] = g["v"].transform(lambda x: x.shift(1).rolling(50).mean())
    df["av30"] = g["v"].transform(lambda x: x.rolling(30).mean())
    last = df["date"].max()
    recent = sorted(df["date"].unique())[-3:]  # آخر 3 أيام حتى لا يضيع يوم تأخرت بياناته
    t = df[df["date"].isin(recent)].copy()
    t["gap"] = t["o"] / t["pc"] - 1
    t["dv"] = t["av30"] * t["c"]
    ev = t[(t["gap"] >= GAP) & (t["v"] >= VOLX * t["av50p"]) & (t["c"] >= t["o"]) & (t["c"] >= MIN_PRICE) & (t["dv"] > MIN_DV)]
    return last, ev.sort_values("gap", ascending=False)


def update_outcomes(log, df):
    by_sym = {s: g.reset_index(drop=True) for s, g in df.groupby("sym", sort=False)}
    for i, r in log.iterrows():
        if r.get("done") in (True, "True"):
            continue
        g = by_sym.get(r["sym"])
        if g is None:
            continue
        after = g[g["date"] > pd.Timestamp(r["date"])].reset_index(drop=True)
        if after.empty:
            continue
        e = float(after.loc[0, "o"])
        log.at[i, "entry_open"] = round(e, 2)
        for h in HORIZONS:
            if len(after) >= h:
                log.at[i, f"chg{h}"] = round(100 * (after.loc[h - 1, "c"] / e - 1), 2)
        seg = after.head(20)
        race = ""
        for hi, lo in zip(seg["h"], seg["l"]):
            if hi >= e * 1.15:
                race = "squeeze"
                break
            if lo <= e * 0.90:
                race = "faded"
                break
        if race or len(after) >= 20:
            log.at[i, "race"] = race or "neither"
        log.at[i, "done"] = len(after) >= 20
    return log


def notify(text):
    token, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if not token or not chat:
        print(text)
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20)


def main():
    key = os.environ["POLYGON_API_KEY"]
    df = daily_panel(key)
    last, ev = find_events(df)
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    have = set(zip(log["date"].astype(str), log["sym"]))
    new = []
    for r in ev.itertuples():
        k = (str(r.date.date()), r.sym)
        if k in have:
            continue
        new.append({"date": str(r.date.date()), "sym": r.sym, "gap_pct": round(100 * r.gap, 1), "close": round(r.c, 2),
                    "dollar_vol_m": round(r.dv / 1e6, 1), "tier": "large" if r.dv > 1e8 else "mid", "done": False})
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
    log = update_outcomes(log, df)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log.reindex(columns=COLS).to_csv(LOG, index=False)

    lines = [f"📉 ماسح تلاشي الفجوات — {last.date()}", "(متابعة ورقية فقط، ليست توصية)", ""]
    if new:
        lines.append(f"فجوات صعود جديدة اليوم ({len(new)}):")
        for n in new[:15]:
            lines.append(f"• {n['sym']}  فجوة +{n['gap_pct']}%  إغلاق {n['close']}$  ({'كبير' if n['tier'] == 'large' else 'متوسط'})")
    else:
        lines.append("لا توجد فجوات صعود جديدة اليوم.")
    done = log[log["done"].astype(str) == "True"]
    if len(done):
        c20 = pd.to_numeric(done["chg20"], errors="coerce")
        lines += ["", f"الحصيلة حتى الآن ({len(done)} حدث مكتمل 20 يوماً):",
                  f"• متوسط التغير بعد 20 يوماً: {c20.mean():+.1f}%  (الوسيط {c20.median():+.1f}%)",
                  f"• تراجع (سالب): {100 * (c20 < 0).mean():.0f}%",
                  f"• نزل -10% أولاً: {100 * (done['race'] == 'faded').mean():.0f}%  |  ارتفع +15% أولاً: {100 * (done['race'] == 'squeeze').mean():.0f}%"]
    lines.append(f"\nإجمالي المسجّل: {len(log)}")
    notify("\n".join(lines))


if __name__ == "__main__":
    main()
