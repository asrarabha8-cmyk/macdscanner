"""
ماسح السوق السعودي (تداول) — اختراق متوسط 100 بهدف متوسط 200 — فاصل يومي
منفصل تماماً عن ماسحات السوق الأمريكي في هذا الريبو.

المخرجات:
  - تنبيه تيليجرام (مرة واحدة لكل جلسة تداول)
  - ملفات JSON في docs/sa/data تقرأها واجهة الويب:
      latest.json    آخر فحص: التوقيت، الإعدادات، الإشارات، المراقبة، المؤشر العام
      universe.json  كل الأسهم المفحوصة ومقاييسها
      history.json   كل الإشارات السابقة وتتبعها (هدف/وقف/مفتوحة/منتهية)
      runs.json      سجل مرات التشغيل
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

# ═════════ الإعدادات ═════════
SETTINGS = {
    "ma_fast": 100,
    "ma_slow": 200,
    "vol_lookback": 20,
    "vol_mult": 2.0,               # فوليوم يوم الاختراق >= ضعف متوسط 20 يوماً
    "min_avg_value_sar": 1_000_000,  # أدنى متوسط قيمة تداول يومية (ريال)
    "require_slow_above_fast": True,  # MA200 فوق MA100
    "exclude_nomu": True,           # استبعاد السوق الموازية (رموز 9xxx)
    "watch_band_pct": 3.0,          # قائمة المراقبة: تحت MA100 بنسبة لا تزيد عن 3%
    "max_hold_days": 60,            # تنتهي متابعة الإشارة بعد 60 يوم تداول
}
SCHEDULE = {
    "days": "الأحد – الخميس",
    "session": "10:00 – 15:00",
    "closing_auction": "15:00 – 15:10",
    "run_time": "15:45",
    "run_weekdays": [0, 1, 2, 3, 4],   # 0=الأحد (بتوقيت الرياض)
    "run_hour": 15,
    "run_minute": 45,
    "tz": "Asia/Riyadh",
}
DASHBOARD_URL = "https://asrarabha8-cmyk.github.io/macdscanner/sa/"

RIYADH = timezone(timedelta(hours=3))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "docs", "sa", "data")
UNIVERSE_FILE = os.path.join(HERE, "sa_universe.txt")

TG_TOKEN = os.getenv("SA_TG_TOKEN") or os.getenv("TG_TOKEN")
TG_CHAT = os.getenv("SA_TG_CHAT") or os.getenv("TG_CHAT")
FORCE_ALERT = os.getenv("SA_FORCE_ALERT") == "1"


# ═════════ أدوات ═════════
def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rnd(x, n=2):
    try:
        if x is None or pd.isna(x):
            return None
        return round(float(x), n)
    except Exception:
        return None


def load_json(name, default):
    p = os.path.join(DATA_DIR, name)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(name, obj):
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


# ═════════ قائمة الأسهم ═════════
def fetch_universe_tradingview():
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "exchange", "operation": "equal", "right": "TADAWUL"},
        ],
        "columns": ["name", "description", "sector", "market_cap_basic"],
        "range": [0, 1000],
    }
    r = requests.post("https://scanner.tradingview.com/ksa/scan", json=payload,
                      timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out = {}
    for row in r.json().get("data", []):
        d = row.get("d", [])
        code = str(d[0]).strip() if d else ""
        if code.isdigit() and len(code) == 4:
            out[code] = {
                "name": (d[1] or code) if len(d) > 1 else code,
                "sector": (d[2] or "") if len(d) > 2 else "",
                "mcap": d[3] if len(d) > 3 else None,
            }
    return out


def load_universe_file():
    out = {}
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",", 1)]
                if parts[0].isdigit():
                    out[parts[0]] = {"name": parts[1] if len(parts) > 1 else parts[0],
                                     "sector": "", "mcap": None}
    return out


def get_universe():
    try:
        uni, src = fetch_universe_tradingview(), "TradingView"
        if len(uni) < 100:
            raise ValueError(f"قائمة قصيرة ({len(uni)})")
    except Exception as e:
        print(f"TradingView غير متاح ({e}) — استخدام القائمة الاحتياطية")
        uni, src = load_universe_file(), "قائمة احتياطية"
    if SETTINGS["exclude_nomu"]:
        uni = {c: v for c, v in uni.items() if not c.startswith("9")}
    return uni, src


# ═════════ البيانات ═════════
def download_history(codes):
    import yfinance as yf
    tickers = [f"{c}.SR" for c in codes]
    frames, failed = {}, []
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        try:
            data = yf.download(chunk, period="14mo", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("download error:", e)
            failed += [t[:-3] for t in chunk]
            continue
        for t in chunk:
            code = t[:-3]
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["Close"])
                if len(df) > SETTINGS["ma_slow"] + 5:
                    frames[code] = df
                else:
                    failed.append(code)
            except Exception:
                failed.append(code)
        time.sleep(1)
    return frames, failed


def get_tasi():
    # المصدر الأول: TradingView
    try:
        r = requests.post("https://scanner.tradingview.com/ksa/scan",
                          json={"symbols": {"tickers": ["TADAWUL:TASI"]},
                                "columns": ["close", "change"]},
                          timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        d = r.json()["data"][0]["d"]
        return {"close": rnd(d[0]), "change_pct": rnd(d[1]), "date": None}
    except Exception as e:
        print("TASI (TradingView) error:", e)
    # احتياطي: Yahoo
    try:
        import yfinance as yf
        df = yf.download("^TASI.SR", period="10d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        c, p = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
        return {"close": rnd(c), "change_pct": rnd((c / p - 1) * 100),
                "date": str(df.index[-1].date())}
    except Exception as e:
        print("TASI error:", e)
        return None


# ═════════ التقييم ═════════
def evaluate(df):
    """يرجع مقاييس السهم وحالته: signal / miss / watch / above / below / illiquid"""
    S = SETTINGS
    c, v = df["Close"], df["Volume"]
    ma_f = c.rolling(S["ma_fast"]).mean()
    ma_s = c.rolling(S["ma_slow"]).mean()
    vol_avg = v.shift(1).rolling(S["vol_lookback"]).mean()
    value_avg = (c * v).rolling(S["vol_lookback"]).mean()

    close, prev = float(c.iloc[-1]), float(c.iloc[-2])
    f_now, f_prev, s_now = float(ma_f.iloc[-1]), float(ma_f.iloc[-2]), float(ma_s.iloc[-1])
    va = float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0
    vol_ratio = float(v.iloc[-1]) / va if va > 0 else 0.0
    avg_val = float(value_avg.iloc[-1]) if not pd.isna(value_avg.iloc[-1]) else 0.0

    m = {
        "close": rnd(close), "chg_pct": rnd((close / prev - 1) * 100),
        "ma100": rnd(f_now), "ma200": rnd(s_now),
        "dist100": rnd((close / f_now - 1) * 100), "dist200": rnd((close / s_now - 1) * 100),
        "vol_ratio": rnd(vol_ratio), "avg_value": rnd(avg_val, 0),
        "low": rnd(df["Low"].iloc[-1]),
    }

    liquid = avg_val >= S["min_avg_value_sar"]
    slow_ok = (s_now > f_now) or not S["require_slow_above_fast"]
    crossed = prev <= f_prev and close > f_now

    if crossed:
        reasons = []
        if vol_ratio < S["vol_mult"]:
            reasons.append(f"الفوليوم ×{vol_ratio:.1f} أقل من ×{S['vol_mult']:g}")
        if not liquid:
            reasons.append("سيولة ضعيفة")
        if not slow_ok:
            reasons.append("MA200 تحت MA100")
        stop, target = float(df["Low"].iloc[-1]), s_now
        if target <= close:
            reasons.append("السعر فوق الهدف MA200")
        if reasons:
            m["status"], m["reasons"] = "miss", reasons
        else:
            risk, reward = close - stop, target - close
            m.update({
                "status": "signal", "stop": rnd(stop), "target": rnd(target),
                "upside_pct": rnd(reward / close * 100), "risk_pct": rnd(risk / close * 100),
                "rr": rnd(reward / risk if risk > 0 else None),
            })
        return m

    if not liquid:
        m["status"] = "illiquid"
    elif close <= f_now and m["dist100"] >= -S["watch_band_pct"] and slow_ok:
        m["status"] = "watch"
    elif close > f_now:
        m["status"] = "above"
    else:
        m["status"] = "below"
    return m


# ═════════ تتبع الإشارات السابقة ═════════
def track(history, frames):
    for h in history:
        if h.get("status") not in (None, "open"):
            continue
        df = frames.get(h["code"])
        if df is None:
            continue
        sig_day = pd.Timestamp(h["date"]).date()
        after = df[[d > sig_day for d in df.index.date]]
        h["days"] = int(len(after))
        h["status"] = "open"
        if len(after):
            h["last_close"] = rnd(after["Close"].iloc[-1])
            h["max_high"] = rnd(after["High"].max())
            h["min_low"] = rnd(after["Low"].min())
            h["return_pct"] = rnd((after["Close"].iloc[-1] / h["entry"] - 1) * 100)
            for ts, bar in after.iterrows():
                if bar["Low"] <= h["stop"]:          # الوقف أولاً (تحفظاً) لو حصلا بنفس اليوم
                    h.update(status="stop", exit_date=str(ts.date()), exit_price=h["stop"],
                             return_pct=rnd((h["stop"] / h["entry"] - 1) * 100))
                    break
                if bar["High"] >= h["target"]:
                    h.update(status="target", exit_date=str(ts.date()), exit_price=h["target"],
                             return_pct=rnd((h["target"] / h["entry"] - 1) * 100))
                    break
            if h["status"] == "open" and h["days"] >= SETTINGS["max_hold_days"]:
                h.update(status="expired", exit_date=str(after.index[-1].date()),
                         exit_price=h["last_close"])
    return history


# ═════════ تيليجرام ═════════
def send_telegram(text):
    if not (TG_TOKEN and TG_CHAT):
        print("لا توجد أسرار تيليجرام — طباعة فقط")
        return False
    ok = True
    for i in range(0, len(text), 3900):
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": text[i:i + 3900],
                                "parse_mode": "HTML", "disable_web_page_preview": True},
                          timeout=30)
        if not r.ok:
            print("Telegram error:", r.text, file=sys.stderr)
            ok = False
    return ok


def build_message(signals, data_date, scanned, watch_n):
    head = (f"🇸🇦 <b>ماسح السعودي — اختراق MA100 ← هدف MA200</b>\n"
            f"📅 جلسة {data_date} | فُحص {scanned} سهم | مراقبة: {watch_n}\n")
    if not signals:
        body = "\nلا توجد إشارات اليوم."
    else:
        rows = [f"\n✅ <b>{len(signals)} إشارة</b>\n"]
        for s in signals:
            rows.append(
                f"<b>{s['code']}</b> — {s['name']}\n"
                f"الإغلاق {s['close']:.2f} | MA100 {s['ma100']:.2f}\n"
                f"🎯 الهدف {s['target']:.2f} (+{s['upside_pct']:.1f}%)\n"
                f"🛑 الوقف {s['stop']:.2f} (-{s['risk_pct']:.1f}%) | R:R {s['rr'] or 0:.1f}\n"
                f"📊 الفوليوم ×{s['vol_ratio']:.1f}\n")
        body = "\n".join(rows)
    return head + body + f"\n🖥 الواجهة: {DASHBOARD_URL}"


# ═════════ التشغيل ═════════
def main():
    t0 = now_utc()
    event = os.getenv("GITHUB_EVENT_NAME", "local")
    runs = load_json("runs.json", [])
    history = load_json("history.json", [])
    state = load_json("state.json", {})

    uni, src = get_universe()
    print(f"عدد الأسهم: {len(uni)} ({src})")
    frames, failed = download_history(list(uni))
    if not frames:
        runs.insert(0, {"run_utc": iso(t0), "status": "error", "event": event,
                        "error": "تعذر جلب البيانات"})
        save_json("runs.json", runs[:150])
        send_telegram("🇸🇦 ماسح السعودي: تعذر جلب البيانات اليوم.")
        sys.exit(1)

    last_dates = pd.Series([df.index[-1].date() for df in frames.values()])
    data_date = last_dates.mode().iloc[0]

    rows, signals, watch, misses = [], [], [], []
    for code, df in frames.items():
        info = uni.get(code, {})
        if df.index[-1].date() != data_date:
            rows.append({"code": code, "name": info.get("name", code),
                         "sector": info.get("sector", ""), "status": "stale",
                         "last_date": str(df.index[-1].date())})
            continue
        try:
            m = evaluate(df)
        except Exception as e:
            print(code, "error", e)
            continue
        m.update(code=code, name=info.get("name", code), sector=info.get("sector", ""))
        rows.append(m)
        if m["status"] == "signal":
            signals.append(m)
        elif m["status"] == "watch":
            watch.append(m)
        elif m["status"] == "miss":
            misses.append(m)

    signals.sort(key=lambda s: -(s.get("rr") or 0))
    watch.sort(key=lambda s: -(s.get("dist100") or -99))

    # إضافة الإشارات الجديدة للسجل
    keys = {(h["code"], h["date"]) for h in history}
    for s in signals:
        k = (s["code"], str(data_date))
        if k not in keys:
            history.insert(0, {"code": s["code"], "name": s["name"], "sector": s["sector"],
                               "date": str(data_date), "entry": s["close"], "stop": s["stop"],
                               "target": s["target"], "rr": s["rr"], "vol_ratio": s["vol_ratio"],
                               "status": "open", "days": 0, "last_close": s["close"],
                               "return_pct": 0.0})
    history = track(history, frames)

    # تنبيه مرة واحدة لكل جلسة
    alert_sent = False
    if FORCE_ALERT or state.get("last_alert_date") != str(data_date):
        alert_sent = send_telegram(build_message(signals, data_date, len(frames), len(watch)))
        if alert_sent:
            state["last_alert_date"] = str(data_date)

    t1 = now_utc()
    tasi = get_tasi()
    run = {"run_utc": iso(t0), "finished_utc": iso(t1), "duration_s": int((t1 - t0).total_seconds()),
           "data_date": str(data_date), "event": event, "status": "ok",
           "universe_source": src, "universe_count": len(uni), "scanned": len(frames),
           "failed": len(failed), "signals": len(signals), "watch": len(watch),
           "misses": len(misses), "alert_sent": alert_sent}
    runs.insert(0, run)

    latest = {"run": run, "settings": SETTINGS, "schedule": SCHEDULE, "tasi": tasi,
              "signals": signals, "watch": watch, "misses": misses,
              "failed_codes": sorted(failed)}
    save_json("latest.json", latest)
    save_json("universe.json", {"data_date": str(data_date), "rows": rows})
    save_json("history.json", history)
    save_json("runs.json", runs[:150])
    save_json("state.json", state)
    print(f"تم: {len(signals)} إشارة، {len(watch)} مراقبة، {len(misses)} قريبة — جلسة {data_date}")


if __name__ == "__main__":
    main()
