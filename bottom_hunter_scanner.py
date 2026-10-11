#!/usr/bin/env python3
"""
صيد القاع — ماسح يومي لاصطياد Call قرب قاع السهم، مع طبقة تأكيد من العقود الدائمة.

يُشغَّل مرة يومياً بعد إغلاق السوق الأمريكي. الإشارة تتطلب الشروط الأربعة معاً:
  1) هبوط قوي ووصول دعم: نزل ≥ 15% من قمة 60 يوماً، وقاع آخر 5 أيام عند دعم
     (أدنى قاع في الـ 120 يوماً السابقة) أو عند متوسط 200 — بفارق ±4%.
  2) إرهاق بيع: RSI(14) نزل ≤ 32 خلال آخر 5 أيام، أو دايفرجنس إيجابي
     (قاع سعري أدنى مع RSI أعلى خلال 30 يوماً).
  3) تجميع: OBV صاعد آخر 10 أيام بينما السعر نازل/مستقر، أو OBV فوق متوسطه 20.
  4) دخول: إغلاق اليوم فوق قمة أمس وفوق EMA 9 (أول يوم ارتداد).

الوقف = أدنى قاع في آخر 5 أيام. الهدف = استرداد 50% من الهبوط (من قمة 60 يوماً).

طبقة العقود الدائمة (perp_filter.py) لا تحذف أي إشارة — ترتّبها وتعلّمها:
  🔥 تمويل سالب (شورتات مزدحمة) + فائدة مفتوحة صاعدة = أقوى تأكيد
  ⚠️ تمويل موجب متطرف = الارتداد قد يكون مستهلكاً
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from perp_filter import PerpData
from swing_options import suggest, render

# ------------------------------------------------------------------
DROP_MIN       = 0.15     # هبوط ≥ 15% من قمة 60 يوماً
SUPPORT_BAND   = 0.04     # ±4% من الدعم
RSI_OVERSOLD   = 32
MIN_PRICE      = 5.0
MIN_DOLLAR_VOL = 20_000_000
LOG_DIR        = "logs/bottom"
SIGNALS_CSV    = os.path.join(LOG_DIR, "signals.csv")

# نفس كون SMC اليومي: أسهم سائلة لها أوبشن + S&P 500
BASE = """
AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO NFLX CRM ORCL ADBE INTC MU QCOM TXN AMAT LRCX KLAC
CSCO IBM ACN NOW PANW CRWD SNOW DDOG NET ZS MDB SHOP PYPL COIN HOOD SOFI UBER ABNB DASH PLTR
RBLX SNAP PINS ROKU TTD APP ANET DELL SMCI WDC STX HPQ F GM NIO RIVN LCID JPM BAC C WFC GS MS
SCHW XOM CVX OXY SLB HAL DVN COP KO PEP WMT COST TGT HD LOW NKE SBUX MCD DIS T VZ PFE MRK JNJ
ABBV LLY UNH CVS BA CAT DE GE LMT RTX MARA RIOT CLSK IREN CCJ FCX AA CLF AAL DAL UAL CCL
ARM MRVL ONON CELH AFRM UPST DKNG CHWY ETSY W LYFT RKLB ASTS IONQ RGTI SOUN BBAI HIMS OKLO
CRWV CRCL NBIS MSTR SNDK QBTS COHR
""".split()


def universe() -> list[str]:
    names = set(BASE)
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        sp = pd.read_html(io.StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False)
        names |= set(sp)
    except Exception as e:  # noqa: BLE001
        print(f"S&P 500 list unavailable: {e}", file=sys.stderr)
    return sorted(names)


# ------------------------------------------------------------------
def rsi(c: pd.Series, n=14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def obv(c: pd.Series, v: pd.Series) -> pd.Series:
    return (np.sign(c.diff().fillna(0)) * v).cumsum()


def slope(x: np.ndarray) -> float:
    return float(np.polyfit(np.arange(len(x)), x, 1)[0])


def bullish_divergence(low: pd.Series, r: pd.Series, window=30, L=3) -> bool:
    """قاعان محوريان خلال آخر window يوم: الثاني أدنى سعراً وأعلى RSI."""
    lo, rr = low.to_numpy()[-window:], r.to_numpy()[-window:]
    piv = [k for k in range(L, len(lo) - 1)
           if lo[k] <= lo[max(0, k - L):k + L + 1].min()]
    if len(piv) < 2:
        return False
    a, b = piv[-2], piv[-1]
    return lo[b] < lo[a] and rr[b] > rr[a] and b >= len(lo) - 8


def detect(df: pd.DataFrame) -> dict | None:
    if len(df) < 210:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    i = len(df) - 1
    close = float(c.iloc[i])
    if close < MIN_PRICE or float((c * v).iloc[-20:].mean()) < MIN_DOLLAR_VOL:
        return None

    # 1) هبوط + دعم
    peak = float(h.iloc[-60:].max())
    drop = 1 - float(l.iloc[-5:].min()) / peak
    if drop < DROP_MIN:
        return None
    recent_low = float(l.iloc[-5:].min())
    support = float(l.iloc[-130:-10].min())
    sma200 = float(c.rolling(200).mean().iloc[i])
    near = lambda lvl: abs(recent_low / lvl - 1) <= SUPPORT_BAND  # noqa: E731
    at_support = near(support) or near(sma200)
    if not at_support:
        return None

    # 2) إرهاق بيع
    r = rsi(c)
    rsi_min = float(r.iloc[-5:].min())
    div = bullish_divergence(l, r)
    if not (rsi_min <= RSI_OVERSOLD or div):
        return None

    # 3) تجميع OBV
    o = obv(c, v)
    obv_up = slope(o.iloc[-10:].to_numpy()) > 0 and slope(c.iloc[-10:].to_numpy()) <= 0
    obv_above = float(o.iloc[i]) > float(o.ewm(span=20).mean().iloc[i])
    if not (obv_up or obv_above):
        return None

    # 4) دخول: فوق قمة أمس و EMA 9
    ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[i])
    if not (close > float(h.iloc[i - 1]) and close > ema9):
        return None

    stop = recent_low
    target = recent_low + 0.5 * (peak - recent_low)
    if target <= close or stop >= close:
        return None
    rr = (target - close) / (close - stop)
    return dict(close=close, stop=stop, target=target, rr=rr, drop=drop * 100,
                rsi_min=rsi_min, div=div, obv="صاعد مع سعر نازل" if obv_up else "فوق متوسطه",
                support="دعم 120 يوم" if near(support) else "متوسط 200",
                risk_pct=(close - stop) / close * 100, date=str(df.index[i].date()))


# ------------------------------------------------------------------
def notify(text: str):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not token or not chat:
        print(text)
        return
    for k in range(0, len(text), 3900):  # حد تيليجرام 4096 حرف
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text[k:k + 3900], "parse_mode": "HTML",
                                "disable_web_page_preview": "true"}, timeout=30)
        if not r.ok:
            print("Telegram error:", r.text, file=sys.stderr)


def already_logged(t, date):
    if not os.path.exists(SIGNALS_CSV):
        return False
    s = pd.read_csv(SIGNALS_CSV)
    return bool(((s.ticker == t) & (s.signal_date == date)).any())


def main():
    perp = PerpData.load()          # طلب واحد + لقطة اليوم في logs/perp/
    tickers = universe()
    print(f"صيد القاع — {len(tickers)} سهم • عقود دائمة: {len(perp.rows)} أصل", flush=True)
    hits, loaded = [], 0
    for k in range(0, len(tickers), 100):
        batch = tickers[k:k + 100]
        raw = yf.download(batch, period="2y", interval="1d", group_by="ticker",
                          auto_adjust=False, progress=False, threads=True)
        for t in batch:
            try:
                df = raw[t].dropna() if len(batch) > 1 else raw.dropna()
            except (KeyError, TypeError):
                continue
            now_ny = pd.Timestamp.now(tz="America/New_York")
            if len(df) and df.index[-1].date() == now_ny.date() and now_ny.hour < 16:
                df = df.iloc[:-1]
            if len(df) >= 210:
                loaded += 1
            try:
                s = detect(df)
            except Exception as e:  # noqa: BLE001
                print(f"{t}: {e}", file=sys.stderr)
                continue
            if s:
                p = perp.get(t)
                s["perp_score"] = p["score"] if p else 0
                s["perp_tags"] = "|".join(p["tags"]) if p else ("" if perp.ok else "n/a")
                hits.append((t, s))
        print(f"  {min(k + 100, len(tickers))}/{len(tickers)} — إشارات: {len(hits)}", flush=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "last_run.txt"), "w", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC • universe {len(tickers)} • loaded {loaded} • "
                f"signals {len(hits)} • perp {'ok' if perp.ok else 'down'} ({len(perp.rows)})\n")

    rows, lines = [], []
    # الترتيب: تأكيد العقود أولاً، ثم R:R
    for t, s in sorted(hits, key=lambda x: (-x[1]["perp_score"], -x[1]["rr"])):
        if already_logged(t, s["date"]):
            continue
        try:
            o = suggest(t, "CALL", s["close"], stop=s["stop"], target=s["target"])
        except Exception as e:  # noqa: BLE001
            print(f"options {t}: {e}", file=sys.stderr)
            o = {"reason": "تعذّر جلب السلسلة"}
        opt_ok = o and "reason" not in o
        rows.append(dict(
            ticker=t, signal_date=s["date"], close=round(s["close"], 4), stop=round(s["stop"], 4),
            target=round(s["target"], 4), rr=round(s["rr"], 2), drop_pct=round(s["drop"], 1),
            rsi_min=round(s["rsi_min"], 1), divergence=s["div"],
            perp_score=s["perp_score"], perp_tags=s["perp_tags"],
            opt_exp=o["exp"] if opt_ok else "", opt_strike=o["leg"]["strike"] if opt_ok else np.nan,
            opt_entry=round(o["leg"]["ask"], 2) if opt_ok else np.nan))
        exhaustion = f"RSI نزل لـ {s['rsi_min']:.0f}" + (" + دايفرجنس" if s["div"] else "")
        lines += [
            f"<b>{t}</b>  إغلاق {s['close']:.2f}  (هبوط {s['drop']:.0f}% من القمة)",
            f"   {s['support']} · {exhaustion} · OBV {s['obv']}",
            f"   الوقف {s['stop']:.2f} (−{s['risk_pct']:.1f}%) · الهدف {s['target']:.2f} · R:R {s['rr']:.1f}",
        ] + perp.render(t) + render(o) + [""]

    if rows:
        pd.DataFrame(rows).to_csv(SIGNALS_CSV, mode="a", header=not os.path.exists(SIGNALS_CSV), index=False)
        head = [f"🎣 <b>صيد القاع — {len(rows)} إشارة</b>  ({rows[0]['signal_date']})",
                "مرتبة حسب تأكيد العقود الدائمة ثم R:R • الدخول على افتتاح الغد", ""]
        notify("\n".join(head + lines) +
               "<i>Data: PerpEquities (perpequities.com) • ليست توصية.</i>")
    else:
        print("لا إشارات قاع اليوم.")


if __name__ == "__main__":
    main()
