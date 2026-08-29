#!/usr/bin/env python3
"""
MACD Cascade Scanner — يفحص أسهماً أمريكية سائلة ويرسل إشارات الدخول إلى تيليجرام.
نفس منطق مؤشر macd_cascade.pine:
  الحالة ١ راقب  : ماكد 4H تحت الصفر، منحنٍ للأعلى، قريب من الصفر
  الحالة ٢ إذن   : ماكد 4H قطع الصفر صاعداً وما زال طازجاً وغير متذبذب
  الحالة ٣ دخول  : إذن مفتوح + تأكيد 1H + تقاطع صاعد على 15m
"""

import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# الإعدادات
# ------------------------------------------------------------------
FAST, SLOW, SIGNAL = 12, 26, 9

PERMIT_BARS   = 10     # عمر الإذن بشمعات 4H
CHOP_LOOK     = 50     # نافذة فحص التذبذب على 4H
CHOP_MAX      = 5      # أقصى عدد تقاطعات للصفر
DIST_CAP_PCT  = 60     # أقصى بُعد لماكد 4H عن الصفر (% من مداه)
ENTRY_TOL_PCT = 50     # حد قرب تقاطع 15m من الصفر
WATCH_TOL_PCT = 25     # حد "قريب من الصفر" لحالة راقب
SWING_LOOK    = 20     # نافذة القاع للستوب (شمعات 15m)
ATR_BUF       = 0.5    # هامش الستوب

MIN_DOLLAR_VOL = 20_000_000   # حد السيولة اليومي
MAX_RISK_PCT   = 3.0          # تجاهل الإشارات ذات الستوب الواسع

UNIVERSE = """
AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA AVGO AMD NFLX ADBE CRM ORCL CSCO INTC
QCOM TXN MU AMAT LRCX KLAC ADI INTU NOW PANW SNOW DDOG CRWD ZS NET MDB TEAM WDAY
SHOP XYZ PYPL COIN HOOD SOFI ABNB UBER LYFT DASH RBLX U PLTR SNAP PINS SPOT ROKU
TTD APP ANET DELL SMCI WDC STX HPQ IBM ACN INFY
JPM BAC WFC GS MS C SCHW BLK AXP V MA COF USB PNC TFC
UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD CVS CI VRTX REGN MRNA ISRG
XOM CVX COP SLB EOG PSX MPC VLO OXY HAL DVN FANG
WMT COST HD LOW TGT NKE SBUX MCD CMG PEP KO PG DIS BKNG MAR
BA CAT DE GE HON UNP UPS FDX LMT RTX NOC MMM EMR ETN PH
T VZ TMUS CMCSA
LIN APD SHW FCX NEM NUE
SPY QQQ IWM DIA SMH XLF XLE XLK XLV ARKK TQQQ SOXL
MSTR MARA RIOT CLSK HUT
"""
TICKERS = sorted(set(UNIVERSE.split()))


# ------------------------------------------------------------------
# أدوات
# ------------------------------------------------------------------
def macd(close: pd.Series):
    ema_f = close.ewm(span=FAST, adjust=False).mean()
    ema_s = close.ewm(span=SLOW, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=SIGNAL, adjust=False).mean()
    return line, sig


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """يجمّع شمعات الساعة إلى 4H بدءاً من افتتاح كل جلسة — مثل تقسيم TradingView للأسهم."""
    idx_ny = df.index.tz_convert("America/New_York")
    out = []
    for _, g in df.groupby(idx_ny.date):
        g = g.sort_index()
        bucket = np.arange(len(g)) // 4
        agg = g.groupby(bucket).agg(
            Open=("Open", "first"), High=("High", "max"),
            Low=("Low", "min"), Close=("Close", "last"),
        )
        agg.index = [g.index[i * 4] for i in agg.index]
        out.append(agg)
    return pd.concat(out).sort_index() if out else pd.DataFrame()


def bars_since_cross_up(line: pd.Series) -> int:
    """كم شمعة مضت منذ آخر قطع صاعد للصفر."""
    crossed = (line > 0) & (line.shift(1) <= 0)
    hits = np.flatnonzero(crossed.to_numpy())
    return 9999 if len(hits) == 0 else len(line) - 1 - hits[-1]


def zero_crossings(line: pd.Series, look: int) -> int:
    tail = line.tail(look)
    return int(((tail > 0) != (tail.shift(1) > 0)).sum())


# ------------------------------------------------------------------
# التحليل
# ------------------------------------------------------------------
def analyse(h1: pd.DataFrame, m15: pd.DataFrame) -> dict | None:
    if len(h1) < 200 or len(m15) < 120:
        return None

    h4 = to_4h(h1)
    if len(h4) < 60:
        return None

    # الفاصل الكبير — نستخدم الشمعة المغلقة فقط
    m4, s4 = macd(h4["Close"])
    m4c, s4c = m4.iloc[-2], s4.iloc[-2]
    rng4 = m4.abs().tail(100).max()
    if rng4 <= 0:
        return None

    age = bars_since_cross_up(m4.iloc[:-1])
    chop = zero_crossings(m4.iloc[:-1], CHOP_LOOK)

    permit = (m4c > 0 and m4c > s4c and age <= PERMIT_BARS
              and chop <= CHOP_MAX and m4c <= rng4 * DIST_CAP_PCT / 100)
    watch = (m4c <= 0 and m4c > m4.iloc[-3] and abs(m4c) <= rng4 * WATCH_TOL_PCT / 100)

    if not permit:
        return {"state": "watch"} if watch else None

    # الفاصل الوسيط
    m1, s1 = macd(h1["Close"])
    if not m1.iloc[-2] > s1.iloc[-2]:
        return {"state": "permit", "age": age}

    # فاصل الدخول — التقاطع على آخر شمعة مغلقة
    m0, s0 = macd(m15["Close"])
    crossed = m0.iloc[-2] > s0.iloc[-2] and m0.iloc[-3] <= s0.iloc[-3]
    rng0 = m0.abs().tail(100).max()
    near0 = rng0 > 0 and abs(m0.iloc[-2]) <= rng0 * ENTRY_TOL_PCT / 100

    if not (crossed and near0):
        return {"state": "permit", "age": age}

    price = float(m15["Close"].iloc[-2])
    stop = float(m15["Low"].tail(SWING_LOOK).min() - ATR_BUF * atr(m15).iloc[-2])
    risk = price - stop
    if risk <= 0:
        return None
    risk_pct = risk / price * 100
    if risk_pct > MAX_RISK_PCT:
        return None

    return {
        "state": "entry", "price": price, "stop": stop, "risk_pct": risk_pct,
        "tp1": price + risk, "tp2": price + 2 * risk, "tp3": price + 3 * risk,
        "age": age, "chop": chop,
    }


# ------------------------------------------------------------------
# البيانات
# ------------------------------------------------------------------
def fetch(tickers: list[str], period: str, interval: str) -> dict:
    raw = yf.download(tickers, period=period, interval=interval,
                      group_by="ticker", auto_adjust=False, prepost=False,
                      progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t].dropna() if len(tickers) > 1 else raw.dropna()
        except (KeyError, TypeError):
            continue
        if df.empty:
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[t] = df
    return out


def liquid(h1: dict) -> list[str]:
    keep = []
    for t, df in h1.items():
        if "Volume" not in df:
            continue
        tail = df.tail(140)  # ~20 جلسة
        dv = float((tail["Close"] * tail["Volume"]).sum() / 20)
        if dv >= MIN_DOLLAR_VOL:
            keep.append(t)
    return keep


# ------------------------------------------------------------------
# التنبيه
# ------------------------------------------------------------------
def notify(text: str):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not token or not chat:
        print(text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=30,
    )
    if not r.ok:
        print("Telegram error:", r.text, file=sys.stderr)


def main():
    now = dt.datetime.now(dt.timezone.utc)
    print(f"scan {now:%Y-%m-%d %H:%M} UTC — {len(TICKERS)} رمز")

    h1 = fetch(TICKERS, "180d", "1h")
    names = liquid(h1)
    print(f"بعد فلتر السيولة: {len(names)}")
    if not names:
        return

    m15 = fetch(names, "10d", "15m")

    entries, permits = [], []
    for t in names:
        if t not in m15:
            continue
        try:
            res = analyse(h1[t], m15[t])
        except Exception as e:
            print(f"{t}: {e}", file=sys.stderr)
            continue
        if not res:
            continue
        if res["state"] == "entry":
            entries.append((t, res))
        elif res["state"] == "permit":
            permits.append(t)

    if not entries:
        print(f"لا إشارات دخول. إذن مفتوح على: {', '.join(permits) or 'لا شيء'}")
        return

    entries.sort(key=lambda x: x[1]["risk_pct"])
    lines = [f"<b>إشارات دخول — {now:%H:%M} UTC</b>", ""]
    for t, r in entries:
        lines += [
            f"<b>{t}</b>  {r['price']:.2f}",
            f"ستوب {r['stop']:.2f}  ({r['risk_pct']:.2f}%)",
            f"أهداف {r['tp1']:.2f} / {r['tp2']:.2f} / {r['tp3']:.2f}",
            f"عمر الإذن {r['age']}/{PERMIT_BARS} · تذبذب {r['chop']}/{CHOP_MAX}",
            "",
        ]
    if permits:
        lines.append("إذن مفتوح بلا تقاطع: " + ", ".join(permits[:15]))
    notify("\n".join(lines))


if __name__ == "__main__":
    main()
