#!/usr/bin/env python3
"""
EMA Cross Scanner — يبحث عن تقاطع EMA8 مع EMA48 على الفاصل الأسبوعي.

مستقل تماماً عن macd_cascade_scanner.py: قائمته وشمعاته ورسالته منفصلة.
يعمل مرة واحدة أسبوعياً بعد إغلاق الجمعة، لأن التقاطع لا يتأكد إلا بإغلاق
الشمعة الأسبوعية. الفحص يجري على آخر شمعة مغلقة فقط.
"""

import os
import sys
import datetime as dt

import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# الإعدادات
# ------------------------------------------------------------------
FAST_EMA = 8
SLOW_EMA = 48
MIN_DOLLAR_VOL = 20_000_000     # حد السيولة اليومي التقريبي
MIN_WEEKS = 60                  # أقل تاريخ مطلوب ليستقر EMA48

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

# مراكزي — من نفس Secret المستخدم في الماسح الآخر
PORTFOLIO = os.environ.get("PORTFOLIO", "")
if not PORTFOLIO.strip():
    print("تنبيه: متغير PORTFOLIO فارغ — تُفحص القائمة العامة فقط.", file=sys.stderr)

CRYPTO = {"BTC-USD", "ETH-USD", "SOL-USD"}
ALWAYS = set(PORTFOLIO.split()) | CRYPTO
TICKERS = sorted(set(UNIVERSE.split()) | ALWAYS)


# ------------------------------------------------------------------
# أدوات
# ------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def check(df: pd.DataFrame) -> dict | None:
    """يفحص آخر شمعة أسبوعية مغلقة بحثاً عن تقاطع طازج."""
    if len(df) < MIN_WEEKS:
        return None

    close = df["Close"]
    f = ema(close, FAST_EMA)
    s = ema(close, SLOW_EMA)

    # -2 آخر شمعة مغلقة، -1 الشمعة الجارية غير المكتملة
    up = f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]
    dn = f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]
    if not (up or dn):
        return None

    price = float(close.iloc[-2])
    gap = abs(f.iloc[-2] - s.iloc[-2]) / price * 100

    return {
        "dir": "up" if up else "down",
        "price": price,
        "ema_fast": float(f.iloc[-2]),
        "ema_slow": float(s.iloc[-2]),
        "gap_pct": gap,
        "date": df.index[-2],
    }


def liquid(t: str, df: pd.DataFrame) -> bool:
    if t in ALWAYS:
        return True
    if "Volume" not in df:
        return False
    tail = df.tail(8)
    weekly_dv = float((tail["Close"] * tail["Volume"]).mean())
    return weekly_dv / 5 >= MIN_DOLLAR_VOL


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


def fmt(x: float) -> str:
    return f"{x:,.4f}" if abs(x) < 10 else f"{x:,.2f}"


# ------------------------------------------------------------------
def main():
    now = dt.datetime.now(dt.timezone.utc)
    print(f"ema-cross {now:%Y-%m-%d %H:%M} UTC — {len(TICKERS)} رمز")

    raw = yf.download(TICKERS, period="5y", interval="1wk", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)

    ups, downs, skipped = [], [], 0
    for t in TICKERS:
        try:
            df = raw[t].dropna()
        except (KeyError, TypeError):
            skipped += 1
            continue
        if df.empty or not liquid(t, df):
            skipped += 1
            continue
        try:
            res = check(df)
        except Exception as e:
            print(f"{t}: {e}", file=sys.stderr)
            continue
        if not res:
            continue
        (ups if res["dir"] == "up" else downs).append((t, res))

    print(f"تُخطّي {skipped} رمزاً · تقاطع صاعد {len(ups)} · هابط {len(downs)}")

    if not ups and not downs:
        print("لا تقاطعات هذا الأسبوع.")
        return

    lines = [f"<b>تقاطع EMA {FAST_EMA}/{SLOW_EMA} — أسبوعي</b>", ""]

    if ups:
        lines.append("🟢 <b>تقاطع صاعد</b>")
        for t, r in sorted(ups, key=lambda x: -x[1]["gap_pct"]):
            lines.append(f"<b>{t}</b>  {fmt(r['price'])}  ·  فجوة {r['gap_pct']:.2f}%")
        lines.append("")

    if downs:
        lines.append("🔴 <b>تقاطع هابط</b>")
        for t, r in sorted(downs, key=lambda x: -x[1]["gap_pct"]):
            lines.append(f"<b>{t}</b>  {fmt(r['price'])}  ·  فجوة {r['gap_pct']:.2f}%")

    notify("\n".join(lines))


if __name__ == "__main__":
    main()
