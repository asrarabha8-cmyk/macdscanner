#!/usr/bin/env python3
"""
MA 100/200 Breakout Scanner — فحص يومي للسوق الأمريكي كاملاً.

الشروط:
  ١) السهم أغلق اليوم فوق متوسط 100 وكان أمس تحته   (اختراق طازج)
  ٢) فوليوم اليوم ≥ ضعف متوسط 20 يوماً
  ٣) متوسط 200 فوق متوسط 100   (حتى يكون الهدف فوق سعر الدخول)

الدخول = إغلاق اليوم · الهدف = متوسط 200 · الوقف = قاع شمعة الاختراق
بلا حد سعري أعلى — الأسهم الرخيصة غير مستثناة.
"""

import io
import os
import sys
import datetime as dt

import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# الإعدادات — كل الافتراضات هنا، تغييرها سطر واحد
# ------------------------------------------------------------------
MA_FAST, MA_SLOW = 100, 200

VOL_MULT       = 2.0          # فوليوم الاختراق مقابل متوسط 20 يوماً
REQUIRE_ORDER  = True         # اشترط متوسط 200 فوق متوسط 100
MIN_PRICE      = 1.0          # بلا حد أعلى
MIN_DOLLAR_VOL = 3_000_000    # سيولة يومية دنيا، لاستبعاد ما لا يُتداول فعلياً
MIN_R          = 1.0          # تجاهل الإشارات التي هدفها أقل من مخاطرتها
MAX_ROWS       = 20           # أقصى عدد إشارات في رسالة تيليجرام

INCLUDE_ETFS   = False
CHUNK          = 150          # عدد الرموز في كل طلب تحميل

NASDAQ_FILES = [
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
]

FALLBACK = """
AAPL MSFT NVDA AMZN META GOOGL TSLA AMD INTC MU MARA RIOT CLSK HUT IREN
SOFI PLTR NIO F GM BBAI OUST ONDS POET BZAI NRXS CBRS ACLS VECO SEDG
""".split()


# ------------------------------------------------------------------
# قائمة الرموز — السوق الأمريكي كاملاً
# ------------------------------------------------------------------
def universe() -> list[str]:
    syms: set[str] = set()
    for url, col in NASDAQ_FILES:
        try:
            txt = requests.get(url, timeout=60).text
            df = pd.read_csv(io.StringIO(txt), sep="|")
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]
            if not INCLUDE_ETFS and "ETF" in df.columns:
                df = df[df["ETF"] != "Y"]
            for s in df[col].dropna().astype(str):
                s = s.strip()
                # استبعاد الوحدات والحقوق والممتازة وسطر التذييل
                if s and s.isalpha() and len(s) <= 5:
                    syms.add(s)
        except Exception as e:
            print(f"تعذّر جلب {url}: {e}", file=sys.stderr)
    if not syms:
        print("رجوع للقائمة الاحتياطية", file=sys.stderr)
        return FALLBACK
    return sorted(syms)


# ------------------------------------------------------------------
# فحص سهم واحد
# ------------------------------------------------------------------
def check(df: pd.DataFrame) -> dict | None:
    if len(df) < MA_SLOW + 25:
        return None

    close = df["Close"]
    ma_f = close.rolling(MA_FAST).mean()
    ma_s = close.rolling(MA_SLOW).mean()
    if pd.isna(ma_s.iloc[-1]) or pd.isna(ma_f.iloc[-2]):
        return None

    price = float(close.iloc[-1])
    if price < MIN_PRICE:
        return None

    # ١) اختراق طازج لمتوسط 100
    if not (close.iloc[-1] > ma_f.iloc[-1] and close.iloc[-2] <= ma_f.iloc[-2]):
        return None

    # ٢) الفوليوم
    vol = float(df["Volume"].iloc[-1])
    vol_avg = float(df["Volume"].iloc[-21:-1].mean())
    if vol_avg <= 0 or vol < vol_avg * VOL_MULT:
        return None
    if price * vol_avg < MIN_DOLLAR_VOL:
        return None

    # ٣) ترتيب المتوسطات — الهدف لازم يكون فوق السعر
    target = float(ma_s.iloc[-1])
    if REQUIRE_ORDER and target <= float(ma_f.iloc[-1]):
        return None
    if target <= price:
        return None

    stop = float(df["Low"].iloc[-1])
    risk = price - stop
    if risk <= 0:
        return None

    reward = target - price
    r = reward / risk
    if r < MIN_R:
        return None

    return {
        "price": price,
        "stop": stop,
        "target": target,
        "risk_pct": risk / price * 100,
        "upside_pct": reward / price * 100,
        "r": r,
        "vol_x": vol / vol_avg,
    }


# ------------------------------------------------------------------
# الفحص على دفعات
# ------------------------------------------------------------------
def scan(tickers: list[str]) -> list[tuple[str, dict]]:
    hits: list[tuple[str, dict]] = []
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            raw = yf.download(batch, period="2y", interval="1d",
                              group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
        except Exception as e:
            print(f"فشل تحميل دفعة: {e}", file=sys.stderr)
            continue

        for t in batch:
            try:
                df = raw[t].dropna() if len(batch) > 1 else raw.dropna()
            except (KeyError, TypeError):
                continue
            if df.empty:
                continue
            try:
                res = check(df)
            except Exception:
                continue
            if res:
                hits.append((t, res))

        done = min(i + CHUNK, len(tickers))
        print(f"  فُحص {done}/{len(tickers)} — إشارات حتى الآن: {len(hits)}", flush=True)
    return hits


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
    today = dt.date.today()
    tickers = universe()
    print(f"فحص {len(tickers)} رمز — {today}", flush=True)

    hits = scan(tickers)
    if not hits:
        print("لا اختراقات اليوم.")
        return

    hits.sort(key=lambda x: -x[1]["r"])
    lines = [f"<b>اختراق متوسط 100 — {today}</b>",
             f"عدد الإشارات: {len(hits)}", ""]
    for t, d in hits[:MAX_ROWS]:
        lines += [
            f"<b>{t}</b>  {d['price']:.2f}",
            f"الهدف (متوسط 200) {d['target']:.2f}  (+{d['upside_pct']:.1f}%)",
            f"الوقف {d['stop']:.2f}  (-{d['risk_pct']:.1f}%)  ·  R = {d['r']:.1f}",
            f"فوليوم {d['vol_x']:.1f}× المتوسط",
            "",
        ]
    if len(hits) > MAX_ROWS:
        lines.append(f"و{len(hits) - MAX_ROWS} إشارة أخرى لم تُعرض — رتّبها R.")
    notify("\n".join(lines))


if __name__ == "__main__":
    main()
