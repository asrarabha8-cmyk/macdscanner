#!/usr/bin/env python3
"""
SMC اليومي — النسخة اللي نجحت في الاختبار التاريخي (backtest/strategy_compare.py، B1).

يُشغَّل مرة يومياً بعد إغلاق السوق الأمريكي، على أسهم سائلة لها أوبشن:
  1) فلاتر "فينفيز" على شمعة اليوم: فوليوم ≥ 2× متوسط 20 يوماً، ارتفاع > 5%، قمة 20 يوماً.
  2) BOS: إغلاق اليوم فوق آخر قمة هيكلية (A) وأمس كان تحتها.
  3) الوقف = القاع الهيكلي B بعد A. الهدف = B + 1.618 × الساق (قاع ← A). R:R ≥ 1.5.
  4) الدخول على افتتاح الغد — تُلغى إذا فتح تحت الوقف أو فوق الهدف.
  5) خروج: الوقف أو الهدف أو بعد 40 يوم تداول (نفس الاختبار).

كل إشارة تُسجَّل في logs/smc_daily/signals.csv مع العقد المقترح، و smc_daily_track.py
يتابعها يومياً على الورق (السهم والعقد) حتى تُغلق.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from swing_options import suggest, render

# ------------------------------------------------------------------
PIVOT_L       = 3
MIN_RR        = 1.5
FIB           = 1.618
RELVOL_MIN    = 2.0
CHANGE_MIN    = 5.0       # %
MIN_PRICE     = 5.0
MIN_DOLLAR_VOL = 20_000_000
MAX_DAYS      = 40
LOG_DIR       = "logs/smc_daily"
SIGNALS_CSV   = os.path.join(LOG_DIR, "signals.csv")

# نفس 109 سهم في الاختبار + أسهم S&P 500 (تُحمَّل تلقائياً) — كلها لها أوبشن سائل غالباً
BASE = """
AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO NFLX CRM ORCL ADBE INTC MU QCOM TXN AMAT LRCX KLAC
CSCO IBM ACN NOW PANW CRWD SNOW DDOG NET ZS MDB SHOP PYPL COIN HOOD SOFI UBER ABNB DASH PLTR
RBLX SNAP PINS ROKU TTD APP ANET DELL SMCI WDC STX HPQ F GM NIO RIVN LCID JPM BAC C WFC GS MS
SCHW XOM CVX OXY SLB HAL DVN COP KO PEP WMT COST TGT HD LOW NKE SBUX MCD DIS T VZ PFE MRK JNJ
ABBV LLY UNH CVS BA CAT DE GE LMT RTX MARA RIOT CLSK IREN CCJ FCX AA CLF AAL DAL UAL CCL
ARM MRVL ONON CELH AFRM UPST DKNG CHWY ETSY W LYFT RKLB ASTS IONQ RGTI SOUN BBAI HIMS OKLO
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
def pivots(h, l, L=PIVOT_L):
    ph, pl = [], []
    for k in range(L, len(h) - L):
        if h[k] >= h[k - L:k + L + 1].max():
            ph.append(k)
        if l[k] <= l[k - L:k + L + 1].min():
            pl.append(k)
    return np.array(ph), np.array(pl)


def detect(df: pd.DataFrame) -> dict | None:
    """نفس منطق strat_smc(strict=True) في الاختبار — على آخر شمعة يومية مغلقة."""
    if len(df) < 80:
        return None
    h, l, c, v = (df[k].to_numpy(float) for k in ("High", "Low", "Close", "Volume"))
    i = len(df) - 1
    vavg = v[i - 20:i].mean()
    relvol = v[i] / vavg if vavg > 0 else 0
    chg = (c[i] / c[i - 1] - 1) * 100
    if not (relvol > RELVOL_MIN and chg > CHANGE_MIN and h[i] >= h[i - 19:i + 1].max()
            and c[i] >= MIN_PRICE and c[i] * vavg >= MIN_DOLLAR_VOL):
        return None
    ph, pl = pivots(h, l)
    conf_h = ph[ph < i - PIVOT_L]
    if len(conf_h) == 0:
        return None
    a = conf_h[-1]
    level = h[a]
    if not (c[i] > level and c[i - 1] <= level):
        return None
    conf_l = pl[pl <= i - PIVOT_L]
    after_a = conf_l[conf_l > a]
    if len(after_a):
        b = after_a[-1]
    elif len(conf_l):
        b = conf_l[-1]
    else:
        return None
    stop = l[b]
    leg = level - l[max(0, a - 40):a + 1].min()
    if stop >= c[i] or leg <= 0:
        return None
    target = stop + leg * FIB
    rr = (target - c[i]) / (c[i] - stop)
    if rr < MIN_RR:
        return None
    return dict(close=c[i], stop=float(stop), target=float(target), rr=float(rr),
                risk_pct=(c[i] - stop) / c[i] * 100, relvol=float(relvol), change=float(chg),
                bos=float(level), date=str(df.index[i].date()))


# ------------------------------------------------------------------
def notify(text: str):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not token or not chat:
        print(text)
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text, "parse_mode": "HTML",
                            "disable_web_page_preview": "true"}, timeout=30)
    if not r.ok:
        print("Telegram error:", r.text, file=sys.stderr)


def already_logged(t, date):
    if not os.path.exists(SIGNALS_CSV):
        return False
    s = pd.read_csv(SIGNALS_CSV)
    return bool(((s.ticker == t) & (s.signal_date == date)).any())


def open_tickers():
    if not os.path.exists(SIGNALS_CSV):
        return set()
    s = pd.read_csv(SIGNALS_CSV)
    return set(s[s.status.isin(["pending", "open"])].ticker)


def main():
    lookback = int(os.getenv("LOOKBACK", "0") or 0)    # >0: كشف إشارات آخر N يوم بدون تسجيل
    tickers = universe()
    print(f"SMC يومي — {len(tickers)} سهم", flush=True)
    hits, loaded, past = [], 0, []
    for k in range(0, len(tickers), 100):
        batch = tickers[k:k + 100]
        raw = yf.download(batch, period="1y", interval="1d", group_by="ticker",
                          auto_adjust=False, progress=False, threads=True)
        for t in batch:
            try:
                df = raw[t].dropna() if len(batch) > 1 else raw.dropna()
            except (KeyError, TypeError):
                continue
            # استبعد شمعة اليوم إذا كانت ناقصة (تشغيل أثناء الجلسة)
            now_ny = pd.Timestamp.now(tz="America/New_York")
            if len(df) and df.index[-1].date() == now_ny.date() and now_ny.hour < 16:
                df = df.iloc[:-1]
            if len(df) >= 80:
                loaded += 1
            if lookback:
                for i in range(max(80, len(df) - lookback), len(df)):
                    try:
                        x = detect(df.iloc[:i + 1])
                    except Exception:  # noqa: BLE001
                        x = None
                    if x:
                        past.append(dict(ticker=t, **x))
                continue
            try:
                s = detect(df)
            except Exception as e:  # noqa: BLE001
                print(f"{t}: {e}", file=sys.stderr)
                continue
            if s:
                hits.append((t, s))
        print(f"  {min(k + 100, len(tickers))}/{len(tickers)} — إشارات: {len(hits)}", flush=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "last_run.txt"), "w", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC • universe {len(tickers)} • loaded {loaded} • "
                f"signals {len(hits)}" + (f" • lookback {lookback}d: {len(past)}" if lookback else "") + "\n")
    if lookback:
        pd.DataFrame(past).to_csv(os.path.join(LOG_DIR, "lookback.csv"), index=False)
        print(f"lookback {lookback}d: {len(past)} signals")
        return
    busy = open_tickers()
    rows, lines = [], []
    for t, s in sorted(hits, key=lambda x: -x[1]["rr"]):
        if t in busy or already_logged(t, s["date"]):
            continue
        try:
            o = suggest(t, "CALL", s["close"], stop=s["stop"], target=s["target"])
        except Exception as e:  # noqa: BLE001
            print(f"options {t}: {e}", file=sys.stderr)
            o = {"reason": "تعذّر جلب السلسلة"}
        opt_ok = o and "reason" not in o
        rows.append(dict(
            ticker=t, signal_date=s["date"], close=round(s["close"], 4), stop=round(s["stop"], 4),
            target=round(s["target"], 4), rr=round(s["rr"], 2), risk_pct=round(s["risk_pct"], 2),
            status="pending", entry_date="", entry=np.nan, exit_date="", exit=np.nan,
            reason="", r=np.nan, days=np.nan,
            opt_exp=o["exp"] if opt_ok else "", opt_strike=o["leg"]["strike"] if opt_ok else np.nan,
            opt_entry=round(o["leg"]["ask"], 2) if opt_ok else np.nan,
            opt_last=np.nan, opt_exit=np.nan, opt_pnl=np.nan))
        lines += [
            f"<b>{t}</b>  إغلاق {s['close']:.2f}  (+{s['change']:.1f}% · فوليوم ×{s['relvol']:.1f})",
            f"   كسر هيكلي فوق {s['bos']:.2f}",
            f"   الوقف {s['stop']:.2f} (−{s['risk_pct']:.1f}%) · الهدف {s['target']:.2f} · R:R {s['rr']:.1f}",
        ] + render(o) + [""]

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(SIGNALS_CSV, mode="a", header=not os.path.exists(SIGNALS_CSV), index=False)
        head = [f"🟢 <b>SMC يومي — {len(rows)} إشارة</b>  ({rows[0]['signal_date']})",
                "الدخول على افتتاح الغد • يُلغى إذا فتح تحت الوقف أو فوق الهدف • أقصى مدة 40 يوم تداول", ""]
        notify("\n".join(head + lines) + "<i>اختبار ورقي — كل إشارة تُتابَع تلقائياً حتى تُغلق.</i>")
    else:
        print("لا إشارات SMC يومية اليوم.")


if __name__ == "__main__":
    main()
