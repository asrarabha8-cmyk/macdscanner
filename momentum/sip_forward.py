#!/usr/bin/env python3
"""
Forward (paper) test of the Stocks-in-Play opening-range breakout.

Runs once per trading day AFTER the close. Uses exactly the same rules as
sip_backtest.py, applied to a day that did not exist when the rules were fixed:
  universe  = top 100 common stocks by 20-day avg $volume (to yesterday), price > $5, ex-ETF
  in play   = top 10 by first-5-min-bar relative volume (> 1)
  trade     = breakout of first-candle high/low, stop at the other side, 10R target or close
Appends every trade to forward_log.csv, keeps a running scoreboard and sends a
Telegram report. Cost assumed: 2 bps per trade (same as the backtest).

  python sip_forward.py                 # today (New York date)
  python sip_forward.py --date 2026-10-02
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import sip_backtest as S

UNIVERSE, TOP, COST_BPS = 100, 10, 2.0
BACKTEST_BPS = 7.37          # expectation from the 3-year backtest (net bps/day)
OUT = "forward"
LOG = os.path.join(OUT, "forward_log.csv")
DAILY = os.path.join(OUT, "forward_daily.csv")
START = pd.Timestamp("2026-10-05")   # first forward day: after the backtest window (ended 2026-10-02)


def session_for(tk: str, day: pd.Timestamp) -> tuple[pd.DataFrame | None, float]:
    """Today's 5m session for tk and its first-bar relative volume vs the prior 14 sessions."""
    bars = S.bars_5m(tk, day - pd.Timedelta(days=30), day)
    if bars.empty:
        return None, np.nan
    sess = {d: g for d, g in bars.groupby(bars.index.normalize())}
    if day not in sess:
        return None, np.nan
    first = pd.Series({d: g["volume"].iloc[0] for d, g in sess.items() if g.index[0].strftime("%H:%M") == "09:30"})
    prior = first[first.index < day].iloc[-14:]
    g = sess[day]
    if len(prior) < 10 or len(g) < 70 or g.index[0].strftime("%H:%M") != "09:30" or prior.mean() <= 0:
        return None, np.nan
    return g, first[day] / prior.mean()


def telegram(text: str):
    tok, chat = os.getenv("TG_TOKEN", ""), os.getenv("TG_CHAT", "")
    if not tok or not chat:
        print("(telegram not configured)")
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=30)
    except Exception as e:
        print("telegram failed:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="run one specific day; default = catch up all missed days")
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    if not S.KEY:
        sys.exit("POLYGON_API_KEY is not set")
    os.makedirs(OUT, exist_ok=True)
    if a.date.strip():
        days = [pd.Timestamp(a.date).normalize()]
    else:
        now = pd.Timestamp.now(tz=S.NY).tz_localize(None)
        last = now.normalize() if now.hour * 60 + now.minute >= 16 * 60 + 20 else now.normalize() - pd.Timedelta(days=1)
        done = set(pd.read_csv(LOG)["date"]) if os.path.exists(LOG) else set()
        days = [d for d in pd.bdate_range(max(START, last - pd.Timedelta(days=10)), last)
                if d.date().isoformat() not in done]
    if not days:
        print("nothing to do (all sessions already logged)")
    for day in days[:5]:
        run_day(day, telegram_on=not a.no_telegram)


def run_day(day: pd.Timestamp, telegram_on: bool = True):
    os.makedirs(OUT, exist_ok=True)

    # 1) full-market daily bars: today + ~45 weekdays back (for the 20-day $volume ranking)
    cal = pd.bdate_range(day - pd.Timedelta(days=45), day)
    with ThreadPoolExecutor(8) as ex:
        frames = [f for f in ex.map(S.grouped_daily, cal) if not f.empty]
    daily = pd.concat(frames, ignore_index=True)
    if day not in set(daily["date"]):
        print(f"{day.date()}: no session (holiday/weekend or data not ready)")
        return
    daily = daily[daily["T"].str.match(S.SYM_OK)]
    try:
        etfs = S.etf_set()
    except Exception as e:
        print("ETF list unavailable:", e)
        etfs = set()
    daily = daily[~daily["T"].isin(etfs)].sort_values(["T", "date"])
    daily["dv"] = daily["c"] * daily["v"]
    g = daily.groupby("T")
    daily["adv20"] = g["dv"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    daily["prev_c"] = g["c"].shift(1)
    today = daily[(daily["date"] == day) & (daily["prev_c"] > 5) & daily["adv20"].notna()]
    universe = today.nlargest(UNIVERSE, "adv20")["T"].tolist()
    print(f"{day.date()}: universe {len(universe)} stocks")

    # 2) today's sessions + relative volume
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(universe, ex.map(lambda t: session_for(t, day), universe)))
    rows = []
    for tk, (sess, rvol) in res.items():
        if sess is None or not np.isfinite(rvol):
            continue
        r = {"date": day.date().isoformat(), "tk": tk, "rvol": rvol}
        for m in ("breakout_candle", "direction_open", "long_drift"):
            d = S.trade_detail(sess, m)
            r[m] = np.nan if d is None else d["ret"]
            if m == "breakout_candle" and d is not None:
                r.update(side=d["side"], entry=d["entry"], stop=d["stop"], exit=d["exit"], how=d["how"],
                         t_entry=d["t_entry"].strftime("%H:%M"))
        rows.append(r)
    allday = pd.DataFrame(rows)
    if allday.empty:
        print(f"{day.date()}: no usable sessions")
        return
    allday["in_play"] = False
    pick = allday[allday["rvol"] > 1].nlargest(TOP, "rvol").index
    allday.loc[pick, "in_play"] = True

    # 3) append to log (replace if this date was already run)
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if not log.empty:
        log = log[log["date"] != allday["date"].iloc[0]]
    log = pd.concat([log, allday], ignore_index=True).sort_values(["date", "in_play", "rvol"],
                                                                 ascending=[True, False, False])
    log.to_csv(LOG, index=False)

    # 4) daily scoreboard
    c = COST_BPS / 1e4
    def day_stats(df):
        ip = df[df["in_play"]]
        return {
            "date": df["date"].iloc[0],
            "inplay_bps": (ip["breakout_candle"].dropna() - c).mean() * 1e4 if ip["breakout_candle"].notna().any() else 0.0,
            "inplay_trades": int(ip["breakout_candle"].notna().sum()),
            "drift_bps": (ip["long_drift"].dropna() - c).mean() * 1e4,
            "all_bps": (df["breakout_candle"].dropna() - c).mean() * 1e4,
        }
    board = pd.DataFrame([day_stats(g) for _, g in log.groupby("date")])
    board.to_csv(DAILY, index=False)

    # 5) report
    t = log[(log["date"] == allday["date"].iloc[0]) & log["in_play"]]
    today_row = board[board["date"] == allday["date"].iloc[0]].iloc[0]
    n = len(board)
    s = board["inplay_bps"]
    tstat = s.mean() / (s.std() / np.sqrt(n)) if n > 2 and s.std() > 0 else float("nan")
    lines = [f"🎯 Stocks in Play ORB — اختبار ورقي | {day.date()}", ""]
    for _, r in t.iterrows():
        if pd.isna(r.get("breakout_candle")):
            lines.append(f"• {r['tk']}  (RVOL {r['rvol']:.1f}) — لا اختراق")
            continue
        arrow = "🟢 شراء" if r["side"] == 1 else "🔴 بيع"
        res = {"stop": "وقف", "target": "هدف 10R", "close": "إغلاق"}[r["how"]]
        lines.append(f"• {r['tk']} {arrow} {r['t_entry']} @ {r['entry']:.2f} | وقف {r['stop']:.2f} | "
                     f"خروج {r['exit']:.2f} ({res}) → {r['breakout_candle'] * 100:+.2f}%")
    lines += ["",
              f"📊 اليوم: {today_row['inplay_bps']:+.1f} نقطة أساس ({today_row['inplay_trades']} صفقات، بعد التكلفة)",
              f"   مقارنة: شراء بسيط {today_row['drift_bps']:+.1f} | كل المئة {today_row['all_bps']:+.1f}",
              "",
              f"📈 التراكمي ({n} يوم): متوسط {s.mean():+.1f} نقطة أساس/يوم | مجموع {s.sum() / 100:+.2f}%",
              f"   أيام رابحة {(s > 0).mean():.0%} | t = {tstat:.2f} | المتوقع من الاختبار الرجعي +{BACKTEST_BPS}",
              f"   شراء بسيط تراكمي {board['drift_bps'].mean():+.1f}/يوم"]
    if n < 40:
        lines.append(f"   (الحكم بعد ~40 يوماً — باقي {40 - n})")
    msg = "\n".join(lines)
    print(msg)
    if telegram_on:
        telegram(msg)


if __name__ == "__main__":
    main()
