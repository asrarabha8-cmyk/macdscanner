#!/usr/bin/env python3
"""
Backtest of the "10 years of trading in 42 seconds" video strategy (Livermore100 on X, Oct 2026).

Video rules:
  1) 9:30 NY: wait for the first 15-minute candle -> range high / range low
  2) Drop to 5-min: wait for a 5-min candle to CLOSE outside the range -> direction (buys above, sells below)
  3) Do not chase: mark recent support (the range high that was broken / trendline) and enter
     when price comes BACK to that level (retest)
  4) Stop below support (range low), target a fixed multiple of risk (the box in the video ~2R)

Variants compared on the same days:
  breakout : enter at the close of the breakout candle (what spy_orb_0dte.py does)
  retest   : limit order at the broken range edge after the breakout close (the video)
Each with and without the compression filter (range < 0.8 x its 20-day average).
Stop = opposite range edge; target = TARGET_R x risk; otherwise exit at 15:55 close.
Signals only until 12:00 NY; a retest must happen by 12:30 NY.
Same-bar ambiguity is resolved against us (stop first). Results in R, before options.

  python video_orb_backtest.py --tickers SPY QQQ --years 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from intraday_momentum import polygon_5m

TARGET_R = float(os.getenv("TARGET_R", "2"))
COST_R = float(os.getenv("COST_R", "0.05"))          # slippage/commission per trade, in R
LAST_SIGNAL = "12:00"
LAST_RETEST = "12:30"


def day_trades(g: pd.DataFrame, compress_ok: bool) -> list[dict]:
    t = g.index.strftime("%H:%M")
    if len(g) < 70 or t[0] != "09:30":
        return []
    rng = g.iloc[:3]
    hi, lo = rng["high"].max(), rng["low"].min()
    if hi <= lo:
        return []
    out = []
    after = g.iloc[3:]
    ta = after.index.strftime("%H:%M")
    brk = None
    for k in range(len(after)):
        if ta[k] > LAST_SIGNAL:
            break
        c = after["close"].iloc[k]
        if c > hi:
            brk = (k, 1)
            break
        if c < lo:
            brk = (k, -1)
            break
    if brk is None:
        return []
    k0, d = brk
    stop = lo if d == 1 else hi

    def run(entry, start):
        risk = (entry - stop) * d
        if risk <= 0:
            return None
        tgt = entry + d * TARGET_R * risk
        for j in range(start, len(after)):
            bh, bl = after["high"].iloc[j], after["low"].iloc[j]
            hit_stop = bl <= stop if d == 1 else bh >= stop
            hit_tgt = bh >= tgt if d == 1 else bl <= tgt
            if hit_stop:
                return -1.0
            if hit_tgt:
                return TARGET_R
        return (after["close"].iloc[-1] - entry) * d / risk

    # breakout: enter at breakout close, manage from next bar
    r = run(after["close"].iloc[k0], k0 + 1)
    if r is not None:
        out.append(dict(variant="breakout", dir=d, r=r - COST_R, compress=compress_ok))
    # retest: limit at broken edge
    level = hi if d == 1 else lo
    for j in range(k0 + 1, len(after)):
        if ta[j] > LAST_RETEST:
            break
        bh, bl = after["high"].iloc[j], after["low"].iloc[j]
        touched = bl <= level if d == 1 else bh >= level
        if touched:
            # fill at level; if the same bar also hits the stop -> loss
            same_stop = bl <= stop if d == 1 else bh >= stop
            if same_stop:
                rr = -1.0
            else:
                rr = run(level, j + 1)
            if rr is not None:
                out.append(dict(variant="retest", dir=d, r=rr - COST_R, compress=compress_ok))
            break
    return out


def stats(r: pd.Series) -> dict:
    if r.empty:
        return dict(n=0)
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), win=round((r > 0).mean() * 100, 1), avgR=round(r.mean(), 3),
                totR=round(r.sum(), 1), pf=round(wins / losses, 2) if losses else np.inf,
                maxDD_R=round(dd, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--years", type=float, default=2)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    key = os.getenv("POLYGON_API_KEY") or sys.exit("POLYGON_API_KEY missing")
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365 * a.years))
    os.makedirs(a.out, exist_ok=True)
    lines, allrows = [f"Video ORB backtest  {start.date()} → {end.date()}  target {TARGET_R}R  cost {COST_R}R", ""]
    for tk in a.tickers:
        print(f"== {tk}", flush=True)
        df = polygon_5m(tk, start, end, key)
        df = df.between_time("09:30", "15:55")
        days = [(d, g) for d, g in df.groupby(df.index.date)]
        widths = pd.Series({d: (g.iloc[:3]["high"].max() - g.iloc[:3]["low"].min()) / g["open"].iloc[0]
                            for d, g in days if len(g) >= 3})
        avg = widths.shift(1).rolling(20).mean()
        rows = []
        for d, g in days:
            ok = bool(d in avg.index and pd.notna(avg[d]) and widths[d] < 0.8 * avg[d])
            for tr in day_trades(g, ok):
                rows.append(dict(ticker=tk, date=d, **tr))
        t = pd.DataFrame(rows)
        if t.empty:
            continue
        allrows.append(t)
        t["year"] = pd.to_datetime(t["date"]).dt.year
        lines.append(f"### {tk}  ({len(days)} days)")
        for v in ("breakout", "retest"):
            for filt in (False, True):
                s = t[(t.variant == v) & (t.compress if filt else True)]
                lines.append(f"  {v:8s} {'+compress' if filt else '         '}  {stats(s.r)}")
        for y, s in t[t.variant == "retest"].groupby("year"):
            lines.append(f"     retest {y}: {stats(s.r)}")
        lines.append("")
    if allrows:
        pd.concat(allrows).to_csv(os.path.join(a.out, "video_orb_trades.csv"), index=False)
        T = pd.concat(allrows)
        lines.append("### ALL tickers")
        for v in ("breakout", "retest"):
            for filt in (False, True):
                s = T[(T.variant == v) & (T.compress if filt else True)]
                lines.append(f"  {v:8s} {'+compress' if filt else '         '}  {stats(s.r)}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, "video_orb_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        os.makedirs("results", exist_ok=True)
        with open("results/video_orb_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise
