#!/usr/bin/env python3
"""
5-minute Opening Range Breakout backtest
========================================
Based on Zarattini & Aziz (2023), "Can Day Trading Really Be Profitable?" (QQQ).

Paper rules (variant "paper"):
  * first 5-min candle 09:30–09:35 sets direction: green -> long, red -> short, doji -> no trade
  * enter at the open of the 09:35 candle
  * stop at the first candle's low (long) / high (short);  R = |entry - stop|
  * target 10R, otherwise exit at the 16:00 close

Classic variant ("breakout"):
  * enter only when price breaks the first candle's high (long) or low (short)
    later in the day (first break wins), same stop / 10R / close exit

Results are shown per trade in R and in % of price with 1x notional (no leverage),
before and after a cost of COST_BPS round trip. Intrabar ambiguity is resolved
conservatively: if a bar touches both stop and target, the stop is assumed first.

  python orb_backtest.py --tickers QQQ SPY --years 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from intraday_momentum import NY, polygon_5m

TARGET_R = 10.0


def simulate_day(g: pd.DataFrame, variant: str) -> dict | None:
    """g = one regular session of 5m bars (09:30..15:55), labelled by open time."""
    if len(g) < 70 or g.index[0].strftime("%H:%M") != "09:30":
        return None
    first = g.iloc[0]
    rest = g.iloc[1:]
    o1, h1, l1, c1 = first[["open", "high", "low", "close"]]
    if variant == "paper":
        if c1 == o1:
            return None
        side = 1 if c1 > o1 else -1
        entry = rest["open"].iloc[0]
        stop = l1 if side == 1 else h1
        i0 = 0
    else:  # breakout
        side, i0, entry = 0, None, None
        for i, (hh, ll, oo) in enumerate(zip(rest["high"], rest["low"], rest["open"])):
            up, dn = hh > h1, ll < l1
            if up and dn:          # both in one bar: ambiguous, skip the day
                return None
            if up:
                side, i0, entry = 1, i, max(oo, h1)
                break
            if dn:
                side, i0, entry = -1, i, min(oo, l1)
                break
        if not side:
            return None
        stop = l1 if side == 1 else h1
    risk = (entry - stop) * side
    if risk <= 0:
        return None
    target = entry + side * TARGET_R * risk
    exit_px, how = rest["close"].iloc[-1], "close"
    bars = rest.iloc[i0:]
    for j, (oo, hh, ll) in enumerate(zip(bars["open"], bars["high"], bars["low"])):
        if variant == "breakout" and j == 0:
            oo = entry                                   # entry bar: already at entry
        if side == 1:
            if oo <= stop:  exit_px, how = oo, "gap-stop";  break
            if ll <= stop:  exit_px, how = stop, "stop";    break
            if hh >= target: exit_px, how = target, "target"; break
        else:
            if oo >= stop:  exit_px, how = oo, "gap-stop";  break
            if hh >= stop:  exit_px, how = stop, "stop";    break
            if ll <= target: exit_px, how = target, "target"; break
    ret = side * (exit_px / entry - 1)
    return {"side": side, "entry": entry, "stop": stop, "exit": exit_px, "how": how,
            "R": (exit_px - entry) * side / risk, "ret": ret, "risk_pct": risk / entry}


def run(bars: pd.DataFrame, variant: str) -> pd.DataFrame:
    rth = bars.between_time("09:30", "15:55")
    out = []
    for day, g in rth.groupby(rth.index.normalize()):
        r = simulate_day(g, variant)
        if r:
            r["date"] = day
            out.append(r)
    return pd.DataFrame(out).set_index("date") if out else pd.DataFrame()


def stats(t: pd.DataFrame, cost_bps: float) -> dict:
    if len(t) < 10:
        return {"n": len(t)}
    net = t["ret"] - cost_bps / 1e4
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {
        "n": len(t),
        "hit": (t["R"] > 0).mean(),
        "avgR": t["R"].mean(),
        "gross_bps": t["ret"].mean() * 1e4,
        "net_bps": net.mean() * 1e4,
        "t": net.mean() / (net.std() / np.sqrt(len(net))),
        "pf": wins / losses if losses > 0 else np.inf,
        "sharpe": net.mean() / net.std() * np.sqrt(252),
        "targets": (t["how"] == "target").mean(),
    }


HDR = f"{'':22s} {'trades':>6s} {'hit':>6s} {'avg R':>6s} {'gross':>7s} {'net':>7s} {'t':>6s} {'PF':>5s} {'Sharpe':>6s} {'10R%':>5s}"


def row(name: str, s: dict) -> str:
    if s["n"] < 10:
        return f"{name:22s} {s['n']:6d}  (too few)"
    return (f"{name:22s} {s['n']:6d} {s['hit']:6.1%} {s['avgR']:6.3f} {s['gross_bps']:7.2f} {s['net_bps']:7.2f} "
            f"{s['t']:6.2f} {s['pf']:5.2f} {s['sharpe']:6.2f} {s['targets']:5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--recent-from", default="2024-01-01")
    ap.add_argument("--cost-bps", type=float, default=1.0, help="round-trip cost in bps of price")
    ap.add_argument("--csv", help="cached 5m bars for a single ticker (testing)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lines = [f"5-MIN OPENING RANGE BREAKOUT  |  stop = first candle  |  target {TARGET_R:.0f}R or close",
             f"returns in bps of price at 1x notional; net = after {a.cost_bps:g} bps round trip",
             "(paper = Zarattini & Aziz 2023 rules; breakout = enter on break of first-candle high/low)"]
    curves = {}
    for tk in a.tickers:
        if a.csv:
            bars = pd.read_csv(a.csv, index_col=0, parse_dates=True)
        else:
            key = os.getenv("POLYGON_API_KEY", "").strip()
            if not key:
                sys.exit("POLYGON_API_KEY is not set")
            end = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
            start = end - pd.Timedelta(days=int(a.years * 365.25))
            print(f"Downloading {tk} {start.date()} → {end.date()}…", flush=True)
            bars = polygon_5m(tk, start, end, key)
        for variant in ("paper", "breakout"):
            t = run(bars, variant)
            if t.empty:
                continue
            t.to_csv(os.path.join(a.out, f"orb_{tk}_{variant}_trades.csv"))
            curves[f"{tk} {variant}"] = (t["ret"] - a.cost_bps / 1e4).cumsum() * 100
            rec = t[t.index >= a.recent_from]
            lines += [f"\n── {tk} · {variant}  ({t.index.min().date()} → {t.index.max().date()}) ──", HDR,
                      row("all", stats(t, a.cost_bps)),
                      row(f"since {a.recent_from}", stats(rec, a.cost_bps)),
                      row("longs", stats(t[t.side == 1], a.cost_bps)),
                      row("shorts", stats(t[t.side == -1], a.cost_bps))]
            for y, g in t.groupby(t.index.year):
                lines.append(row(f"  {y}", stats(g, a.cost_bps)))
    lines += ["",
              "PF = profit factor (>1.2 interesting), t > 2 ≈ significant, 10R% = share of trades hitting target.",
              "Paper reported strong results on QQQ 2016–2023 with leverage; here 1x, so focus on net bps, t, PF."]
    summary = "\n".join(lines)
    print("\n" + summary)
    open(os.path.join(a.out, "orb_summary.txt"), "w").write(summary + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#0d1422")
        ax.set_facecolor("#0d1422")
        palette = ["#e0b25c", "#7fb6e6", "#6fcf97", "#c58af9"]
        for (name, c), col in zip(curves.items(), palette):
            ax.plot(c.index, c.values, label=name, color=col, lw=1.8)
        ax.axhline(0, color="#8a96a8", lw=0.8)
        ax.set_title(f"5-min ORB — cumulative % after {a.cost_bps:g} bps costs (1x, no leverage)",
                     color="#e8edf5", loc="left", fontsize=14, weight="bold")
        ax.tick_params(colors="#8a96a8")
        ax.grid(color="#1e2a3d", lw=0.6)
        for sp in ax.spines.values():
            sp.set_color("#1e2a3d")
        leg = ax.legend(facecolor="#0d1422", edgecolor="#1e2a3d")
        for txt in leg.get_texts():
            txt.set_color("#e8edf5")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "orb.png"), dpi=130, facecolor="#0d1422")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
