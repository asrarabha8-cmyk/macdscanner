#!/usr/bin/env python3
"""
Intraday momentum backtest (Gao, Han, Li & Zhou, 2018, "Market Intraday Momentum")
================================================================================

Claim to test on SPY:  the first half-hour return (yesterday's close -> 10:00 NY)
predicts the LAST half-hour return (15:30 -> 16:00). Agreement with the
second-to-last half hour (15:00 -> 15:30) is said to strengthen the signal.

Signals tested (position held 15:30 -> 16:00 only):
  r1      : sign of (prev close -> 10:00)
  r12     : sign of (15:00 -> 15:30)
  agree   : trade only when r1 and r12 agree
  base    : always long (to show the plain drift of the last half hour)

Reports: hit rate, avg return per trade (bps), annualized Sharpe, by year, on
high-volatility days, and on the most recent period — plus a regression
r13 ~ r1 + r12 with t-stats. Data: Polygon/Massive 5-minute bars (POLYGON_API_KEY).

  python intraday_momentum.py --ticker SPY --years 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

NY = "America/New_York"


# ─────────────────────────────── data ───────────────────────────────
def polygon_5m(ticker: str, start: pd.Timestamp, end: pd.Timestamp, key: str) -> pd.DataFrame:
    base = (os.getenv("POLYGON_BASE_URL") or "https://api.polygon.io").rstrip("/")
    rows = []
    # one request chain per quarter keeps each response small
    edges = pd.date_range(start, end, freq="QS").tolist()
    edges = [start] + [e for e in edges if e > start] + [end + pd.Timedelta(days=1)]
    for a, b in zip(edges[:-1], edges[1:]):
        url = (f"{base}/v2/aggs/ticker/{urllib.parse.quote(ticker)}/range/5/minute/"
               f"{a.date()}/{(b - pd.Timedelta(days=1)).date()}?adjusted=true&sort=asc&limit=50000")
        while url:
            sep = "&" if "?" in url else "?"
            for attempt in range(5):
                try:
                    req = urllib.request.Request(url + f"{sep}apiKey={key}", headers={"User-Agent": "intraday-momentum"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        d = json.loads(r.read().decode())
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        time.sleep(15)
                        continue
                    raise
            rows += d.get("results") or []
            url = d.get("next_url")
        print(f"  fetched through {(b - pd.Timedelta(days=1)).date()}  ({len(rows)} bars)", flush=True)
    if not rows:
        sys.exit("Polygon returned no data")
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(NY).dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def daily_segments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per full regular session with the half-hour returns we need.
    Bars are labelled by their OPEN time, so the price 'at 10:00' is the close
    of the 09:55 bar, 'at 16:00' is the close of the 15:55 bar."""
    rth = df.between_time("09:30", "15:55")
    out = []
    for day, g in rth.groupby(rth.index.normalize()):
        t = g.index.strftime("%H:%M")
        need = {"09:30", "09:55", "14:55", "15:25", "15:55"}
        if not need.issubset(set(t)) or len(g) < 70:      # skips half-days & gaps
            continue
        px = dict(zip(t, g["close"].values))
        out.append({
            "date": day,
            "open": g["open"].iloc[0],
            "p1000": px["09:55"],
            "p1500": px["14:55"],
            "p1530": px["15:25"],
            "close": px["15:55"],
            "day_range": g["high"].max() - g["low"].min(),
        })
    d = pd.DataFrame(out).set_index("date")
    d["prev_close"] = d["close"].shift(1)
    d = d.dropna()
    d["r1"] = d["p1000"] / d["prev_close"] - 1          # first half hour incl. overnight
    d["r12"] = d["p1530"] / d["p1500"] - 1              # second-to-last half hour
    d["r13"] = d["close"] / d["p1530"] - 1              # last half hour (target)
    # volatility regime known before 15:30: yesterday's range relative to its 20-day average
    rng_pct = d["day_range"] / d["prev_close"]
    d["vol_prev"] = rng_pct.shift(1)
    d["vol_rel"] = d["vol_prev"] / d["vol_prev"].rolling(20).mean()
    return d


# ─────────────────────────────── stats ──────────────────────────────
def strat_stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    ret = ret[ret != 0] if (ret == 0).mean() > 0.3 else ret      # filtered strategies sit out
    n = len(ret)
    if n < 10:
        return {"n": n}
    mu, sd = ret.mean(), ret.std()
    return {
        "n": n,
        "hit": (ret > 0).mean(),
        "avg_bps": mu * 1e4,
        "t": mu / (sd / np.sqrt(n)) if sd > 0 else np.nan,
        "sharpe": mu / sd * np.sqrt(252) if sd > 0 else np.nan,
        "total_pct": ret.sum() * 100,
    }


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # White heteroskedasticity-robust standard errors
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = X.T @ (X * resid[:, None] ** 2)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, beta / se, r2


def signals(d: pd.DataFrame) -> dict[str, pd.Series]:
    s1, s12 = np.sign(d["r1"]), np.sign(d["r12"])
    return {
        "always long (baseline)": d["r13"],
        "sign(r1)": s1 * d["r13"],
        "sign(r12)": s12 * d["r13"],
        "r1 & r12 agree": np.where(s1 == s12, s1, 0) * d["r13"],
    }


def table(d: pd.DataFrame, title: str) -> list[str]:
    lines = [f"\n── {title}  ({d.index.min().date()} → {d.index.max().date()}, {len(d)} days) ──",
             f"{'signal':24s} {'trades':>6s} {'hit':>6s} {'avg bps':>8s} {'t':>6s} {'Sharpe':>7s} {'total%':>7s}"]
    for name, r in signals(d).items():
        s = strat_stats(pd.Series(r, index=d.index))
        if s["n"] < 10:
            lines.append(f"{name:24s} {s['n']:6d}   (too few)")
            continue
        lines.append(f"{name:24s} {s['n']:6d} {s['hit']:6.1%} {s['avg_bps']:8.2f} {s['t']:6.2f} "
                     f"{s['sharpe']:7.2f} {s['total_pct']:7.2f}")
    return lines


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--csv", help="cached 5m bars (timestamp index) instead of Polygon")
    ap.add_argument("--recent-from", default="2024-01-01", help="start of the 'recent' sub-period")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.csv:
        bars = pd.read_csv(a.csv, index_col=0, parse_dates=True)
    else:
        key = os.getenv("POLYGON_API_KEY", "").strip()
        if not key:
            sys.exit("POLYGON_API_KEY is not set")
        end = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
        start = end - pd.Timedelta(days=int(a.years * 365.25))
        print(f"Downloading {a.ticker} 5m bars {start.date()} → {end.date()} from Polygon…")
        bars = polygon_5m(a.ticker, start, end, key)

    d = daily_segments(bars)
    if len(d) < 60:
        sys.exit(f"Only {len(d)} usable sessions — not enough")
    tag = a.ticker.replace("^", "")
    d.to_csv(os.path.join(a.out, f"momentum_{tag}_daily.csv"))

    lines = [f"INTRADAY MOMENTUM — {a.ticker}  |  trade 15:30→16:00 NY  |  costs not included"]
    lines += table(d, "Full period")
    recent = d[d.index >= a.recent_from]
    if len(recent) >= 60:
        lines += table(recent, f"Recent (since {a.recent_from})")
    hv = d[d["vol_rel"] >= d["vol_rel"].quantile(2 / 3)]
    lines += table(hv, "High-volatility days (top third, known before the trade)")
    big = d[d["r1"].abs() >= d["r1"].abs().quantile(2 / 3)]
    lines += table(big, "Big first-half-hour move (top third |r1|)")

    # by year, main signal only
    lines.append("\n── sign(r1) by year ──")
    lines.append(f"{'year':6s} {'days':>5s} {'hit':>6s} {'avg bps':>8s} {'t':>6s}")
    for y, g in d.groupby(d.index.year):
        s = strat_stats(np.sign(g["r1"]) * g["r13"])
        if s["n"] >= 10:
            lines.append(f"{y:<6d} {s['n']:5d} {s['hit']:6.1%} {s['avg_bps']:8.2f} {s['t']:6.2f}")

    # regression
    y = d["r13"].values * 1e4
    b, t, r2 = ols(y, d[["r1", "r12"]].values * 1e4)
    b1, t1, r21 = ols(y, d[["r1"]].values * 1e4)
    lines += [
        "\n── Regression (bps):  r13 = a + b1·r1 + b2·r12 ──",
        f"r1 only      : b1={b1[1]:.4f} (t={t1[1]:.2f})  R²={r21:.2%}",
        f"r1 + r12     : b1={b[1]:.4f} (t={t[1]:.2f})  b2={b[2]:.4f} (t={t[2]:.2f})  R²={r2:.2%}",
        "(paper: positive, significant b1; R² around 1–2% for SPY.  |t| > 2 ≈ significant)",
        "",
        "Note: SPY round-trip cost ≈ 0.5–1 bps on the ETF; options cost far more —",
        "an edge of 1–2 bps per trade is not tradable with 0DTE options.",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    open(os.path.join(a.out, f"momentum_{tag}_summary.txt"), "w").write(summary + "\n")

    # equity curve chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#0d1422")
        ax.set_facecolor("#0d1422")
        colors = {"always long (baseline)": "#5d6b80", "sign(r1)": "#e0b25c",
                  "sign(r12)": "#7fb6e6", "r1 & r12 agree": "#6fcf97"}
        for name, r in signals(d).items():
            ax.plot(pd.Series(r, index=d.index).cumsum() * 100, label=name, color=colors[name], lw=1.8)
        ax.axhline(0, color="#8a96a8", lw=0.8)
        ax.set_title(f"Intraday momentum — {a.ticker} last half hour (cumulative %, no costs)",
                     color="#e8edf5", loc="left", fontsize=14, weight="bold")
        ax.tick_params(colors="#8a96a8")
        ax.grid(color="#1e2a3d", lw=0.6)
        for sp in ax.spines.values():
            sp.set_color("#1e2a3d")
        leg = ax.legend(facecolor="#0d1422", edgecolor="#1e2a3d")
        for txt in leg.get_texts():
            txt.set_color("#e8edf5")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"momentum_{tag}.png"), dpi=130, facecolor="#0d1422")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
