#!/usr/bin/env python3
"""
Deep-dive on the Stocks-in-Play breakout backtest (reads results/sip_all_records.csv).

Question: WHERE does the edge come from? Each table splits the in-play trades
(top-10 by first-bar RVOL) by one property and reports per-trade net bps and t.
Purely descriptive: anything found here is a hypothesis to test forward, NOT a
rule change for the running forward test.

  python sip_research.py --cost-bps 2
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def st(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) < 30:
        return f"{len(x):6d}   (few)"
    t = x.mean() / (x.std() / np.sqrt(len(x)))
    return f"{len(x):6d} {x.mean() * 1e4:8.2f} {t:6.2f} {(x > 0).mean():6.1%}"


H = f"{'':26s} {'trades':>6s} {'net bps':>8s} {'t':>6s} {'win%':>6s}"


def daily(df: pd.DataFrame, col: str = "net") -> pd.Series:
    return df.groupby("date")[col].mean()


def dstat(s: pd.Series) -> str:
    t = s.mean() / (s.std() / np.sqrt(len(s)))
    return f"{len(s):5d} days {s.mean() * 1e4:8.2f} bps/day  t={t:5.2f}  Sharpe={s.mean() / s.std() * np.sqrt(252):5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/sip_all_records.csv")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="results/sip_research.txt")
    a = ap.parse_args()
    c = a.cost_bps / 1e4

    allr = pd.read_csv(a.inp, parse_dates=["date"]).dropna(subset=["rvol"])
    allr["rank"] = allr.groupby("date")["rvol"].rank(ascending=False, method="first")
    allr["net"] = allr["breakout_candle"] - c
    ip = allr[(allr["rank"] <= a.top) & (allr["rvol"] > 1)].copy()
    tr = ip.dropna(subset=["breakout_candle"]).copy()
    L = [f"STOCKS IN PLAY — WHERE DOES THE EDGE COME FROM?   cost {a.cost_bps:g} bps/trade",
         f"{tr['date'].min().date()} → {tr['date'].max().date()}  |  {len(tr)} breakout trades on {tr['date'].nunique()} days",
         "Descriptive only — hypotheses for forward testing, not rule changes."]

    def table(title, groups):
        L.append(f"\n── {title} ──")
        L.append(H)
        for name, x in groups:
            L.append(f"{name:26s} {st(x)}")

    # 1. how many stocks / RVOL rank
    table("RVOL rank (1 = most in play)", [
        ("rank 1-3", tr.loc[tr["rank"] <= 3, "net"]),
        ("rank 4-6", tr.loc[tr["rank"].between(4, 6), "net"]),
        ("rank 7-10", tr.loc[tr["rank"].between(7, 10), "net"]),
        ("rank 11-30 (not selected)", allr.loc[allr["rank"].between(11, 30), "net"]),
        ("rank 31-100 (not selected)", allr.loc[allr["rank"] > 30, "net"]),
    ])
    table("RVOL level", [
        ("RVOL 1-2", tr.loc[tr["rvol"] < 2, "net"]),
        ("RVOL 2-4", tr.loc[tr["rvol"].between(2, 4), "net"]),
        ("RVOL > 4", tr.loc[tr["rvol"] > 4, "net"]),
    ])
    L.append("\n── Portfolio size (daily equal-weight, top-K by RVOL) ──")
    for k in (3, 5, 10, 20):
        sub = allr[(allr["rank"] <= k) & (allr["rvol"] > 1)].dropna(subset=["breakout_candle"])
        L.append(f"top {k:<3d} {dstat(daily(sub))}")

    # 2. direction
    if "side" in tr:
        table("Direction", [("long (break of high)", tr.loc[tr["side"] == 1, "net"]),
                            ("short (break of low)", tr.loc[tr["side"] == -1, "net"])])
        tr["with_candle"] = tr["side"] == tr["first_dir"]
        table("Breakout vs first-candle colour", [("same direction", tr.loc[tr["with_candle"], "net"]),
                                                  ("against", tr.loc[~tr["with_candle"], "net"])])

    # 3. entry time
    if "t_entry" in tr:
        te = tr["t_entry"].fillna("99:99")
        table("Entry time (NY)", [("09:35-09:45", tr.loc[te <= "09:45", "net"]),
                                  ("09:50-10:30", tr.loc[(te > "09:45") & (te <= "10:30"), "net"]),
                                  ("10:35-12:00", tr.loc[(te > "10:30") & (te <= "12:00"), "net"]),
                                  ("after 12:00", tr.loc[te > "12:00", "net"])])

    # 4. exit type
    if "how" in tr:
        table("Exit type", [(h, tr.loc[tr["how"] == h, "net"]) for h in ("stop", "target", "close")])
        L.append("share of trades: " + ", ".join(f"{h} {v:.0%}" for h, v in tr["how"].value_counts(normalize=True).items()))

    # 5. opening-range width & gap
    for col, title in (("or_pct", "First-candle range (% of price) terciles"),
                       ("risk_pct", "Stop distance (% of price) terciles")):
        if col in tr and tr[col].notna().sum() > 90:
            q = tr[col].quantile([1 / 3, 2 / 3]).values
            table(title, [(f"narrow  (<{q[0]:.2%})", tr.loc[tr[col] < q[0], "net"]),
                          (f"middle", tr.loc[tr[col].between(q[0], q[1]), "net"]),
                          (f"wide    (>{q[1]:.2%})", tr.loc[tr[col] > q[1], "net"])])
    if "gap_pct" in tr and tr["gap_pct"].notna().sum() > 90:
        g = tr["gap_pct"]
        q = g.abs().quantile([1 / 3, 2 / 3]).values
        table("Overnight gap size (|gap|)", [(f"small (<{q[0]:.2%})", tr.loc[g.abs() < q[0], "net"]),
                                             ("middle", tr.loc[g.abs().between(q[0], q[1]), "net"]),
                                             (f"big   (>{q[1]:.2%})", tr.loc[g.abs() > q[1], "net"])])
        if "side" in tr:
            agree = np.sign(g) == tr["side"]
            table("Breakout vs gap direction", [("with the gap", tr.loc[agree, "net"]),
                                                ("against the gap (fade)", tr.loc[~agree & g.notna(), "net"])])

    # 6. concentration
    pnl = tr.groupby("tk")["net"].sum().sort_values(ascending=False)
    top5 = pnl.head(5)
    L.append("\n── Concentration ──")
    L.append(f"stocks traded: {len(pnl)} | total net (sum of trade %): {pnl.sum() * 100:.1f}")
    L.append("top 5 contributors: " + ", ".join(f"{k} {v * 100:+.1f}" for k, v in top5.items()))
    L.append("worst 5: " + ", ".join(f"{k} {v * 100:+.1f}" for k, v in pnl.tail(5).items()))
    ex = tr[~tr["tk"].isin(top5.index)]
    L.append(f"without top-5 stocks: {dstat(daily(ex))}")
    L.append(f"share of stocks with positive total: {(pnl > 0).mean():.0%}")

    # 7. cost sensitivity
    L.append("\n── Cost sensitivity (top-10 daily portfolio) ──")
    for cb in (0, 2, 5, 10):
        s = (tr["breakout_candle"] - cb / 1e4).groupby(tr["date"]).mean()
        L.append(f"{cb:>3d} bps/trade: {dstat(s)}")

    # 8. market context (descriptive: same-day universe drift, known only after the close)
    mkt = allr.groupby("date")["long_drift"].mean()
    tr["mkt"] = tr["date"].map(mkt)
    q = mkt.quantile([1 / 3, 2 / 3]).values
    table("Market day (universe 09:35→close drift; hindsight, descriptive)", [
        ("down day", tr.loc[tr["mkt"] < q[0], "net"]),
        ("flat day", tr.loc[tr["mkt"].between(q[0], q[1]), "net"]),
        ("up day", tr.loc[tr["mkt"] > q[1], "net"])])

    # 9. weekday
    table("Weekday", [(n, tr.loc[tr["date"].dt.weekday == i, "net"])
                      for i, n in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])])

    # ═════════ idea tests (2026-10-06): each vs A and B, same days, no-trade day = 0 ═════════
    days = pd.Index(sorted(ip["date"].unique()))
    tr["B"] = (tr["side"] == tr["first_dir"]) & (tr["t_entry"].astype(str) <= "09:45")

    def port(df, col="breakout_candle", cost=c, weight=None):
        x = df.dropna(subset=[col])
        net = x[col] - cost
        if weight is None:
            s = net.groupby(x["date"]).mean()
        else:
            w = weight.loc[x.index]
            s = (net * w).groupby(x["date"]).sum() / w.groupby(x["date"]).sum()
        return s.reindex(days).fillna(0.0)

    def fmt(s):
        t = s.mean() / (s.std() / np.sqrt(len(s)))
        yrs = s.groupby(s.index.year).mean()
        return f"{s.mean() * 1e4:7.2f} {t:6.2f} {int((yrs > 0).sum())}/{len(yrs)}"

    inv = 1.0 / tr["risk_pct"].clip(lower=0.002)
    has_spy = "spy_dir" in tr and tr["spy_dir"].notna().any()
    rows = [("A  baseline", tr, "breakout_candle", None),
            ("B  aligned+early", tr[tr["B"]], "breakout_candle", None)]
    if has_spy:
        rows += [("A + 1 market direction", tr[tr["side"] == tr["spy_dir"]], "breakout_candle", None),
                 ("B + 1 market direction", tr[tr["B"] & (tr["side"] == tr["spy_dir"])], "breakout_candle", None)]
    if "breakout_trail" in tr:
        rows += [("A + 2 trailing stop", tr, "breakout_trail", None),
                 ("B + 2 trailing stop", tr[tr["B"]], "breakout_trail", None)]
    rows += [("A + 3 equal-risk sizing", tr, "breakout_candle", inv),
             ("B + 3 equal-risk sizing", tr[tr["B"]], "breakout_candle", inv)]
    L.append("\n══ IDEA TESTS (daily portfolio; years+ = years with positive average) ══")
    L.append(f"{'':26s} {'@2bps':>7s} {'t':>6s} {'yrs+':>4s} | {'@5bps':>7s} {'t':>6s} {'yrs+':>4s}")
    for name, df, col, w in rows:
        L.append(f"{name:26s} {fmt(port(df, col, c, w))} | {fmt(port(df, col, 5e-4, w))}")

    L += ["", "t > 2 ≈ significant. With ~12 tables, expect ~1 false 'significant' cell by chance —",
          "treat isolated wins with suspicion; consistent patterns across tables matter more."]
    txt = "\n".join(L)
    print(txt)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
