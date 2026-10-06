#!/usr/bin/env python3
"""
"Stocks in Play" opening-range-breakout backtest — survivorship-free universe
============================================================================
Answers the two doubts left by the hand-picked ticker test:
  1. selection bias  -> the universe is chosen by a fixed rule, point-in-time,
                        from Polygon's full-market daily data (delisted names included)
  2. long/bull bias  -> every strategy is compared with simply buying the same
                        stocks at 09:35 and selling at the close

Daily procedure (all information known at decision time):
  universe(D)  = top N common stocks by 20-day average dollar volume up to D-1,
                 prior close > $5, ETFs/ETNs excluded
  in play(D)   = top K of the universe by relative volume of the first 5-min bar
                 (first-bar volume / its average over the previous 14 sessions)

Strategies, equal-weight across the day's selected stocks, 1x notional:
  breakout_candle : enter on break of first-candle high/low, stop at the other side,
                    10R target or close  (the rule that worked on QQQ/tech)
  breakout_atr    : same entry, stop 10% of 14-day ATR, exit at close
                    (Zarattini, Barbon & Aziz 2024, "A Profitable Day Trading Strategy
                    for the U.S. Equity Market")
  direction_open  : first-candle colour, enter 09:35 open, candle stop, 10R / close
  long_drift      : buy 09:35 open, sell close (bull-bias control)
  breakout_all    : breakout_candle on the WHOLE universe, not only stocks in play
                    (is "in play" what matters?)

  python sip_backtest.py --years 3 --universe 100 --top 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

NY = "America/New_York"
BASE = (os.getenv("POLYGON_BASE_URL") or "https://api.polygon.io").rstrip("/")
KEY = os.getenv("POLYGON_API_KEY", "").strip()
SYM_OK = re.compile(r"^[A-Z]{1,5}$")


# ─────────────────────────────── polygon ────────────────────────────
def _get(url: str) -> dict:
    sep = "&" if "?" in url else "?"
    for attempt in range(6):
        try:
            req = urllib.request.Request(f"{url}{sep}apiKey={KEY}", headers={"User-Agent": "sip-backtest"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url[:120]}")


def grouped_daily(day: pd.Timestamp) -> pd.DataFrame:
    d = _get(f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{day.date()}?adjusted=true")
    res = d.get("results") or []
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)[["T", "o", "h", "l", "c", "v"]]
    df["date"] = day
    return df


def etf_set() -> set[str]:
    out = set()
    for typ in ("ETF", "ETN", "ETV", "ETS", "FUND"):
        for active in ("true", "false"):
            url = f"{BASE}/v3/reference/tickers?market=stocks&type={typ}&active={active}&limit=1000"
            pages = 0
            while url and pages < 50:
                d = _get(url)
                out |= {r["ticker"] for r in d.get("results") or []}
                url, pages = d.get("next_url"), pages + 1
    return out


def bars_5m(tk: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, a = [], start
    while a <= end:
        b = min(a + pd.Timedelta(days=120), end)
        url = (f"{BASE}/v2/aggs/ticker/{urllib.parse.quote(tk)}/range/5/minute/"
               f"{a.date()}/{b.date()}?adjusted=true&sort=asc&limit=50000")
        while url:
            d = _get(url)
            rows += d.get("results") or []
            url = d.get("next_url")
        a = b + pd.Timedelta(days=1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(NY).dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].between_time("09:30", "15:55").sort_index()


# ─────────────────────────────── trade logic ────────────────────────
def trade_detail(g: pd.DataFrame, mode: str, atr: float = np.nan) -> dict | None:
    """One trade on one session (5m bars from 09:30). Returns side/entry/stop/exit/ret
    (ret = fraction at 1x notional) or None if no trade. Stop-before-target on ambiguous bars."""
    first, rest = g.iloc[0], g.iloc[1:]
    o1, h1, l1, c1 = first["open"], first["high"], first["low"], first["close"]
    if mode == "long_drift":
        e, x = rest["open"].iloc[0], rest["close"].iloc[-1]
        return {"side": 1, "entry": e, "stop": np.nan, "exit": x, "how": "close",
                "t_entry": rest.index[0], "ret": x / e - 1}
    if mode == "direction_open":
        if c1 == o1:
            return None
        side, i0, entry = (1 if c1 > o1 else -1), 0, rest["open"].iloc[0]
    else:  # breakout_candle / breakout_atr
        side = 0
        for i, (hh, ll, oo) in enumerate(zip(rest["high"].values, rest["low"].values, rest["open"].values)):
            up, dn = hh > h1, ll < l1
            if up and dn:
                return None
            if up:
                side, i0, entry = 1, i, max(oo, h1)
                break
            if dn:
                side, i0, entry = -1, i, min(oo, l1)
                break
        if not side:
            return None
    if mode == "breakout_atr":
        if not np.isfinite(atr) or atr <= 0:
            return None
        stop, target = entry - side * 0.10 * atr, None
    else:
        stop = l1 if side == 1 else h1
        risk = (entry - stop) * side
        if risk <= 0:
            return None
        target = entry + side * 10 * risk
    exit_px, how = rest["close"].iloc[-1], "close"
    seg = rest.iloc[i0:]
    for j, (oo, hh, ll) in enumerate(zip(seg["open"].values, seg["high"].values, seg["low"].values)):
        if j == 0 and mode != "direction_open":
            oo = entry
        if side == 1:
            if oo <= stop:  exit_px, how = oo, "stop";     break
            if ll <= stop:  exit_px, how = stop, "stop";   break
            if target is not None and hh >= target: exit_px, how = target, "target"; break
        else:
            if oo >= stop:  exit_px, how = oo, "stop";     break
            if hh >= stop:  exit_px, how = stop, "stop";   break
            if target is not None and ll <= target: exit_px, how = target, "target"; break
    return {"side": side, "entry": entry, "stop": stop, "exit": exit_px, "how": how,
            "t_entry": rest.index[i0], "ret": side * (exit_px / entry - 1)}


def trade(g: pd.DataFrame, mode: str, atr: float = np.nan) -> float | None:
    d = trade_detail(g, mode, atr)
    return None if d is None else d["ret"]


def trade_trail(g: pd.DataFrame) -> float | None:
    """Idea 2 — same entry/initial stop as breakout_candle, but a moving stop:
    once price has gone +1R in our favour the stop moves to break-even, and from
    then on it trails 1R behind the best price reached. Otherwise exit at close.
    Stop updates take effect from the NEXT bar (no look-ahead inside a bar)."""
    first, rest = g.iloc[0], g.iloc[1:]
    h1, l1 = first["high"], first["low"]
    side = 0
    for i, (hh, ll, oo) in enumerate(zip(rest["high"].values, rest["low"].values, rest["open"].values)):
        up, dn = hh > h1, ll < l1
        if up and dn:
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
    R = (entry - stop) * side
    if R <= 0:
        return None
    best = entry
    seg = rest.iloc[i0:]
    exit_px = rest["close"].iloc[-1]
    for j, (oo, hh, ll) in enumerate(zip(seg["open"].values, seg["high"].values, seg["low"].values)):
        if j == 0:
            oo = entry
        if side == 1:
            if oo <= stop: exit_px = oo; break
            if ll <= stop: exit_px = stop; break
            best = max(best, hh)
            if best - entry >= R:
                stop = max(stop, entry, best - R)
        else:
            if oo >= stop: exit_px = oo; break
            if hh >= stop: exit_px = stop; break
            best = min(best, ll)
            if entry - best >= R:
                stop = min(stop, entry, best + R)
    return side * (exit_px / entry - 1)


MODES = ["breakout_candle", "breakout_atr", "direction_open", "long_drift"]


def ticker_records(tk: str, days: list[pd.Timestamp], atr: pd.Series, start: pd.Timestamp,
                   end: pd.Timestamp) -> list[dict]:
    bars = bars_5m(tk, start, end)
    if bars.empty:
        return []
    out = []
    sessions = {d: g for d, g in bars.groupby(bars.index.normalize())}
    order = sorted(sessions)
    first_vol = pd.Series({d: sessions[d]["volume"].iloc[0] if sessions[d].index[0].strftime("%H:%M") == "09:30"
                           else np.nan for d in order})
    avg14 = first_vol.shift(1).rolling(14, min_periods=10).mean()
    want = set(days)
    prev_close = pd.Series({d: sessions[d]["close"].iloc[-1] for d in order}).shift(1)
    for d in order:
        if d not in want:
            continue
        g = sessions[d]
        if len(g) < 70 or g.index[0].strftime("%H:%M") != "09:30" or not np.isfinite(avg14.get(d, np.nan)):
            continue
        rec = {"date": d, "tk": tk, "rvol": first_vol[d] / avg14[d] if avg14[d] > 0 else np.nan}
        for m in MODES:
            rec[m] = trade(g, m, atr.get(d, np.nan))
        # descriptive fields for research (known at/after entry; not used for selection)
        f = g.iloc[0]
        pc = prev_close.get(d, np.nan)
        rec["gap_pct"] = f["open"] / pc - 1 if np.isfinite(pc) and pc > 0 else np.nan
        rec["or_pct"] = (f["high"] - f["low"]) / f["open"]
        rec["first_dir"] = int(np.sign(f["close"] - f["open"]))
        rec["atr_pct"] = atr.get(d, np.nan) / pc if np.isfinite(pc) and pc > 0 else np.nan
        rec["breakout_trail"] = trade_trail(g)
        det = trade_detail(g, "breakout_candle")
        if det is not None:
            rec.update(side=det["side"], how=det["how"], t_entry=det["t_entry"].strftime("%H:%M"),
                       risk_pct=abs(det["entry"] - det["stop"]) / det["entry"])
        out.append(rec)
    return out


# ─────────────────────────────── stats ──────────────────────────────
def daily_portfolio(trades: pd.DataFrame, col: str, cost_bps: float) -> pd.Series:
    t = trades.dropna(subset=[col])
    net = t[col] - cost_bps / 1e4
    return net.groupby(t["date"]).mean()


def line(name: str, s: pd.Series) -> str:
    if len(s) < 20:
        return f"{name:28s} (too few: {len(s)})"
    t = s.mean() / (s.std() / np.sqrt(len(s)))
    return (f"{name:28s} {len(s):5d} {s.mean() * 1e4:8.2f} {t:6.2f} {s.mean() / s.std() * np.sqrt(252):7.2f} "
            f"{(s > 0).mean():6.1%} {s.sum() * 100:8.1f}")


HDR = f"{'strategy (daily equal-weight)':28s} {'days':>5s} {'net bps':>8s} {'t':>6s} {'Sharpe':>7s} {'up%':>6s} {'total%':>8s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--universe", type=int, default=100)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--recent-from", default="2025-01-01")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if not KEY:
        sys.exit("POLYGON_API_KEY is not set")
    os.makedirs(a.out, exist_ok=True)

    end = pd.Timestamp.now(tz=NY).tz_localize(None).normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(a.years * 365.25))
    cal = pd.bdate_range(start - pd.Timedelta(days=45), end)

    print(f"1/4 full-market daily bars for {len(cal)} weekdays…", flush=True)
    frames = []
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(grouped_daily, d) for d in cal]):
            df = f.result()
            if not df.empty:
                frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily[daily["T"].str.match(SYM_OK)]
    print(f"   {daily['date'].nunique()} sessions, {daily['T'].nunique()} symbols", flush=True)

    print("2/4 excluding ETFs/ETNs…", flush=True)
    try:
        etfs = etf_set()
    except Exception as e:
        print(f"   ETF list unavailable ({e}); continuing without the filter", flush=True)
        etfs = set()
    daily = daily[~daily["T"].isin(etfs)]
    print(f"   removed {len(etfs)} fund symbols", flush=True)

    # point-in-time universe
    daily = daily.sort_values(["T", "date"])
    daily["dv"] = daily["c"] * daily["v"]
    g = daily.groupby("T")
    daily["adv20"] = g["dv"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    daily["prev_c"] = g["c"].shift(1)
    tr = np.maximum(daily["h"] - daily["l"],
                    np.maximum((daily["h"] - daily["prev_c"]).abs(), (daily["l"] - daily["prev_c"]).abs()))
    daily["atr14"] = tr.groupby(daily["T"]).transform(lambda s: s.shift(1).rolling(14, min_periods=10).mean())
    elig = daily[(daily["date"] >= start) & (daily["prev_c"] > 5) & daily["adv20"].notna()]
    uni = (elig.sort_values("adv20", ascending=False).groupby("date").head(a.universe))
    members = uni.groupby("T")["date"].apply(list).to_dict()
    atr_map = {tk: s.set_index("date")["atr14"] for tk, s in daily[daily["T"].isin(members)].groupby("T")}
    print(f"   universe: {len(members)} distinct stocks over {uni['date'].nunique()} sessions", flush=True)

    print(f"3/4 5-minute bars for {len(members)} stocks…", flush=True)
    recs, done = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ticker_records, tk, ds, atr_map.get(tk, pd.Series(dtype=float)),
                          min(ds) - pd.Timedelta(days=30), max(ds)): tk for tk, ds in members.items()}
        for f in as_completed(futs):
            done += 1
            try:
                recs += f.result()
            except Exception as e:
                print(f"   {futs[f]} failed: {e}", flush=True)
            if done % 25 == 0:
                print(f"   {done}/{len(members)}", flush=True)
    allt = pd.DataFrame(recs)
    # Idea 1 — direction of the whole market's first 5-min candle (SPY), known at 09:35
    try:
        spy = bars_5m("SPY", start - pd.Timedelta(days=5), end)
        f = spy[spy.index.strftime("%H:%M") == "09:30"]
        spy_dir = pd.Series(np.sign(f["close"].values - f["open"].values), index=f.index.normalize())
        allt["spy_dir"] = pd.to_datetime(allt["date"]).map(spy_dir)
    except Exception as e:
        print(f"   SPY first candle unavailable: {e}", flush=True)
        allt["spy_dir"] = np.nan
    allt.to_csv(os.path.join(a.out, "sip_all_records.csv"), index=False)

    print("4/4 selecting stocks in play & scoring…", flush=True)
    allt = allt.dropna(subset=["rvol"])
    inplay = allt[allt["rvol"] > 1].sort_values("rvol", ascending=False).groupby("date").head(a.top)
    inplay.to_csv(os.path.join(a.out, "sip_inplay_trades.csv"), index=False)

    def block(df: pd.DataFrame, title: str) -> list[str]:
        L = [f"\n── {title} ──", HDR]
        for m in MODES:
            L.append(line(m, daily_portfolio(df, m, a.cost_bps)))
        L.append(line("breakout_all (whole univ.)",
                      daily_portfolio(allt[allt["date"].isin(df["date"].unique())], "breakout_candle", a.cost_bps)))
        return L

    lines = [f"STOCKS IN PLAY — ORB  |  universe top {a.universe} by $volume (point-in-time, ex-ETF)  |  "
             f"top {a.top} by first-bar RVOL  |  cost {a.cost_bps:g} bps/trade",
             f"{allt['date'].min().date()} → {allt['date'].max().date()}  |  {allt['tk'].nunique()} stocks traded"]
    lines += block(inplay, "All days")
    lines += block(inplay[inplay["date"] >= a.recent_from], f"Since {a.recent_from}")
    lines.append("\n── breakout_candle by year (stocks in play) ──")
    pb = daily_portfolio(inplay, "breakout_candle", a.cost_bps)
    pd_ = daily_portfolio(inplay, "long_drift", a.cost_bps)
    lines.append(f"{'year':6s} {'days':>5s} {'net bps':>8s} {'t':>6s} {'| drift bps':>11s}")
    for y in sorted(set(pb.index.year)):
        s, dd = pb[pb.index.year == y], pd_[pd_.index.year == y]
        lines.append(f"{y:<6d} {len(s):5d} {s.mean() * 1e4:8.2f} {s.mean() / (s.std() / np.sqrt(len(s))):6.2f} "
                     f"{dd.mean() * 1e4:11.2f}")
    # edge over drift on identical days
    j = pd.concat([pb, pd_], axis=1, keys=["b", "d"]).dropna()
    diff = j["b"] - j["d"]
    lines += ["",
              f"Breakout minus long-drift (same days): {diff.mean() * 1e4:.2f} bps/day, "
              f"t={diff.mean() / (diff.std() / np.sqrt(len(diff))):.2f}  (>0 and t>2 → edge is not just bull drift)",
              "net bps = average daily portfolio return after costs; t > 2 ≈ significant; 1x notional, no leverage."]
    summary = "\n".join(lines)
    print("\n" + summary)
    open(os.path.join(a.out, "sip_summary.txt"), "w").write(summary + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#0d1422")
        ax.set_facecolor("#0d1422")
        cols = {"breakout_candle": "#e0b25c", "breakout_atr": "#6fcf97", "direction_open": "#7fb6e6",
                "long_drift": "#5d6b80"}
        for m, c in cols.items():
            s = daily_portfolio(inplay, m, a.cost_bps)
            ax.plot(s.index, s.cumsum() * 100, label=m, color=c, lw=1.8)
        s = daily_portfolio(allt, "breakout_candle", a.cost_bps)
        ax.plot(s.index, s.cumsum() * 100, label="breakout_all (whole universe)", color="#c58af9", lw=1.2, ls="--")
        ax.axhline(0, color="#8a96a8", lw=0.8)
        ax.set_title(f"Stocks in Play ORB — cumulative % after {a.cost_bps:g} bps (1x, equal weight)",
                     color="#e8edf5", loc="left", fontsize=14, weight="bold")
        ax.tick_params(colors="#8a96a8")
        ax.grid(color="#1e2a3d", lw=0.6)
        for sp in ax.spines.values():
            sp.set_color("#1e2a3d")
        leg = ax.legend(facecolor="#0d1422", edgecolor="#1e2a3d")
        for txt in leg.get_texts():
            txt.set_color("#e8edf5")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "sip.png"), dpi=130, facecolor="#0d1422")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
