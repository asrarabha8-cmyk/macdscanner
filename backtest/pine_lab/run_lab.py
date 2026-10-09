"""
Pine Lab — fair comparison of the Arun K Bhaskar Pine strategies on the S&P 500.

Same universe, same period, same exits for every strategy:
  * entry at the next bar's open after the signal bar closes
  * fixed holds of 5 / 10 / 20 bars (exit at that bar's close)
  * ATR bracket: stop 1.5×ATR14, target 3×ATR14, max 20 bars (stop wins on same-bar hits)
Baseline: every trade is compared with the average 10-bar return of ALL S&P 500
stocks entered on the same day ("date-matched excess"), which removes market timing.

Usage:  python run_lab.py --years 10 --out results
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from signals import all_signals, skipper_regime, atr

HOLDS = (5, 10, 20)
STOP_ATR, TGT_ATR, MAX_BARS = 1.5, 3.0, 20


def download(symbols, years):
    frames = {}
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                                 group_by="ticker", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print("download retry", attempt, e)
                time.sleep(5)
        for s in chunk:
            try:
                d = df[s][["Open", "High", "Low", "Close", "Volume"]].dropna()
            except KeyError:
                continue
            if len(d) > 300:
                frames[s] = d
        print(f"downloaded {min(k + 100, len(symbols))}/{len(symbols)}", flush=True)
    return frames


def bracket(o, h, l, c, a, i, side):
    """ATR bracket from entry at open[i+1]. Returns (return, R multiple) or None."""
    e = i + 1
    if e >= len(c) or np.isnan(a[i]) or a[i] <= 0:
        return None
    entry = o[e]
    risk = STOP_ATR * a[i]
    stop = entry - side * risk
    tgt = entry + side * TGT_ATR * a[i]
    last = min(e + MAX_BARS - 1, len(c) - 1)
    if last - e < MAX_BARS - 1:
        return None
    def done(px):
        ret = px / entry - 1 if side == 1 else entry / px - 1
        return ret, side * (px - entry) / risk

    for j in range(e, last + 1):
        if side == 1:
            if l[j] <= stop:                       # gaps through the stop fill at the open
                return done(min(stop, o[j]))
            if h[j] >= tgt:
                return done(max(tgt, o[j]))
        else:
            if h[j] >= stop:
                return done(max(stop, o[j]))
            if l[j] <= tgt:
                return done(min(tgt, o[j]))
    return done(c[last])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N symbols")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in Path(__file__).with_name("sp500_symbols.txt").read_text().split() if s.strip()]
    if args.limit:
        symbols = symbols[:args.limit]
    data = download(symbols, args.years)
    print("symbols with data:", len(data), flush=True)

    # forward returns for every stock-day (baseline)
    fwd = {}
    for s, d in data.items():
        o = d["Open"].to_numpy(); c = d["Close"].to_numpy()
        f = {}
        for H in HOLDS:
            r = np.full(len(c), np.nan)
            if len(c) > H + 1:
                r[:-H - 1] = c[H:-1] / o[1:-H] - 1   # enter open[i+1], exit close[i+H]
            f[H] = pd.Series(r, index=d.index)
        fwd[s] = f
    day_mean = {H: pd.concat([fwd[s][H] for s in fwd], axis=1).mean(axis=1) for H in HOLDS}
    base = {H: float(np.nanmean(np.concatenate([fwd[s][H].to_numpy() for s in fwd]))) for H in HOLDS}

    trades = []
    t0 = time.time()
    for n_done, (s, d) in enumerate(data.items(), 1):
        o, h, l, c, v = (d[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
        try:
            sig = all_signals(o, h, l, c, v)
            regL, regS = skipper_regime(o, h, l, c, v)
        except Exception as e:  # noqa: BLE001
            print("signal error", s, e, flush=True)
            continue
        a = atr(h, l, c, 14)
        idx = d.index
        for name, (L, S) in sig.items():
            for side, arr, reg in ((1, L, regL), (-1, S, regS)):
                for i in np.flatnonzero(arr):
                    if i < 260 or i + 1 >= len(c):   # warm-up: 250-bar lookback indicators
                        continue
                    rec = {"strategy": name, "side": "Long" if side == 1 else "Short",
                           "symbol": s, "date": idx[i], "regime_ok": bool(reg[i])}
                    for H in HOLDS:
                        r = fwd[s][H].iat[i]
                        rec[f"r{H}"] = side * r if not np.isnan(r) else np.nan
                        dm = day_mean[H].get(idx[i], np.nan)
                        rec[f"x{H}"] = side * (r - dm) if not (np.isnan(r) or np.isnan(dm)) else np.nan
                    b = bracket(o, h, l, c, a, i, side)
                    rec["br_ret"], rec["br_R"] = (b if b else (np.nan, np.nan))
                    trades.append(rec)
        if n_done % 50 == 0:
            print(f"signals {n_done}/{len(data)}  {time.time() - t0:.0f}s", flush=True)

    tr = pd.DataFrame(trades)
    tr.to_csv(out / "trades.csv.gz", index=False, compression="gzip")
    years_span = max(1e-9, (max(d.index[-1] for d in data.values()) - min(d.index[0] for d in data.values())).days / 365.25)

    def summarize(g):
        x10 = g["x10"].dropna()
        br = g["br_R"].dropna()
        gains = g["br_ret"][g["br_ret"] > 0].sum(); losses = -g["br_ret"][g["br_ret"] < 0].sum()
        half = g["date"] >= pd.Timestamp(g["date"].min()) + (pd.Timestamp(g["date"].max()) - pd.Timestamp(g["date"].min())) / 2
        return pd.Series({
            "trades": len(g),
            "per_year": len(g) / years_span,
            "win10": (g["r10"] > 0).mean(),
            "avg_r5": g["r5"].mean(), "avg_r10": g["r10"].mean(), "avg_r20": g["r20"].mean(),
            "excess10": x10.mean(),
            "t_excess10": x10.mean() / (x10.std(ddof=1) / math.sqrt(len(x10))) if len(x10) > 2 and x10.std() > 0 else np.nan,
            "excess10_1st_half": g.loc[~half, "x10"].mean(),
            "excess10_2nd_half": g.loc[half, "x10"].mean(),
            "excess20": g["x20"].mean(),
            "bracket_win": (br > 0).mean() if len(br) else np.nan,
            "bracket_avgR": br.mean() if len(br) else np.nan,
            "bracket_PF": gains / losses if losses > 0 else np.nan,
        })

    tr["date"] = pd.to_datetime(tr["date"])
    summ = tr.groupby(["strategy", "side"]).apply(summarize).reset_index()
    filt = tr[tr["regime_ok"]].groupby(["strategy", "side"]).apply(summarize).reset_index()
    filt["strategy"] = filt["strategy"] + " + Skipper filter"
    filt = filt[filt["strategy"] != "Sideways Market Skipper + Skipper filter"]
    allres = pd.concat([summ, filt], ignore_index=True).sort_values("excess10", ascending=False)
    allres.to_csv(out / "summary.csv", index=False)

    # random-entry baseline (all stock-days)
    meta = {"symbols": len(data), "years": round(years_span, 1),
            "baseline_long_avg_r5": base[5], "baseline_long_avg_r10": base[10], "baseline_long_avg_r20": base[20],
            "baseline_long_win10": float(np.nanmean(np.concatenate([(fwd[s][10] > 0).to_numpy()[~np.isnan(fwd[s][10].to_numpy())] for s in fwd])))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    fmt = allres.copy()
    for col in fmt.columns:
        if col.startswith(("win", "avg_r", "excess", "bracket_win")):
            fmt[col] = (fmt[col] * 100).round(2)
    for col in ("per_year", "t_excess10", "bracket_avgR", "bracket_PF"):
        fmt[col] = fmt[col].round(2)
    txt = ["# Pine Lab results", "", "```", json.dumps(meta, indent=2), "```", "",
           "Percent columns are in %. excess10 = 10-bar return minus the same-day average of all S&P 500 stocks.",
           "", fmt.to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
