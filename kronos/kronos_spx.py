#!/usr/bin/env python3
"""
Kronos SPX — intraday forecast experiment + walk-forward backtest
=================================================================

Re-creates the "Kronos / SPX" experiment: feed the model only the candles
available up to a cutoff time (default 11:00 New York), ask it for the rest
of the session on 5-minute candles, then compare with what really happened.

Improvements over the single-path demo:
  * N independent sampled paths -> mean path + 10–90% band (Monte Carlo)
  * Walk-forward backtest over every available session
  * Naive baseline ("price stays at the last observed close") for honesty

Usage
-----
  # one session, same setup as the original chart
  python kronos_spx.py single --date 2026-10-02

  # backtest every session in the last ~60 days
  python kronos_spx.py backtest

  # other assets / settings
  python kronos_spx.py single --ticker QQQ --date 2026-10-02 --cutoff 10:30 --paths 50
  python kronos_spx.py backtest --model small --paths 10

Setup (once)
------------
  git clone https://github.com/shiyu-coder/Kronos.git
  pip install -r requirements.txt
  (put this file next to the Kronos folder, or pass --kronos-dir)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
SESSION_OPEN = "09:30"
LAST_BAR = "15:55"           # opening time of the final 5m candle
BAR_MIN = 5

MODELS = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
}


# ─────────────────────────────── data ───────────────────────────────
def load_candles(ticker: str, csv: str | None = None) -> pd.DataFrame:
    """5-minute regular-session candles, index = naive New York time.

    Source: a CSV (columns: timestamps/open/high/low/close[/volume]) or
    yfinance (Yahoo keeps ~60 days of 5m history).
    """
    if csv:
        df = pd.read_csv(csv)
        ts_col = next(c for c in df.columns if c.lower() in ("timestamps", "timestamp", "datetime", "date", "time"))
        df.index = pd.to_datetime(df.pop(ts_col))
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_convert(NY).tz_localize(None)
    else:
        import time
        import yfinance as yf
        raw = pd.DataFrame()
        for attempt in range(4):
            try:
                raw = yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=False)
                if raw.empty:
                    raw = yf.Ticker(ticker).history(period="60d", interval="5m", auto_adjust=False)
            except Exception as e:
                print(f"yfinance attempt {attempt + 1} failed: {e}")
            if not raw.empty:
                break
            time.sleep(15 * (attempt + 1))
        if raw.empty:
            sys.exit(f"No data returned for {ticker} from Yahoo (rate-limited or bad symbol)")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        df = raw
        df.index = df.index.tz_convert(NY).tz_localize(None)

    df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])].astype(float)
    df = df.between_time(SESSION_OPEN, LAST_BAR)
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"]).sort_index()
    return df


@dataclass
class Window:
    day: pd.Timestamp
    ctx: pd.DataFrame        # what the model sees
    fut: pd.DataFrame        # hidden truth (cutoff -> 15:55)
    morning: pd.DataFrame    # same-day candles before cutoff (for the chart)


def make_window(df: pd.DataFrame, day: str | pd.Timestamp, cutoff: str, lookback: int,
                pred_len: int | None) -> Window | None:
    day = pd.Timestamp(day).normalize()
    cut = day + pd.Timedelta(cutoff + ":00")
    end = day + pd.Timedelta(LAST_BAR + ":00")
    ctx = df[df.index < cut].iloc[-lookback:]
    fut = df[(df.index >= cut) & (df.index <= end)]
    if pred_len:
        fut = fut.iloc[:pred_len]
    if len(ctx) < lookback or len(fut) < 2:
        return None
    morning = ctx[ctx.index >= day]
    return Window(day, ctx, fut, morning)


# ─────────────────────────────── model ──────────────────────────────
def load_predictor(model_key: str, kronos_dir: str, device: str | None):
    sys.path.insert(0, os.path.abspath(kronos_dir))
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        sys.exit(f"Cannot import Kronos from '{kronos_dir}'. Clone it first:\n"
                 f"  git clone https://github.com/shiyu-coder/Kronos.git\n({e})")
    m_id, t_id, ctx_len = MODELS[model_key]
    tok = KronosTokenizer.from_pretrained(t_id)
    mdl = Kronos.from_pretrained(m_id)
    return KronosPredictor(mdl, tok, device=device, max_context=ctx_len)


def forecast_paths(predictor, w: Window, n_paths: int, seed: int,
                   T: float = 1.0, top_p: float = 0.9, use_volume: bool = False) -> np.ndarray:
    """Return array (n_paths, pred_len, 4) of sampled OHLC paths.

    Each path is an independent sample (sample_count=1 per batch item), so
    path 0 with the same seed matches a single-run experiment, and the set of
    paths gives a distribution instead of one guess.
    """
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = w.ctx[["open", "high", "low", "close"]].copy()
    if use_volume and "volume" in w.ctx and w.ctx["volume"].sum() > 0:
        x["volume"] = w.ctx["volume"].values
    else:                                   # indices have no volume -> zero-fill
        x["volume"] = 0.0
        x["amount"] = 0.0

    x_ts = pd.Series(w.ctx.index)
    y_ts = pd.Series(w.fut.index)
    pred_len = len(w.fut)

    out = []
    batch = 16
    for i in range(0, n_paths, batch):
        k = min(batch, n_paths - i)
        res = predictor.predict_batch([x] * k, [x_ts] * k, [y_ts] * k, pred_len=pred_len,
                                      T=T, top_k=0, top_p=top_p, sample_count=1, verbose=False)
        out += [r[["open", "high", "low", "close"]].values for r in res]
    return np.stack(out)


# ─────────────────────────────── metrics ────────────────────────────
def score(w: Window, paths: np.ndarray) -> dict:
    last = w.ctx["close"].iloc[-1]
    actual = w.fut["close"].values
    p0 = paths[0, :, 3]
    mean = paths[:, :, 3].mean(0)
    lo, hi = np.percentile(paths[:, :, 3], [10, 90], axis=0)
    a_end = actual[-1]
    return {
        "date": w.day.date().isoformat(),
        "last_observed": last,
        "actual_close": a_end,
        "path0_close": p0[-1],
        "mean_close": mean[-1],
        "err_path0": abs(p0[-1] - a_end),
        "err_mean": abs(mean[-1] - a_end),
        "err_naive": abs(last - a_end),
        "path_mae_mean": np.abs(mean - actual).mean(),
        "path_mae_naive": np.abs(last - actual).mean(),
        "dir_hit": int(np.sign(mean[-1] - last) == np.sign(a_end - last)),
        "prob_up": float((paths[:, -1, 3] > last).mean()),
        "actual_up": int(a_end > last),
        "band_cover": float(((actual >= lo) & (actual <= hi)).mean()),
        "end_in_band": int(lo[-1] <= a_end <= hi[-1]),
    }


# ─────────────────────────────── charts ─────────────────────────────
BG, PANEL, GRID = "#0d1422", "#0f1828", "#1e2a3d"
TXT, MUTED = "#e8edf5", "#8a96a8"
GOLD, BLUE, CANDLE = "#e0b25c", "#7fb6e6", "#d4dae4"


def _candles(ax, xs, o, h, l, c, color, width=0.6, alpha=1.0):
    from matplotlib.patches import Rectangle
    for x, oo, hh, ll, cc in zip(xs, o, h, l, c):
        ax.vlines(x, ll, hh, color=color, lw=0.8, alpha=alpha * 0.8)
        ax.add_patch(Rectangle((x - width / 2, min(oo, cc)), width, max(abs(cc - oo), 1e-6),
                               facecolor=color, edgecolor=color, lw=0.8, alpha=alpha))


def plot_single(w: Window, paths: np.ndarray, s: dict, ticker: str, model_key: str,
                cutoff: str, seed: int, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_m, n_f = len(w.morning), len(w.fut)
    xm = np.arange(n_m)
    xf = np.arange(n_m, n_m + n_f)
    p0 = paths[0]
    mean = paths[:, :, 3].mean(0)
    lo, hi = np.percentile(paths[:, :, 3], [10, 90], axis=0)
    last = s["last_observed"]

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.06, 0.92, f"KRONOS / {ticker.replace('^GSPC', 'SPX')}", color=TXT, fontsize=26, weight="bold")
    fig.text(0.06, 0.875,
             f"{w.day:%d %b %Y}".upper() + f"  |  Historical forecast at {cutoff} ET  |  "
             f"{len(w.ctx)} input candles / {n_f} forecast candles  |  {paths.shape[0]} sampled paths",
             color=MUTED, fontsize=11)

    ax = fig.add_axes([0.06, 0.24, 0.88, 0.6], facecolor=BG)
    if n_m:
        _candles(ax, xm, *(w.morning[c].values for c in ("open", "high", "low", "close")), CANDLE)
    ax.fill_between(xf, lo, hi, color="#c9a24f", alpha=0.22, lw=0, label="Kronos 10–90% band")
    _candles(ax, xf, p0[:, 0], p0[:, 1], p0[:, 2], p0[:, 3], GOLD, alpha=0.55)
    ax.plot(np.r_[xf[0] - 1, xf], np.r_[last, p0[:, 3]], color=GOLD, lw=1.2, alpha=0.6,
            label=f"Kronos-{model_key} (one sampled path, seed {seed})")
    ax.plot(np.r_[xf[0] - 1, xf], np.r_[last, mean], color=GOLD, lw=2.4, label="Kronos mean path")
    ax.plot(np.r_[xf[0] - 1, xf], np.r_[last, w.fut["close"].values], color=BLUE, lw=1.8, ls="--",
            label="Actual close path (hidden from model)")
    ax.axvline(n_m - 0.5, color=MUTED, ls=":", lw=1)

    idx = list(w.morning.index) + list(w.fut.index)
    ticks = [i for i, t in enumerate(idx) if t.minute == 0 or (t.hour == 9 and t.minute == 30)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([idx[i].strftime("%H:%M") for i in ticks], color=MUTED)
    ax.set_xlim(-1, n_m + n_f)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_ylabel("points", color=MUTED)
    ax.set_xlabel("5-minute candle opening time | New York", color=MUTED)
    leg = ax.legend(loc="lower left", facecolor=BG, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TXT)

    tiles = [("LAST OBSERVED", f"{last:,.2f}", TXT),
             ("MEAN FORECAST CLOSE", f"{s['mean_close']:,.2f}", GOLD),
             (f"ACTUAL {LAST_BAR} BAR CLOSE", f"{s['actual_close']:,.2f}", BLUE),
             ("ABS ERROR (MEAN / 1 PATH)", f"{s['err_mean']:.2f} / {s['err_path0']:.2f}", TXT),
             ("P(CLOSE > LAST)", f"{s['prob_up']:.0%}", TXT)]
    for i, (lab, val, col) in enumerate(tiles):
        x = 0.06 + i * 0.18
        fig.text(x, 0.135, lab, color=MUTED, fontsize=9)
        fig.text(x, 0.08, val, color=col, fontsize=22, weight="bold")
    fig.text(0.06, 0.03,
             f"Official pre-trained weights | T=1.0 | top-p=0.9 | OHLC only; volume/amount zero-filled | "
             f"naive 'no change' error = {s['err_naive']:.2f} | not investment advice",
             color=MUTED, fontsize=8)
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def plot_backtest(res: pd.DataFrame, ticker: str, model_key: str, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), facecolor=BG, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(res))
    ax = axes[0]
    ax.bar(x - 0.2, res["err_mean"], 0.4, color=GOLD, label="Kronos mean-path error")
    ax.bar(x + 0.2, res["err_naive"], 0.4, color="#5d6b80", label="Naive 'no change' error")
    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in res["date"]], rotation=90, fontsize=7, color=MUTED)
    ax.set_ylabel("abs error at session close (points)", color=MUTED)
    ax.set_title(f"KRONOS-{model_key.upper()} / {ticker.replace('^GSPC', 'SPX')} — walk-forward, {len(res)} sessions",
                 color=TXT, loc="left", fontsize=15, weight="bold")

    ax2 = axes[1]
    cum = (res["err_naive"] - res["err_mean"]).cumsum()
    ax2.plot(x, cum, color=GOLD, lw=2)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_ylabel("cumulative edge vs naive\n(points, >0 = Kronos better)", color=MUTED)
    ax2.set_xticks([])
    for a in axes:
        a.set_facecolor(BG)
        a.tick_params(colors=MUTED)
        a.grid(color=GRID, lw=0.5, axis="y")
        for sp in a.spines.values():
            sp.set_color(GRID)
    leg = ax.legend(facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TXT)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def summarize(res: pd.DataFrame) -> str:
    n = len(res)
    win = (res["err_mean"] < res["err_naive"]).mean()
    # direction only matters when the model takes a side with some conviction
    conv = res[(res["prob_up"] >= 0.65) | (res["prob_up"] <= 0.35)]
    conv_hit = ((conv["prob_up"] > 0.5).astype(int) == conv["actual_up"]).mean() if len(conv) else float("nan")
    lines = [
        f"Sessions tested                 : {n}",
        f"Median close error  Kronos mean : {res['err_mean'].median():.2f}",
        f"Median close error  single path : {res['err_path0'].median():.2f}",
        f"Median close error  naive       : {res['err_naive'].median():.2f}",
        f"Path MAE  Kronos / naive        : {res['path_mae_mean'].mean():.2f} / {res['path_mae_naive'].mean():.2f}",
        f"Sessions Kronos beat naive      : {win:.0%}",
        f"Direction hit rate (all)        : {res['dir_hit'].mean():.0%}",
        f"Direction hit rate (P>=65%/<=35%): {conv_hit:.0%}  on {len(conv)} sessions",
        f"Actual inside 10–90% band       : {res['band_cover'].mean():.0%} of bars (ideal ≈ 80%)",
    ]
    return "\n".join(lines)


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["single", "backtest"])
    ap.add_argument("--ticker", default="^GSPC", help="Yahoo symbol: ^GSPC (SPX), ^NDX, SPY, QQQ, AAPL…")
    ap.add_argument("--csv", help="use your own 5m CSV instead of yfinance")
    ap.add_argument("--date", help="session date for single mode (default: latest)")
    ap.add_argument("--cutoff", default="11:00", help="New York time the model stops seeing data")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--pred-len", type=int, default=None, help="default: until the 15:55 candle")
    ap.add_argument("--model", choices=MODELS, default="base")
    ap.add_argument("--paths", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-volume", action="store_true", help="feed real volume (stocks/ETFs only)")
    ap.add_argument("--device", default=None, help="cpu / cuda:0 / mps (auto if omitted)")
    ap.add_argument("--kronos-dir", default="Kronos")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    df = load_candles(a.ticker, a.csv)
    days = sorted(set(df.index.normalize()))
    print(f"Loaded {len(df)} candles, {days[0].date()} → {days[-1].date()}")
    tag = a.ticker.replace("^", "")

    predictor = load_predictor(a.model, a.kronos_dir, a.device)

    if a.mode == "single":
        day = pd.Timestamp(a.date) if a.date else days[-1]
        w = make_window(df, day, a.cutoff, a.lookback, a.pred_len)
        if w is None:
            sys.exit("Not enough data for that date/cutoff (need lookback candles before and the session after).")
        paths = forecast_paths(predictor, w, a.paths, a.seed, use_volume=a.use_volume)
        s = score(w, paths)
        png = os.path.join(a.out, f"kronos_{tag}_{s['date']}.png")
        plot_single(w, paths, s, a.ticker, a.model, a.cutoff, a.seed, png)
        pd.DataFrame({"actual": w.fut["close"].values, "path0": paths[0, :, 3],
                      "mean": paths[:, :, 3].mean(0),
                      "p10": np.percentile(paths[:, :, 3], 10, 0),
                      "p90": np.percentile(paths[:, :, 3], 90, 0)},
                     index=w.fut.index).to_csv(png.replace(".png", ".csv"))
        for k, v in s.items():
            print(f"{k:15s}: {v:.2f}" if isinstance(v, float) else f"{k:15s}: {v}")
        print("Chart →", png)
    else:
        rows = []
        for i, d in enumerate(days):
            w = make_window(df, d, a.cutoff, a.lookback, a.pred_len)
            if w is None:
                continue
            paths = forecast_paths(predictor, w, a.paths, a.seed + i, use_volume=a.use_volume)
            s = score(w, paths)
            rows.append(s)
            print(f"{s['date']}  kronos {s['err_mean']:7.2f}  naive {s['err_naive']:7.2f}  "
                  f"P(up) {s['prob_up']:.0%}  {'✓' if s['dir_hit'] else '✗'}", flush=True)
        if not rows:
            sys.exit("No session had enough history; lower --lookback.")
        res = pd.DataFrame(rows)
        csv = os.path.join(a.out, f"backtest_{tag}_{a.model}.csv")
        res.to_csv(csv, index=False)
        png = csv.replace(".csv", ".png")
        plot_backtest(res, a.ticker, a.model, png)
        summary = summarize(res)
        open(csv.replace(".csv", "_summary.txt"), "w").write(summary + "\n")
        print("\n" + summary + f"\n\nResults → {csv}\nChart   → {png}")


if __name__ == "__main__":
    main()
