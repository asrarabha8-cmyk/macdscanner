"""Alerts review — rebuild every Telegram alert the scanners sent and score it.

Outcome rule for every alert with a stop and a target:
  win     -> target touched before the stop
  loss    -> stop touched before the target
  open    -> neither yet (current move reported)
If both are touched inside the same bar we count it as a loss (conservative).

Runs inside GitHub Actions (needs yfinance). Writes review/out/*.csv + summary.json
"""
from __future__ import annotations

import datetime as dt
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "review" / "raw"
OUT = ROOT / "review" / "out"
OUT.mkdir(parents=True, exist_ok=True)
NOW = pd.Timestamp.now(tz="UTC")
sys.path.insert(0, str(ROOT))


def load(mod_name: str, path: str):
    os.environ.setdefault("PORTFOLIO", "")
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ prices
_cache: dict = {}


def bars(ticker: str, interval: str, period: str) -> pd.DataFrame:
    key = (ticker, interval, period)
    if key not in _cache:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                         prepost=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("UTC")
        _cache[key] = df
    return _cache[key]


def score(ticker, t0, entry, stop, target, side="long", interval="1h", period="730d", max_days=None):
    """Walk forward from t0 (exclusive). Returns dict(result, days, exit, move_pct, mfe_pct, mae_pct)."""
    df = bars(ticker, interval, period)
    if df.empty:
        return dict(result="no-data")
    after = df[df.index > pd.Timestamp(t0)]
    if max_days:
        after = after[after.index <= pd.Timestamp(t0) + pd.Timedelta(days=max_days)]
    if after.empty:
        return dict(result="open", move_pct=0.0, mfe_pct=0.0, mae_pct=0.0, days=0)
    sgn = 1 if side == "long" else -1
    hi, lo = after["High"].to_numpy(), after["Low"].to_numpy()
    for i in range(len(after)):
        hit_stop = lo[i] <= stop if sgn > 0 else hi[i] >= stop
        hit_tgt = hi[i] >= target if sgn > 0 else lo[i] <= target
        if hit_stop or hit_tgt:
            res = "loss" if hit_stop else "win"
            px = stop if hit_stop else target
            seg = after.iloc[: i + 1]
            break
    else:
        res, px, seg = "open", float(after["Close"].iloc[-1]), after
    days = (seg.index[-1] - pd.Timestamp(t0)).total_seconds() / 86400
    mfe = (seg["High"].max() / entry - 1) * 100 * sgn if sgn > 0 else (1 - seg["Low"].min() / entry) * 100
    mae = (seg["Low"].min() / entry - 1) * 100 if sgn > 0 else (1 - seg["High"].max() / entry) * 100
    return dict(result=res, exit=round(px, 4), move_pct=round((px / entry - 1) * 100 * sgn, 2),
                mfe_pct=round(mfe, 2), mae_pct=round(mae, 2), days=round(days, 1))


rows: list[dict] = []
notes: dict = {}


def add(scanner, ticker, t0, entry, stop, target, side="long", interval="1h", period="730d", extra=None,
        max_days=None):
    try:
        r = score(ticker, t0, entry, stop, target, side, interval, period, max_days)
    except Exception as e:  # noqa
        r = dict(result="error", err=str(e)[:80])
    rows.append(dict(scanner=scanner, ticker=ticker, time_utc=str(pd.Timestamp(t0))[:16], side=side,
                     entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4), **r,
                     **(extra or {})))


# ------------------------------------------------------------------ run logs
def runs(folder: str):
    """[(created_utc, text)] for one scanner's dumped run logs."""
    out = []
    for f in sorted(glob.glob(str(RAW / folder / "*.txt"))):
        txt = Path(f).read_text(errors="replace")
        head = txt.splitlines()[0]
        m = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", head)
        if m:
            out.append((pd.Timestamp(m.group(1)), txt))
    return out


# ------------------------------------------------------------------ sources
def src_csv_logged():
    # Bottom hunter — daily signals with stop/target
    p = ROOT / "logs/bottom/signals.csv"
    if p.exists():
        df = pd.read_csv(p).drop_duplicates(["ticker", "signal_date"])
        for r in df.itertuples():
            t0 = pd.Timestamp(r.signal_date, tz="America/New_York") + pd.Timedelta(hours=16)
            add("صيد القاع", r.ticker, t0, r.close, r.stop, r.target, interval="1d", period="1y")
    # SMC daily
    p = ROOT / "logs/smc_daily/lookback.csv"
    if p.exists():
        df = pd.read_csv(p).drop_duplicates(["ticker", "date"])
        for r in df.itertuples():
            t0 = pd.Timestamp(r.date, tz="America/New_York") + pd.Timedelta(hours=16)
            add("SMC يومي", r.ticker, t0, r.close, r.stop, r.target, interval="1d", period="1y")
    # SMC breakout (artifact)
    for p in glob.glob(str(RAW / "artifacts" / "*alerts_log*.csv")):
        df = pd.read_csv(p)
        notes["smc_breakout_cols"] = list(df.columns)
        tcol = next((c for c in ("time", "time_utc", "date", "ts") if c in df.columns), None)
        df = df.drop_duplicates(["ticker", tcol] if tcol else ["ticker"])
        for r in df.to_dict("records"):
            t0 = pd.Timestamp(r[tcol])
            t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
            if t0.hour == 0:  # date only -> signal on that close
                t0 = t0.tz_convert("America/New_York").normalize() + pd.Timedelta(hours=16)
            add("اختراق SMC (فلتر خالد)", r["ticker"], t0, float(r["entry"]), float(r["stop"]),
                float(r.get("t1") or r.get("target")), interval="1d", period="1y")


def src_cascade():
    """MACD cascade: runs that printed no 'no-signal' line sent a Telegram message -> replay at that time."""
    slow = load("slow_pf", "macd_slow_scanner.py")
    os.environ["PORTFOLIO"] = " ".join(sorted(slow.ALWAYS - slow.CRYPTO))
    m = load("cascade", "macd_cascade_scanner.py")
    sent = [(t, x) for t, x in runs("macd_cascade_scan") if "success" in x.splitlines()[0]
            and "لا إشارات دخول" not in x and "بعد فلتر السيولة" in x]
    notes["cascade_runs_total"] = len(runs("macd_cascade_scan"))
    notes["cascade_runs_with_alert"] = len(sent)
    if not sent:
        return
    tick = sorted(set(m.TICKERS))
    h1_all = m.fetch(tick, "180d", "1h")
    m15_all = m.fetch(tick, "60d", "15m")
    seen = set()
    for t_run, _ in sent:
        session = m.us_session(t_run.to_pydatetime())
        names = tick if session else sorted(m.CRYPTO)
        for tk in names:
            if tk not in h1_all or tk not in m15_all:
                continue
            h1 = h1_all[tk][h1_all[tk].index <= t_run]
            m15 = m15_all[tk][m15_all[tk].index <= t_run]
            if len(m15) < 120:
                continue
            try:
                res = m.analyse(h1, m15, tk in m.CRYPTO)
            except Exception:
                continue
            if not res or res.get("state") != "entry":
                continue
            bar_t = m15.index[-2]
            if (tk, bar_t) in seen:
                continue
            seen.add((tk, bar_t))
            t0 = bar_t + pd.Timedelta(minutes=15)
            ex = dict(tp2=round(res["tp2"], 4), risk_pct=round(res["risk_pct"], 2))
            add("ماكد المتسلسل (هدف ١R)", tk, t0, res["price"], res["stop"], res["tp1"],
                interval="15m", period="60d", extra=ex)
            add("ماكد المتسلسل (هدف ٢R)", tk, t0, res["price"], res["stop"], res["tp2"],
                interval="15m", period="60d", extra=ex)


def src_slow():
    m = load("slow", "macd_slow_scanner.py")
    sent = [(t, x) for t, x in runs("macd_slow_scan") if "success" in x.splitlines()[0]
            and "لا إشارات دخول" not in x and "بعد فلتر السيولة" in x]
    notes["slow_runs_total"] = len(runs("macd_slow_scan"))
    notes["slow_runs_with_alert"] = len(sent)
    if not sent:
        return
    tick = sorted(set(m.SCAN_TICKERS))
    d1_all = m.fetch(tick, "2y", "1d")
    h1_all = m.fetch(tick, "180d", "1h")
    seen = set()
    for t_run, _ in sent:
        for tk in tick:
            if tk not in d1_all or tk not in h1_all:
                continue
            d1 = d1_all[tk][d1_all[tk].index <= t_run]
            h1 = h1_all[tk][h1_all[tk].index <= t_run]
            try:
                res = m.analyse(d1, h1, tk in m.CRYPTO)
            except Exception:
                continue
            if not res or res.get("state") != "entry":
                continue
            bar_t = h1.index[-2]
            if (tk, bar_t) in seen:
                continue
            seen.add((tk, bar_t))
            t0 = bar_t + pd.Timedelta(hours=1)
            add("ماكد البطيء (هدف ٢R)", tk, t0, res["price"], res["stop"], res["tp2"],
                interval="1h", period="730d", extra=dict(risk_pct=round(res["risk_pct"], 2)))


def src_ma():
    m = load("ma", "ma100_200_scanner.py")
    sent = [(t, x) for t, x in runs("ma_100_200_breakout_scan") if "success" in x.splitlines()[0]
            and "لا اختراقات اليوم" not in x]
    notes["ma_runs_with_alert"] = len(sent)
    if not sent:
        return
    tick = m.universe()
    d_all = {}
    for i in range(0, len(tick), 200):
        chunk = tick[i:i + 200]
        raw = yf.download(chunk, period="2y", interval="1d", group_by="ticker", auto_adjust=False,
                          progress=False, threads=True)
        for t in chunk:
            try:
                df = raw[t].dropna()
            except Exception:
                continue
            if not df.empty:
                df.index = pd.DatetimeIndex(df.index).tz_localize("America/New_York")
                d_all[t] = df
    seen = set()
    for t_run, _ in sent:
        day = t_run.tz_convert("America/New_York").normalize() + pd.Timedelta(hours=23)
        for tk, df in d_all.items():
            sl = df[df.index <= day]
            if len(sl) < 210:
                continue
            try:
                res = m.check(sl)
            except Exception:
                continue
            if not res:
                continue
            key = (tk, sl.index[-1])
            if key in seen:
                continue
            seen.add(key)
            t0 = sl.index[-1] + pd.Timedelta(hours=16)
            add("اختراق متوسط ١٠٠", tk, t0, res["price"], res["stop"], res["target"],
                interval="1d", period="1y")


def src_ema():
    slow = load("slow_pf2", "macd_slow_scanner.py")
    os.environ["PORTFOLIO"] = " ".join(sorted(slow.ALWAYS - slow.CRYPTO))
    m = load("ema", "ema_cross_scanner.py")
    sent = [(t, x) for t, x in runs("ema_cross_scan") if "success" in x.splitlines()[0]
            and "لا تقاطعات" not in x]
    notes["ema_runs_with_alert"] = len(sent)
    if not sent:
        return
    w_all = m_fetch_weekly(m.TICKERS)
    seen = set()
    for t_run, _ in sent:
        for tk, df in w_all.items():
            sl = df[df.index <= t_run]
            try:
                res = m.check(sl)
            except Exception:
                continue
            if not res or (tk, res["date"]) in seen:
                continue
            seen.add((tk, res["date"]))
            side = "long" if res["dir"] == "up" else "short"
            entry, stop = res["price"], res["ema_slow"]
            risk = abs(entry - stop) or entry * 0.01
            tgt = entry + 2 * risk if side == "long" else entry - 2 * risk
            # weekly crossover: judged on daily bars; stop = EMA48 at the signal, target = 2R
            add("تقاطع EMA 8/48 أسبوعي (هدف ٢R)", tk, t_run, entry, stop, tgt, side, interval="1d",
                period="1y", extra=dict(gap_pct=round(res["gap_pct"], 2)))


def m_fetch_weekly(tickers):
    out = {}
    raw = yf.download(sorted(tickers), period="5y", interval="1wk", group_by="ticker", auto_adjust=False,
                      progress=False, threads=True)
    for t in tickers:
        try:
            df = raw[t].dropna()
        except Exception:
            continue
        if not df.empty:
            df.index = pd.DatetimeIndex(df.index)
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            out[t] = df
    return out


def src_harmonic():
    for p in glob.glob(str(RAW / "artifacts" / "harmonic-scan__*harmonic_hits.json")):
        hits = json.load(open(p))
        t_run = pd.Timestamp("2026-10-13 23:20", tz="UTC")
        for h in hits:
            if h.get("stage") != "SIGNAL" or not h.get("tp1"):
                continue
            side = "long" if h["direction"] == "bullish" else "short"
            add("هارمونيك كارني (TP1)", h["symbol"], t_run, h["entry"], h["stop"], h["tp1"], side,
                interval="1d", period="1y", extra=dict(pattern=h["pattern"]))


def src_sa():
    p = ROOT / "docs/sa/data/history.json"
    if p.exists():
        notes["sa_history"] = json.load(open(p))


def src_optitrader():
    p = ROOT / "logs/trades.csv"
    if p.exists():
        df = pd.read_csv(p)
        notes["optitrader"] = df[["date", "symbol", "kind", "entry", "exit", "pnl", "pnl_pct", "reason"]] \
            .to_dict("records")


def src_flow():
    p = ROOT / "flow_confirm/data/signals.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    notes["flow_decisions"] = df["decision"].value_counts().to_dict()
    for r in df.itertuples():
        if str(r.decision) not in ("yes", "go", "enter", "دخول") or pd.isna(r.stop) or pd.isna(r.target):
            continue
        t0 = pd.Timestamp(r.time_ny, tz="America/New_York")
        side = "long" if r.direction == "bull" else "short"
        add("نظام التأكيد", r.ticker, t0, r.spot, r.stop, r.target, side, interval="5m", period="30d")


def src_sip():
    p = ROOT / "momentum/forward/forward_log.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["in_play"] == True]  # noqa
    pnl = np.where(df["side"] > 0, df["exit"] / df["entry"] - 1, 1 - df["exit"] / df["entry"])
    notes["sip"] = dict(trades=int(len(df)), wins=int((pnl > 0).sum()), losses=int((pnl <= 0).sum()),
                        avg_pct=round(float(pnl.mean() * 100), 3), days=int(df["date"].nunique()))


def src_spy0dte():
    hits = []
    for t, txt in runs("spy_0dte_orb_alert"):
        keep = [ln for ln in txt.splitlines()[1:] if re.search(r"(Call|Put|CALL|PUT|مؤهل|وقف|خروج|ربح|خسار|ORB|range|entry)", ln)]
        hits.append(dict(t=str(t)[:16], lines=keep[:12]))
    notes["spy0dte"] = hits


def main():
    for fn in (src_csv_logged, src_cascade, src_slow, src_ma, src_ema, src_harmonic, src_sa, src_optitrader,
               src_flow, src_sip, src_spy0dte):
        try:
            fn()
            print(fn.__name__, "ok", len(rows))
        except Exception as e:  # noqa
            notes[f"error_{fn.__name__}"] = repr(e)[:300]
            print(fn.__name__, "FAILED", e)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "alerts_scored.csv", index=False)
    if not df.empty:
        summ = df.groupby("scanner")["result"].value_counts().unstack(fill_value=0)
        summ.to_csv(OUT / "summary.csv")
        print(summ)
    (OUT / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
