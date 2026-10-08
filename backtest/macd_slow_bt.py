#!/usr/bin/env python3
"""
اختبار تاريخي لاستراتيجية MACD Slow — نفس منطق macd_slow_scanner.py حرفياً، شمعة شمعة.

  الإذن  (يومي، آخر يوم مغلق): ماكد فوق الصفر وفوق الإشارة، عمر التقاطع ≤ 10 أيام،
          تقاطعات الصفر ≤ 5 خلال 50 يوماً، وبُعده عن الصفر ≤ 60% من مداه.
  التأكيد (4H، آخر شمعة 4 ساعات مغلقة): ماكد فوق الإشارة.
  الزناد  (ساعة، آخر شمعة مغلقة): تقاطع صاعد قريب من الصفر (≤ 50% من المدى).
  الوقف  = أدنى قاع 20 ساعة − 0.5 ATR، مخاطرة ≤ 7%. الأهداف 1R / 2R / 3R.

البيانات: شموع الساعة من Yahoo متاحة لآخر ~730 يوماً فقط ← الاختبار على سنتين.
الدخول على افتتاح الساعة التالية، تكلفة 0.1% لكل جهة، خروج بالوقت بعد 70 ساعة (~10 جلسات).
ضابط عشوائي: نفس الوقف والهدف والمدة، لكن الدخول في ساعات عشوائية.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strategy_compare import simulate, stats  # noqa: E402

FAST, SLOW, SIGNAL = 12, 26, 9
PERMIT_BARS, CHOP_LOOK, CHOP_MAX, DIST_CAP_PCT = 10, 50, 5, 60
ENTRY_TOL_PCT, SWING_LOOK, ATR_BUF = 50, 20, 0.5
MIN_DOLLAR_VOL, MAX_RISK_PCT = 20_000_000, 7.0
MAX_BARS = 70


def macd(close):
    f = close.ewm(span=FAST, adjust=False).mean()
    s = close.ewm(span=SLOW, adjust=False).mean()
    line = f - s
    return line, line.ewm(span=SIGNAL, adjust=False).mean()


def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def daily_permit(d1):
    """لكل يوم D: هل الإذن مفتوح بناءً على الأيام حتى D (يُستخدم في جلسة اليوم التالي)."""
    md, sd = macd(d1["Close"])
    rng = md.abs().rolling(100, min_periods=100).max()
    cross = ((md > 0) & (md.shift(1) <= 0)).to_numpy()
    age = np.full(len(md), 9999)
    last = -1
    for k in range(len(md)):
        if cross[k]:
            last = k
        age[k] = 9999 if last < 0 else k - last
    pos = (md > 0)
    chop = (pos != pos.shift(1)).astype(int).rolling(CHOP_LOOK).sum()
    dv = (d1["Close"] * d1["Volume"]).rolling(20).mean()
    permit = (md > 0) & (md > sd) & (pd.Series(age, index=md.index) <= PERMIT_BARS) \
        & (chop <= CHOP_MAX) & (md <= rng * DIST_CAP_PCT / 100)
    return permit.fillna(False), dv


def to_4h_ok(h1):
    """لكل ساعة: هل ماكد آخر شمعة 4H مغلقة (قبل الساعة التالية) فوق الإشارة."""
    ny = h1.index.tz_convert("America/New_York")
    day = ny.date
    pos_in_day = h1.groupby(day).cumcount().to_numpy()
    bucket_in_day = pos_in_day // 4
    # مفتاح فريد لكل شمعة 4H
    keys = pd.Series(list(zip(day, bucket_in_day)), index=h1.index)
    uniq = pd.Index(pd.unique(keys))
    bid = uniq.get_indexer(keys)
    agg = h1.assign(b=bid).groupby("b").agg(Close=("Close", "last"))
    m4, s4 = macd(agg["Close"])
    above = (m4 > s4).to_numpy()
    # آخر شمعة مكتملة عند إغلاق الساعة k = شمعة الساعة k إذا كانت آخر ساعة فيها، وإلا التي قبلها
    nxt_bid = np.append(bid[1:], bid[-1] + 1)
    done_bid = np.where(nxt_bid != bid, bid, bid - 1)
    ok = np.zeros(len(h1), bool)
    valid = done_bid >= 60
    ok[valid] = above[done_bid[valid]]
    return ok


def signals(h1, d1, always=False):
    permit_d, dv = daily_permit(d1)
    ny_dates = pd.Index(h1.index.tz_convert("America/New_York").date)
    d_dates = pd.Index(d1.index.tz_convert("America/New_York").date if d1.index.tz is not None else d1.index.date)
    pser = pd.Series(permit_d.to_numpy(), index=d_dates)
    dvser = pd.Series(dv.to_numpy(), index=d_dates)
    # الإذن والسيولة من آخر يوم مغلق قبل يوم الساعة
    prev_permit = np.zeros(len(h1), bool)
    prev_dv = np.zeros(len(h1))
    pos = d_dates.searchsorted(ny_dates) - 1
    ok = pos >= 0
    prev_permit[ok] = pser.to_numpy()[pos[ok]]
    prev_dv[ok] = np.nan_to_num(dvser.to_numpy()[pos[ok]])
    ok4 = to_4h_ok(h1)
    m1, s1 = macd(h1["Close"])
    m1v, s1v = m1.to_numpy(), s1.to_numpy()
    rng1 = m1.abs().rolling(100, min_periods=100).max().to_numpy()
    lo20 = h1["Low"].rolling(SWING_LOOK).min().to_numpy()
    a = atr(h1).to_numpy()
    close = h1["Close"].to_numpy()
    out = []
    for k in range(200, len(h1) - 1):
        if not (prev_permit[k] and ok4[k]):
            continue
        if not (always or prev_dv[k] >= MIN_DOLLAR_VOL):
            continue
        if not (m1v[k] > s1v[k] and m1v[k - 1] <= s1v[k - 1]):
            continue
        if not (rng1[k] > 0 and abs(m1v[k]) <= rng1[k] * ENTRY_TOL_PCT / 100):
            continue
        stop = lo20[k] - ATR_BUF * a[k]
        risk = close[k] - stop
        if risk <= 0 or risk / close[k] * 100 > MAX_RISK_PCT:
            continue
        out.append((k, stop))
    return out


def run_ticker(t, h1, d1, always=False):
    trades = {f"MACD_Slow_{n}R": [] for n in (1, 2, 3)}
    sigs = signals(h1, d1, always)
    for n in (1, 2, 3):
        name = f"MACD_Slow_{n}R"
        nxt = 0
        for k, stop in sigs:
            if k < nxt:
                continue
            entry = h1["Open"].iat[k + 1]
            tr, j = simulate(h1, k, stop, entry + n * (entry - stop), MAX_BARS, name, t)
            if tr:
                trades[name].append(tr)
                nxt = j + 1
    # ضابط عشوائي بنفس الوقف/الهدف 2R/المدة
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    lo20 = h1["Low"].rolling(SWING_LOOK).min().to_numpy()
    a = atr(h1).to_numpy()
    rand, k = [], 200
    while k < len(h1) - 1:
        if rng.random() < 0.01:
            stop = lo20[k] - ATR_BUF * a[k]
            c = h1["Close"].iat[k]
            if 0 < c - stop <= c * MAX_RISK_PCT / 100:
                entry = h1["Open"].iat[k + 1]
                tr, j = simulate(h1, k, stop, entry + 2 * (entry - stop), MAX_BARS, "Random_2R", t)
                if tr:
                    rand.append(tr)
                    k = j + 1
                    continue
        k += 1
    trades["Random_2R"] = rand
    return trades


def load(tickers):
    import yfinance as yf
    h, d = {}, {}
    for k in range(0, len(tickers), 30):
        b = tickers[k:k + 30]
        rh = yf.download(b, period="730d", interval="1h", group_by="ticker", auto_adjust=False,
                         prepost=False, progress=False, threads=True)
        rd = yf.download(b, period="4y", interval="1d", group_by="ticker", auto_adjust=False,
                         progress=False, threads=True)
        for t in b:
            try:
                x, y = rh[t].dropna(), rd[t].dropna()
            except (KeyError, TypeError):
                continue
            if len(x) < 500 or len(y) < 300:
                continue
            if x.index.tz is None:
                x.index = x.index.tz_localize("UTC")
            h[t], d[t] = x, y
        print(f"  تحميل {min(k + 30, len(tickers))}/{len(tickers)}", flush=True)
    return h, d


def synthetic(tickers, seed=3):
    rng = np.random.default_rng(seed)
    h, d = {}, {}
    days = pd.bdate_range("2023-01-02", periods=500)
    for t in tickers:
        idx = []
        for dd in days:
            base = pd.Timestamp(dd).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
            idx += [base + pd.Timedelta(hours=x) for x in range(7)]
        idx = pd.DatetimeIndex(idx).tz_convert("UTC")
        n = len(idx)
        r = rng.normal(0, 0.006, n)
        c = 50 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.001, n))
        hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.003, n)))
        lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
        v = rng.lognormal(13, 0.4, n)
        x = pd.DataFrame(dict(Open=o, High=hi, Low=lo, Close=c, Volume=v), index=idx)
        dd = x.groupby(x.index.tz_convert("America/New_York").date).agg(
            Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum"))
        dd.index = pd.to_datetime(dd.index)
        h[t], d[t] = x, dd
    return h, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backtest/results_macd_slow")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "..", "macd_slow_scanner.py"))
    if a.synthetic:
        tickers, always = [f"T{i}" for i in range(40)], set()
        h, d = synthetic(tickers)
    else:
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)
        tickers = [t for t in ms.TICKERS if t not in ms.CRYPTO]
        always = ms.ALWAYS
        h, d = load(tickers)
    print(f"أسهم: {len(h)}", flush=True)

    allt = []
    for t in h:
        try:
            res = run_ticker(t, h[t], d[t], t in always)
            for v in res.values():
                allt += v
        except Exception as e:  # noqa: BLE001
            print(f"{t}: {e}", file=sys.stderr)
    tr = pd.DataFrame([asdict(x) for x in allt])
    os.makedirs(a.out, exist_ok=True)
    tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)

    names = ["MACD_Slow_1R", "MACD_Slow_2R", "MACD_Slow_3R", "Random_2R"]
    lab = {"MACD_Slow_1R": "MACD Slow هدف 1R", "MACD_Slow_2R": "MACD Slow هدف 2R",
           "MACD_Slow_3R": "MACD Slow هدف 3R", "Random_2R": "عشوائي 2R (ضابط)"}
    lines = [f"# MACD Slow — اختبار تاريخي ({len(h)} سهم، شموع ساعة لآخر سنتين)", "",
             "| الاستراتيجية | الصفقات | نسبة الربح % | التوقع R/صفقة | ±95% | معامل الربح | أطول سلسلة خسائر | متوسط الساعات |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        g = tr[(tr.strategy == n) & (tr.reason != "open")] if not tr.empty else tr
        if g.empty:
            lines.append(f"| {lab[n]} | 0 | — | — | — | — | — | — |")
            continue
        s = stats(g)
        ci = 1.96 * g.r.std() / np.sqrt(len(g))
        lines.append(f"| {lab[n]} | {s['trades']} | {s['win_rate']:.0f} | {s['expectancy']:+.3f} | {ci:.2f} | "
                     f"{s['pf']:.2f} | {s['max_lose_streak']} | {s['avg_days']:.0f} |")
    md = "\n".join(lines)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
