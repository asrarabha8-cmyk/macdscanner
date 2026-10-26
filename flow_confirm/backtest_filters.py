"""تجربة الفلاتر المقترحة على إشارات "دخول" المسجلة في docs/flow/data/history.json.

المتغيرات:
  base   الوضع الحالي (وقف أدنى 0.3٪)
  time   + لا دخول قبل 10:00 نيويورك
  mkt    + فلتر السوق: صعود فقط إذا SPY فوق VWAP، وهبوط فقط إذا تحته
  stop6  وقف أدنى 0.6٪ (مع إعادة حساب R:R ومنع ما يقل عن 1.2)
  all    الثلاثة معاً

python backtest_filters.py   (يحتاج POLYGON_API_KEY)
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from flowconfirm import config as C
from flowconfirm.data import NY, _polygon_aggs

ROOT = Path(__file__).resolve().parent.parent
HIST = json.loads((ROOT / "docs/flow/data/history.json").read_text(encoding="utf-8"))
_cache = {}


def bars(ticker, day):
    k = (ticker, day)
    if k not in _cache:
        df = _polygon_aggs(ticker, 5, "minute", day, day)
        t = df.index.time
        _cache[k] = df[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))]
    return _cache[k]


def spy_side(t0):
    """SPY فوق/تحت VWAP باستخدام الشموع المكتملة قبل وقت الإشارة."""
    b = bars("SPY", t0.date())
    b = b[b.index + pd.Timedelta(minutes=5) <= t0]
    tp = (b["high"] + b["low"] + b["close"]) / 3
    vwap = float((tp * b["volume"]).sum() / b["volume"].sum())
    last = float(b["close"].iloc[-1])
    return last > vwap, last, vwap


def simulate(h, t0, stop):
    bull = h["direction"] == "bull"
    s = 1 if bull else -1
    b = bars(h["ticker"], t0.date())
    b = b[b.index > t0]  # نفس منطق المتابعة الحالي
    for ts, r in b.iterrows():
        hs = r["low"] <= stop if bull else r["high"] >= stop
        ht = r["high"] >= h["target"] if bull else r["low"] <= h["target"]
        if hs or ht:
            px = stop if hs else h["target"]
            return ("stop" if hs else "target"), px, ts.strftime("%H:%M")
    return "expired", float(b["close"].iloc[-1]), "16:00"


def main():
    rows = []
    for h in reversed(HIST):
        if h["mode"] != "intraday":
            continue
        t0 = pd.Timestamp(h["time_utc"]).tz_convert(NY)
        bull = h["direction"] == "bull"
        s = 1 if bull else -1
        e = h["entry"]
        up, spy, vw = spy_side(t0)
        early = t0.time() < dt.time(10, 0)
        mkt_bad = up != bull

        stop6 = h["stop"]
        if (e - stop6) * s < 0.006 * e:
            stop6 = e * (1 - 0.006 * s)
        rr6 = abs(h["target"] - e) / abs(e - stop6)

        base = simulate(h, t0, h["stop"])
        wide = simulate(h, t0, stop6)
        ret = lambda o: round(s * (o[1] / e - 1) * 100, 2)
        rows.append({
            "time": t0.strftime("%m-%d %H:%M"), "tk": h["ticker"], "dir": h["direction"],
            "spy": f"{spy:.2f}/{vw:.2f}{'↑' if up else '↓'}",
            "base": f"{base[0]}@{base[2]}", "base_r": ret(base),
            "stop6": f"{wide[0]}@{wide[2]}", "stop6_r": ret(wide), "rr6": round(rr6, 2),
            "early": early, "mkt_block": mkt_bad, "rr_block": rr6 < C.MODES["intraday"].min_rr,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))

    def stats(name, mask, col):
        d = df[mask]
        st = d[col.replace("_r", "")].str.split("@").str[0] if col != "base_r" else d["base"].str.split("@").str[0]
        w, l = (st == "target").sum(), (st == "stop").sum()
        print(f"{name:7s} n={len(d):2d} هدف={w} وقف={l} منتهية={len(d) - w - l} "
              f"مجموع={d[col].sum():+.2f}% متوسط={d[col].mean() if len(d) else 0:+.2f}%")

    print()
    allm = pd.Series(True, index=df.index)
    stats("base", allm, "base_r")
    stats("time", ~df.early, "base_r")
    stats("mkt", ~df.mkt_block, "base_r")
    stats("stop6", ~df.rr_block, "stop6_r")
    stats("all", ~df.early & ~df.mkt_block & ~df.rr_block, "stop6_r")


if __name__ == "__main__":
    main()
