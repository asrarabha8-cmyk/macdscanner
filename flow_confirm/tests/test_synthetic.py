"""اختبار شامل على بيانات محاكاة (بدون إنترنت): python tests/test_synthetic.py"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowconfirm import engine, notify  # noqa: E402
from flowconfirm.data import NY, Snapshot, finalize_chain  # noqa: E402


def make_chain(now, spot, bullish=True):
    rows = []
    for dte in (1, 3):
        exp = now.date() + dt.timedelta(days=dte)
        for k in np.arange(85, 116, 1.0):
            for typ in ("C", "P"):
                intrinsic = max(spot - k, 0) if typ == "C" else max(k - spot, 0)
                mid = intrinsic + 1.2 * np.exp(-((k - spot) / 4) ** 2) + 0.05
                oi = 2000 * np.exp(-((k - spot) / 6) ** 2)
                if typ == "C" and k == 105:
                    oi += 15000           # جدار كول
                if typ == "P" and k == 95:
                    oi += 15000           # جدار بوت
                vol = 300 * np.exp(-((k - spot) / 5) ** 2)
                last = mid
                hot = (typ == "C") if bullish else (typ == "P")
                if hot and spot < k <= spot + 3 and dte == 1:
                    vol, last = 8000, mid + 0.04   # شراء عدواني على Ask
                rows.append({"type": typ, "expiry": exp, "strike": k, "bid": mid - 0.05,
                             "ask": mid + 0.05, "last": last, "volume": vol, "oi": oi,
                             "iv": 0.35, "last_trade": pd.Timestamp(now)})
    return finalize_chain(pd.DataFrame(rows), now)


def make_bars(now, up=True):
    idx = pd.date_range(now.replace(hour=9, minute=30, second=0, microsecond=0), periods=60, freq="5min")
    drift = np.linspace(97.5, 100, 60) if up else np.linspace(102.5, 100, 60)
    rng = np.random.default_rng(1)
    c = drift + rng.normal(0, 0.05, 60)
    bars = pd.DataFrame({"open": c - 0.05, "high": c + 0.15, "low": c - 0.15, "close": c,
                         "volume": 400_000.0}, index=idx)
    didx = pd.date_range(end=now.date() - dt.timedelta(days=1), periods=150, freq="B", tz=NY)
    d = np.linspace(80, 98, 150) if up else np.linspace(120, 102, 150)
    daily = pd.DataFrame({"open": d, "high": d + 1, "low": d - 1, "close": d,
                          "volume": 30_000_000.0}, index=didx)
    return bars, daily


def run(name, bullish, up, side="auto", mode="intraday"):
    now = dt.datetime(2026, 10, 20, 12, 30, tzinfo=NY)
    bars, daily = make_bars(now, up)
    spot = float(bars["close"].iloc[-1])
    snap = Snapshot("TEST", spot, make_chain(now, spot, bullish), bars, daily, None, now)
    r = engine.evaluate(snap, mode, side)
    print(f"===== {name} =====")
    print(notify.plain(engine.format_message(r)), "\n")
    return r


if __name__ == "__main__":
    r1 = run("تدفق صاعد + سعر صاعد", True, True)
    assert r1.bull and r1.decision == "enter", r1.decision
    assert r1.gex["call_wall"] == 105 and r1.gex["put_wall"] == 95

    r2 = run("تدفق صاعد لكن السعر يهبط تحت VWAP", True, False)
    assert r2.decision == "no"

    r3 = run("طلب كول والتدفق هابط", False, False, side="call")
    assert r3.decision == "no"

    r4 = run("تدفق هابط + سعر هابط", False, False)
    assert not r4.bull and r4.decision in ("enter", "wait")
    print("✅ كل الاختبارات نجحت")
