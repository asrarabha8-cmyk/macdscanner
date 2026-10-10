"""
سجل الاستراتيجيات. كل استراتيجية دالة تأخذ البيانات العريضة d ومعاملاتها،
وترجع dict فيه:
    long  : DataFrame منطقي (True = إشارة شراء على إغلاق ذلك اليوم)
    short : اختياري، إشارات بيع
    sl/tp : اختياري، الوقف والهدف كنسبة من السعر (لاختبار المحفظة)

لإضافة استراتيجية جديدة: اكتب الدالة ثم سجّلها في STRATEGIES بالأسفل
مع إعداداتها الافتراضية وشبكة التجربة (grid) ومدد الاحتفاظ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ───────────────────────── مؤشرات على جداول عريضة ─────────────────────────

def rma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def mfi(d: dict, n: int = 14) -> pd.DataFrame:
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    flow = tp * d["Volume"]
    ch = tp.diff()
    pos = flow.where(ch > 0, 0.0).rolling(n).sum()
    neg = flow.where(ch < 0, 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def cci(d: dict, n: int = 20) -> pd.DataFrame:
    tp = ((d["High"] + d["Low"] + d["Close"]) / 3)
    ma = tp.rolling(n).mean()
    arr = tp.to_numpy()
    md = np.full(arr.shape, np.nan)
    if len(arr) >= n:
        w = sliding_window_view(arr, n, axis=0)              # (T-n+1, N, n)
        md[n - 1:] = np.abs(w - w.mean(axis=2, keepdims=True)).mean(axis=2)
    md = pd.DataFrame(md, index=tp.index, columns=tp.columns)
    return (tp - ma) / (0.015 * md)


def atr(d: dict, n: int = 14) -> pd.DataFrame:
    c1 = d["Close"].shift(1)
    tr = np.fmax(d["High"] - d["Low"], np.fmax((d["High"] - c1).abs(), (d["Low"] - c1).abs()))
    return rma(tr, n)


# ───────────────────────── 1) Ankush Bajaj Momentum ─────────────────────────
# المرشح الوحيد الذي أظهر تفوقاً (غير مؤكد) في اختبار مكتبة Pine.
# الأصل: تغير 250 يوماً > 8% و RSI>60 و MFI>60 و CCI>100 وفوليوم > 5× EMA20.

def ankush(d, pchg=8.0, osc=60, cci_th=100, vol_mult=5.0):
    C, V = d["Close"], d["Volume"]
    ch = (C - C.shift(250)) / C * 100
    r, m, cc = rsi(C), mfi(d), cci(d)
    vma = V.ewm(span=20, adjust=False).mean()
    long = (ch > pchg) & (r > osc) & (m > osc) & (cc > cci_th) & (V > vma * vol_mult)
    a = atr(d)
    return {"long": long, "sl": (1.5 * a / C), "tp": (3.0 * a / C)}


# ───────────────────────── 2) MA100/200 breakout (منطق ma100_200_scanner.py) ─────────────────────────

def ma_breakout(d, fast=100, slow=200, vol_mult=2.0, require_order=True, min_r=1.0):
    C, L, V = d["Close"], d["Low"], d["Volume"]
    mf, ms = C.rolling(fast).mean(), C.rolling(slow).mean()
    vavg = V.shift(1).rolling(20).mean()
    cross = (C > mf) & (C.shift(1) <= mf.shift(1))
    ok = cross & (V >= vavg * vol_mult) & (C * vavg >= 3_000_000) & (ms > C)
    if require_order:
        ok &= ms > mf
    risk = C - L
    reward = ms - C
    ok &= (risk > 0) & (reward / risk.replace(0, np.nan) >= min_r)
    return {"long": ok, "sl": risk / C, "tp": reward / C}


# ───────────────────────── 3) EMA 8/48 weekly cross (منطق ema_cross_scanner.py) ─────────────────────────

def ema_cross_weekly(d, fast=8, slow=48):
    C = d["Close"]
    wk = C.resample("W-FRI").last()
    f = wk.ewm(span=fast, adjust=False).mean()
    s = wk.ewm(span=slow, adjust=False).mean()
    up_w = (f > s) & (f.shift(1) <= s.shift(1))
    dn_w = (f < s) & (f.shift(1) >= s.shift(1))
    up_w.iloc[:slow] = False; dn_w.iloc[:slow] = False
    up_w.index = up_w.index.to_period("W-FRI"); dn_w.index = dn_w.index.to_period("W-FRI")
    # الإشارة تُعرف على إغلاق آخر يوم تداول فعلي في الأسبوع
    per = C.index.to_period("W-FRI")
    is_last = pd.Series(per != per.to_series().shift(-1).to_numpy(), index=C.index).to_numpy()
    up = up_w.reindex(per).fillna(False).to_numpy() & is_last[:, None]
    dn = dn_w.reindex(per).fillna(False).to_numpy() & is_last[:, None]
    mk = lambda a: pd.DataFrame(a.astype(bool), index=C.index, columns=C.columns)  # noqa: E731
    return {"long": mk(up), "short": mk(dn)}


# ───────────────────────── السجل ─────────────────────────

STRATEGIES = {
    "ankush": {
        "fn": ankush,
        "title": "Ankush Bajaj Momentum",
        "defaults": dict(pchg=8.0, osc=60, cci_th=100, vol_mult=5.0),
        "grid": dict(pchg=[0.0, 8.0, 25.0], osc=[55, 60, 70], cci_th=[100], vol_mult=[1.5, 2.0, 3.0, 5.0]),
        "holds": (5, 10, 20), "main_hold": 10, "max_hold": 20,
    },
    "ma100": {
        "fn": ma_breakout,
        "title": "MA100/200 Breakout",
        "defaults": dict(fast=100, slow=200, vol_mult=2.0, require_order=True, min_r=1.0),
        "grid": dict(fast=[50, 100], slow=[200], vol_mult=[1.0, 1.5, 2.0, 3.0], require_order=[True], min_r=[0.0, 1.0, 2.0]),
        "holds": (5, 10, 20), "main_hold": 10, "max_hold": 20,
    },
    "ema_cross": {
        "fn": ema_cross_weekly,
        "title": "EMA 8/48 Weekly Cross",
        "defaults": dict(fast=8, slow=48),
        "grid": dict(fast=[5, 8, 10, 13], slow=[21, 34, 48]),
        "holds": (10, 20, 60), "main_hold": 20, "max_hold": 60,
    },
}
