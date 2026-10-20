"""تأكيد السعر: VWAP والحجم النسبي والهيكل (يومي) — أو الاتجاه والاختراق (سوينغ)."""
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Check:
    label: str
    ok: bool
    pts: float
    max: float


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.fillna(100.0).where(up > 0, out.fillna(50.0))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _session_frac(last_bar_ts) -> float:
    open_ = last_bar_ts.replace(hour=9, minute=30, second=0, microsecond=0)
    mins = (last_bar_ts + dt.timedelta(minutes=5) - open_).total_seconds() / 60
    return float(np.clip(mins / 390, 15 / 390, 1.0))


def _rvol_check(rv, cfg):
    pts = 10 if rv >= cfg.rvol_good else 5 if rv >= 1.0 else 0
    return Check(f"حجم نسبي {rv:.1f}x", pts >= 10, pts, 10)


def intraday(bars, daily, spot, bull, cfg: C.ModeCfg):
    if bars.empty or len(bars) < 3:
        return [Check("لا توجد شموع كافية للجلسة", False, 0, 30)], {}
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap = float(((tp * bars["volume"]).cumsum() / bars["volume"].cumsum()).iloc[-1])
    above = spot > vwap
    checks = [Check(f"{'فوق' if above else 'تحت'} VWAP {vwap:.2f}",
                    above == bull, 10 if above == bull else 0, 10)]

    day = bars.index[-1].date()
    prior = daily[daily.index.date < day]["volume"].tail(20).mean()
    rv = bars["volume"].sum() / (prior * _session_frac(bars.index[-1])) if prior else 0.0
    checks.append(_rvol_check(rv, cfg))

    orb = bars.iloc[:3]  # أول ١٥ دقيقة
    orh, orl = float(orb["high"].max()), float(orb["low"].min())
    brk = spot > orh if bull else spot < orl
    checks.append(Check(f"كسر نطاق الافتتاح ({orl:.2f}–{orh:.2f})", brk, 5 if brk else 0, 5))
    e9, e21 = ema(bars["close"], 9).iloc[-1], ema(bars["close"], 21).iloc[-1]
    tr = e9 > e21 if bull else e9 < e21
    checks.append(Check("EMA 9/21 على ٥ دقائق", tr, 5 if tr else 0, 5))

    return checks, {"vwap": vwap, "orh": orh, "orl": orl, "rvol": rv}


def swing(daily, spot, bull, cfg: C.ModeCfg, live_now=None):
    """live_now: وقت نيويورك إذا كانت الجلسة مفتوحة (آخر شمعة يومية جزئية)، وإلا None."""
    if len(daily) < 60:
        return [Check("بيانات يومية غير كافية", False, 0, 30)], {}
    c = daily["close"]
    e20, e50 = float(ema(c, 20).iloc[-1]), float(ema(c, 50).iloc[-1])
    if bull:
        full, half = spot > e20 > e50, spot > e20
    else:
        full, half = spot < e20 < e50, spot < e20
    pts = 10 if full else 5 if half else 0
    checks = [Check(f"الاتجاه EMA20 {e20:.2f} / EMA50 {e50:.2f}", full, pts, 10)]

    vol = daily["volume"]
    frac = 1.0
    if live_now is not None and daily.index[-1].date() == live_now.date():
        frac = _session_frac(live_now - dt.timedelta(minutes=5))
    rv = vol.iloc[-1] / frac / vol.iloc[-21:-1].mean()
    checks.append(_rvol_check(rv, cfg))

    hi20, lo20 = float(daily["high"].iloc[-21:-1].max()), float(daily["low"].iloc[-21:-1].min())
    brk = spot > hi20 if bull else spot < lo20
    checks.append(Check(f"اختراق قمة/قاع ٢٠ يوماً ({lo20:.2f}–{hi20:.2f})", brk, 5 if brk else 0, 5))
    r = float(rsi(c).iloc[-1])
    rok = 50 <= r <= 70 if bull else 30 <= r <= 50
    checks.append(Check(f"RSI {r:.0f}", rok, 5 if rok else 0, 5))

    return checks, {"ema20": e20, "ema50": e50, "atr": float(atr(daily).iloc[-1]),
                    "rsi": r, "rvol": rv}
