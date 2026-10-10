"""
محرّك الاختبار الموحّد.

طريقتان للقياس، ولكل منهما سؤال:
  1) event_study — "هل الإشارة نفسها لها قيمة؟"
     عائد السهم بعد الإشارة (دخول افتتاح اليوم التالي، خروج إغلاق اليوم H)،
     مطروحاً منه متوسط عائد كل أسهم القائمة الداخلة في نفس اليوم (excess).
     هذا يلغي أثر اتجاه السوق: إشارة لا تتفوق على "شراء أي سهم ذلك اليوم" لا قيمة لها.
  2) portfolio (vectorbt) — "كيف تبدو الصفقات بقواعد الخروج الفعلية؟"
     وقف وهدف وحد زمني ورسوم 0.1% لكل جهة، صفقة واحدة لكل سهم في نفس الوقت.

لا غش بالمستقبل: الإشارة تُعرف على إغلاق يومها، والتنفيذ على افتتاح اليوم التالي.
"""
from __future__ import annotations

import itertools
import math
import traceback

import numpy as np
import pandas as pd

FEES = 0.001


# ───────────────────────── 1) Event study ─────────────────────────

def forward_returns(d: dict, holds) -> dict[int, pd.DataFrame]:
    O, C = d["Open"], d["Close"]
    return {H: C.shift(-H) / O.shift(-1) - 1 for H in holds}


def event_study(d: dict, entries: pd.DataFrame, holds=(5, 10, 20), side: int = 1,
                fwd: dict | None = None) -> pd.DataFrame:
    """جدول الصفقات: تاريخ، سهم، والعائد والتفوق لكل مدة احتفاظ."""
    fwd = fwd or forward_returns(d, holds)
    E = entries.reindex_like(d["Close"]).fillna(False).astype(bool)
    stacked = E.stack()
    idx = stacked[stacked].index
    if len(idx) == 0:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.DataFrame({"date": idx.get_level_values(0), "symbol": idx.get_level_values(1)})
    for H in holds:
        f = fwd[H]
        xs = f.sub(f.mean(axis=1), axis=0)
        r = f.stack(future_stack=True).reindex(idx).to_numpy()
        x = xs.stack(future_stack=True).reindex(idx).to_numpy()
        out[f"r{H}"] = side * r
        out[f"x{H}"] = side * x
    return out


def summarize(trades: pd.DataFrame, holds, main_hold: int, years: float) -> dict:
    t = trades.dropna(subset=[f"r{main_hold}"])
    n = len(t)
    if n == 0:
        return {"trades": 0}
    x = t[f"x{main_hold}"].dropna()
    mid = t["date"].min() + (t["date"].max() - t["date"].min()) / 2
    res = {"trades": n, "per_year": n / years, f"win{main_hold}": (t[f"r{main_hold}"] > 0).mean()}
    for H in holds:
        res[f"avg_r{H}"] = t[f"r{H}"].mean()
        res[f"excess{H}"] = t[f"x{H}"].mean()
    res["t_stat"] = (x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
                     if len(x) > 2 and x.std() > 0 else np.nan)
    res["excess_1st_half"] = t.loc[t["date"] < mid, f"x{main_hold}"].mean()
    res["excess_2nd_half"] = t.loc[t["date"] >= mid, f"x{main_hold}"].mean()
    yearly = t.groupby(t["date"].dt.year)[f"x{main_hold}"].mean()
    res["good_years"] = f"{int((yearly > 0).sum())}/{len(yearly)}"
    return res


def baseline(d: dict, holds, fwd=None) -> dict:
    fwd = fwd or forward_returns(d, holds)
    out = {}
    for H in holds:
        v = fwd[H].to_numpy().ravel()
        v = v[~np.isnan(v)]
        out[f"avg_r{H}"] = float(v.mean())
        out[f"win{H}"] = float((v > 0).mean())
    return out


# ───────────────────────── 2) Portfolio via vectorbt ─────────────────────────

def portfolio_stats(d: dict, entries: pd.DataFrame, side: int = 1, max_hold: int = 20,
                    sl: pd.DataFrame | float | None = None, tp: pd.DataFrame | float | None = None) -> dict:
    """صفقات فعلية: دخول افتتاح اليوم التالي، وقف/هدف (كنسبة من الدخول)، خروج بالوقت."""
    try:
        import vectorbt as vbt
    except ImportError:
        return {"vbt": "not installed"}
    try:
        C = d["Close"].ffill()
        O = d["Open"].fillna(C); H = d["High"].fillna(C); L = d["Low"].fillna(C)
        E = entries.reindex_like(C).fillna(False).astype(bool).shift(1, fill_value=False)
        X = E.shift(max_hold, fill_value=False)                       # خروج بالوقت
        kw = dict(open=O, high=H, low=L, price=O, fees=FEES, freq="1D", init_cash=100.0,
                  size=1.0, size_type="percent", accumulate=False)
        if sl is not None:
            kw["sl_stop"] = sl.shift(1) if isinstance(sl, pd.DataFrame) else sl
        if tp is not None:
            kw["tp_stop"] = tp.shift(1) if isinstance(tp, pd.DataFrame) else tp
        if side == 1:
            pf = vbt.Portfolio.from_signals(C, entries=E, exits=X, direction="longonly", **kw)
        else:
            pf = vbt.Portfolio.from_signals(C, entries=E, exits=X, direction="shortonly", **kw)
        tr = pf.trades.records_readable
        if len(tr) == 0:
            return {"pf_trades": 0}
        ret = tr["Return"].astype(float)
        gains, losses = ret[ret > 0].sum(), -ret[ret < 0].sum()
        return {
            "pf_trades": len(tr),
            "pf_win": float((ret > 0).mean()),
            "pf_avg_trade": float(ret.mean()),
            "pf_profit_factor": float(gains / losses) if losses > 0 else np.nan,
            "pf_avg_win": float(ret[ret > 0].mean()),
            "pf_avg_loss": float(ret[ret < 0].mean()),
        }
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return {"vbt": "error"}


# ───────────────────────── 3) Parameter sweep ─────────────────────────

def grid(params: dict) -> list[dict]:
    keys = list(params)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(params[k] for k in keys))]
