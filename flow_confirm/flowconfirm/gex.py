"""خريطة الغاما: GEX، نقطة الانقلاب، جدار الكول والبوت، Max Pain — محسوبة محلياً من OI."""
import math

import numpy as np
import pandas as pd

from . import config as C

_erf = np.vectorize(math.erf)


def _d1(S, K, T, iv, r=C.RISK_FREE):
    return (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))


def ncdf(x):
    return 0.5 * (1 + _erf(np.asarray(x, dtype=float) / math.sqrt(2)))


def bs_gamma(S, K, T, iv):
    d1 = _d1(S, K, T, iv)
    return np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) / (S * iv * np.sqrt(T))


def bs_delta(typ, S, K, T, iv):
    nd = ncdf(_d1(S, K, T, iv))
    return np.where(np.asarray(typ) == "C", nd, nd - 1)


def dealer_gex(chain: pd.DataFrame, S: float) -> np.ndarray:
    """تعرض صناع السوق بالدولار لكل حركة ١٪ (كول موجب، بوت سالب)."""
    g = bs_gamma(S, chain["strike"].values, chain["T"].values, chain["iv"].values)
    sign = np.where(chain["type"].values == "C", 1.0, -1.0)
    return g * chain["oi"].values * 100 * S * S * 0.01 * sign


def max_pain(chain: pd.DataFrame):
    if chain.empty:
        return None
    near = chain[chain["expiry"] == chain["expiry"].min()]
    strikes = np.sort(near["strike"].unique())
    c = near[near["type"] == "C"]
    p = near[near["type"] == "P"]
    pain = [(c["oi"] * np.maximum(k - c["strike"], 0)).sum()
            + (p["oi"] * np.maximum(p["strike"] - k, 0)).sum() for k in strikes]
    return float(strikes[int(np.argmin(pain))])


def analyze(chain: pd.DataFrame, S: float) -> dict:
    out = {"total": 0.0, "flip": None, "call_wall": None, "put_wall": None, "max_pain": None}
    if chain.empty or chain["oi"].sum() == 0:
        return out
    df = chain.assign(gex=dealer_gex(chain, S))
    by = df.groupby(["strike", "type"])["gex"].sum().unstack(fill_value=0.0)
    calls = by["C"] if "C" in by else pd.Series(dtype=float)
    puts = by["P"] if "P" in by else pd.Series(dtype=float)

    above = calls[calls.index > S]
    if not above.empty and above.max() > 0:
        out["call_wall"] = float(above.idxmax())
    below = puts[puts.index < S]
    if not below.empty and below.min() < 0:
        out["put_wall"] = float(below.idxmin())

    out["total"] = float(df["gex"].sum())

    grid = np.linspace(0.85 * S, 1.15 * S, 121)
    totals = np.array([dealer_gex(chain, s).sum() for s in grid])
    sgn = np.sign(totals)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if len(idx):
        flips = [grid[i] - totals[i] * (grid[i + 1] - grid[i]) / (totals[i + 1] - totals[i])
                 for i in idx]
        out["flip"] = float(min(flips, key=lambda x: abs(x - S)))

    out["max_pain"] = max_pain(chain)
    return out
