"""قراءة التدفق: البريميوم، جانب التنفيذ التقريبي، والعقود غير الاعتيادية."""
import numpy as np
import pandas as pd

from . import config as C


def classify(chain: pd.DataFrame) -> pd.DataFrame:
    """يضيف: السعر المستخدم، البريميوم، جانب التنفيذ (A/B/M/?)، واتجاه الصفقة."""
    df = chain.copy()
    spread = df["ask"] - df["bid"]
    valid = (df["bid"] > 0) & (df["ask"] > 0) & (spread > 0) & (df["last"] > 0)
    pos = ((df["last"] - df["bid"]) / spread.where(spread > 0, np.nan)).clip(-0.5, 1.5)
    df["side"] = np.where(~valid, "?",
                          np.where(pos >= C.ASK_SIDE, "A",
                                   np.where(pos <= C.BID_SIDE, "B", "M")))
    mid = (df["bid"] + df["ask"]) / 2
    px = df["last"].where(df["last"] > 0, mid)
    df["premium"] = df["volume"] * px * 100

    # الشراء على Ask = رهان باتجاه العقد ، البيع على Bid = رهان عكسه (وزن أقل)
    is_c = df["type"] == "C"
    df["dir"] = np.select(
        [(is_c & (df["side"] == "A")) | (~is_c & (df["side"] == "B")),
         (~is_c & (df["side"] == "A")) | (is_c & (df["side"] == "B"))],
        ["bull", "bear"], default="neutral")
    df["weight"] = np.select([df["side"] == "A", df["side"] == "B"], [1.0, 0.5], default=0.0)
    return df


def summarize(chain: pd.DataFrame, cfg: C.ModeCfg) -> dict:
    if chain.empty:
        return {"ok": False, "net": 0.0, "call_prem": 0, "put_prem": 0,
                "bull": 0, "bear": 0, "u_bull": 0, "u_bear": 0, "unusual": pd.DataFrame()}
    df = classify(chain)
    w = df["premium"] * df["weight"]
    bull = float(w[df["dir"] == "bull"].sum())
    bear = float(w[df["dir"] == "bear"].sum())
    net = (bull - bear) / (bull + bear) if bull + bear > 0 else 0.0

    unusual = df[(df["premium"] >= cfg.min_premium)
                 & (df["volume"] >= cfg.min_contracts)
                 & (df["volume"] >= cfg.min_vol_oi * df["oi"].clip(lower=1))
                 & (df["dir"] != "neutral")].sort_values("premium", ascending=False)

    return {
        "ok": True,
        "net": net,
        "call_prem": float(df.loc[df["type"] == "C", "premium"].sum()),
        "put_prem": float(df.loc[df["type"] == "P", "premium"].sum()),
        "bull": bull,
        "bear": bear,
        "u_bull": float(unusual.loc[unusual["dir"] == "bull", "premium"].sum()),
        "u_bear": float(unusual.loc[unusual["dir"] == "bear", "premium"].sum()),
        "unusual": unusual,
    }
