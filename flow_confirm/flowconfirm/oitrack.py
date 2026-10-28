"""متابعة الـOI للعقود غير الاعتيادية: هل الأموال فتحت مراكز واحتفظت بها، أم كانت مضاربة يومية؟

الفكرة (من GammaGrid): الحجم الكبير وحده لا يكفي. إذا ارتفع الـOI في اليوم التالي بقدر
كبير من حجم ذلك العقد، فالصفقة فُتحت واحتُفظ بها — وهذا تأكيد أقوى للاتجاه.
إذا لم يرتفع الـOI، فالحجم كان إغلاقاً أو مضاربة داخل اليوم.

الماسح يسجّل العقود غير الاعتيادية كل يوم في data/unusual_history.json،
وفي الأيام التالية نقارن الـOI الحالي بالـOI يوم التسجيل.
ملاحظة: yfinance يحدّث الـOI مرة يومياً (مراكز إغلاق أمس)، لذلك النتيجة تظهر من اليوم التالي.
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
PATH = DATA / "unusual_history.json"
KEEP_DAYS = 10      # أيام نحتفظ بها في السجل
HELD_RATIO = 0.5    # ارتفاع OI ≥ نصف الحجم → مركز محتفظ به


def _key(r) -> str:
    exp = r["expiry"]
    exp = exp.isoformat() if hasattr(exp, "isoformat") else str(exp)
    return f"{r['type']}{float(r['strike']):g}@{exp[:10]}"


def _load() -> dict:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def record(ticker: str, unusual: pd.DataFrame, now: dt.datetime):
    """يسجّل العقود غير الاعتيادية لهذا اليوم (يحتفظ بأعلى حجم لكل عقد في اليوم)."""
    if unusual is None or unusual.empty:
        return
    hist = _load()
    day = now.date().isoformat()
    t = hist.setdefault(ticker, {}).setdefault(day, {})
    for _, r in unusual.iterrows():
        k = _key(r)
        if k in t and t[k]["vol"] >= r["volume"]:
            continue
        t[k] = {"vol": int(r["volume"]), "oi": int(r["oi"]), "dir": r["dir"],
                "prem": round(float(r["premium"]))}
    cutoff = (now.date() - dt.timedelta(days=KEEP_DAYS)).isoformat()
    for tk in list(hist):
        hist[tk] = {d: v for d, v in hist[tk].items() if d >= cutoff}
        if not hist[tk]:
            del hist[tk]
    DATA.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(hist, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def follow_through(ticker: str, chain: pd.DataFrame, now: dt.datetime, hist: dict | None = None) -> dict:
    """يقارن الـOI الحالي بآخر تسجيل سابق لكل عقد غير اعتيادي (قبل اليوم)."""
    out = {"held_bull": 0.0, "held_bear": 0.0, "faded_bull": 0.0, "faded_bear": 0.0, "items": []}
    days = (hist if hist is not None else _load()).get(ticker, {})
    today = now.date().isoformat()
    latest = {}
    for d in sorted(x for x in days if x < today):
        for k, v in days[d].items():
            latest[k] = (d, v)  # الأحدث يغلب
    if not latest or chain is None or chain.empty:
        return out
    now_oi = {_key(r): float(r["oi"]) for _, r in chain[["type", "strike", "expiry", "oi"]].iterrows()}
    for k, (d, v) in latest.items():
        if k not in now_oi or v["vol"] <= 0:
            continue  # العقد خارج السلسلة الحالية أو انتهى
        d_oi = now_oi[k] - v["oi"]
        held = d_oi >= HELD_RATIO * v["vol"]
        bucket = ("held_" if held else "faded_") + v["dir"]
        if bucket in out:
            out[bucket] += v["prem"]
        out["items"].append({"contract": k, "day": d, "vol": v["vol"], "d_oi": int(d_oi),
                             "dir": v["dir"], "prem": v["prem"], "held": held})
    out["items"].sort(key=lambda x: x["prem"], reverse=True)
    return out
