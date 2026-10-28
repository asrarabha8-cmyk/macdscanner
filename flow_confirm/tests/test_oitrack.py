"""اختبار متابعة الـOI (بدون إنترنت): python tests/test_oitrack.py"""
import datetime as dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowconfirm import oitrack as OT  # noqa: E402
from flowconfirm.data import NY  # noqa: E402


def main():
    OT.PATH = Path(tempfile.mkdtemp()) / "u.json"
    d1 = dt.datetime(2026, 10, 26, 11, 0, tzinfo=NY)
    d2 = dt.datetime(2026, 10, 27, 11, 0, tzinfo=NY)
    exp = dt.date(2026, 10, 30)
    unusual = pd.DataFrame([
        {"type": "C", "strike": 105.0, "expiry": exp, "volume": 8000, "oi": 1000, "dir": "bull", "premium": 900_000},
        {"type": "P", "strike": 95.0, "expiry": exp, "volume": 4000, "oi": 2000, "dir": "bear", "premium": 300_000},
    ])
    OT.record("XYZ", unusual, d1)
    # تكرار التسجيل بحجم أقل لا يغيّر شيئاً
    OT.record("XYZ", unusual.assign(volume=10), d1)
    # نفس اليوم: لا متابعة (الـOI لم يتحدث بعد)
    chain = pd.DataFrame([
        {"type": "C", "strike": 105.0, "expiry": exp, "oi": 7000.0},   # +6000 ≥ نصف 8000 → محتفظ
        {"type": "P", "strike": 95.0, "expiry": exp, "oi": 1900.0},    # -100 → لم تُحفظ
    ])
    assert OT.follow_through("XYZ", chain, d1)["items"] == []
    ft = OT.follow_through("XYZ", chain, d2)
    assert ft["held_bull"] == 900_000 and ft["held_bear"] == 0, ft
    assert ft["faded_bear"] == 300_000, ft
    assert [x["held"] for x in ft["items"]] == [True, False], ft
    # التنظيف بعد KEEP_DAYS
    OT.record("ABC", unusual, d1 + dt.timedelta(days=OT.KEEP_DAYS + 1))
    assert "XYZ" not in OT._load()
    print("✅ oitrack OK")


if __name__ == "__main__":
    main()
