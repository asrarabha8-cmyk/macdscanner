"""اختبار بيانات الواجهة ومتابعة الإشارات (بدون إنترنت): python tests/test_web.py [مجلد_إخراج]"""
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.environ["FLOW_WEB_DIR"] = out
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_synthetic as T  # noqa: E402
from flowconfirm import web  # noqa: E402
from flowconfirm.data import NY  # noqa: E402

UTC = dt.timezone.utc


def bars(start_ny, closes):
    idx = pd.date_range(start_ny, periods=len(closes), freq="5min", tz=NY)
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1e5}, index=idx)


if __name__ == "__main__":
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = [T.run("a", True, True), T.run("b", True, False), T.run("c", False, False),
               T.run("d", True, True, mode="swing")]
    items = [web.result_dict(r) for r in res]
    for x, tk in zip(items, ["NVDA", "AMD", "TSLA", "META"]):
        x["ticker"] = tk
    items.append(web.error_dict("XYZ", "intraday", "no data", dt.datetime.now(UTC)))
    started = dt.datetime.now(UTC) - dt.timedelta(seconds=40)
    web.save_scan("intraday", [i for i in items if i["mode"] == "intraday"],
                  web.make_run("scan", "intraday", started, items[:3] + items[4:], alerts=2))
    web.save_scan("swing", [items[3]], web.make_run("scan", "swing", started, [items[3]], alerts=1))
    chk = dict(items[0], ticker="SMCI", side_req="auto")
    web.save_checks([chk], web.make_run("check", "intraday", started, [chk], alerts=1))
    n = web.add_signals(items, "scan")
    assert n == 3, n  # NVDA bull, TSLA bear (intraday), META swing
    assert web.add_signals(items, "scan") == 0  # لا تكرار وهي مفتوحة

    # متابعة: نجعل NVDA تلمس الهدف، TSLA تلمس الوقف، META تبقى مفتوحة
    hist = web.load("history.json", [])
    sig_t = dt.datetime(2026, 10, 20, 11, 0, tzinfo=NY)
    for h in hist:
        h["time_utc"] = sig_t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    web.save("history.json", hist)
    H = {h["ticker"]: h for h in hist}
    path = {
        "NVDA": bars("2026-10-20 11:05", np.linspace(H["NVDA"]["entry"], H["NVDA"]["target"] + 0.5, 30)),
        "TSLA": bars("2026-10-20 11:05", np.linspace(H["TSLA"]["entry"], H["TSLA"]["stop"] + 0.5, 30)),
        "META": bars("2026-10-20 11:05", np.full(30, H["META"]["entry"] + 0.3)),
    }
    web.update_history(lambda tk, start: path[tk])
    H = {h["ticker"]: h for h in web.load("history.json", [])}
    print({k: (v["status"], v["return_pct"]) for k, v in H.items()})
    assert H["NVDA"]["status"] == "target" and H["NVDA"]["return_pct"] > 0
    assert H["TSLA"]["status"] == "stop" and H["TSLA"]["return_pct"] < 0
    assert H["META"]["status"] == "open"
    for f in ("latest.json", "checks.json", "runs.json", "history.json", "settings.json"):
        assert (Path(out) / f).exists(), f
    print("✅ بيانات الواجهة سليمة →", out)
