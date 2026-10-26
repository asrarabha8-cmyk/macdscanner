"""بيانات الواجهة: يكتب ملفات JSON في docs/flow/data لتقرأها صفحة GitHub Pages.

  latest.json    آخر فحص للماسح لكل إطار (يومي/سوينغ) بنتائجه الكاملة
  checks.json    آخر الفحوصات اليدوية (من الاختصار أو Actions)
  history.json   كل إشارات "دخول" ومتابعتها (هدف/وقف/مفتوحة/منتهية) — على سعر السهم
  runs.json      سجل مرات التشغيل
  blocked.json   إشارات منعها فلتر السوق فقط، تُتابع بنفس الطريقة لدراسة الفلتر (لا تظهر في الإحصائيات)
  settings.json  العتبات والمواعيد المستخدمة
"""
import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent.parent  # جذر المستودع
WEB = Path(os.getenv("FLOW_WEB_DIR", ROOT / "docs" / "flow" / "data"))
UTC = dt.timezone.utc
NY = ZoneInfo("America/New_York")


def _iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (dt.date, dt.datetime, pd.Timestamp)):
        return x.isoformat()
    return x


def load(name, default):
    p = WEB / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except Exception:
        return default


def save(name, obj):
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / name).write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=1), encoding="utf-8")


def event_name() -> str:
    return os.getenv("GITHUB_EVENT_NAME", "local")


# ───────────────────────── تحويل النتيجة ─────────────────────────
def result_dict(r) -> dict:
    unusual = []
    u = r.flow.get("unusual")
    if isinstance(u, pd.DataFrame) and not u.empty:
        for _, x in u.head(6).iterrows():
            unusual.append({"contract": f"{x['strike']:g}{x['type']}", "expiry": x["expiry"].isoformat(),
                            "volume": int(x["volume"]), "oi": int(x["oi"]), "premium": float(x["premium"]),
                            "side": {"A": "Ask", "B": "Bid"}.get(x["side"], x["side"]), "dir": x["dir"]})
    return {
        "ticker": r.ticker, "mode": r.mode, "direction": "bull" if r.bull else "bear",
        "score": r.score, "decision": r.decision, "spot": r.spot, "ext_price": r.ext_price,
        "asof": _iso(r.asof), "src": r.src,
        "components": [{"name": c.name, "pts": c.pts, "max": c.max, "lines": c.lines} for c in r.components],
        "vetoes": r.vetoes, "warnings": r.warnings,
        "stock": r.stock, "option": r.option,
        "flow": {k: r.flow.get(k) for k in ("net", "call_prem", "put_prem", "bull", "bear", "u_bull", "u_bear")},
        "unusual": unusual,
        "gex": r.gex,
        "market": r.market, "mkt_blocked": r.mkt_blocked,
    }


def error_dict(ticker, mode, err, now) -> dict:
    return {"ticker": ticker, "mode": mode, "decision": "error", "error": str(err)[:200], "asof": _iso(now)}


# ───────────────────────── الكتابة ─────────────────────────
def write_settings():
    save("settings.json", {
        "enter_score": C.ENTER_SCORE, "wait_score": C.WAIT_SCORE, "neutral_flow": C.NEUTRAL_FLOW,
        "gap_veto_pct": C.GAP_VETO_PCT,
        "market_filter": C.MARKET_FILTER, "market_ticker": C.MARKET_TICKER, "swing_max_days": C.SWING_MAX_DAYS,
        "cooldown_hours": C.ALERT_COOLDOWN_HOURS,
        "modes": {k: {"label": m.label, "dte": [m.dte_min, m.dte_max], "min_premium": m.min_premium,
                      "min_vol_oi": m.min_vol_oi, "min_wall_room_pct": m.min_wall_room_pct,
                      "target_delta": list(m.target_delta), "max_spread_pct": m.max_spread_pct,
                      "min_oi": m.min_oi, "rvol_good": m.rvol_good, "min_rr": m.min_rr}
                  for k, m in C.MODES.items()},
        "schedule": {"tz": "Asia/Riyadh", "minute": 50, "hours": list(range(16, 24)),
                     "weekdays": [1, 2, 3, 4, 5],
                     "intraday_window_ny": ["09:45", "15:50"], "swing_window_ny": ["15:00", "16:10"]},
        "watchlist": _watchlist(),
    })


def _watchlist():
    p = ROOT / "flow_confirm" / "watchlist.txt"
    if not p.exists():
        return []
    return [ln.split("#")[0].strip().upper() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.split("#")[0].strip()]


def save_scan(mode, items, run):
    latest = load("latest.json", {})
    latest[mode] = {"run": run, "results": items}
    save("latest.json", latest)
    add_run(run)
    write_settings()


def save_checks(items, run):
    checks = load("checks.json", [])
    checks = list(items) + checks
    save("checks.json", checks[:60])
    add_run(run)
    write_settings()


def add_run(run):
    runs = load("runs.json", [])
    runs.insert(0, run)
    save("runs.json", runs[:200])


def make_run(kind, mode, started, items, alerts=0):
    fin = dt.datetime.now(UTC)
    ok = [x for x in items if x.get("decision") != "error"]
    return {"kind": kind, "mode": mode, "run_utc": _iso(started), "finished_utc": _iso(fin),
            "duration_s": int((fin - started).total_seconds()), "event": event_name(),
            "tickers": [x["ticker"] for x in items], "scanned": len(ok),
            "errors": len(items) - len(ok),
            "enter": sum(x.get("decision") == "enter" for x in items),
            "wait": sum(x.get("decision") == "wait" for x in items),
            "alerts": alerts, "status": "ok" if ok or not items else "error"}


# ───────────────────────── متابعة إشارات الدخول ─────────────────────────
def add_signals(items, source, blocked=False):
    """يضيف إشارات الدخول الجديدة لمتابعتها (بدون تكرار نفس السهم/الاتجاه/الإطار وهو مفتوح).
    blocked=True: يضيف بدلها الإشارات التي منعها فلتر السوق فقط إلى blocked.json."""
    fname = "blocked.json" if blocked else "history.json"
    hist = load(fname, [])
    open_keys = {(h["ticker"], h["mode"], h["direction"]) for h in hist if h["status"] == "open"}
    added = 0
    for x in items:
        if (not x.get("mkt_blocked")) if blocked else x.get("decision") != "enter":
            continue
        k = (x["ticker"], x["mode"], x["direction"])
        if k in open_keys:
            continue
        s = x["stock"]
        hist.insert(0, {"id": f"{x['ticker']}-{x['mode']}-{x['asof']}", "ticker": x["ticker"],
                        "mode": x["mode"], "direction": x["direction"], "source": source,
                        "time_utc": x["asof"], "score": x["score"],
                        "entry": s["entry"], "stop": s["stop"], "target": s["target"], "rr": s["rr"],
                        "contract": (x.get("option") or {}).get("name"),
                        "contract_mid": (x.get("option") or {}).get("mid"),
                        "status": "open", "last_price": s["entry"], "return_pct": 0.0,
                        "exit_price": None, "exit_utc": None, "days": 0}
                       | ({"market": x.get("market")} if blocked else {}))
        open_keys.add(k)
        added += 1
    if added:
        save(fname, hist)
    return added


def _evaluate(h, bars: pd.DataFrame, now: dt.datetime):
    """يحدّث إشارة مفتوحة من شموع ما بعد وقت الإشارة. الوقف قبل الهدف إذا لمسهما نفس الشمعة."""
    t0 = pd.Timestamp(h["time_utc"])
    b = bars[bars.index > t0]
    if b.empty:
        return
    bull = h["direction"] == "bull"
    sgn = 1 if bull else -1
    ny = b.index.tz_convert("America/New_York")
    if h["mode"] == "intraday":  # الإشارة اليومية تنتهي مع إغلاق يومها
        day = t0.tz_convert("America/New_York").date()
        b = b[(ny.date == day) & (ny.time < dt.time(16, 0))]
        if b.empty:
            return
    for ts, row in b.iterrows():
        hit_stop = row["low"] <= h["stop"] if bull else row["high"] >= h["stop"]
        hit_tgt = row["high"] >= h["target"] if bull else row["low"] <= h["target"]
        if hit_stop or hit_tgt:
            px = h["stop"] if hit_stop else h["target"]
            h.update(status="stop" if hit_stop else "target", exit_price=px, last_price=px,
                     exit_utc=_iso(ts.to_pydatetime()))
            break
    else:
        h["last_price"] = float(b["close"].iloc[-1])
    sessions = sorted(set(b.index.tz_convert("America/New_York").date))
    h["days"] = len(sessions)
    if h["status"] == "open":
        now_ny = now.astimezone(NY)
        sig_day = t0.tz_convert("America/New_York").date()
        expired = (h["mode"] == "intraday" and (now_ny.date() > sig_day or now_ny.time() >= dt.time(16, 0))) \
            or (h["mode"] == "swing" and len(sessions) > C.SWING_MAX_DAYS)
        if expired:
            h.update(status="expired", exit_price=h["last_price"], exit_utc=_iso(b.index[-1].to_pydatetime()))
    px = h["exit_price"] if h["exit_price"] is not None else h["last_price"]
    h["return_pct"] = round(sgn * (px / h["entry"] - 1) * 100, 2)


def update_history(fetch_bars):
    """fetch_bars(ticker, start_utc) → شموع ٥ دقائق (بتوقيت مع منطقة زمنية) من وقت البداية حتى الآن."""
    for fname in ("history.json", "blocked.json"):
        _update_file(fname, fetch_bars)


def _update_file(fname, fetch_bars):
    hist = load(fname, [])
    now = dt.datetime.now(UTC)
    changed = False
    for h in hist:
        if h["status"] != "open":
            continue
        try:
            bars = fetch_bars(h["ticker"], pd.Timestamp(h["time_utc"]).to_pydatetime())
            if bars is not None and not bars.empty:
                _evaluate(h, bars, now)
                changed = True
        except Exception as e:
            print(f"متابعة {h['ticker']}: {e}")
    if changed:
        save(fname, hist)
