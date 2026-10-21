"""الماسح التلقائي: يفحص قائمة المراقبة ويرسل لتليجرام الأسهم التي اكتمل تأكيدها فقط.

  python scan.py --mode intraday
  python scan.py --mode swing
  python scan.py --mode intraday --force   (تجاهل وقت السوق — للتجربة)
"""
import argparse
import datetime as dt
import time
import traceback
from pathlib import Path

from flowconfirm import engine, notify, web
from flowconfirm.data import bars_since, get_snapshot, now_ny

WATCHLIST = Path(__file__).resolve().parent / "watchlist.txt"


def load_watchlist():
    out = []
    for line in WATCHLIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip().upper()
        if line:
            out.append(line)
    return out


def in_window(mode: str, now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    if mode == "intraday":
        return dt.time(9, 45) <= t <= dt.time(15, 50)
    return dt.time(15, 0) <= t <= dt.time(16, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intraday", "swing"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-decision", choices=["enter", "wait"], default="enter",
                    help="enter = نبّه على الدخول فقط")
    a = ap.parse_args()

    now = now_ny()
    try:  # متابعة إشارات الدخول المفتوحة في كل تشغيل، حتى خارج النافذة
        web.update_history(bars_since)
    except Exception:
        traceback.print_exc()
    if not a.force and not in_window(a.mode, now):
        print(f"خارج نافذة التشغيل ({now:%a %H:%M} NY) — لا شيء")
        return

    started = dt.datetime.now(dt.timezone.utc)
    items = []

    allowed = {"enter"} if a.min_decision == "enter" else {"enter", "wait"}
    hits, errors = [], []
    for t in load_watchlist():
        try:
            r = engine.evaluate(get_snapshot(t, a.mode), a.mode)
            notify.log_signal(engine.log_row(r) | {"source": "scan"})
            print(f"{t:6} {r.decision:5} {r.score:3.0f} {'bull' if r.bull else 'bear'}")
            items.append(web.result_dict(r))
            key = f"{a.mode}:{t}:{'bull' if r.bull else 'bear'}"
            if r.decision in allowed and notify.should_alert(key, a.mode, now):
                notify.send_telegram(engine.format_message(r))
                notify.mark_alert(key, now)
                hits.append(t)
        except Exception as e:
            traceback.print_exc()
            errors.append(f"{t}: {e}")
            items.append(web.error_dict(t, a.mode, e, now))
        time.sleep(1)  # تهدئة الطلبات على yfinance

    try:
        web.save_scan(a.mode, items, web.make_run("scan", a.mode, started, items, alerts=len(hits)))
        web.add_signals(items, "scan")
    except Exception:
        traceback.print_exc()

    print(f"\nتنبيهات: {hits or 'لا شيء'}")
    if errors:
        print("أخطاء:", *errors, sep="\n  ")


if __name__ == "__main__":
    main()
