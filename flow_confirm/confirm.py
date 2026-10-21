"""فحص سهم (أو أكثر) قبل الدخول.

أمثلة:
  python confirm.py NVDA
  python confirm.py SPY --mode intraday --side call
  python confirm.py AAPL,MSFT --mode swing --telegram
"""
import argparse
import datetime as dt
import traceback

from flowconfirm import engine, notify, web
from flowconfirm.data import get_snapshot, now_ny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", help="رمز أو عدة رموز مفصولة بفاصلة")
    ap.add_argument("--mode", choices=["intraday", "swing", "both"], default="intraday")
    ap.add_argument("--side", choices=["auto", "call", "put"], default="auto",
                    help="auto = يحدد الاتجاه من التدفق")
    ap.add_argument("--telegram", action="store_true", help="أرسل النتيجة لتليجرام")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    modes = ["intraday", "swing"] if a.mode == "both" else [a.mode]
    started = dt.datetime.now(dt.timezone.utc)
    items = []
    for t in [x.strip().upper() for x in a.tickers.replace(" ", ",").split(",") if x.strip()]:
        for m in modes:
            try:
                r = engine.evaluate(get_snapshot(t, m), m, a.side)
                msg = engine.format_message(r)
                print(notify.plain(msg), "\n")
                items.append(web.result_dict(r) | {"side_req": a.side})
                if not a.no_log:
                    notify.log_signal(engine.log_row(r) | {"source": "check"})
                if a.telegram:
                    notify.send_telegram(msg)
            except Exception as e:
                traceback.print_exc()
                err = f"⚠️ {t} ({m}): تعذر الفحص — {e}"
                items.append(web.error_dict(t, m, e, now_ny()))
                print(err)
                if a.telegram:
                    notify.send_telegram(err)

    if not a.no_log and items:
        try:
            web.save_checks(items, web.make_run("check", a.mode, started, items,
                                                alerts=len(items) if a.telegram else 0))
            web.add_signals(items, "check")
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
