"""فحص سهم (أو أكثر) قبل الدخول.

أمثلة:
  python confirm.py NVDA
  python confirm.py SPY --mode intraday --side call
  python confirm.py AAPL,MSFT --mode swing --telegram
"""
import argparse
import traceback

from flowconfirm import engine, notify
from flowconfirm.data import get_snapshot


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
    for t in [x.strip().upper() for x in a.tickers.replace(" ", ",").split(",") if x.strip()]:
        for m in modes:
            try:
                r = engine.evaluate(get_snapshot(t, m), m, a.side)
                msg = engine.format_message(r)
                print(notify.plain(msg), "\n")
                if not a.no_log:
                    notify.log_signal(engine.log_row(r) | {"source": "check"})
                if a.telegram:
                    notify.send_telegram(msg)
            except Exception as e:
                traceback.print_exc()
                err = f"⚠️ {t} ({m}): تعذر الفحص — {e}"
                print(err)
                if a.telegram:
                    notify.send_telegram(err)


if __name__ == "__main__":
    main()
