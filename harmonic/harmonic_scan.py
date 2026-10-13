"""
harmonic_scan.py — ماسح يومي لنماذج كارني التوافقية

    python harmonic_scan.py                      # القائمة الافتراضية، فاصل يومي
    python harmonic_scan.py --tf 1h --period 3mo
    python harmonic_scan.py --symbols AAPL,NVDA,SPY --stage SIGNAL
    python harmonic_scan.py --tickers my_list.txt --json out.json

المراحل:
    WATCH   ٤ من ٥ نقاط مثبتة، الـ PRZ مُسقطة ولم تُختبر   → مراقبة فقط
    TBAR    السعر اختبر آخر رقم في الـ PRZ                  → النموذج مكتمل رسمياً
    SIGNAL  T-Bar+1 مع تغيّر طابع السعر                     → الفرصة حيّة بنص الكتاب
    FAILED  تجاوز الـ PRZ دون أي شمعة بنّاءة                → تُترك
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from harmonic_carney import Config, Stage, find_patterns

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "NFLX", "CRM", "ADBE", "QCOM", "MU", "INTC", "PLTR",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "XLE", "XLF",
]

STAGE_ORDER = {Stage.SIGNAL: 0, Stage.TBAR: 1, Stage.WATCH: 2, Stage.FAILED: 3}


# ─────────────────────────────────────────────────────────────────────────────
# التنبيه — نفس اصطلاح بقية ماسحات الريبو (TG_TOKEN / TG_CHAT)
# ─────────────────────────────────────────────────────────────────────────────

def notify(text: str) -> None:
    token = os.environ.get("HARMONIC_TG_TOKEN") or os.environ.get("TG_TOKEN")
    chat = os.environ.get("HARMONIC_TG_CHAT") or os.environ.get("TG_CHAT")
    if not token or not chat:
        print(text)
        return
    try:
        import requests
    except ImportError:
        print(text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=30,
    )
    if not r.ok:
        print("Telegram error:", r.text, file=sys.stderr)


def build_alert(hits: list, interval: str) -> str:
    """رسالة تيليجرام — الإشارات أولاً، والمراقبة مجمّعة في سطر."""
    sig = [p for p in hits if p.stage is Stage.SIGNAL]
    tb = [p for p in hits if p.stage is Stage.TBAR]
    watch = [p for p in hits if p.stage is Stage.WATCH]

    lines = [f"<b>نماذج كارني التوافقية</b> · {interval}"]

    for title, group in (("🎯 إشارة (T-Bar+1)", sig), ("⏳ شمعة الإتمام", tb)):
        if not group:
            continue
        lines.append(f"\n<b>{title}</b>")
        for p in group:
            d = "صاعد" if p.direction == "bullish" else "هابط"
            lines.append(
                f"<code>{p.symbol:<6}</code> {p.pattern} {d}"
                f" · طبقات {p.confluence}/3"
                f"\n   دخول {p.entry:.2f} · وقف {p.stop:.2f}"
                f" · TP1 {p.tp1:.2f} · TP2 {p.tp2:.2f}"
            )

    if watch:
        names = ", ".join(f"{p.symbol}({p.pattern[:4]})" for p in watch[:12])
        more = f" +{len(watch) - 12}" if len(watch) > 12 else ""
        lines.append(f"\n<b>👁 مراقبة</b> ({len(watch)}): {names}{more}")

    if not sig and not tb and not watch:
        return ""

    lines.append("\n<i>الرسم ليس إشارة — الدخول عند T-Bar+1 بنص الكتاب.</i>")
    return "\n".join(lines)


def fetch(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ينقص yfinance:  pip install yfinance --break-system-packages")

    raw = yf.download(
        symbols, period=period, interval=interval,
        group_by="ticker", auto_adjust=False, progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        if len(d) >= 60:
            out[s] = d
    return out


def read_tickers(path: Path) -> list[str]:
    """
    يقرأ ملف رموز سطراً سطراً، ويتجاهل التعليقات والفراغ.

    ملفات watchlist في هذا الريبو تبدأ بأسطر تعليق عربية بـ `#`، فالقراءة
    بـ split() على الملف كله كانت تعدّ كل كلمة رمزاً.
    """
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(line.split()[0].upper())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ماسح نماذج كارني التوافقية")
    ap.add_argument("--symbols", help="رموز مفصولة بفاصلة")
    ap.add_argument("--tickers", type=Path, help="ملف نصي، رمز في كل سطر")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--tf", dest="interval", default="1d", help="1d · 1h · 15m …")
    ap.add_argument("--stage", choices=[s.value for s in Stage], action="append",
                    help="اقتصر على مرحلة (يمكن تكرارها)")
    ap.add_argument("--pivot", type=int, default=5, help="قوة المحور يميناً ويساراً")
    ap.add_argument("--min-bars", type=int, default=30, help="أدنى امتداد للنموذج")
    ap.add_argument("--require-rsi", action="store_true",
                    help="اشترط تشبع RSI عند شمعة الإتمام")
    ap.add_argument("--json", type=Path, help="اكتب النتائج JSON")
    ap.add_argument("--notify", action="store_true",
                    help="أرسل تنبيه تيليجرام (TG_TOKEN / TG_CHAT)")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.tickers:
        symbols = read_tickers(args.tickers)
    else:
        symbols = DEFAULT_SYMBOLS

    cfg = Config(
        pivot_left=args.pivot,
        pivot_right=args.pivot,
        min_pattern_bars=args.min_bars,
        require_rsi=args.require_rsi,
    )

    print(f"جلب {len(symbols)} رمزاً · {args.period} · {args.interval}", file=sys.stderr)
    data = fetch(symbols, args.period, args.interval)
    print(f"نجح {len(data)} رمزاً\n", file=sys.stderr)

    hits = []
    for sym, df in data.items():
        try:
            hits.extend(find_patterns(df, cfg, symbol=sym))
        except Exception as e:                      # noqa: BLE001
            print(f"  تخطّي {sym}: {e}", file=sys.stderr)

    if args.stage:
        keep = {Stage(s) for s in args.stage}
        hits = [p for p in hits if p.stage in keep]

    hits.sort(key=lambda p: (STAGE_ORDER[p.stage], p.prz_width_pct))

    if not hits:
        print("لا نماذج مطابقة. هذا متوقع — شرط ٣٠ شمعة وتسامح نقطة B يحرقان معظم البنى.")
        return

    print(f"{'الرمز':<7}{'النموذج':<15}{'المرحلة':<8}{'الاتجاه':<9}"
          f"{'دخول':>9}{'وقف':>9}{'TP1':>9}{'TP2':>9}  {'B':>6} {'PRZ%':>6} {'شمعات':>6}")
    print("─" * 110)
    for p in hits:
        e = f"{p.entry:.2f}" if p.entry else f"{p.prz_low:.2f}"
        print(
            f"{p.symbol:<7}{p.pattern:<15}{p.stage.value:<8}"
            f"{('صاعد' if p.direction == 'bullish' else 'هابط'):<9}"
            f"{e:>9}{p.stop:>9.2f}"
            f"{(f'{p.tp1:.2f}' if p.tp1 else '—'):>9}"
            f"{(f'{p.tp2:.2f}' if p.tp2 else '—'):>9}"
            f"  {p.b_ratio:>6.3f} {p.prz_width_pct*100:>5.1f}% {p.bars_span:>6}"
        )

    by_stage = {}
    for p in hits:
        by_stage[p.stage.value] = by_stage.get(p.stage.value, 0) + 1
    print("\n" + " · ".join(f"{k}: {v}" for k, v in by_stage.items()))

    if args.notify:
        msg = build_alert(hits, args.interval)
        if msg:
            notify(msg)

    if args.json:
        args.json.write_text(
            json.dumps([p.to_dict() for p in hits], ensure_ascii=False, indent=2,
                       default=float),
            encoding="utf-8",
        )
        print(f"كُتبت النتائج في {args.json}")


if __name__ == "__main__":
    main()
