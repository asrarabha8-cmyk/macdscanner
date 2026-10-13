"""
backtest.py — باك تست لنماذج كارني التوافقية

يسدّ أخطر فراغ في الكتاب: كارني لا ينشر رقماً واحداً عن نسبة نجاح نماذجه في
٣٥١ صفحة، وكل أمثلته منتقاة بعد وقوعها.

═══ لا تسريب مستقبلي ═══════════════════════════════════════════════════════════

هنا المكان الذي يفسد فيه أغلب باك تست الهارمونيك. نقطة D محور كسري، ولا تُثبَّت
إلا بعد مرور `pivot_right` شمعة عليها. فمن يقيس النتيجة من سعر D يقيس من سعر لم
يكن يعرفه وقتها — وتخرج نتائج ممتازة وكاذبة.

لذلك تسلسل الدخول هنا:

    known_at  = D.index + pivot_right      ← أول شمعة صار فيها المحور معلوماً
    tbar      ≥ D.index                    ← شمعة اختبار آخر رقم في الـ PRZ
    entry_bar = max(known_at, tbar) + 1    ← الكتاب: T-Bar+1، والواقع: بعد التثبيت
    entry_px  = open[entry_bar]            ← افتتاح الشمعة التالية، لا سعر D

وعند تساوي اللمس في شمعة واحدة (الهدف والوقف معاً) يُحتسب **الوقف** أولاً.
الاحتياط هنا متعمد: النتيجة المتحفظة أنفع من المتفائلة.

═══ التشغيل ═══════════════════════════════════════════════════════════════════

    python backtest.py --symbols AAPL,NVDA,SPY --period 5y
    python backtest.py --csv data/aapl.csv --tf 1d
    python backtest.py --tickers ../watchlist.txt --pivot 7 --json bt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from harmonic_carney import Config, Stage, find_patterns


# ─────────────────────────────────────────────────────────────────────────────
# الصفقة
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    pattern: str
    direction: str
    confluence: int

    d_bar: int
    tbar: int
    entry_bar: int
    entry: float
    stop: float
    tp1: float
    tp2: float

    outcome: str          # tp2 · tp1 · stop · open · timeout
    exit_bar: int
    exit_px: float
    bars_held: int

    r_multiple: float     # الربح/الخسارة مقسوماً على مخاطرة الصفقة
    pct: float            # النسبة المئوية للحركة
    mfe_r: float          # أقصى ربح غير محقق بوحدات R
    mae_r: float          # أقصى خسارة غير محققة بوحدات R

    hit_tp1: bool
    hit_tp2: bool
    stage_at_entry: str

    # ما كان سيحدث لو أمكن الدخول عند نقطة D نفسها — للمقارنة فقط.
    # هذا دخول مستحيل عملياً (D لا تُعرف وقتها)، وفارقه عن r_multiple هو
    # ثمن التوقيت السليم: كم تأكل اللحظةُ الواقعيةُ من الميزة النظرية.
    r_if_entered_at_d: float = 0.0
    slip_pct: float = 0.0          # انزلاق الدخول عن D بالنسبة المئوية

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# محاكاة صفقة واحدة
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    entry_bar: int, entry: float, stop: float, tp1: float, tp2: float,
    bull: bool, max_hold: int,
) -> tuple[str, int, float, float, float, float, bool, bool]:
    """
    يمشي للأمام شمعة شمعة من شمعة الدخول.

    الاحتساب يطابق إدارة الكتاب: **نصف المركز يُغلق تلقائياً عند TP1** (ص ٢٩٥:
    جزء من المركز بأمر تلقائي يُوضع مع الدخول) والنصف الآخر يسعى لـ TP2 بنفس
    الوقف. فالنتيجة النهائية مركّبة:

        لمس TP1 ثم TP2    →  0.5·R₁ + 0.5·R₂
        لمس TP1 ثم الوقف  →  0.5·R₁ + 0.5·(−1)      ← لا −1 كاملة
        لمس TP1 ثم المدة  →  0.5·R₁ + 0.5·R_إغلاق
        الوقف قبل TP1     →  −1
        TP2 بفجوة بلا TP1 →  R₂

    وعند تساوي اللمس في شمعة واحدة (هدف ووقف معاً) يُحتسب **الوقف** أولاً.
    الاحتياط متعمد: النتيجة المتحفظة أنفع من المتفائلة.

    يعيد (النتيجة، شمعة الخروج، سعر الخروج، R، أقصى ربح R، أقصى خسارة R،
          لمس TP1، لمس TP2)
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return "open", entry_bar, entry, 0.0, 0.0, 0.0, False, False

    r1 = abs(tp1 - entry) / risk
    r2 = abs(tp2 - entry) / risk

    last = len(c) - 1
    end = min(last, entry_bar + max_hold)
    hit1 = hit2 = False
    mfe = mae = 0.0

    for i in range(entry_bar, end + 1):
        if bull:
            up = (h[i] - entry) / risk
            dn = (l[i] - entry) / risk
        else:
            up = (entry - l[i]) / risk
            dn = (entry - h[i]) / risk
        mfe = max(mfe, up)
        mae = min(mae, dn)

        touched_stop = (l[i] <= stop) if bull else (h[i] >= stop)
        touched_tp1  = (h[i] >= tp1)  if bull else (l[i] <= tp1)
        touched_tp2  = (h[i] >= tp2)  if bull else (l[i] <= tp2)

        # الوقف أولاً عند التساوي
        if touched_stop:
            r = 0.5 * r1 + 0.5 * (-1.0) if hit1 else -1.0
            return "stop", i, stop, r, mfe, mae, hit1, hit2

        if touched_tp2:
            r = 0.5 * r1 + 0.5 * r2 if hit1 else r2
            return "tp2", i, tp2, r, mfe, mae, True, True

        if touched_tp1 and not hit1:
            hit1 = True          # نصف المركز خرج — والباقي يكمل بنفس الوقف

    # انتهت المدة
    px = float(c[end])
    r_close = (px - entry) / risk if bull else (entry - px) / risk
    if hit1:
        return "tp1", end, px, 0.5 * r1 + 0.5 * r_close, mfe, mae, True, False
    return "timeout", end, px, r_close, mfe, mae, False, False


# ─────────────────────────────────────────────────────────────────────────────
# باك تست رمز واحد
# ─────────────────────────────────────────────────────────────────────────────

def backtest_symbol(
    df: pd.DataFrame, cfg: Config, symbol: str = "",
    max_hold: int = 60, min_stage: Stage = Stage.TBAR,
) -> list[Trade]:
    """
    يمرّ على كل نماذج الرمز ويحاكي كل واحد بتوقيت دخول سليم.

    min_stage: أدنى مرحلة تُحتسب صفقةً. Stage.SIGNAL أكثر التزاماً بالكتاب،
    و Stage.TBAR أوسع ويعطي عيّنة أكبر لقياس أثر شرط تغيّر الطابع نفسه.
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    last = len(c) - 1

    # نرفع سقف المحاور ليمسح التاريخ كله لا آخر نافذة فقط
    scan_cfg = Config(**{**cfg.__dict__, "max_lookback_pivots": 10_000})
    pats = find_patterns(df, scan_cfg, symbol=symbol)

    accept = {Stage.TBAR, Stage.SIGNAL} if min_stage is Stage.TBAR else {Stage.SIGNAL}

    trades: list[Trade] = []
    seen: set[tuple[int, str]] = set()

    for p in pats:
        if p.D is None or p.stage not in accept or p.tbar_index is None:
            continue
        key = (p.D[0], p.pattern)
        if key in seen:
            continue
        seen.add(key)

        d_bar = p.D[0]
        bull = p.direction == "bullish"

        # ── التوقيت السليم: المحور لا يُعرف قبل تثبيته ──────────────────────
        known_at = d_bar + cfg.pivot_right
        entry_bar = max(known_at, p.tbar_index) + 1
        if entry_bar > last:
            continue                       # الإشارة في آخر البيانات — لا نتيجة بعد

        entry = float(o[entry_bar])        # افتتاح الشمعة التالية، لا سعر D

        # الوقف والأهداف يُعاد حسابها على سعر الدخول الفعلي لا على D
        risk_abs = abs(entry - p.stop)
        if risk_abs <= 0:
            continue
        a_px = p.A[1]
        tp1 = entry + 0.382 * (a_px - entry)
        tp2 = entry + 0.618 * (a_px - entry)

        outcome, ex_bar, ex_px, r, mfe, mae, hit1, hit2 = _simulate(
            o, h, l, c, entry_bar, entry, p.stop, tp1, tp2, bull, max_hold
        )

        pct = ((ex_px - entry) / entry if bull else (entry - ex_px) / entry) * 100.0

        # المقارنة: نفس الوقف والأهداف الأصلية، لكن بدخول عند D ومن شمعة D
        d_px = p.D[1]
        d_tp1 = d_px + 0.382 * (a_px - d_px)
        d_tp2 = d_px + 0.618 * (a_px - d_px)
        r_at_d = _simulate(o, h, l, c, d_bar, d_px, p.stop,
                           d_tp1, d_tp2, bull, max_hold)[3]
        slip = ((entry - d_px) / d_px if bull else (d_px - entry) / d_px) * 100.0

        trades.append(Trade(
            symbol=symbol, pattern=p.pattern, direction=p.direction,
            confluence=p.confluence,
            d_bar=d_bar, tbar=p.tbar_index, entry_bar=entry_bar,
            entry=entry, stop=float(p.stop), tp1=tp1, tp2=tp2,
            outcome=outcome, exit_bar=ex_bar, exit_px=float(ex_px),
            bars_held=ex_bar - entry_bar,
            r_multiple=r, pct=pct, mfe_r=mfe, mae_r=mae,
            hit_tp1=hit1, hit_tp2=hit2, stage_at_entry=p.stage.value,
            r_if_entered_at_d=r_at_d, slip_pct=slip,
        ))

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# مقياس الأساس — دخول عشوائي بنفس الهندسة
# ─────────────────────────────────────────────────────────────────────────────

def random_baseline(
    df: pd.DataFrame, trades: list[Trade], max_hold: int,
    n_per_trade: int = 20, seed: int = 0,
) -> dict:
    """
    لكل صفقة حقيقية: ندخل في شمعات عشوائية بنفس الاتجاه، وبنفس **نسب**
    الوقف والهدف من سعر الدخول، ونفس المدة القصوى.

    هذا ما يفرّق الميزة الحقيقية عن انحياز السوق: سوق صاعد يجعل أي شراء
    عشوائي ناجحاً. فإن تساوت النتيجتان فالنماذج لا تضيف شيئاً.
    """
    if not trades:
        return {}

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    last = len(c) - 1
    rng = np.random.default_rng(seed)

    rs: list[float] = []
    wins = 0
    for t in trades:
        bull = t.direction == "bullish"
        # النسب الهندسية للصفقة الحقيقية
        stop_f = (t.stop - t.entry) / t.entry
        tp1_f = (t.tp1 - t.entry) / t.entry
        tp2_f = (t.tp2 - t.entry) / t.entry

        lo_bar, hi_bar = 30, last - max_hold - 1
        if hi_bar <= lo_bar:
            continue
        for _ in range(n_per_trade):
            eb = int(rng.integers(lo_bar, hi_bar))
            e = float(o[eb])
            out = _simulate(
                o, h, l, c, eb, e, e * (1 + stop_f),
                e * (1 + tp1_f), e * (1 + tp2_f), bull, max_hold
            )
            rs.append(out[3])
            wins += 1 if out[3] > 0 else 0

    if not rs:
        return {}
    return {
        "n": len(rs),
        "avg_r": float(np.mean(rs)),
        "win_rate": wins / len(rs) * 100.0,
        "median_r": float(np.median(rs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# التجميع والعرض
# ─────────────────────────────────────────────────────────────────────────────

def summarize(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        return pd.Series({
            "صفقات": n,
            "TP1 %": g["hit_tp1"].mean() * 100.0,
            "TP2 %": g["hit_tp2"].mean() * 100.0,
            "وقف %": (g["outcome"] == "stop").mean() * 100.0,
            "متوسط R": g["r_multiple"].mean(),
            "وسيط R": g["r_multiple"].median(),
            "مجموع R": g["r_multiple"].sum(),
            "شمعات": g["bars_held"].mean(),
            "أقصى ربح R": g["mfe_r"].mean(),
            "أقصى خسارة R": g["mae_r"].mean(),
            "R عند D": g["r_if_entered_at_d"].mean(),
            "ثمن التوقيت": g["r_multiple"].mean() - g["r_if_entered_at_d"].mean(),
        })

    return df.groupby("pattern", group_keys=True).apply(agg, include_groups=False)


def summarize_by(trades: list[Trade], col: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    return df.groupby(col).agg(
        صفقات=("r_multiple", "size"),
        TP1=("hit_tp1", lambda s: s.mean() * 100.0),
        TP2=("hit_tp2", lambda s: s.mean() * 100.0),
        متوسط_R=("r_multiple", "mean"),
        مجموع_R=("r_multiple", "sum"),
    )


def fetch(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ينقص yfinance:  pip install yfinance")
    raw = yf.download(symbols, period=period, interval=interval,
                      group_by="ticker", auto_adjust=False, progress=False)
    out = {}
    for s in symbols:
        try:
            d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        if len(d) >= 120:
            out[s] = d.reset_index(drop=True)
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
    ap = argparse.ArgumentParser(description="باك تست نماذج كارني التوافقية")
    ap.add_argument("--symbols", help="رموز مفصولة بفاصلة")
    ap.add_argument("--tickers", type=Path, help="ملف نصي، رمز في كل سطر")
    ap.add_argument("--csv", type=Path, help="ملف OHLC واحد بدل الجلب")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--tf", dest="interval", default="1d")
    ap.add_argument("--pivot", type=int, default=5)
    ap.add_argument("--min-bars", type=int, default=30)
    ap.add_argument("--max-hold", type=int, default=60, help="أقصى شمعات للصفقة")
    ap.add_argument("--signal-only", action="store_true",
                    help="احتسب مرحلة SIGNAL فقط (أكثر التزاماً بالكتاب)")
    ap.add_argument("--require-rsi", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    cfg = Config(pivot_left=args.pivot, pivot_right=args.pivot,
                 min_pattern_bars=args.min_bars, require_rsi=args.require_rsi)
    stage = Stage.SIGNAL if args.signal_only else Stage.TBAR

    if args.csv:
        d = pd.read_csv(args.csv).rename(columns=str.lower)
        data = {args.csv.stem.upper(): d[["open", "high", "low", "close"]].dropna()}
    else:
        if args.symbols:
            syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        elif args.tickers:
            syms = read_tickers(args.tickers)
        else:
            syms = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY", "QQQ"]
        print(f"جلب {len(syms)} رمزاً · {args.period} · {args.interval}", file=sys.stderr)
        data = fetch(syms, args.period, args.interval)

    all_trades: list[Trade] = []
    base_rs: list[float] = []
    base_wins = base_n = 0

    for sym, df in data.items():
        tr = backtest_symbol(df, cfg, sym, args.max_hold, stage)
        all_trades.extend(tr)
        if tr and not args.no_baseline:
            b = random_baseline(df, tr, args.max_hold)
            if b:
                base_rs.append(b["avg_r"] * b["n"])
                base_wins += b["win_rate"] / 100.0 * b["n"]
                base_n += b["n"]

    if not all_trades:
        print("لا صفقات. وسّع المدة، أو أنقص --pivot، أو أزل --signal-only.")
        return

    print(f"\n{'═' * 78}")
    print(f"باك تست نماذج كارني · {len(data)} رمز · مرحلة الدخول {stage.value}"
          f" · محور {args.pivot} · أقصى حمل {args.max_hold} شمعة")
    print("═" * 78)

    s = summarize(all_trades)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:7.2f}"):
        print("\nحسب النموذج:")
        print(s.to_string())
        print("\nحسب درجة الطبقات (تقارب مستويات الـ PRZ):")
        print(summarize_by(all_trades, "confluence").to_string())
        print("\nحسب الاتجاه:")
        print(summarize_by(all_trades, "direction").to_string())

    rs = np.array([t.r_multiple for t in all_trades])
    wins = (rs > 0).sum()
    print(f"\n{'─' * 78}")
    print(f"الإجمالي: {len(rs)} صفقة · ناجحة {wins / len(rs) * 100:.1f}%"
          f" · متوسط {rs.mean():+.3f}R · وسيط {np.median(rs):+.3f}R"
          f" · مجموع {rs.sum():+.1f}R")

    rd = np.array([t.r_if_entered_at_d for t in all_trades])
    sl = np.array([t.slip_pct for t in all_trades])
    print(f"ثمن التوقيت السليم: الدخول عند D كان يعطي {rd.mean():+.3f}R"
          f" — أي أن الانتظار حتى تثبيت المحور و T-Bar+1 يكلّف"
          f" {rs.mean() - rd.mean():+.3f}R لكل صفقة")
    print(f"متوسط انزلاق الدخول عن نقطة D: {sl.mean():+.2f}%")

    if base_n:
        b_avg = sum(base_rs) / base_n
        b_win = base_wins / base_n * 100.0
        print(f"الأساس العشوائي: {base_n} دخول · ناجح {b_win:.1f}%"
              f" · متوسط {b_avg:+.3f}R")
        edge = rs.mean() - b_avg
        print(f"\nالفرق: {edge:+.3f}R لكل صفقة")
        if abs(edge) < 0.05:
            print("  → لا ميزة تُذكر: النماذج تساوي الدخول العشوائي بنفس الهندسة.")
        elif edge > 0:
            print("  → ميزة موجبة. تحقق أنها تصمد على عيّنة أكبر وفترة أخرى قبل الاعتماد.")
        else:
            print("  → ميزة سالبة: الدخول العشوائي أفضل من النماذج على هذه العيّنة.")

    print(f"\nتحفّظ: العيّنة هنا {len(rs)} صفقة. أقل من ١٠٠ صفقة لكل نموذج لا تكفي"
          f" لحكم إحصائي، وشروط الكتاب تُنتج نماذج قليلة بطبعها.")

    if args.json:
        args.json.write_text(json.dumps(
            {"trades": [t.to_dict() for t in all_trades],
             "config": {"pivot": args.pivot, "min_bars": args.min_bars,
                        "max_hold": args.max_hold, "stage": stage.value}},
            ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"كُتبت الصفقات في {args.json}")


if __name__ == "__main__":
    main()
