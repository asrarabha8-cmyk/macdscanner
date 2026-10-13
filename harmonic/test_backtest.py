"""
test_backtest.py — تحقق من ميكانيكا الباك تست

أهم اختبار هنا هو **الثالث**: لا تسريب مستقبلي. إن سقط فكل نتيجة يخرجها
الباك تست بلا قيمة، وهو بالضبط ما يفسد أغلب باك تست الهارمونيك المنشور.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harmonic_carney import Config, Stage
from backtest import Trade, _simulate, backtest_symbol, random_baseline, summarize


def frame(o, h, l, c) -> pd.DataFrame:
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


# ═══ ١. محاكاة الصفقة: هل تحتسب النتائج صحيحاً؟ ══════════════════════════════

def test_simulate() -> bool:
    print("\n" + "=" * 74)
    print("١. محاكاة الصفقة — TP و SL والأولوية عند التساوي")
    print("=" * 74)

    # صفقة شراء: دخول 100، وقف 95، TP1 102، TP2 105  →  مخاطرة 5
    #   R₁ = 2/5 = 0.40   ·   R₂ = 5/5 = 1.00
    # والاحتساب بنصف مركز عند TP1 كما ينص الكتاب:
    #   TP1 ثم TP2   →  0.5(0.40) + 0.5(1.00) = 0.70
    #   TP1 ثم الوقف →  0.5(0.40) + 0.5(−1.00) = −0.30
    base = dict(entry=100.0, stop=95.0, tp1=102.0, tp2=105.0, bull=True, max_hold=20)
    cases = []

    # (أ) يصعد لـ 105 مباشرة → tp2 بـ +1.0R
    o = np.array([100.0, 101, 103]); h = np.array([100.5, 102.5, 105.5])
    l = np.array([99.8, 100.5, 102]); c = np.array([100.2, 102, 105])
    cases.append(("TP1 ثم TP2", _simulate(o, h, l, c, 0, **base), "tp2", 0.70))

    # (ب) يهبط لـ 95 → stop بـ −1.0R
    o = np.array([100.0, 99, 97]); h = np.array([100.2, 99.5, 97.5])
    l = np.array([99.0, 97, 94.5]); c = np.array([99.2, 97.5, 95])
    cases.append(("يبلغ الوقف", _simulate(o, h, l, c, 0, **base), "stop", -1.0))

    # (ج) شمعة واحدة تلمس الهدف والوقف → الوقف أولاً (احتياط متعمد)
    o = np.array([100.0, 100]); h = np.array([100.2, 106])
    l = np.array([99.5, 94]);  c = np.array([100.0, 100])
    cases.append(("تساوي اللمس", _simulate(o, h, l, c, 0, **base), "stop", -1.0))

    # (د) يلمس TP1 ثم يُوقف → النصف الأول ربح، فلا تُحتسب −1 كاملة
    o = np.array([100.0, 100, 100]); h = np.array([100.2, 102.5, 100.5])
    l = np.array([99.5, 100, 94.0]); c = np.array([100.0, 101, 95])
    cases.append(("TP1 ثم الوقف", _simulate(o, h, l, c, 0, **base), "stop", -0.30))

    # (هـ) الوقف قبل أي هدف → −1 كاملة
    o = np.array([100.0, 98]); h = np.array([100.1, 98.5])
    l = np.array([99.6, 94.0]); c = np.array([99.8, 95])
    cases.append(("وقف بلا TP1", _simulate(o, h, l, c, 0, **base), "stop", -1.0))

    ok = 0
    for name, res, want_out, want_r in cases:
        got_out, _, _, got_r = res[0], res[1], res[2], res[3]
        good = got_out == want_out and abs(got_r - want_r) < 0.01
        ok += good
        print(f"  {'✓' if good else '✗'} {name:<16} → {got_out:<8} {got_r:+.2f}R"
              f"   (مطلوب {want_out} {want_r:+.2f}R)")
    print(f"\n  النتيجة: {ok}/{len(cases)}")
    return ok == len(cases)


# ═══ ٢. حساب R: هل الوقف دائماً −1R بالتعريف؟ ════════════════════════════════

def test_r_definition() -> bool:
    print("\n" + "=" * 74)
    print("٢. تعريف R — الوقف يجب أن يساوي −1.00R دائماً، أي كانت المسافة")
    print("=" * 74)
    ok = True
    for stop_dist in (2.0, 5.0, 11.37):
        e = 100.0
        o = np.array([e, e - stop_dist * 0.5, e - stop_dist * 1.2])
        h = o + 0.1
        l = np.array([e - 0.1, e - stop_dist * 0.6, e - stop_dist * 1.3])
        c = o - 0.05
        res = _simulate(o, h, l, c, 0, e, e - stop_dist, e + 2, e + 4, True, 10)
        good = res[0] == "stop" and abs(res[3] + 1.0) < 1e-9
        ok &= good
        print(f"  {'✓' if good else '✗'} مسافة الوقف {stop_dist:6.2f} → {res[3]:+.4f}R")
    return ok


# ═══ ٣. لا تسريب مستقبلي — الاختبار الحاسم ═══════════════════════════════════

def test_no_lookahead() -> bool:
    print("\n" + "=" * 74)
    print("٣. لا تسريب مستقبلي — الاختبار الحاسم، على الطرق الثلاث")
    print("=" * 74)
    print("  نقطة D محور، ولا تُعرف قبل تثبيتها. ولحظة التثبيت تختلف بالطريقة:")
    print("  الكسرية تتأخر pivot_right دائماً، والـ zigzag تأخرها متغير.")
    print("  افتراض رقم ثابت لكل الطرق = تسريب مستقبلي صامت.\n")

    rng = np.random.default_rng(3)
    methods = [
        ("fractal",    Config(pivot_method="fractal", pivot_left=5, pivot_right=5)),
        ("zigzag_pct", Config(pivot_method="zigzag_pct", zz_pct=6.0)),
        ("zigzag_atr", Config(pivot_method="zigzag_atr", zz_atr_mult=2.5)),
    ]

    series = []
    for _ in range(400):
        ret = rng.normal(0.0003, 0.015, 600)
        c = 100 * np.exp(np.cumsum(ret))
        o = np.concatenate([[c[0]], c[:-1]])
        w = np.abs(rng.normal(0, 0.007, 600)) * c
        series.append(frame(o, np.maximum(o, c) + w, np.minimum(o, c) - w, c))

    all_ok = True
    for name, cfg in methods:
        checked = violations = 0
        examples = []
        lags = []
        for df in series:
            for t in backtest_symbol(df, cfg, "T", max_hold=60, min_stage=Stage.TBAR):
                checked += 1
                lags.append(t.d_confirmed_at - t.d_bar)
                need = max(t.d_confirmed_at, t.tbar) + 1
                if t.entry_bar < need:
                    violations += 1
                    if len(examples) < 2:
                        examples.append((t.d_bar, t.d_confirmed_at, t.tbar,
                                         t.entry_bar, need))
                # الدخول بافتتاح شمعة الدخول لا بسعر D
                if abs(t.entry - df["open"].iloc[t.entry_bar]) > 1e-9:
                    violations += 1
                # لحظة التثبيت لا تسبق المحور نفسه
                if t.d_confirmed_at < t.d_bar:
                    violations += 1
                if t.exit_bar < t.entry_bar:
                    violations += 1

        ok = violations == 0
        all_ok &= ok
        lag_txt = (f"تأخر التثبيت: وسيط {int(np.median(lags))} أقصى {max(lags)}"
                   if lags else "—")
        print(f"  {'✓' if ok else '✗'} {name:<12} صفقات {checked:>4}"
              f"  خروق {violations:>3}   {lag_txt}")
        for d, ca, tb, eb, need in examples:
            print(f"      D={d} تثبيت={ca} T-Bar={tb} دخول={eb} (الأدنى {need})")

    print(f"\n  {'✓ لا تسريب في أي طريقة' if all_ok else '✗ يوجد تسريب — النتائج بلا قيمة'}")
    return all_ok


# ═══ ٤. المقياس العشوائي على سوق صاعد اصطناعياً ══════════════════════════════

def test_baseline_catches_drift() -> bool:
    print("\n" + "=" * 74)
    print("٤. المقياس العشوائي — هل يكشف انحياز السوق الصاعد؟")
    print("=" * 74)
    print("  على سلسلة صاعدة بقوة، أي شراء عشوائي سينجح. فإن لم يُظهر المقياس")
    print("  ذلك فهو لا يحمينا من نسبة نجاح كاذبة.\n")

    rng = np.random.default_rng(5)
    # انحياز صاعد قوي: +0.15% يومياً
    ret = rng.normal(0.0015, 0.012, 1500)
    c = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[c[0]], c[:-1]])
    w = np.abs(rng.normal(0, 0.006, 1500)) * c
    df = frame(o, np.maximum(o, c) + w, np.minimum(o, c) - w, c)

    fake = [Trade(
        symbol="T", pattern="Gartley", direction="bullish", confluence=3,
        d_bar=100, d_confirmed_at=105, tbar=100, entry_bar=100, entry=float(o[100]),
        stop=float(o[100]) * 0.95, tp1=float(o[100]) * 1.02,
        tp2=float(o[100]) * 1.04, outcome="tp1", exit_bar=110,
        exit_px=float(c[110]), bars_held=10, r_multiple=0.4, pct=2.0,
        mfe_r=0.5, mae_r=-0.2, hit_tp1=True, hit_tp2=False,
        stage_at_entry="TBAR",
    )]
    b = random_baseline(df, fake, max_hold=60, n_per_trade=300, seed=1)
    print(f"  دخول عشوائي: {b['n']} · ناجح {b['win_rate']:.1f}%"
          f" · متوسط {b['avg_r']:+.3f}R")
    total = (c[-1] / c[0] - 1) * 100
    print(f"  (السلسلة ارتفعت {total:.0f}% إجمالاً)")
    good = b["win_rate"] > 55.0
    print(f"\n  {'✓ المقياس يكشف الانحياز' if good else '✗ المقياس لا يكشفه'}")
    return good


# ═══ ٥. التجميع ══════════════════════════════════════════════════════════════

def test_summary() -> bool:
    print("\n" + "=" * 74)
    print("٥. جدول التجميع")
    print("=" * 74)
    rng = np.random.default_rng(9)
    cfg = Config(pivot_left=4, pivot_right=4, min_pattern_bars=30)
    trades = []
    for _ in range(300):
        ret = rng.normal(0.0004, 0.015, 700)
        c = 100 * np.exp(np.cumsum(ret))
        o = np.concatenate([[c[0]], c[:-1]])
        w = np.abs(rng.normal(0, 0.007, 700)) * c
        df = frame(o, np.maximum(o, c) + w, np.minimum(o, c) - w, c)
        trades.extend(backtest_symbol(df, cfg, "RND", 60, Stage.TBAR))

    if not trades:
        print("  لا صفقات على البيانات العشوائية — شروط الكتاب صارمة")
        return True
    s = summarize(trades)
    with pd.option_context("display.width", 190, "display.float_format",
                           lambda v: f"{v:7.2f}"):
        print(s.to_string())
    rs = np.array([t.r_multiple for t in trades])
    print(f"\n  {len(rs)} صفقة · متوسط {rs.mean():+.3f}R"
          f" · ناجحة {(rs > 0).mean() * 100:.1f}%")
    print("  (على مسار عشوائي يُتوقع متوسط قريب من الصفر — وهذا فحص سلامة)")
    return True


if __name__ == "__main__":
    r1 = test_simulate()
    r2 = test_r_definition()
    r3 = test_no_lookahead()
    r4 = test_baseline_catches_drift()
    test_summary()

    print("\n" + "=" * 74)
    print("الخلاصة")
    print("=" * 74)
    print(f"  محاكاة الصفقة        : {'ناجح' if r1 else 'فاشل'}")
    print(f"  تعريف R              : {'ناجح' if r2 else 'فاشل'}")
    print(f"  لا تسريب مستقبلي     : {'ناجح' if r3 else 'فاشل ← النتائج بلا قيمة'}")
    print(f"  المقياس يكشف الانحياز: {'ناجح' if r4 else 'فاشل'}")
