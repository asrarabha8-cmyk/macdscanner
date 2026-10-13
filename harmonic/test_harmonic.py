"""
test_harmonic.py — تحقق من نواة الكشف

ثلاثة أسئلة:
  ١. هل يكتشف نموذجاً كتابياً مضبوط النسب ويسمّيه صحيحاً؟
  ٢. هل يرفض نموذجاً خارج حد تسامح نقطة B؟
  ٣. كم نموذجاً يرى على مسار عشوائي؟ (معدل الضجيج)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harmonic_carney import Config, Stage, SPECS, find_patterns, find_pivots


# ─── بناء نموذج صناعي بنسب مضبوطة ────────────────────────────────────────────

def build_bullish(
    b_ratio: float, c_frac: float, xa_ratio: float,
    x_price: float = 100.0, xa_leg: float = 20.0,
    bars_per_leg: int = 12, tail: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    يبني OHLC صاعداً X(قاع) A(قمة) B(قاع) C(قمة) D(قاع).

    c_frac = موضع C بين B و A (0 عند B، 1 عند A). بناء C بهذه الطريقة
    يضمن هندسة صحيحة دائماً، وإسقاط BC يُحسب تبعاً لا العكس.

    يعيد (البيانات، إسقاط BC الناتج).
    """
    X = x_price
    A = X + xa_leg
    B = A - b_ratio * xa_leg
    D = A - xa_ratio * xa_leg
    C = B + c_frac * (A - B)
    assert B < C < A, f"هندسة غير متسقة: B={B:.2f} C={C:.2f} A={A:.2f}"
    bc_ratio = (C - D) / (C - B)

    pts = [X, A, B, C, D]
    # هبوط تمهيدي حتى X — بلا هذا لا تُكتشف X كمحور (لا شمعات على يسارها)
    lead = 8
    closes: list[float] = np.linspace(X + 0.35 * xa_leg, X, lead, endpoint=False).tolist()
    offset = lead
    for i in range(len(pts) - 1):
        closes.extend(np.linspace(pts[i], pts[i + 1], bars_per_leg, endpoint=False).tolist())
    closes.append(D)
    # ذيل انعكاسي صاعد واضح — ليتحقق شرط تغيّر الطابع بعد الـ T-Bar
    closes.extend(np.linspace(D, D + 0.45 * (A - D), tail).tolist())

    c = np.asarray(closes)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + 0.01
    l = np.minimum(o, c) - 0.01
    # ظل صغير عند كل نقطة انعطاف ليصير محوراً حقيقياً بأقل تشويه للنسب
    for k, px in enumerate(pts):
        idx = offset + k * bars_per_leg
        if idx >= len(c):
            continue
        if k % 2 == 0:                      # X, B, D = قيعان
            l[idx] = px - 0.08
        else:                               # A, C = قمم
            h[idx] = px + 0.08
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}), bc_ratio


CFG = Config(pivot_left=3, pivot_right=3, min_pattern_bars=30, prz_entry_tol=0.04)


def test_textbook_patterns() -> None:
    print("\n" + "=" * 72)
    print("١. نماذج كتابية مضبوطة النسب — هل تُكتشف وتُسمّى صحيحاً؟")
    print("=" * 72)

    # النسب الثلاث لكل حالة من كتلة Elements الخاصة بالنموذج:
    #   (نقطة B ، إسقاط BC ، نسبة XA عند D)
    # (نقطة B ، موضع C بين B و A ، نسبة XA عند D) — من كتلة Elements لكل نموذج
    cases = [
        ("Gartley",       0.618, 0.90, 0.786),
        ("Bat",           0.450, 0.90, 0.886),
        ("Alternate Bat", 0.370, 0.90, 1.000),
        ("Butterfly",     0.786, 0.85, 1.270),
        ("Crab",          0.500, 0.90, 1.618),
        ("Deep Crab",     0.900, 0.70, 1.618),
    ]
    passed = 0
    for want, b, cf, xa in cases:
        df, bc = build_bullish(b, cf, xa)
        hits = find_patterns(df, CFG, symbol="TEST")
        five = [p for p in hits if p.D is not None]
        names = {p.pattern for p in five}
        ok = want in names
        passed += ok
        mark = "✓" if ok else "✗"
        detail = ""
        if five:
            p = five[0]
            detail = (f"B={p.b_ratio:.3f} BC={p.bc_ratio:.3f} "
                      f"XA={p.xa_ratio:.3f} ABCD={p.abcd_ratio:.3f} "
                      f"طبقات={p.confluence} PRZ={p.prz_width_pct*100:.1f}% [{p.stage.value}]")
        print(f"  {mark} {want:<15} BC={bc:5.3f} → {sorted(names) or '— مرفوض'}\n        {detail}")
    print(f"\n  النتيجة: {passed}/{len(cases)}")
    return passed == len(cases)


def test_b_point_rejection() -> None:
    print("\n" + "=" * 72)
    print("٢. حد تسامح نقطة B — هل يُرفض ما خرج عنه؟")
    print("=" * 72)
    print("  Gartley صالح فقط بين 0.588 و 0.648 بنص الكتاب (ص ٩١)\n")

    rows = []
    for b in (0.520, 0.560, 0.600, 0.618, 0.640, 0.670, 0.720):
        df, _ = build_bullish(b, 0.90, 0.786)
        hits = [p for p in find_patterns(df, CFG, symbol="T") if p.D is not None]
        names = sorted({p.pattern for p in hits})
        # نحكم على النسبة **المقيسة** لا المطلوبة: ظل الشمعة يزحزحها قليلاً
        meas = hits[0].b_ratio if hits else b
        in_band = 0.588 <= meas <= 0.648
        ok = ("Gartley" in names) == in_band
        rows.append((b, meas, in_band, names, ok))

    for b, meas, in_band, names, ok in rows:
        mark = "✓" if ok else "✗"
        band = "داخل" if in_band else "خارج"
        print(f"  {mark} B مطلوب {b:.3f} / مقيس {meas:.3f} ({band}) → {names or '— مرفوض'}")
    allok = all(r[-1] for r in rows)
    print(f"\n  النتيجة: {'كل الحدود محترمة' if allok else 'خلل في الحدود'}")
    return allok


def test_noise_rate() -> None:
    print("\n" + "=" * 72)
    print("٣. معدل الضجيج — كم نموذجاً على مسار عشوائي؟")
    print("=" * 72)

    rng = np.random.default_rng(7)
    n_series, n_bars = 200, 500
    counts = {s.value: 0 for s in Stage}
    total_complete = 0

    for _ in range(n_series):
        ret = rng.normal(0, 0.012, n_bars)
        c = 100 * np.exp(np.cumsum(ret))
        o = np.concatenate([[c[0]], c[:-1]])
        wig = np.abs(rng.normal(0, 0.006, n_bars)) * c
        h = np.maximum(o, c) + wig
        l = np.minimum(o, c) - wig
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
        for p in find_patterns(df, Config(), symbol="RND"):
            counts[p.stage.value] += 1
            if p.D is not None:
                total_complete += 1

    bars_total = n_series * n_bars
    print(f"  {n_series} مسار × {n_bars} شمعة = {bars_total:,} شمعة")
    print(f"  نماذج مكتملة (٥ نقاط): {total_complete}")
    for k, v in counts.items():
        if v:
            print(f"    {k:<8} {v}")
    per_1k = total_complete / bars_total * 1000
    print(f"\n  الكثافة: {per_1k:.2f} نموذج لكل ١٠٠٠ شمعة")
    print(f"  أي ما يعادل ~{per_1k * 252 / 1000:.1f} نموذج في السنة للسهم الواحد (فاصل يومي)")
    return per_1k


def test_pivots() -> None:
    print("\n" + "=" * 72)
    print("٤. المحاور — هل تتناوب؟")
    print("=" * 72)
    df, _ = build_bullish(0.618, 0.90, 0.786)
    pv = find_pivots(df["high"], df["low"], 3, 3)
    kinds = [p.kind for p in pv]
    alt = all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))
    print(f"  عدد المحاور: {len(pv)}")
    print(f"  التسلسل: {' → '.join('قمة' if k == 'high' else 'قاع' for k in kinds)}")
    print(f"  {'✓' if alt else '✗'} التناوب {'محفوظ' if alt else 'مكسور'}")
    return alt


if __name__ == "__main__":
    print(f"عدد النماذج في الجدول: {len(SPECS)}")
    print(" · ".join(s.name for s in SPECS))

    r1 = test_textbook_patterns()
    r2 = test_b_point_rejection()
    r4 = test_pivots()
    dens = test_noise_rate()

    print("\n" + "=" * 72)
    print("الخلاصة")
    print("=" * 72)
    print(f"  اكتشاف النماذج الكتابية : {'ناجح' if r1 else 'فاشل'}")
    print(f"  احترام حدود نقطة B      : {'ناجح' if r2 else 'فاشل'}")
    print(f"  تناوب المحاور           : {'ناجح' if r4 else 'فاشل'}")
    print(f"  كثافة الضجيج            : {dens:.2f} / ١٠٠٠ شمعة")
