"""
harmonic_carney.py
==================
كاشف نماذج توافقية بالالتزام الحرفي بكتاب:
    Scott M. Carney — Harmonic Trading Volume 3: Reaction vs. Reversal (2016)

النماذج المشمولة (بنية M/W ذات خمس نقاط X-A-B-C-D):
    Gartley · Bat · Alternate Bat · Butterfly · Crab · Deep Crab

غير مشمولة بقصد:
    Shark و 5-0  → بنيتهما 0-X-A-B-C وتحتاج كاشفاً منفصلاً.
    Cypher       → غير موجود في الكتاب إطلاقاً (نموذج دارين أوجلسبي).
    HSI          → معادلة ملكية غير منشورة في الكتاب؛ البديل المنصوص هو RSI القياسي.

كل رقم في SPECS أدناه منقول من كتلة "Elements" الخاصة بالنموذج في الفصل الثاني.
كل معامل من اختياري أنا — لا من الكتاب — مُعلَّم بـ  # [param]
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PatternSpec",
    "SPECS",
    "Config",
    "Stage",
    "Pivot",
    "HarmonicPattern",
    "find_pivots",
    "find_patterns",
    "rsi",
]


# ─────────────────────────────────────────────────────────────────────────────
# جدول النسب — منقول حرفياً من الفصل الثاني
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternSpec:
    """مواصفة نموذج واحد. كل الحدود شاملة (inclusive)."""

    name: str
    name_ar: str

    # نقطة B كنسبة من ضلع XA  →  AB / XA
    b_min: float
    b_max: float

    # إسقاط BC  →  CD / BC
    bc_min: float
    bc_max: float

    # نقطة الإتمام D كنسبة من ضلع XA  →  AD / XA
    xa_ideal: float          # النسبة الرئيسية المعلنة في الكتاب لهذا النموذج
    xa_min: float
    xa_max: float

    # نوع AB=CD — تعليمة بناء الـ PRZ لا قيد على ضلع CD.
    # "AB=CD–1.618AB=CD" تعني أن مستوى الـ PRZ يُحسب بإسقاط ضلع AB من C
    # بمعامل بين 1.0 و 1.618، ويُختار المعامل الذي يتقارب مع مستوى XA.
    abcd_min: float
    abcd_max: float

    # نسبة XA التي يوضع وقف الخسارة خلفها (جدول "Standard Stop Loss Pattern Limits" ص ١٧٤)
    stop_xa: float

    # ملاحظة الكتاب الإلزامية على هذا النموذج
    note: str = ""


# تسامح نقطة B عند كارني يُقاس بنقاط مئوية مطلقة لا نسبةً:
# "+/-3%" على 61.8% يعني 58.8%–64.8% بنص الكتاب (ص ٩١).
_T3 = 0.03
_T5 = 0.05

# حيث يعطي الكتاب نطاقاً لنقطة B (Bat 0.382–0.50، Crab 0.382–0.618) فالنطاق
# هو التسامح نفسه، ولا يُضاف إليه شيء. وحيث يعطي قيمة مفردة يُضاف التسامح المنصوص.

SPECS: tuple[PatternSpec, ...] = (
    PatternSpec(
        name="Gartley", name_ar="جارتلي",
        b_min=0.618 - _T3, b_max=0.618 + _T3,      # 0.588–0.648 منصوص صريحاً
        bc_min=1.13, bc_max=1.618,
        xa_ideal=0.786, xa_min=0.786 - _T3, xa_max=0.886,   # يمتد إلى 0.886 في Deep Gartley
        abcd_min=1.0, abcd_max=1.27,
        stop_xa=1.0,
        note="يجب اختبار الـ PRZ كاملة قبل الدخول. الاختبار الأول يجب ألا يتجاوز الـ PRZ.",
    ),
    PatternSpec(
        name="Bat", name_ar="الخفاش",
        b_min=0.382, b_max=0.50,                   # نطاق منصوص — لا تسامح إضافي
        bc_min=1.618, bc_max=2.618,
        xa_ideal=0.886, xa_min=0.886 - _T3, xa_max=0.886 + _T3,
        abcd_min=1.0, abcd_max=1.618,
        stop_xa=1.13,
        note="يُتوقع اختبار 0.886 مع حد أدنى AB=CD (عادةً 1.27AB=CD). يجوز الاقتراب من 1.0XA.",
    ),
    PatternSpec(
        name="Alternate Bat", name_ar="الخفاش البديل",
        b_min=0.382 - _T3, b_max=0.382,            # "0.382 كحد أقصى، تسامح -3%"
        bc_min=2.0, bc_max=3.618,
        xa_ideal=1.0, xa_min=0.886, xa_max=1.13,
        abcd_min=1.618 - 0.10, abcd_max=1.618 + 0.10,   # [param] 1.618AB=CD مفردة
        stop_xa=1.27,
        note="الحالة يجب أن تشمل 1.0XA، والانعكاس فوري بعد تجاوزها.",
    ),
    PatternSpec(
        name="Butterfly", name_ar="الفراشة",
        b_min=0.786 - _T3, b_max=0.786 + _T3,      # 0.756–0.816
        bc_min=1.618, bc_max=2.24,
        xa_ideal=1.27, xa_min=1.27 - _T3, xa_max=1.414,     # يُفضّل 1.414 إن تجاوز AB=CD مستوى 1.27
        abcd_min=1.0, abcd_max=1.27,
        stop_xa=1.414,
        note="نقطة B يجب أن تكون 0.786 داخل ٣%؛ خارجها تتحول البنية عادةً إلى Crab عند 1.618.",
    ),
    PatternSpec(
        name="Crab", name_ar="السلطعون",
        b_min=0.382, b_max=0.618,                  # نطاق منصوص
        bc_min=2.618, bc_max=3.618,
        xa_ideal=1.618, xa_min=1.618 - _T5, xa_max=1.902,   # 1.902 تسامح إضافي منصوص (Bryce Gilmore)
        abcd_min=1.0, abcd_max=1.618,
        stop_xa=2.0,
        note="يُتوقع تجاوز 1.618XA بحركة عنيفة، والانعكاس يجب أن يكون فورياً بعدها.",
    ),
    PatternSpec(
        name="Deep Crab", name_ar="السلطعون العميق",
        b_min=0.886, b_max=0.886 + _T5,            # "0.886 بتسامح +5%" — كتلة Elements
        bc_min=2.0, bc_max=3.618,
        xa_ideal=1.618, xa_min=1.618 - _T5, xa_max=1.902,
        abcd_min=1.0, abcd_max=1.618,
        stop_xa=2.0,
        note="0.886 عند B حد أدنى يجب اختباره لتشغيل البنية.",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# الإعدادات
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    معاملات التشغيل.

    القيم المعلَّمة [كتاب] منصوصة في Harmonic Trading Volume 3.
    القيم المعلَّمة [param] اختياري أنا — الكتاب صامت عنها، وهي أكبر متغير
    وحده في أي ماسح هارمونيك. اضبطها بصرياً على شارتات تعرفها قبل الاعتماد عليها.
    """

    # منطق المحاور — [param] الكتاب لا يذكر طريقة تحديد المحاور أبداً
    pivot_left: int = 5
    pivot_right: int = 5

    # [كتاب] ص ٣١٨: النموذج يجب أن يمتد على ٣٠ شمعة على الأقل (نظرية النهاية المركزية)
    min_pattern_bars: int = 30

    # [كتاب] ص ١٥٢: "من الشائع أن تشكل الـ PRZ ٣–٥% فقط من مدى النموذج الكلي"
    # شرط تقارب يرفض المناطق المتباعدة التي لا تصلح منطقةَ انعكاس واحدة
    max_prz_width: float = 0.05

    # [param] سماح لنقطة D أن تقع خارج حدود الـ PRZ المحسوبة، كنسبة من مدى النموذج.
    # الكتاب لا يعطي رقماً هنا بقصد: الـ PEZ = PRZ + طرف الـ T-Bar، أي أن التسامح
    # يحدده سلوك السعر لا رقم ثابت. هذا تقريب برمجي لا بديل عن قراءة الشمعة.
    prz_entry_tol: float = 0.015

    # [param] تسامح على شرط AB=CD الأدنى (CD ≥ AB). الكتاب يجعله مطلقاً،
    # وهذا هامش صغير يمنع رفض بنية سليمة بفارق ١–٢% في القياس.
    abcd_tol: float = 0.03

    # [كتاب] ص ١٩٧: الاستمرار الواضح يجب أن يتحقق خلال ٣–٥ شمعات بعد الـ T-Bar
    confirm_bars: int = 5

    # [كتاب] ص ٢١٤–٢١٥: ٣٠ تشبع بيعي، ٧٠ تشبع شرائي، والقراءة يجب أن تكون < ٣١ فعلياً
    rsi_period: int = 14
    rsi_oversold: float = 31.0
    rsi_overbought: float = 69.0
    require_rsi: bool = False          # [param] الكتاب يجعله مُرجَّحاً لا شرطاً مطلقاً

    # [param] أقصى عدد محاور يُرجَع للخلف بحثاً عن بنية
    max_lookback_pivots: int = 40


class Stage(str, Enum):
    """مراحل الكتاب الثلاث. الرسم ليس إشارة — الإشارة تبدأ عند T-Bar+1."""

    WATCH = "WATCH"      # ٤ من ٥ نقاط مثبتة، الـ PRZ مُسقطة، السعر لم يصلها بعد ("أثر المغناطيس")
    TBAR = "TBAR"        # السعر اختبر آخر رقم في الـ PRZ — النموذج مكتمل رسمياً
    SIGNAL = "SIGNAL"    # T-Bar+1 مع تغيّر طابع السعر في اتجاه الانعكاس
    FAILED = "FAILED"    # تجاوز الـ PRZ دون أي شمعة بنّاءة — فرصة معيبة تُترك


Direction = Literal["bullish", "bearish"]


# ─────────────────────────────────────────────────────────────────────────────
# المحاور
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pivot:
    index: int
    price: float
    kind: Literal["high", "low"]


def find_pivots(
    high: Sequence[float],
    low: Sequence[float],
    left: int,
    right: int,
) -> list[Pivot]:
    """
    محاور كسرية (fractal) متناوبة.

    محور قمة = أعلى من `left` شمعة قبله و `right` شمعة بعده.
    ثم يُفرض التناوب: عند توالي محورين من نفس النوع يُستبقى الأكثر تطرفاً.

    التناوب شرط بنيوي: نموذج X-A-B-C-D يتطلب تعاقب قمة/قاع بالضرورة.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(h)
    raw: list[Pivot] = []

    for i in range(left, n - right):
        window_h = h[i - left : i + right + 1]
        if h[i] == window_h.max() and (window_h == h[i]).sum() == 1:
            raw.append(Pivot(i, float(h[i]), "high"))
            continue
        window_l = l[i - left : i + right + 1]
        if l[i] == window_l.min() and (window_l == l[i]).sum() == 1:
            raw.append(Pivot(i, float(l[i]), "low"))

    # فرض التناوب
    out: list[Pivot] = []
    for p in raw:
        if not out or out[-1].kind != p.kind:
            out.append(p)
            continue
        prev = out[-1]
        more_extreme = p.price > prev.price if p.kind == "high" else p.price < prev.price
        if more_extreme:
            out[-1] = p
    return out


# ─────────────────────────────────────────────────────────────────────────────
# النموذج المكتشف
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HarmonicPattern:
    symbol: str
    pattern: str
    pattern_ar: str
    direction: Direction
    stage: Stage

    # النقاط: (index, price)
    X: tuple[int, float]
    A: tuple[int, float]
    B: tuple[int, float]
    C: tuple[int, float]
    D: tuple[int, float] | None          # None في مرحلة WATCH

    # النسب المقيسة
    b_ratio: float
    bc_ratio: float | None
    xa_ratio: float | None
    abcd_ratio: float | None
    time_symmetry: float | None          # (مدة CD) / (مدة AB) — الكتاب يشترط التماثل الزمني بلا رقم

    # منطقة الانعكاس
    prz_low: float
    prz_high: float
    prz_width_pct: float
    confluence: int = 1                  # كم مستوى من الثلاثة تقارب (٣ أفضل)
    prz_levels: dict[str, float] = field(default_factory=dict)

    # الإدارة
    entry: float | None = None
    stop: float | None = None
    tp1: float | None = None             # D + 0.382 × (A − D)
    tp2: float | None = None             # D + 0.618 × (A − D)

    tbar_index: int | None = None
    rsi_at_tbar: float | None = None
    bars_span: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    def __str__(self) -> str:
        arrow = "▲" if self.direction == "bullish" else "▼"
        d_px = f"{self.D[1]:.2f}" if self.D else f"{self.prz_low:.2f}–{self.prz_high:.2f}"
        return (
            f"{arrow} {self.symbol:<6} {self.pattern:<14} {self.stage.value:<7} "
            f"D={d_px:<16} B={self.b_ratio:.3f} "
            f"PRZ عرض={self.prz_width_pct*100:.1f}% شمعات={self.bars_span}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RSI القياسي (Wilder) — البديل المنصوص عن HSI غير المنشور
# ─────────────────────────────────────────────────────────────────────────────

def rsi(close: Sequence[float], period: int = 14) -> np.ndarray:
    c = np.asarray(close, dtype=float)
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_g = np.full_like(c, np.nan)
    avg_l = np.full_like(c, np.nan)
    if len(c) <= period:
        return np.full_like(c, 50.0)

    avg_g[period] = gain[1 : period + 1].mean()
    avg_l[period] = loss[1 : period + 1].mean()
    for i in range(period + 1, len(c)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_l == 0, np.inf, avg_g / np.where(avg_l == 0, 1, avg_l))
    out = 100.0 - 100.0 / (1.0 + rs)
    return np.nan_to_num(out, nan=50.0)


# ─────────────────────────────────────────────────────────────────────────────
# حساب الـ PRZ
# ─────────────────────────────────────────────────────────────────────────────

# جدول النسب العكسية — منقول حرفياً من الفصل الثاني (ص ٨٠)
# ارتداد نقطة C  →  إسقاط BC المثالي
RECIPROCAL: dict[float, tuple[float, ...]] = {
    0.382: (2.24, 2.618, 3.14, 3.618),
    0.50:  (2.0,),
    0.618: (1.618,),
    0.707: (1.414,),
    0.786: (1.272,),
    0.886: (1.13,),
    1.0:   (2.0,),
}


def _bc_projection(c_retr: float, bc_leg: float, abcd_level: float,
                   c: float, sign: float) -> tuple[float, float]:
    """
    إسقاط BC بجدول الكتاب العكسي، لا بمنتصف نطاق اعتباطي.

    ارتداد نقطة C الفعلي يحدّد إسقاط BC المثالي. وحين يعطي الجدول عدة
    احتمالات (كما عند 0.382) نختار ما يُكمّل نقطة إتمام AB=CD أقربَ ما يكون
    — وهي الطريقة التي يصفها الكتاب في استخدام الطبقات (layering).
    """
    key = min(RECIPROCAL, key=lambda k: abs(k - c_retr))
    best_ratio, best_level, best_gap = None, None, float("inf")
    for ratio in RECIPROCAL[key]:
        level = c + sign * ratio * bc_leg
        gap = abs(level - abcd_level)
        if gap < best_gap:
            best_ratio, best_level, best_gap = ratio, level, gap
    return float(best_ratio), float(best_level)


def _prz_levels(
    spec: PatternSpec,
    x: float, a: float, b: float, c: float,
    direction: Direction,
    window: float,
) -> dict:
    """
    تبني الـ PRZ من ثلاثة مستويات، مرساتها ضلع XA.

    الكتاب يجعل ارتداد/امتداد XA هو المستوى الحاكم ("أكبر ضلع في النموذج"،
    ص ١٨٣)، والـ AB=CD و BC "يُكمّلان المنطقة". ولذلك لا نرفض بنيةً لأن
    مستوياتها الثلاثة لم تتقارب — نماذج الامتداد (Crab، Butterfly) لا تتقارب
    أصلاً لأن إسقاط AB=CD يقع فوق امتداد XA. بدلاً من الرفض نُرجع عدد
    المستويات المتقاربة كـ **درجة طبقات (confluence)** تُرتَّب بها النتائج:
    ٣ = تقارب تام، ١ = ضلع XA وحده.
    """
    xa_leg = abs(a - x)
    ab_leg = abs(a - b)
    bc_leg = abs(c - b)
    sign = -1.0 if direction == "bullish" else 1.0   # D يتحرك بعيداً عن A

    anchor = a + sign * spec.xa_ideal * xa_leg       # مستوى XA — المرساة

    # إسقاط AB=CD: يُختار المعامل الكتابي الأقرب للمرساة
    cands = [m for m in (1.0, 1.27, 1.618) if spec.abcd_min <= m <= spec.abcd_max] \
            or [spec.abcd_min]
    abcd_mult = min(cands, key=lambda m: abs((c + sign * m * ab_leg) - anchor))
    abcd = c + sign * abcd_mult * ab_leg

    # إسقاط BC بجدول النسب العكسية
    c_retr = bc_leg / ab_leg if ab_leg else 0.0
    bc_mult, bc = _bc_projection(c_retr, bc_leg, anchor, c, sign)

    levels = {"XA": anchor, "ABCD": abcd, "BC": bc}
    kept = {k: v for k, v in levels.items() if abs(v - anchor) <= window}
    # عند تقارب مستوى واحد فقط تكون الـ PRZ هي مستوى XA نفسه — بلا نطاق
    # وهمي: اختراع حدٍّ أدنى غير موجود يُنشئ "آخر رقم" لا يختبره السعر أبداً،
    # فلا تتشكل الـ T-Bar ويتعطل التصنيف.
    lo, hi = min(kept.values()), max(kept.values())

    return {
        "levels": levels,
        "lo": lo, "hi": hi,
        "confluence": len(kept),
        "abcd_mult": abcd_mult,
        "bc_mult": bc_mult,
    }


def _in(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


# ─────────────────────────────────────────────────────────────────────────────
# الكاشف
# ─────────────────────────────────────────────────────────────────────────────

def find_patterns(
    df: pd.DataFrame,
    cfg: Config | None = None,
    symbol: str = "",
) -> list[HarmonicPattern]:
    """
    يبحث في OHLC عن نماذج توافقية ويعيدها مع مرحلتها.

    df يجب أن يحمل الأعمدة: open, high, low, close  (حروف صغيرة)
    """
    cfg = cfg or Config()
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"الأعمدة المطلوبة: {sorted(need)} — الموجود: {list(df.columns)}")
    if len(df) < cfg.min_pattern_bars + cfg.pivot_left + cfg.pivot_right + 5:
        return []

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    r = rsi(c, cfg.rsi_period)
    last = len(df) - 1

    pivots = find_pivots(h, l, cfg.pivot_left, cfg.pivot_right)
    if len(pivots) < 4:
        return []
    pivots = pivots[-cfg.max_lookback_pivots :]

    found: list[HarmonicPattern] = []

    # ── نماذج مكتملة بخمس نقاط ───────────────────────────────────────────────
    for i in range(len(pivots) - 4):
        quint = pivots[i : i + 5]
        if quint[0].kind == "low":
            direction: Direction = "bullish"      # X قاع → A قمة → B قاع → C قمة → D قاع
        else:
            direction = "bearish"
        # تأكيد التناوب الصحيح
        expect = ["low", "high", "low", "high", "low"] if direction == "bullish" else \
                 ["high", "low", "high", "low", "high"]
        if [p.kind for p in quint] != expect:
            continue

        X, A, B, C, D = quint
        span = D.index - X.index
        if span < cfg.min_pattern_bars:
            continue

        found.extend(
            _validate_quint(X, A, B, C, D, direction, span, cfg, symbol, h, l, c, o, r, last)
        )

    # ── مرحلة المراقبة: ٤ نقاط مثبتة و D لم تتشكل بعد ───────────────────────
    if len(pivots) >= 4:
        quad = pivots[-4:]
        direction = "bullish" if quad[0].kind == "low" else "bearish"
        expect = ["low", "high", "low", "high"] if direction == "bullish" else \
                 ["high", "low", "high", "low"]
        if [p.kind for p in quad] == expect:
            found.extend(_project_watch(quad, direction, cfg, symbol, h, l, c, last))

    return found


def _validate_quint(
    X: Pivot, A: Pivot, B: Pivot, C: Pivot, D: Pivot,
    direction: Direction, span: int, cfg: Config, symbol: str,
    h: np.ndarray, l: np.ndarray, c: np.ndarray, o: np.ndarray, r: np.ndarray,
    last: int,
) -> list[HarmonicPattern]:

    xa_leg = abs(A.price - X.price)
    ab_leg = abs(A.price - B.price)
    bc_leg = abs(C.price - B.price)
    cd_leg = abs(C.price - D.price)
    if min(xa_leg, ab_leg, bc_leg, cd_leg) <= 0:
        return []

    b_ratio = ab_leg / xa_leg
    bc_ratio = cd_leg / bc_leg
    xa_ratio = abs(A.price - D.price) / xa_leg
    abcd_ratio = cd_leg / ab_leg
    t_ab = max(B.index - A.index, 1)
    t_cd = max(D.index - C.index, 1)
    time_sym = t_cd / t_ab

    # شرط بنيوي: C بين A و B، و D تتجاوز B في اتجاه الضلع
    if direction == "bullish":
        if not (B.price < C.price < A.price):
            return []
    else:
        if not (A.price < C.price < B.price):
            return []

    out: list[HarmonicPattern] = []

    for spec in SPECS:
        if not _in(b_ratio, spec.b_min, spec.b_max):
            continue
        if not _in(bc_ratio, spec.bc_min, spec.bc_max):
            continue
        if not _in(xa_ratio, spec.xa_min, spec.xa_max):
            continue
        # الكتاب ص ٧٥: "AB=CD حد أدنى بنيوي إلزامي" ويُتجاوَز عادةً.
        # لذا الشرط هو CD ≥ AB، وليس نطاقاً مغلقاً — النطاق أعلاه لبناء الـ PRZ.
        if abcd_ratio < 1.0 - cfg.abcd_tol:
            continue

        # الكتاب ص ١٥٢: الـ PRZ تشكل ٣–٥% من **مدى النموذج الكلي**
        pat_range = max(X.price, A.price, B.price, C.price, D.price) - \
                    min(X.price, A.price, B.price, C.price, D.price)
        if pat_range <= 0:
            continue
        prz = _prz_levels(spec, X.price, A.price, B.price, C.price, direction,
                          window=cfg.max_prz_width * pat_range)
        lo, hi, levels = prz["lo"], prz["hi"], prz["levels"]
        width = (hi - lo) / pat_range

        # D يجب أن تقع داخل الـ PRZ (مع تسامح برمجي معلن)
        tol = cfg.prz_entry_tol * pat_range
        if not (lo - tol <= D.price <= hi + tol):
            continue

        stage, tbar_idx = _classify_stage(
            D, direction, lo, hi, cfg, h, l, c, o, last, touch_tol=tol
        )

        sign = -1.0 if direction == "bullish" else 1.0
        stop = A.price + sign * spec.stop_xa * xa_leg
        ad = A.price - D.price
        tp1 = D.price + 0.382 * ad
        tp2 = D.price + 0.618 * ad

        rsi_tbar = float(r[tbar_idx]) if tbar_idx is not None else None
        if cfg.require_rsi and rsi_tbar is not None:
            ok = (direction == "bullish" and rsi_tbar <= cfg.rsi_oversold) or \
                 (direction == "bearish" and rsi_tbar >= cfg.rsi_overbought)
            if not ok and stage is Stage.SIGNAL:
                stage = Stage.TBAR

        out.append(HarmonicPattern(
            symbol=symbol, pattern=spec.name, pattern_ar=spec.name_ar,
            direction=direction, stage=stage,
            X=(X.index, X.price), A=(A.index, A.price), B=(B.index, B.price),
            C=(C.index, C.price), D=(D.index, D.price),
            b_ratio=b_ratio, bc_ratio=bc_ratio, xa_ratio=xa_ratio,
            abcd_ratio=abcd_ratio, time_symmetry=time_sym,
            prz_low=lo, prz_high=hi, prz_width_pct=width,
            confluence=prz["confluence"], prz_levels=levels,
            entry=D.price, stop=stop, tp1=tp1, tp2=tp2,
            tbar_index=tbar_idx, rsi_at_tbar=rsi_tbar,
            bars_span=span, note=spec.note,
        ))
        break   # نموذج واحد لكل بنية — أول مطابقة في ترتيب SPECS

    return out


def _classify_stage(
    D: Pivot, direction: Direction, prz_lo: float, prz_hi: float,
    cfg: Config, h: np.ndarray, l: np.ndarray, c: np.ndarray, o: np.ndarray,
    last: int, touch_tol: float = 0.0,
) -> tuple[Stage, int | None]:
    """
    الـ T-Bar هي الشمعة التي تختبر **آخر رقم** في الـ PRZ — لا أولها.
    والإشارة تبدأ بعدها (T-Bar+1) بشرط تغيّر طابع السعر.

    touch_tol يسمح بأن يرتد السعر قبل المستوى بفرق ضئيل. الكتاب يعالج هذا
    بمفهوم الـ PEZ (التسامح = طرف الـ T-Bar نفسها) لا برقم ثابت، فهذا تقريب
    برمجي معلن وليس شرطاً كتابياً.
    """
    far = prz_lo if direction == "bullish" else prz_hi

    tbar = None
    for i in range(D.index, min(last, D.index + cfg.confirm_bars) + 1):
        tested = (l[i] <= far + touch_tol) if direction == "bullish" \
                 else (h[i] >= far - touch_tol)
        if tested:
            tbar = i
            break
    if tbar is None:
        return Stage.WATCH, None

    # تغيّر الطابع: شمعات فردية في اتجاه الانعكاس خلال ٣–٥ شمعات
    nxt = range(tbar + 1, min(last, tbar + cfg.confirm_bars) + 1)
    if not nxt:
        return Stage.TBAR, tbar

    bull = direction == "bullish"
    directional = sum(1 for i in nxt if ((c[i] > o[i]) if bull else (c[i] < o[i])))
    progressed = (
        max(c[i] for i in nxt) > c[tbar] if bull
        else min(c[i] for i in nxt) < c[tbar]
    )
    # تجاوز قاطع للـ PRZ دون أي بناء = فرصة معيبة
    if bull:
        breached = min(l[i] for i in nxt) < far - 0.005 * abs(far) and directional == 0
    else:
        breached = max(h[i] for i in nxt) > far + 0.005 * abs(far) and directional == 0
    if breached:
        return Stage.FAILED, tbar
    if directional >= 2 and progressed:
        return Stage.SIGNAL, tbar
    return Stage.TBAR, tbar


def _project_watch(
    quad: list[Pivot], direction: Direction, cfg: Config, symbol: str,
    h: np.ndarray, l: np.ndarray, c: np.ndarray, last: int,
) -> list[HarmonicPattern]:
    """
    ٤ من ٥ نقاط مثبتة — ما يسميه الكتاب "أثر المغناطيس" (ص ٢٣٧):
    السعر يتسارع نحو نقطة الإتمام في الضلع الأخير. تُسقط الـ PRZ ويُراقب.
    """
    X, A, B, C = quad
    xa_leg = abs(A.price - X.price)
    ab_leg = abs(A.price - B.price)
    if xa_leg <= 0 or ab_leg <= 0:
        return []
    if C.index - X.index < cfg.min_pattern_bars * 0.6:   # [param] الضلع الأخير لم يأتِ بعد
        return []

    b_ratio = ab_leg / xa_leg
    if direction == "bullish" and not (B.price < C.price < A.price):
        return []
    if direction == "bearish" and not (A.price < C.price < B.price):
        return []

    out: list[HarmonicPattern] = []
    for spec in SPECS:
        if not _in(b_ratio, spec.b_min, spec.b_max):
            continue
        pat_range = max(X.price, A.price, B.price, C.price) - \
                    min(X.price, A.price, B.price, C.price)
        if pat_range <= 0:
            continue
        prz = _prz_levels(spec, X.price, A.price, B.price, C.price, direction,
                          window=cfg.max_prz_width * pat_range)
        lo, hi, levels = prz["lo"], prz["hi"], prz["levels"]
        width = (hi - lo) / pat_range
        # السعر لم يصل الـ PRZ بعد
        reached = l[-1] <= hi if direction == "bullish" else h[-1] >= lo
        sign = -1.0 if direction == "bullish" else 1.0

        out.append(HarmonicPattern(
            symbol=symbol, pattern=spec.name, pattern_ar=spec.name_ar,
            direction=direction, stage=Stage.WATCH,
            X=(X.index, X.price), A=(A.index, A.price), B=(B.index, B.price),
            C=(C.index, C.price), D=None,
            b_ratio=b_ratio, bc_ratio=None, xa_ratio=None,
            abcd_ratio=None, time_symmetry=None,
            prz_low=lo, prz_high=hi, prz_width_pct=width,
            confluence=prz["confluence"], prz_levels=levels,
            entry=None,
            stop=A.price + sign * spec.stop_xa * xa_leg,
            tp1=None, tp2=None,
            bars_span=C.index - X.index,
            note=("السعر داخل الـ PRZ — راقب شمعة الإتمام" if reached
                  else f"{spec.note} | الـ PRZ مُسقطة ولم تُختبر بعد"),
        ))
        break
    return out
