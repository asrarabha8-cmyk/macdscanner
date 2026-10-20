"""إعدادات نظام التأكيد — عدّل العتبات من هنا."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeCfg:
    label: str
    dte_min: int              # أقل أيام للانتهاء
    dte_max: int              # أعلى أيام للانتهاء
    max_expiries: int         # كم تاريخ انتهاء نقرأ
    min_premium: float        # أقل بريميوم ($) ليعتبر العقد "غير اعتيادي"
    min_vol_oi: float         # أقل نسبة حجم / OI للعقد غير الاعتيادي
    min_contracts: int        # أقل عدد عقود للعقد غير الاعتيادي
    min_wall_room_pct: float  # أقل مسافة (٪) بين السعر والجدار المعاكس
    target_delta: tuple       # نطاق الدلتا للعقد المقترح
    max_spread_pct: float     # أعلى سبريد مسموح للعقد المقترح
    min_oi: int               # أقل OI للعقد المقترح
    rvol_good: float          # حجم نسبي يعتبر قوياً
    min_rr: float             # أقل عائد/مخاطرة مقبول


MODES = {
    "intraday": ModeCfg("يومي / 0DTE", 0, 7, 2, 50_000, 1.0, 100,
                        0.4, (0.40, 0.60), 0.10, 50, 1.3, 1.2),
    "swing": ModeCfg("سوينغ", 7, 60, 4, 100_000, 0.5, 100,
                     2.0, (0.45, 0.65), 0.08, 200, 1.5, 1.5),
}

# حدود القرار
ENTER_SCORE = 70   # ≥ 70 بدون موانع  → دخول
WAIT_SCORE = 50    # 50–69            → انتظار ، أقل → ممنوع
NEUTRAL_FLOW = 0.15  # صافي تدفق أقل من هذا = محايد (لا دخول)

# تصنيف جانب التنفيذ: موقع آخر سعر بين Bid و Ask
ASK_SIDE = 0.65
BID_SIDE = 0.35

RISK_FREE = 0.04
ALERT_COOLDOWN_HOURS = {"intraday": 3, "swing": 24}

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE = os.getenv("POLYGON_BASE", "https://api.polygon.io")
TG_TOKEN = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID", "")
