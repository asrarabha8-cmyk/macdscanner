#!/usr/bin/env python3
"""
فلتر العقود الدائمة (PerpEquities) — مشترك بين الماسحات.

يجلب كون العقود الدائمة على الأسهم بطلب واحد من الـ API المجاني (بدون مفتاح):
    https://perpequities.com/api/v1/assets?assetClass=equity&live=1&limit=2000
ولكل سهم يحسب:
  • وسيط التمويل السنوي بين المنصات السائلة (APR)
  • الفائدة المفتوحة الحالية + تغيّرها مقارنة بلقطات الأيام السابقة
  • الحجم المفلتر بالجودة وعدد المنصات الحية

الـ API يعطي الفائدة المفتوحة الحالية فقط — لذلك كل تشغيل يحفظ لقطة يومية في
logs/perp/oi_YYYY-MM-DD.csv، ومنها يُحسب الاتجاه (يكتمل بعد ~5 أيام تشغيل).

الاستخدام داخل أي ماسح:
    from perp_filter import PerpData
    perp = PerpData.load()          # طلب واحد + حفظ لقطة اليوم
    info = perp.get("NVDA")         # None إذا ما له عقود سائلة
    lines += perp.render("NVDA")

تشغيل مستقل (لحفظ اللقطة فقط):  python perp_filter.py
Data: PerpEquities (perpequities.com)
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import statistics
import sys

import pandas as pd
import requests

API_URL = "https://perpequities.com/api/v1/assets"
SNAP_DIR = "logs/perp"

# ------------------------------------------------------------------
# العتبات — تتغير من env في ملف yml بدون تعديل الكود
# ------------------------------------------------------------------
def _env(name, default):
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


MIN_QVOL        = _env("PERP_MIN_QVOL", 1_000_000)   # حجم مفلتر بالجودة ≥ مليون $
MIN_VENUES      = _env("PERP_MIN_VENUES", 5)         # منصات حية
MIN_VENUE_VOL   = _env("PERP_MIN_VENUE_VOL", 50_000) # منصة تدخل حساب التمويل إذا حجمها ≥ 50 ألف $
SHORT_CROWDED   = _env("PERP_SHORT_APR", -0.10)      # وسيط تمويل ≤ −10% سنوياً = شورتات مزدحمة
LONG_CROWDED    = _env("PERP_LONG_APR", 0.50)        # وسيط تمويل ≥ +50% سنوياً = شراء مزدحم
OI_UP           = _env("PERP_OI_UP", 0.10)           # فائدة مفتوحة +10% = مراكز جديدة
OI_DOWN         = _env("PERP_OI_DOWN", -0.10)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


class PerpData:
    def __init__(self, rows: dict[str, dict], ok: bool, note: str = ""):
        self.rows = rows      # ticker → info
        self.ok = ok
        self.note = note

    # --------------------------------------------------------------
    @classmethod
    def load(cls, save=True, timeout=40) -> "PerpData":
        try:
            r = requests.get(API_URL, params={"assetClass": "equity", "live": 1, "limit": 2000},
                             headers={"User-Agent": "macdscanner/1.0 (+https://perpequities.com)"},
                             timeout=timeout)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:  # noqa: BLE001 — الفلتر اختياري: الماسح يكمل بدونه
            print(f"PerpEquities unavailable: {e}", file=sys.stderr)
            return cls({}, False, "بيانات العقود غير متاحة اليوم")
        rows = cls._parse(payload.get("data", []))
        stale = bool(payload.get("stale"))
        obj = cls(rows, True, "⚠️ بيانات متأخرة" if stale else "")
        if save:
            obj._save_snapshot()
        obj._attach_history()
        return obj

    @staticmethod
    def _parse(data: list[dict]) -> dict[str, dict]:
        out = {}
        for a in data:
            t = (a.get("ticker") or "").upper().replace(".", "-")
            if not t or a.get("country") not in (None, "US"):
                continue   # نركّز على الأسهم الأمريكية (SKHYNIX الكوري ≠ SKHY مثلاً)
            venues = [v for v in a.get("venues") or [] if v.get("live") and v.get("status") == "trading"]
            liquid = [v for v in venues if (v.get("volume24hUsd") or 0) >= MIN_VENUE_VOL]
            qvol = a.get("qualityFilteredVolume24hUsd")
            if qvol is None:
                qvol = a.get("volume24hUsd")
            out[t] = dict(
                ticker=t,
                price=a.get("price"),
                oi=a.get("openInterestUsd"),
                qvol=qvol or 0,
                venues=len({v.get("venue") for v in venues}),
                funding=_median([v.get("fundingApr") for v in liquid]),
                funding_venues=len(liquid),
                url=a.get("url"),
            )
        return out

    # --------------------------------------------------------------
    def _snap_path(self, day=None):
        day = day or dt.datetime.now(dt.timezone.utc).date()
        return os.path.join(SNAP_DIR, f"oi_{day}.csv")

    def _save_snapshot(self):
        if not self.rows:
            return
        os.makedirs(SNAP_DIR, exist_ok=True)
        pd.DataFrame(self.rows.values())[["ticker", "price", "oi", "qvol", "venues", "funding"]] \
            .to_csv(self._snap_path(), index=False)

    def _attach_history(self):
        """يقارن الفائدة المفتوحة بلقطة سابقة (~1 يوم و~5 أيام) — أي لقطات موجودة."""
        today = self._snap_path()
        files = sorted(f for f in glob.glob(os.path.join(SNAP_DIR, "oi_*.csv")) if f != today)
        for key, back in (("oi_1d", 1), ("oi_5d", 5)):
            if len(files) < back:
                continue
            try:
                old = pd.read_csv(files[-back]).set_index("ticker")["oi"].to_dict()
            except Exception:  # noqa: BLE001
                continue
            for t, info in self.rows.items():
                o, n = old.get(t), info["oi"]
                if o and n and o > 0:
                    info[key] = n / o - 1

    # --------------------------------------------------------------
    def get(self, ticker: str) -> dict | None:
        """معلومات السهم إذا عقوده سائلة كفاية، وإلا None."""
        info = self.rows.get(ticker.upper().replace(".", "-"))
        if not info or info["qvol"] < MIN_QVOL or info["venues"] < MIN_VENUES:
            return None
        return info | {"tags": self._tags(info), "score": self._score(info)}

    @staticmethod
    def _oi_change(info):
        return info.get("oi_5d", info.get("oi_1d"))

    def _tags(self, info):
        tags = []
        f = info["funding"]
        if f is not None and f <= SHORT_CROWDED:
            tags.append("short_crowded")
        if f is not None and f >= LONG_CROWDED:
            tags.append("long_crowded")
        ch = self._oi_change(info)
        if ch is not None and ch >= OI_UP:
            tags.append("oi_up")
        if ch is not None and ch <= OI_DOWN:
            tags.append("oi_down")
        return tags

    def _score(self, info):
        """لماسح القاع: +2 شورتات مزدحمة، +1 فائدة مفتوحة صاعدة، −2 شراء مزدحم."""
        t = self._tags(info)
        return 2 * ("short_crowded" in t) + ("oi_up" in t) - 2 * ("long_crowded" in t)

    # --------------------------------------------------------------
    def render(self, ticker: str) -> list[str]:
        if not self.ok:
            return [f"   🔸 العقود الدائمة: {self.note}"]
        info = self.get(ticker)
        if not info:
            return ["   🔸 العقود الدائمة: لا توجد عقود سائلة"]
        f = info["funding"]
        f_txt = "—" if f is None else f"{f * 100:+.0f}% سنوياً"
        ch = self._oi_change(info)
        ch_txt = "تاريخ غير كافٍ بعد" if ch is None else f"{ch * 100:+.0f}%"
        tags = info["tags"]
        if "short_crowded" in tags and "oi_up" in tags:
            v = "🔥 شورتات مزدحمة + مراكز جديدة — وقود ارتداد محتمل"
        elif "short_crowded" in tags:
            v = "🟢 شورتات مزدحمة — دعم لفكرة الارتداد"
        elif "long_crowded" in tags:
            v = "⚠️ شراء مزدحم — الارتداد قد يكون مستهلكاً"
        elif "oi_up" in tags:
            v = "🟡 مراكز جديدة تُفتح"
        else:
            v = "⚪ محايد"
        lines = [
            f"   🔸 العقود: تمويل {f_txt} · فائدة مفتوحة ${info['oi'] / 1e6:,.1f}M ({ch_txt}) · "
            f"حجم ${info['qvol'] / 1e6:,.1f}M · {info['venues']} منصة",
            f"   {v}",
        ]
        if self.note:
            lines.append(f"   {self.note}")
        return lines


if __name__ == "__main__":
    p = PerpData.load()
    print(f"assets: {len(p.rows)} • liquid: {sum(1 for t in p.rows if p.get(t))} • ok={p.ok}")
    for t in sorted(p.rows, key=lambda t: p.rows[t]["funding"] if p.rows[t]["funding"] is not None else 0)[:5]:
        if p.get(t):
            print(t, *p.render(t), sep="\n")
