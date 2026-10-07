#!/usr/bin/env python3
"""
اختيار عقد أوبشن "سوينق" لإشارات الماسحات — مشترك بين كل الماسحات.

الفكرة: الصفقة تمتد أيام إلى أسابيع، فنختار عقداً يتحمّل الوقت:
  • انتهاء بعد 21–60 يوماً (التآكل الزمني بطيء في هذا المدى)
  • دلتا 0.55–0.75 (داخل السعر قليلاً: يتحرك مع السهم وقيمته الزمنية أقل)
  • سبريد ضيق نسبياً + حد أدنى للعقود المفتوحة (سيولة حقيقية)
  • التكلفة ضمن الميزانية — وإن كان العقد غالياً: سبريد شرائي بيعه عند الهدف

ولكل عقد يحسب تقديرياً: قيمته لو وصل السهم الهدف أو ضرب الوقف خلال ~10 أيام،
بنفس التذبذب الضمني الحالي (تقدير، لا ضمان).

الاستخدام داخل أي ماسح:
    from swing_options import option_lines
    lines += option_lines("NVDA", "CALL", price, stop=stop, target=target)
"""
from __future__ import annotations

import datetime as dt
import math
import os

# ------------------------------------------------------------------
# الإعدادات — تتغير من env في ملف yml بدون تعديل الكود
# ------------------------------------------------------------------
def _env(name, default, cast=float):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default

BUDGET        = _env("SWING_BUDGET", 200.0)   # أقصى تكلفة للعقد/السبريد بالدولار
MIN_DTE       = _env("SWING_MIN_DTE", 21, int)
MAX_DTE       = _env("SWING_MAX_DTE", 60, int)
DELTA_MIN     = _env("SWING_DELTA_MIN", 0.55)
DELTA_MAX     = _env("SWING_DELTA_MAX", 0.75)
DELTA_AIM     = 0.65
MAX_SPREAD_PCT = _env("SWING_MAX_SPREAD_PCT", 0.06)   # سبريد ≤ 6% من منتصف السعر
MIN_SPREAD_ABS = 0.15                                  # أو 0.15 للعقود الرخيصة
MIN_OI        = _env("SWING_MIN_OI", 100, int)
HOLD_DAYS     = _env("SWING_HOLD_DAYS", 10, int)      # مدة الصفقة المتوقعة للتقدير
MAX_LOOKUPS   = _env("SWING_MAX_LOOKUPS", 8, int)     # حد البحث لكل تشغيل (سرعة)
ALLOW_SPREADS = _env("SWING_ALLOW_SPREADS", "0", str) == "1"   # منصة يحيى لا تدعم العقود المركبة
FALLBACK_DELTA_MIN = 0.40     # إذا العقد الأعمق غالٍ: عقد عند السعر (أرخص، يحتاج حركة أكبر)
RISK_FREE     = 0.045

_lookups = 0


# ------------------------------------------------------------------
# Black-Scholes
# ------------------------------------------------------------------
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, iv, kind, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == "CALL" else (K - S))
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CALL":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S, K, T, iv, kind, r=RISK_FREE):
    if T <= 0 or iv <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    return _ncdf(d1) if kind == "CALL" else _ncdf(d1) - 1


def implied_vol(price, S, K, T, kind):
    intrinsic = max(0.0, (S - K) if kind == "CALL" else (K - S))
    if price <= intrinsic + 1e-4 or T <= 0:
        return float("nan")
    lo, hi = 0.01, 5.0
    if bs_price(S, K, T, hi, kind) < price:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------
# مصدر البيانات (yfinance افتراضياً — قابل للاستبدال في الاختبار)
# ------------------------------------------------------------------
def _yf_expiries(t):
    import yfinance as yf
    return list(yf.Ticker(t).options)


def _yf_chain(t, exp, kind):
    import yfinance as yf
    oc = yf.Ticker(t).option_chain(exp)
    return (oc.calls if kind == "CALL" else oc.puts)


GET_EXPIRIES = _yf_expiries
GET_CHAIN = _yf_chain


def _rows(t, exp, kind, S, today):
    """سلسلة نظيفة: أسعار حقيقية + تذبذب محسوب من السعر + دلتا، وتحذف الأسعار الشاذة."""
    df = GET_CHAIN(t, exp, kind)
    d = dt.date.fromisoformat(exp)
    T = max((d - today).days, 1) / 365
    rows = []
    for _, r in df.iterrows():
        bid, ask = float(r.get("bid", 0) or 0), float(r.get("ask", 0) or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        K = float(r["strike"])
        mid = (bid + ask) / 2
        iv = implied_vol(mid, S, K, T, kind)
        if iv != iv:
            continue
        oi = r.get("openInterest", None)
        rows.append(dict(strike=K, bid=bid, ask=ask, mid=mid, spread=ask - bid, iv=iv,
                         delta=bs_delta(S, K, T, iv, kind),
                         oi=float(oi) if oi == oi and oi is not None else None, T=T))
    if not rows:
        return []
    near = sorted(r["iv"] for r in rows if abs(r["strike"] / S - 1) < 0.10)
    if near:
        med = near[len(near) // 2]
        rows = [r for r in rows if 0.4 * med <= r["iv"] <= 2.5 * med]
    return rows


def _liquid(r):
    ok_spread = r["spread"] <= max(MIN_SPREAD_ABS, MAX_SPREAD_PCT * r["mid"])
    ok_oi = r["oi"] is None or r["oi"] >= MIN_OI
    return ok_spread and ok_oi


# ------------------------------------------------------------------
# الاختيار
# ------------------------------------------------------------------
def suggest(t, kind, S, stop=None, target=None, today=None):
    """يرجع dict بالعقد المقترح أو None (مع سبب في المفتاح reason)."""
    today = today or dt.date.today()
    exps = []
    for e in GET_EXPIRIES(t):
        dte = (dt.date.fromisoformat(e) - today).days
        if MIN_DTE <= dte <= MAX_DTE:
            exps.append(e)
    if not exps:
        return {"reason": f"ما فيه انتهاء بين {MIN_DTE} و{MAX_DTE} يوم"}

    best_single, best_spread, why = None, None, "ما فيه عقد سيولته كافية"
    for e in exps[:2]:
        rows = _rows(t, e, kind, S, today)
        if not rows:
            continue
        liquid = [r for r in rows if _liquid(r)]
        if not liquid:
            continue
        # ١) عقد منفرد
        singles = [r for r in liquid if DELTA_MIN <= abs(r["delta"]) <= DELTA_MAX]
        cheap = [r for r in singles if r["ask"] * 100 <= BUDGET]
        if cheap:
            r = min(cheap, key=lambda r: (abs(abs(r["delta"]) - DELTA_AIM), r["spread"] / r["mid"]))
            best_single = best_single or dict(type="single", exp=e, leg=r)
            break
        # ١ب) بديل أرخص: عقد عند السعر (دلتا 0.40–0.55)
        atm = [r for r in liquid if FALLBACK_DELTA_MIN <= abs(r["delta"]) < DELTA_MIN and r["ask"] * 100 <= BUDGET]
        if atm:
            r = max(atm, key=lambda r: abs(r["delta"]))
            best_single = dict(type="single", exp=e, leg=r, atm=True)
            break
        pool = singles or [r for r in liquid if FALLBACK_DELTA_MIN <= abs(r["delta"]) <= DELTA_MAX]
        if pool:
            why = (f"أرخص عقد مناسب ${min(r['ask'] for r in pool) * 100:.0f} — فوق الميزانية ${BUDGET:.0f}")
        if not ALLOW_SPREADS:
            continue
        # ٢) سبريد شرائي: شراء دلتا ~0.6 وبيع عند الهدف (أو ~10% أبعد)
        aim = target if target else (S * 1.10 if kind == "CALL" else S * 0.90)
        longs = [r for r in liquid if 0.40 <= abs(r["delta"]) <= DELTA_MAX]
        for L in sorted(longs, key=lambda r: abs(abs(r["delta"]) - 0.60)):
            further = [r for r in liquid if (r["strike"] > L["strike"] if kind == "CALL" else r["strike"] < L["strike"])
                       and L["bid"] >= r["ask"]]
            # من الأبعد (عند الهدف) للأقرب: أعرض سبريد تكلفته داخل الميزانية
            further = [r for r in further if (r["strike"] <= aim if kind == "CALL" else r["strike"] >= aim)] or further[:1]
            further.sort(key=lambda r: -abs(r["strike"] - L["strike"]))
            for Sh in further:
                debit = L["ask"] - Sh["bid"]
                width = abs(Sh["strike"] - L["strike"])
                if debit <= 0 or debit >= width or debit * 100 > BUDGET:
                    continue
                if (width - debit) / debit < 1.0:     # أقصى ربح لازم ≥ التكلفة
                    continue
                best_spread = dict(type="spread", exp=e, leg=L, short=Sh, debit=debit, width=width)
                break
            if best_spread:
                break
        if best_spread:
            break

    pick = best_single or best_spread
    if not pick:
        return {"reason": why}
    return _finish(t, kind, S, stop, target, pick, today)


def _value(pick, kind, price, T):
    L = pick["leg"]
    v = bs_price(price, L["strike"], T, L["iv"], kind)
    if pick["type"] == "spread":
        Sh = pick["short"]
        v -= bs_price(price, Sh["strike"], T, Sh["iv"], kind)
    return max(v, 0.0)


def _finish(t, kind, S, stop, target, pick, today):
    L = pick["leg"]
    d = dt.date.fromisoformat(pick["exp"])
    dte = (d - today).days
    T_after = max(dte - HOLD_DAYS, 1) / 365
    cost = (pick["debit"] if pick["type"] == "spread" else L["ask"])
    out = dict(pick, ticker=t, kind=kind, dte=dte, cost=cost * 100,
               delta=L["delta"] - (pick["short"]["delta"] if pick["type"] == "spread" else 0))
    if pick["type"] == "single":
        out["breakeven"] = L["strike"] + cost if kind == "CALL" else L["strike"] - cost
    else:
        out["breakeven"] = L["strike"] + cost if kind == "CALL" else L["strike"] - cost
        out["max_profit"] = (pick["width"] - cost) * 100
    if target:
        out["at_target"] = _value(pick, kind, target, T_after) * 100 - out["cost"]
    if stop:
        out["at_stop"] = _value(pick, kind, stop, T_after) * 100 - out["cost"]
    return out


# ------------------------------------------------------------------
# النص للتيليجرام
# ------------------------------------------------------------------
def _name(o):
    d = dt.date.fromisoformat(o["exp"])
    k = "C" if o["kind"] == "CALL" else "P"
    if o["type"] == "spread":
        return f"{o['ticker']} {o['leg']['strike']:g}/{o['short']['strike']:g}{k} {d:%d/%m}"
    return f"{o['ticker']} {o['leg']['strike']:g}{k} {d:%d/%m}"


def render(o) -> list[str]:
    if not o or "reason" in o:
        return [f"   ⚪ أوبشن: {o.get('reason', 'غير متاح') if o else 'غير متاح'}"]
    head = "🎯 سبريد سوينق" if o["type"] == "spread" else ("🎯 عقد سوينق (عند السعر — يحتاج حركة أقوى)" if o.get("atm") else "🎯 عقد سوينق")
    lines = [f"   {head}: <b>{_name(o)}</b> ({o['dte']} يوم)"]
    if o["type"] == "spread":
        lines.append(f"   شراء {o['leg']['strike']:g} + بيع {o['short']['strike']:g} • التكلفة ${o['cost']:.0f} • أقصى ربح ${o['max_profit']:.0f}")
    else:
        L = o["leg"]
        lines.append(f"   Bid {L['bid']:.2f} / Ask {L['ask']:.2f} • دلتا {o['delta']:+.2f} • التكلفة ${o['cost']:.0f}")
    est = []
    if "at_target" in o:
        est.append(f"عند الهدف ≈ {o['at_target']:+.0f}$ ({o['at_target'] / o['cost'] * 100:+.0f}%)")
    if "at_stop" in o:
        est.append(f"عند الوقف ≈ {o['at_stop']:+.0f}$ ({o['at_stop'] / o['cost'] * 100:+.0f}%)")
    if est:
        lines.append(f"   تقدير خلال ~{HOLD_DAYS} أيام: " + " • ".join(est))
    return lines


def option_lines(t, kind, price, stop=None, target=None) -> list[str]:
    """واجهة آمنة للماسحات: أي خطأ = سطر قصير بدل ما يوقف الماسح."""
    global _lookups
    if _lookups >= MAX_LOOKUPS:
        return []
    _lookups += 1
    try:
        return render(suggest(t, kind, float(price), stop=stop, target=target))
    except Exception as e:  # noqa: BLE001
        print(f"options {t}: {e}")
        return ["   ⚪ أوبشن: تعذّر جلب السلسلة"]
