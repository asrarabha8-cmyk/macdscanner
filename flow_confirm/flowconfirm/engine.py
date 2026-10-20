"""المحرك: يجمع التدفق + الغاما + السعر في درجة من ١٠٠ وقرار، ويقترح خطة للسهم والعقد."""
import datetime as dt
import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import flow as F
from . import gex as G
from . import price as P
from .data import Snapshot

DECISION = {"enter": ("🟢", "دخول"), "wait": ("🟡", "انتظار"), "no": ("🔴", "ممنوع")}


@dataclass
class Component:
    name: str
    pts: float
    max: float
    lines: list = field(default_factory=list)


@dataclass
class Result:
    ticker: str
    mode: str
    bull: bool
    score: float
    decision: str
    spot: float
    components: list
    vetoes: list
    warnings: list
    stock: dict
    option: dict | None
    flow: dict
    gex: dict
    asof: dt.datetime


def money(x: float) -> str:
    x = float(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.1f}{suf}"
    return f"{x:.0f}"


def session_open(now: dt.datetime) -> bool:
    return now.weekday() < 5 and dt.time(9, 30) <= now.time() < dt.time(16, 0)


# ───────────────────────── المكوّنات ─────────────────────────
def _flow_component(fl, bull, cfg, vetoes):
    aligned = fl["net"] if bull else -fl["net"]
    pts = float(np.clip(aligned / 0.6, 0, 1) * 30)
    c = Component("💰 التدفق", pts, 30, [
        f"صافي التدفق {fl['net']:+.2f} | صاعد {money(fl['bull'])}$ مقابل هابط {money(fl['bear'])}$",
        f"بريميوم الكول {money(fl['call_prem'])}$ | البوت {money(fl['put_prem'])}$",
    ])
    if aligned < -0.3:
        vetoes.append("التدفق بقوة عكس اتجاه الصفقة")
    return c


def _unusual_component(fl, bull, cfg):
    al, op = (fl["u_bull"], fl["u_bear"]) if bull else (fl["u_bear"], fl["u_bull"])
    if al + op == 0:
        return Component("🐋 غير اعتيادي", 0, 20, ["لا توجد عقود غير اعتيادية"])
    share = al / (al + op)
    size = min(1.0, al / (3 * cfg.min_premium))
    pts = 20 * share * size
    lines = [f"باتجاهك {money(al)}$ | عكسك {money(op)}$"]
    for _, r in fl["unusual"].head(4).iterrows():
        icon = "🟢" if r["dir"] == "bull" else "🔴"
        side = {"A": "Ask", "B": "Bid"}.get(r["side"], r["side"])
        lines.append(f"{icon} {r['strike']:g}{r['type']} {r['expiry']:%d/%m} | "
                     f"{int(r['volume']):,} عقد (OI {int(r['oi']):,}) | {money(r['premium'])}$ | {side}")
    return Component("🐋 غير اعتيادي", pts, 20, lines)


def _gamma_component(gx, spot, bull, cfg, vetoes):
    lines, pts = [], 0.0
    flip = gx["flip"]
    if flip:
        good = spot > flip if bull else spot < flip
        pts += 10 if good else 0
        regime = "موجبة (تهدئة)" if spot > flip else "سالبة (تسارع)"
        lines.append(f"{'✅' if good else '❌'} نقطة الانقلاب {flip:.2f} — الغاما {regime}")
    else:
        pos = gx["total"] > 0
        pts += (6 if pos else 3) if bull else (3 if pos else 6)
        lines.append(f"لا توجد نقطة انقلاب قريبة — إجمالي GEX {money(gx['total'])}$")

    wall = gx["call_wall"] if bull else gx["put_wall"]
    name = "جدار الكول" if bull else "جدار البوت"
    if wall is None:
        pts += 10
        lines.append(f"✅ لا يوجد {name} يعيق الحركة")
    else:
        room = abs(wall - spot) / spot * 100
        if room >= 2 * cfg.min_wall_room_pct:
            pts += 10
        elif room >= cfg.min_wall_room_pct:
            pts += 5
        else:
            vetoes.append(f"{name} قريب جداً ({wall:g}، {room:.1f}٪)")
        signed = (wall - spot) / spot * 100
        lines.append(f"{'✅' if room >= cfg.min_wall_room_pct else '❌'} {name} {wall:g} ({signed:+.1f}٪)")
    other = gx["put_wall"] if bull else gx["call_wall"]
    if other:
        lines.append(f"الدعم/المقاومة المقابلة: {other:g}" + (f" | Max Pain {gx['max_pain']:g}" if gx["max_pain"] else ""))
    return Component("🧲 الغاما", pts, 20, lines)


# ───────────────────────── الخطط ─────────────────────────
def stock_plan(mode, bull, spot, info, gx):
    s = 1 if bull else -1
    if mode == "intraday":
        vw = info.get("vwap", spot)
        stop = vw * (1 - 0.001 * s)
        if (spot - stop) * s < 0.003 * spot:
            stop = spot * (1 - 0.003 * s)
        risk = abs(spot - stop)
        dflt = spot + 2 * risk * s
    else:
        a = info.get("atr", spot * 0.02)
        stop = spot - 1.5 * a * s
        risk = abs(spot - stop)
        dflt = spot + 3 * a * s
    wall = gx["call_wall"] if bull else gx["put_wall"]
    target = wall if wall and (wall - spot) * s > 0 else dflt
    return {"entry": spot, "stop": stop, "target": target,
            "rr": abs(target - spot) / risk if risk else 0.0}


def pick_contract(chain, spot, bull, cfg, mode, now, plan):
    if chain.empty:
        return None
    df = chain[(chain["type"] == ("C" if bull else "P")) & (chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    if mode == "intraday" and now.time() >= dt.time(14, 0) and (df["dte"] >= 1).any():
        df = df[df["dte"] >= 1]  # بعد الساعة ٢ لا نقترح 0DTE
    if df.empty:
        return None
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"]
    df["delta"] = G.bs_delta(df["type"].values, spot, df["strike"].values, df["T"].values, df["iv"].values)
    df["gamma"] = G.bs_gamma(spot, df["strike"].values, df["T"].values, df["iv"].values)
    lo, hi = cfg.target_delta
    ad = df["delta"].abs()
    for relax in (1, 2):
        cand = df[(ad >= lo) & (ad <= hi) & (df["spread"] <= cfg.max_spread_pct * relax)
                  & (df["oi"] >= cfg.min_oi / relax)]
        if not cand.empty:
            break
    else:
        return None
    if mode == "intraday":
        cand = cand[cand["dte"] == cand["dte"].min()]
    else:
        best = cand.loc[(cand["dte"] - 30).abs().idxmin(), "dte"]
        cand = cand[cand["dte"] == best]
    r = cand.loc[(cand["delta"].abs() - (lo + hi) / 2).abs().idxmin()]
    dS = plan["target"] - spot
    est = r["mid"] + r["delta"] * dS + 0.5 * r["gamma"] * dS * dS
    return {"name": f"{r['strike']:g}{r['type']} {r['expiry']:%d/%m/%Y}", "dte": int(r["dte"]),
            "mid": float(r["mid"]), "delta": float(r["delta"]), "spread": float(r["spread"]),
            "oi": int(r["oi"]), "volume": int(r["volume"]), "est_target": max(float(est), 0.0),
            "relaxed": relax > 1}


# ───────────────────────── التقييم ─────────────────────────
def evaluate(snap: Snapshot, mode: str, side: str = "auto") -> Result:
    cfg = C.MODES[mode]
    vetoes, warnings = [], []
    fl = F.summarize(snap.chain, cfg)
    gx = G.analyze(snap.chain, snap.spot)

    if side in ("call", "bull", "long"):
        bull = True
    elif side in ("put", "bear", "short"):
        bull = False
    else:
        bull = fl["net"] >= 0
    if abs(fl["net"]) < C.NEUTRAL_FLOW:
        warnings.append("التدفق محايد — لا توجد أفضلية واضحة للأموال")
    if not fl["ok"]:
        warnings.append("لم تتوفر سلسلة خيارات للفترة المختارة")

    comps = [_flow_component(fl, bull, cfg, vetoes),
             _unusual_component(fl, bull, cfg),
             _gamma_component(gx, snap.spot, bull, cfg, vetoes)]

    if mode == "intraday":
        checks, info = P.intraday(snap.intraday, snap.daily, snap.spot, bull, cfg)
        if checks and "VWAP" in checks[0].label and not checks[0].ok:
            vetoes.append("السعر في الجهة الخطأ من VWAP")
    else:
        checks, info = P.swing(snap.daily, snap.spot, bull, cfg,
                               snap.asof if session_open(snap.asof) else None)
    comps.append(Component("📊 السعر", sum(c.pts for c in checks), 30,
                           [f"{'✅' if c.ok else '❌'} {c.label}" for c in checks]))

    plan = stock_plan(mode, bull, snap.spot, info, gx)
    if plan["rr"] < cfg.min_rr:
        vetoes.append(f"العائد/المخاطرة ضعيف ({plan['rr']:.1f})")
    opt = pick_contract(snap.chain, snap.spot, bull, cfg, mode, snap.asof, plan)

    if snap.earnings:
        days = (snap.earnings - snap.asof.date()).days
        horizon = opt["dte"] if opt else (1 if mode == "intraday" else 30)
        if days <= max(horizon, 1):
            warnings.append(f"⚠️ إعلان نتائج {snap.earnings:%d/%m} قبل انتهاء العقد — خطر انهيار التذبذب")

    score = round(sum(c.pts for c in comps))
    neutral = abs(fl["net"]) < C.NEUTRAL_FLOW
    if vetoes or score < C.WAIT_SCORE:
        decision = "no"
    elif score >= C.ENTER_SCORE and not neutral:
        decision = "enter"
    else:
        decision = "wait"

    return Result(snap.ticker, mode, bull, score, decision, snap.spot, comps, vetoes,
                  warnings, plan, opt, fl, gx, snap.asof)


# ───────────────────────── الرسالة ─────────────────────────
def format_message(r: Result) -> str:
    e = html.escape
    icon, word = DECISION[r.decision]
    cfg = C.MODES[r.mode]
    lines = [
        f"{icon} <b>{r.ticker} — {word} ({r.score:.0f}/100)</b> | {cfg.label}",
        f"الاتجاه: {'صعود 📈' if r.bull else 'هبوط 📉'} | السعر {r.spot:.2f} | {r.asof:%d/%m %H:%M} NY",
        "━━━━━━━━━━",
    ]
    for c in r.components:
        lines.append(f"<b>{c.name} {c.pts:.0f}/{c.max:.0f}</b>")
        lines += [e(x) for x in c.lines]
    lines.append("━━━━━━━━━━")
    s = r.stock
    lines.append(f"<b>السهم:</b> دخول {s['entry']:.2f} | وقف {s['stop']:.2f} | هدف {s['target']:.2f} | R:R {s['rr']:.1f}")
    if r.option:
        o = r.option
        lines.append(f"<b>العقد:</b> {r.ticker} {o['name']} | دلتا {o['delta']:+.2f} | منتصف {o['mid']:.2f} | "
                     f"سبريد {o['spread'] * 100:.0f}٪ | OI {o['oi']:,}")
        lines.append(f"قيمة تقديرية عند الهدف ≈ {o['est_target']:.2f} (بدون تآكل الوقت)")
        if o["relaxed"]:
            lines.append("⚠️ لم يوجد عقد مثالي — خُففت شروط السيولة")
        lines.append(f"الخروج: إذا كسر السهم {s['stop']:.2f} أو وصل {s['target']:.2f}")
    else:
        lines.append("<b>العقد:</b> لا يوجد عقد بسيولة كافية ضمن الشروط")
    if r.vetoes:
        lines.append("━━━━━━━━━━")
        lines.append("<b>⛔ موانع:</b>")
        lines += [f"• {e(v)}" for v in r.vetoes]
    if r.warnings:
        lines.append("━━━━━━━━━━")
        lines += [e(w) for w in r.warnings]
    return "\n".join(lines)


def log_row(r: Result) -> dict:
    o = r.option or {}
    return {"time_ny": r.asof.strftime("%Y-%m-%d %H:%M"), "ticker": r.ticker, "mode": r.mode,
            "direction": "bull" if r.bull else "bear", "score": r.score, "decision": r.decision,
            "spot": round(r.spot, 2), "stop": round(r.stock["stop"], 2),
            "target": round(r.stock["target"], 2), "rr": round(r.stock["rr"], 2),
            "net_flow": round(r.flow["net"], 3),
            "flip": round(r.gex["flip"], 2) if r.gex["flip"] else None,
            "call_wall": r.gex["call_wall"], "put_wall": r.gex["put_wall"],
            "contract": o.get("name", ""), "contract_mid": round(o.get("mid", 0), 2),
            "vetoes": " | ".join(r.vetoes)}
