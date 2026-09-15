#!/usr/bin/env python3
"""
MCP server — يعرض الماسحات الثلاثة (MACD Cascade / EMA Cross / SMC Breakout)
كأدوات MCP. يعيد استخدام دوال التحليل النقية من كل سكربت مباشرة، بلا إرسال
تنبيهات تيليجرام ولا حفظ حالة أو سجلات — فقط يرجّع النتائج للمتصل.

التشغيل محلياً (stdio، مناسب لعميل MCP على نفس الجهاز):
    pip install -r requirements.txt
    python mcp_server.py

للنشر كخادم HTTP بعيد (مثل mcp.abyancapital.sa):
    MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 python mcp_server.py
"""

import os
import datetime as dt

import yfinance as yf
from mcp.server.mcpserver import MCPServer

import ema_cross_scanner as ec
import macd_cascade_scanner as mc
import smc_breakout_scanner as smc

mcp = MCPServer("macdscanner")


@mcp.tool()
def macd_cascade_scan(tickers: list[str] | None = None) -> dict:
    """يفحص رموزاً بمنطق MACD Cascade (4H/1H/15m) ويرجّع حالات الدخول والإذن.

    tickers: قائمة رموز مخصصة؛ إن تُركت فارغة تُستخدم قائمة الماسح الافتراضية
    (الأسهم أثناء الجلسة الأمريكية، أو العملات فقط خارجها) مع فلتر السيولة.
    """
    now = dt.datetime.now(dt.timezone.utc)
    names = tickers if tickers else (mc.TICKERS if mc.us_session(now) else sorted(mc.CRYPTO))

    h1 = mc.fetch(names, "180d", "1h")
    liquid_names = names if tickers else mc.liquid(h1)
    m15 = mc.fetch(liquid_names, "10d", "15m")

    entries, permits = [], []
    for t in liquid_names:
        if t not in m15 or t not in h1:
            continue
        try:
            res = mc.analyse(h1[t], m15[t], t in mc.CRYPTO)
        except Exception:
            continue
        if not res:
            continue
        if res["state"] == "entry":
            entries.append({"ticker": t, **res})
        elif res["state"] == "permit":
            permits.append(t)

    return {"entries": entries, "permits": permits}


@mcp.tool()
def ema_cross_scan(tickers: list[str] | None = None) -> dict:
    """يفحص تقاطع EMA8/EMA48 على آخر شمعة أسبوعية مغلقة.

    tickers: قائمة رموز مخصصة؛ إن تُركت فارغة تُستخدم قائمة الماسح الافتراضية
    مع فلتر السيولة.
    """
    names = tickers if tickers else ec.TICKERS

    raw = yf.download(names, period="5y", interval="1wk", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)

    ups, downs = [], []
    for t in names:
        try:
            df = raw[t].dropna() if len(names) > 1 else raw.dropna()
        except (KeyError, TypeError):
            continue
        if df.empty or (not tickers and not ec.liquid(t, df)):
            continue
        try:
            res = ec.check(df)
        except Exception:
            continue
        if not res:
            continue
        res = {**res, "date": res["date"].isoformat()}
        (ups if res["dir"] == "up" else downs).append({"ticker": t, **res})

    return {"up": ups, "down": downs}


@mcp.tool()
def smc_breakout_scan(tickers: list[str] | None = None) -> dict:
    """يفحص اختراقاً هيكلياً (BOS) على شموع 4 ساعات ويرجّع الإشارات المؤكدة.

    tickers: قائمة رموز مخصصة؛ إن تُركت فارغة تُستخدم مرشحو فلتر فينفيز الافتراضي.
    """
    names = tickers if tickers else smc.finviz_screen()

    signals, watch, skipped = [], [], []
    for t in names:
        df = smc.fetch_4h(t)
        if df is None or len(df) < 40:
            skipped.append(t)
            continue

        spike = smc.violent_candle(df)
        if spike > smc.MAX_CANDLE_GAIN:
            skipped.append(t)
            continue

        try:
            sig = smc.find_bos(df)
        except Exception:
            continue
        if not sig:
            watch.append(t)
            continue

        targets = smc.abc_targets(df, sig)
        if not targets:
            continue

        risk = sig["entry"] - sig["stop"]
        if risk <= 0:
            continue
        rr = (targets[0] - sig["entry"]) / risk
        risk_pct = risk / sig["entry"] * 100
        if rr < smc.MIN_RR:
            skipped.append(t)
            continue

        signals.append({
            "ticker": t,
            "entry": round(sig["entry"], 4),
            "stop": round(sig["stop"], 4),
            "targets": targets,
            "rr": round(rr, 2),
            "risk_pct": round(risk_pct, 1),
            "bar_time": sig["bar_time"].isoformat(),
        })

    return {"signals": signals, "watch": watch, "skipped": skipped}


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=transport,
            host=os.environ.get("MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_PORT", "8000")),
        )
