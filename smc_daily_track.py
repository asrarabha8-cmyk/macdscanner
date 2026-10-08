#!/usr/bin/env python3
"""
متابعة ورقية لإشارات SMC اليومي — تُشغَّل يومياً بعد الإغلاق (قبل الماسح).

  pending → open : الدخول على افتتاح أول يوم بعد الإشارة (يُلغى إذا فتح تحت الوقف أو فوق الهدف)
  open → closed  : الوقف أو الهدف أو 40 يوم تداول — نفس قواعد الاختبار التاريخي حرفياً
  العقد          : يُسجَّل سعره الحالي يومياً (Bid) وعند الإغلاق يُحسب ربحه/خسارته

يرسل تيليجرام عند كل دخول وكل إغلاق، مع حصيلة الاختبار حتى الآن.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from smc_daily_scanner import SIGNALS_CSV, MAX_DAYS, notify

COST = 0.001


def option_bid(t, exp, strike):
    try:
        ch = yf.Ticker(t).option_chain(exp).calls
        row = ch[np.isclose(ch["strike"], float(strike))]
        if row.empty:
            return np.nan
        b = float(row["bid"].iloc[0])
        return b if b > 0 else float(row["lastPrice"].iloc[0])
    except Exception as e:  # noqa: BLE001
        print(f"option {t}: {e}", file=sys.stderr)
        return np.nan


def main():
    if not os.path.exists(SIGNALS_CSV):
        print("لا إشارات بعد.")
        return
    s = pd.read_csv(SIGNALS_CSV, dtype={"entry_date": str, "exit_date": str, "reason": str})
    for col in ("entry_date", "exit_date", "reason", "opt_exp"):
        s[col] = s[col].fillna("").astype(str)
    active = s[s.status.isin(["pending", "open"])]
    if active.empty:
        print("لا صفقات نشطة.")
        return
    now_ny = pd.Timestamp.now(tz="America/New_York")
    msgs = []
    for idx, r in active.iterrows():
        t = r.ticker
        df = yf.download(t, start=str(pd.Timestamp(r.signal_date) - pd.Timedelta(days=3)),
                         interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) and df.index[-1].date() == now_ny.date() and now_ny.hour < 16:
            df = df.iloc[:-1]                         # شمعة اليوم غير مكتملة
        after = df[df.index > pd.Timestamp(r.signal_date)]
        if after.empty:
            continue
        stop, target = float(r.stop), float(r.target)

        if r.status == "pending":
            o = float(after["Open"].iloc[0])
            d0 = str(after.index[0].date())
            if o <= stop or o >= target:
                s.loc[idx, ["status", "reason", "entry_date"]] = ["cancelled", "فتح تحت الوقف" if o <= stop else "فتح فوق الهدف", d0]
                msgs.append(f"⚪ {t}: أُلغيت — {s.loc[idx, 'reason']} ({o:.2f})")
                continue
            s.loc[idx, ["status", "entry_date", "entry"]] = ["open", d0, o]
            r = s.loc[idx]
            msgs.append(f"🔵 <b>{t}</b> دخول ورقي على الافتتاح {o:.2f} • الوقف {stop:.2f} • الهدف {target:.2f}")

        entry = float(r.entry)
        bars = df[df.index >= pd.Timestamp(r.entry_date)]
        exit_px = why = exit_d = None
        for n, (d, b) in enumerate(bars.iterrows()):
            if b.Low <= stop:
                exit_px = b.Open if (n > 0 and b.Open < stop) else stop
                why, exit_d = "stop", d
                break
            if b.High >= target:
                exit_px = b.Open if (n > 0 and b.Open > target) else target
                why, exit_d = "target", d
                break
            if n + 1 >= MAX_DAYS:
                exit_px, why, exit_d = b.Close, "time", d
                break

        bid = option_bid(t, r.opt_exp, r.opt_strike) if r.opt_exp else np.nan
        s.loc[idx, "opt_last"] = bid
        if why:
            risk = entry - stop
            R = (exit_px * (1 - COST) - entry * (1 + COST)) / risk
            pnl = (bid - float(r.opt_entry)) * 100 if bid == bid and r.opt_entry == r.opt_entry else np.nan
            s.loc[idx, ["status", "exit_date", "exit", "reason", "r", "days", "opt_exit", "opt_pnl"]] = [
                "closed", str(exit_d.date()), round(float(exit_px), 4), why, round(R, 3),
                int(len(bars.loc[:exit_d])), bid, pnl]
            icon = {"target": "✅", "stop": "❌", "time": "⏰"}[why]
            word = {"target": "الهدف", "stop": "الوقف", "time": "انتهاء المدة"}[why]
            opt = f" • العقد {pnl:+.0f}$" if pnl == pnl else ""
            msgs.append(f"{icon} <b>{t}</b> أُغلقت عند {word} {exit_px:.2f} → <b>{R:+.2f}R</b>{opt}")

    s.to_csv(SIGNALS_CSV, index=False)
    closed = s[s.status == "closed"]
    if msgs:
        tally = ""
        if len(closed):
            pnl = closed.opt_pnl.dropna()
            tally = (f"\n\n📒 الحصيلة: {len(closed)} صفقة مغلقة • ربح {(closed.r > 0).mean() * 100:.0f}% • "
                     f"متوسط {closed.r.mean():+.2f}R"
                     + (f" • العقود {pnl.sum():+.0f}$" if len(pnl) else "")
                     + f" • مفتوحة {int((s.status == 'open').sum())}")
        notify("🧾 <b>متابعة SMC اليومي</b>\n" + "\n".join(msgs) + tally)
    print("\n".join(msgs) or "لا تغييرات.")


if __name__ == "__main__":
    main()
