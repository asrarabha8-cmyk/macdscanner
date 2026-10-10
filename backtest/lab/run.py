#!/usr/bin/env python3
"""
مختبر الاستراتيجيات — اختبار أي استراتيجية مسجلة على S&P 500.

أمثلة:
    python run.py --strategy ankush              # الإعدادات الافتراضية
    python run.py --strategy ankush --sweep      # + تجربة كل إعدادات الشبكة
    python run.py --strategy all --sweep

النتيجة: results/<strategy>.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import get_data                                         # noqa: E402
from engine import baseline, event_study, forward_returns, grid, portfolio_stats, summarize  # noqa: E402
from strategies import STRATEGIES                                # noqa: E402

PCT_KEYS = ("win", "avg_r", "excess", "pf_win", "pf_avg")


def fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "_لا توجد نتائج_"
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            if c.startswith(PCT_KEYS):
                df[c] = (df[c] * 100).round(2)
            else:
                df[c] = df[c].round(2)
    return df.to_markdown(index=False)


def run_one(key: str, d: dict, do_sweep: bool, years: float) -> str:
    spec = STRATEGIES[key]
    holds, H = spec["holds"], spec["main_hold"]
    fwd = forward_returns(d, holds)
    base = baseline(d, holds, fwd)
    lines = [f"# {spec['title']}", "",
             f"الأسهم: {d['Close'].shape[1]} · الفترة: {d['Close'].index[0].date()} → {d['Close'].index[-1].date()}",
             "", "النسب بالمئة. excess = عائد الصفقة ناقص متوسط عائد كل الأسهم في نفس يوم الدخول.",
             f"الضابط (شراء عشوائي لأي سهم في أي يوم): " +
             " · ".join(f"{h} يوم: {base[f'avg_r{h}']*100:+.2f}% (ربح {base[f'win{h}']*100:.0f}%)" for h in holds),
             ""]

    # الإعدادات الافتراضية
    t0 = time.time()
    sig = spec["fn"](d, **spec["defaults"])
    rows = []
    for side_name, side in (("long", 1), ("short", -1)):
        if side_name not in sig or sig[side_name] is None:
            continue
        tr = event_study(d, sig[side_name], holds, side, fwd)
        row = {"side": "Long" if side == 1 else "Short", **summarize(tr, holds, H, years)}
        if side == 1:
            row.update(portfolio_stats(d, sig["long"], 1, spec["max_hold"], sig.get("sl"), sig.get("tp")))
        else:
            row.update(portfolio_stats(d, sig["short"], -1, spec["max_hold"]))
        rows.append(row)
    lines += ["## الإعدادات الافتراضية", "", f"`{json.dumps(spec['defaults'])}`", "", fmt_table(rows), "",
              "pf_* = صفقات فعلية (vectorbt): دخول افتتاح اليوم التالي، وقف/هدف، "
              f"خروج بالوقت بعد {spec['max_hold']} يوماً، رسوم 0.1% لكل جهة.", ""]
    print(f"{key}: defaults done in {time.time() - t0:.0f}s", flush=True)

    # الشبكة
    if do_sweep:
        combos = grid(spec["grid"])
        srows = []
        for p in combos:
            s = spec["fn"](d, **p)
            for side_name, side in (("long", 1), ("short", -1)):
                if side_name not in s or s[side_name] is None:
                    continue
                tr = event_study(d, s[side_name], holds, side, fwd)
                sm = summarize(tr, holds, H, years)
                srows.append({**p, "side": side_name, **sm})
        sdf = pd.DataFrame(srows)
        if not sdf.empty and "t_stat" in sdf:
            sdf = sdf.sort_values("t_stat", ascending=False)
        keep = [c for c in sdf.columns if c in list(spec["grid"]) + ["side", "trades", "per_year", f"win{H}",
                                                                      f"excess{H}", "t_stat", "excess_1st_half",
                                                                      "excess_2nd_half", "good_years"]]
        lines += [f"## تجربة {len(combos)} إعداداً (مرتبة حسب قوة التفوق t)", "",
                  f"تحذير: مع {len(srows)} تجربة، ظهور t≈2 في واحدة أو اثنتين متوقع بالصدفة. "
                  "الإعداد الموثوق: t ≥ 3، وتفوق موجب في نصفي الفترة، وأغلب السنوات رابحة.", "",
                  fmt_table(sdf[keep].head(25).to_dict("records")), ""]
        print(f"{key}: sweep of {len(combos)} done", flush=True)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all", help="|".join(STRATEGIES) + "|all")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--universe", default="sp500")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    d = get_data(args.universe, args.years)
    span = (d["Close"].index[-1] - d["Close"].index[0]).days / 365.25
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keys = list(STRATEGIES) if args.strategy == "all" else args.strategy.split(",")
    for k in keys:
        md = run_one(k, d, args.sweep, span)
        (out / f"{k}.md").write_text(md)
        print(md, flush=True)


if __name__ == "__main__":
    main()
