"""news_lab — هل الأخبار وقت التنبيه تتنبأ بنتيجته؟

ياخذ كل تنبيه انحسم (ربح/خسارة) من review/out/alerts_scored.csv،
ويجيب أخبار السهم خلال ٣ أيام قبل وقت التنبيه بالضبط (بدون نظر للمستقبل)،
ثم يقارن نسبة الفوز حسب حالة الأخبار.

لو التنبيهات اللي معها أخبار سلبية/خطيرة تخسر أكثر بوضوح ← نفعّل التحذير في الماسحات.
لو ما فيه فرق ← ما نضيف شي ونوفر التعقيد.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from news_guard import check  # noqa: E402

SRC = ROOT / "review" / "out" / "alerts_scored.csv"
OUT_CSV = ROOT / "review" / "out" / "news_vs_outcome.csv"
OUT_MD = ROOT / "research" / "news_lab.md"

AR = {"red": "خبر خطير 🚨", "negative": "أخبار سلبية ⚠️", "positive": "أخبار إيجابية 📰",
      "mixed": "أخبار متعادلة", "quiet": "بدون أخبار"}
ORDER = ["red", "negative", "mixed", "quiet", "positive"]


def ar_num(x) -> str:
    return str(x).translate(str.maketrans("0123456789.-", "٠١٢٣٤٥٦٧٨٩٫-"))


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["result"].isin(["win", "loss"])].copy()
    rows = []
    for _, a in df.iterrows():
        when = pd.Timestamp(a["time_utc"]).tz_localize("UTC").to_pydatetime()
        info = check(str(a["ticker"]), when, days=3)
        rows.append({**a.to_dict(), "news_n": info["n"], "news_pos": info["pos"],
                     "news_neg": info["neg"], "news_red": info["red"],
                     "flag": info.get("flag", "error") if not info["error"] else "error",
                     "red_title": (info["red_titles"] or [""])[0],
                     "error": info["error"]})
        time.sleep(0.15)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    ok = res[res["flag"] != "error"]
    res["win"] = res["result"].eq("win")
    ok = ok.assign(win=ok["result"].eq("win"))
    base = ok["win"].mean() * 100 if len(ok) else 0

    lines = ["# أثر الأخبار على نتيجة التنبيهات", "",
             f"تنبيهات محسومة: {len(res)} — بأخبار مقروءة: {len(ok)} — "
             f"أخطاء: {int((res['flag'] == 'error').sum())}",
             f"نسبة الفوز العامة: {base:.1f}%", "",
             "| حالة الأخبار | عدد | فوز% | متوسط الحركة% |", "|---|---|---|---|"]
    tg = [f"🧪 مختبر الأخبار — {len(ok)} تنبيه محسوم",
          f"نسبة الفوز العامة: {ar_num(f'{base:.0f}')}٪", ""]
    for f in ORDER:
        g = ok[ok["flag"] == f]
        if g.empty:
            continue
        wr = g["win"].mean() * 100
        mv = g["move_pct"].mean()
        lines.append(f"| {AR[f]} | {len(g)} | {wr:.1f} | {mv:.2f} |")
        tg.append(f"{AR[f]}: {ar_num(len(g))} تنبيه — فوز {ar_num(f'{wr:.0f}')}٪")

    bad = ok[ok["flag"].isin(["red", "negative"])]
    rest = ok[~ok["flag"].isin(["red", "negative"])]
    if len(bad) and len(rest):
        d = rest["win"].mean() * 100 - bad["win"].mean() * 100
        verdict = (f"التنبيهات بدون أخبار سلبية تفوز أكثر بـ {d:.1f} نقطة"
                   if d > 0 else f"ما فيه أفضلية لتجنب الأخبار السلبية ({d:.1f} نقطة)")
        lines += ["", f"**الخلاصة:** {verdict} (عينة سلبية = {len(bad)})"]
        tg += ["", "الخلاصة: " + verdict.replace(".", "٫")]
        if len(bad) < 20:
            lines.append("⚠️ العينة السلبية صغيرة — النتيجة مبدئية.")
            tg.append("⚠️ العينة صغيرة — النتيجة مبدئية")

    lines += ["", "## حسب الماسح", "", "| الماسح | حالة | عدد | فوز% |", "|---|---|---|---|"]
    for (s, f), g in ok.groupby(["scanner", "flag"]):
        lines.append(f"| {s} | {AR.get(f, f)} | {len(g)} | {g['win'].mean()*100:.0f} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": "\n".join(tg)}, timeout=20)


if __name__ == "__main__":
    main()
