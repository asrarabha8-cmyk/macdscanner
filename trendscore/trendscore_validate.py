"""
اختبار TrendScore: هل الدرجة العالية في الماضي تنبأت بأداء أفضل بعدها؟
التشغيل:
    pip install pandas yfinance
    python trendscore_validate.py
ضع الملفين بجانب السكربت:
    trend-score-sp500-2026-10-10_2.csv  -> قبل 3 أشهر
    trend-score-sp500-2026-10-10_3.csv  -> قبل شهر
"""
import pandas as pd
import yfinance as yf

AS_OF = pd.Timestamp("2026-10-09")  # آخر إغلاق
TESTS = {
    "3M": ("trend-score-sp500-2026-10-10_2.csv", AS_OF - pd.DateOffset(months=3)),
    "1M": ("trend-score-sp500-2026-10-10_3.csv", AS_OF - pd.DateOffset(months=1)),
}


def load(path):
    d = pd.read_csv(path, encoding="utf-8-sig").set_index("Symbol")
    d.index = d.index.str.replace(".", "-", regex=False)  # BRK.B -> BRK-B لياهو
    return d


def price_on(px, date):
    """أول إغلاق في أو بعد التاريخ المحدد."""
    return px.loc[px.index >= date].iloc[0]


all_syms = set()
frames = {}
for k, (f, _) in TESTS.items():
    frames[k] = load(f)
    all_syms |= set(frames[k].index)

start = min(d for _, d in TESTS.values()) - pd.Timedelta(days=7)
px = yf.download(sorted(all_syms | {"SPY"}), start=start,
                 end=AS_OF + pd.Timedelta(days=1), auto_adjust=True,
                 progress=False)["Close"]

for k, (_, start_date) in TESTS.items():
    d = frames[k].copy()
    p0, p1 = price_on(px, start_date), px.loc[:AS_OF].iloc[-1]
    ret = (p1 / p0 - 1)
    d["fwd"] = ret.reindex(d.index)
    d["excess"] = d["fwd"] - ret["SPY"]
    d = d.dropna(subset=["fwd"])

    d["bucket"] = pd.qcut(d["Score"].rank(method="first"), 5,
                          labels=["Q1 أضعف", "Q2", "Q3", "Q4", "Q5 أقوى"])
    g = d.groupby("bucket", observed=True).agg(
        n=("fwd", "size"),
        avg_ret=("fwd", "mean"),
        med_excess=("excess", "median"),
        beat_spy=("excess", lambda s: (s > 0).mean()),
    )
    ic = d["Score"].corr(d["fwd"], method="spearman")

    print(f"\n=== اختبار {k}: من {start_date.date()} إلى {AS_OF.date()} ===")
    print(f"عائد SPY: {ret['SPY']:+.1%} | ارتباط الترتيب (IC): {ic:+.3f}")
    print(g.to_string(float_format=lambda x: f"{x:+.3f}"))

    by_q = d.groupby("Quality")["excess"].median().sort_values()
    print("متوسط التفوق حسب Quality:")
    print(by_q.to_string(float_format=lambda x: f"{x:+.2%}"))

print("\nالقراءة: لو Q5 تتفوق بوضوح على Q1 والـ IC موجب (>0.05)، المؤشر له قيمة.")
