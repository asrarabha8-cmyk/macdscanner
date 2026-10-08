#!/usr/bin/env python3
"""
مقارنة استراتيجيات يحيى على بيانات تاريخية — نفس الأسهم، نفس السنوات، نفس طريقة القياس.

الاستراتيجيات (شراء فقط، فاصل يومي):
  A) MA100/200  — نفس منطق ma100_200_scanner.py: اختراق طازج لمتوسط 100 + فوليوم ≥ 2× +
                  متوسط 200 فوق 100. الهدف = متوسط 200، الوقف = قاع شمعة الاختراق.
  B1) SMC صارم  — منطق smc_breakout_scanner.py على شموع يومية (بدل 4 ساعات): BOS فوق آخر قمة
                  هيكلية، الوقف = القاع B، الهدف = ABC 1.618، R:R ≥ 1.5 + فلاتر فينفيز الممكنة
                  تاريخياً (فوليوم نسبي > 2، تغيّر اليوم > 5%، قمة 20 يوم). فلتر الفلوت غير ممكن.
  B2) SMC مبسّط — نفس BOS والأهداف بدون فلاتر فينفيز (عينة أكبر).
  C) تصحيح داخل الترند — السهم فوق 50 و200، SPY فوق 200، لمس EMA20 خلال 3 أيام، ثم إغلاق
                  فوق قمة أمس. الوقف = أدنى قاع 5 أيام، الهدف = 2R.

قواعد التنفيذ (موحّدة لكل الاستراتيجيات — لا غش بالمستقبل):
  • الإشارة على إغلاق اليوم، الدخول على افتتاح اليوم التالي.
  • إذا فتح تحت الوقف تُلغى الصفقة. إذا فتح فوق الهدف تُلغى.
  • في نفس الشمعة: الوقف يُفحص قبل الهدف (افتراض متحفظ). الفجوة تُنفّذ على الافتتاح.
  • تكلفة 0.10% دخولاً و0.10% خروجاً. صفقة واحدة لكل سهم في نفس الوقت.
  • خروج بالوقت إذا لم يُضرب الوقف ولا الهدف.

القياس بوحدة R (الربح ÷ المخاطرة) — يقارن الاستراتيجيات بعدل مهما اختلف سعر السهم.
محاكاة رأس المال: 1% مخاطرة لكل صفقة.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

COST = 0.001
MIN_RISK = 0.0075   # وقف أقرب من 0.75% من سعر الدخول = غير واقعي (الانزلاق يأكله)
MAX_OPEN = 10       # أقصى صفقات مفتوحة في محاكاة رأس المال

UNIVERSE = """
AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO NFLX CRM ORCL ADBE INTC MU QCOM TXN AMAT LRCX KLAC
CSCO IBM ACN NOW PANW CRWD SNOW DDOG NET ZS MDB SHOP PYPL COIN HOOD SOFI UBER ABNB DASH PLTR
RBLX SNAP PINS ROKU TTD APP ANET DELL SMCI WDC STX HPQ F GM NIO RIVN LCID JPM BAC C WFC GS MS
SCHW XOM CVX OXY SLB HAL DVN COP KO PEP WMT COST TGT HD LOW NKE SBUX MCD DIS T VZ PFE MRK JNJ
ABBV LLY UNH CVS BA CAT DE GE LMT RTX MARA RIOT CLSK IREN CCJ FCX AA CLF AAL DAL UAL CCL
""".split()


# ------------------------------------------------------------------
@dataclass
class Trade:
    strategy: str
    ticker: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry: float
    stop: float
    target: float
    exit: float
    reason: str
    r: float
    days: int


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def simulate(df, i, stop, target, max_days, strat, ticker):
    """دخول على افتتاح i+1. يرجع (Trade أو None، فهرس الخروج)."""
    n = len(df)
    if i + 1 >= n:
        return None, i
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    entry = o[i + 1]
    if not (entry > stop) or (target is not None and entry >= target):
        return None, i
    risk = entry - stop
    if risk <= 0 or risk / entry < MIN_RISK:
        return None, i
    end = min(n - 1, i + max_days)
    for j in range(i + 1, end + 1):
        if l[j] <= stop:
            px, why = (o[j] if o[j] < stop and j > i + 1 else stop), "stop"
            break
        if target is not None and h[j] >= target:
            px, why = (o[j] if o[j] > target and j > i + 1 else target), "target"
            break
    else:
        j, px, why = end, c[end], "time"
        if end == n - 1 and end < i + max_days:
            why = "open"          # الصفقة ما زالت مفتوحة عند نهاية البيانات
    r = (px * (1 - COST) - entry * (1 + COST)) / risk
    idx = df.index
    t = Trade(strat, ticker, str(idx[i].date()), str(idx[i + 1].date()), str(idx[j].date()),
              round(entry, 4), round(stop, 4), round(target, 4) if target else float("nan"),
              round(px, 4), why, round(r, 3), int(j - i))
    return t, j


# ------------------------------------------------------------------
# A) MA100/200
def strat_ma(df, ticker, spy=None):
    c, v, lo = df["Close"], df["Volume"], df["Low"]
    ma100, ma200 = c.rolling(100).mean(), c.rolling(200).mean()
    vavg = v.shift(1).rolling(20).mean()
    trades, i, n = [], 201, len(df)
    while i < n - 1:
        ok = (c.iat[i] > ma100.iat[i] and c.iat[i - 1] <= ma100.iat[i - 1]
              and vavg.iat[i] > 0 and v.iat[i] >= 2 * vavg.iat[i]
              and c.iat[i] * vavg.iat[i] >= 3e6
              and ma200.iat[i] > ma100.iat[i] and ma200.iat[i] > c.iat[i])
        if ok:
            stop, target = lo.iat[i], ma200.iat[i]
            if c.iat[i] - stop > 0 and (target - c.iat[i]) / (c.iat[i] - stop) >= 1.0:
                t, j = simulate(df, i, stop, target, 60, "A_MA100_200", ticker)
                if t:
                    trades.append(t)
                    i = j + 1
                    continue
        i += 1
    return trades


# ------------------------------------------------------------------
# B) SMC — نفس find_bos / abc_targets لكن تراكمياً على اليومي
def _pivots(h, l, L=3):
    ph, pl = [], []
    for k in range(L, len(h) - L):
        if h[k] >= h[k - L:k + L + 1].max():
            ph.append(k)
        if l[k] <= l[k - L:k + L + 1].min():
            pl.append(k)
    return np.array(ph), np.array(pl)


def strat_smc(df, ticker, spy=None, strict=True):
    L = 3
    h, l, c, o, v = (df[k].values for k in ("High", "Low", "Close", "Open", "Volume"))
    ph, pl = _pivots(h, l, L)
    vavg = pd.Series(v).shift(1).rolling(20).mean().values
    hi20 = pd.Series(h).rolling(20).max().values
    name = "B1_SMC_strict" if strict else "B2_SMC_simple"
    trades, i, n = [], 60, len(df)
    while i < n - 1:
        conf_h = ph[ph < i - L]                    # قمم مؤكدة قبل الشمعة الحالية
        if len(conf_h) == 0:
            i += 1
            continue
        a = conf_h[-1]
        level = h[a]
        if not (c[i] > level and c[i - 1] <= level):
            i += 1
            continue
        conf_l = pl[pl <= i - L]
        after_a = conf_l[conf_l > a]
        if len(after_a):
            b = after_a[-1]
        elif len(conf_l):
            b = conf_l[-1]
        else:
            i += 1
            continue
        stop = l[b]
        leg_low = l[max(0, a - 40):a + 1].min()
        leg = level - leg_low
        if stop >= c[i] or leg <= 0:
            i += 1
            continue
        t1 = l[b] + leg * 1.618
        rr = (t1 - c[i]) / (c[i] - stop)
        if rr < 1.5:
            i += 1
            continue
        if strict:
            relvol = v[i] / vavg[i] if vavg[i] > 0 else 0
            chg = (c[i] / c[i - 1] - 1) * 100
            if not (relvol > 2 and chg > 5 and h[i] >= hi20[i] and c[i] > 1):
                i += 1
                continue
        t, j = simulate(df, i, stop, t1, 40, name, ticker)
        if t:
            trades.append(t)
            i = j + 1
            continue
        i += 1
    return trades


# ------------------------------------------------------------------
# C) تصحيح داخل الترند
def strat_pullback(df, ticker, spy=None):
    c, h, lo = df["Close"], df["High"], df["Low"]
    s50, s200, e20 = c.rolling(50).mean(), c.rolling(200).mean(), ema(c, 20)
    spy_ok = None
    if spy is not None:
        sc = spy["Close"]
        spy_ok = (sc > sc.rolling(200).mean()).reindex(df.index).ffill().fillna(False)
    trades, i, n = [], 201, len(df)
    while i < n - 1:
        ok = (c.iat[i] > s50.iat[i] > s200.iat[i]
              and (spy_ok is None or bool(spy_ok.iat[i]))
              and (lo.iloc[i - 2:i + 1] <= e20.iloc[i - 2:i + 1]).any()
              and c.iat[i] > h.iat[i - 1] and c.iat[i] > e20.iat[i])
        if ok:
            stop = lo.iloc[i - 4:i + 1].min()
            risk_pct = (c.iat[i] - stop) / c.iat[i]
            if 0.01 <= risk_pct <= 0.10:
                entry_est = c.iat[i]
                # الهدف 2R يُحسب من سعر الدخول الفعلي داخل simulate
                t, j = _sim_2r(df, i, stop, 30, ticker)
                if t:
                    trades.append(t)
                    i = j + 1
                    continue
        i += 1
    return trades


def _sim_2r(df, i, stop, max_days, ticker):
    if i + 1 >= len(df):
        return None, i
    entry = df["Open"].iat[i + 1]
    if entry <= stop:
        return None, i
    target = entry + 2 * (entry - stop)
    return simulate(df, i, stop, target, max_days, "C_Trend_Pullback", ticker)


# ------------------------------------------------------------------
# Z) ضابط عشوائي — نفس وقف وهدف الاستراتيجية C لكن الدخول في أيام عشوائية.
#    أي استراتيجية ما تتفوق عليه = ما عندها ميزة حقيقية غير اتجاه السوق العام.
def strat_random(df, ticker, spy=None):
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    lo = df["Low"]
    trades, i, n = [], 201, len(df)
    while i < n - 1:
        if rng.random() < 0.03:
            stop = lo.iloc[i - 4:i + 1].min()
            t, j = _sim_2r(df, i, stop, 30, ticker)
            if t:
                t.strategy = "Z_Random"
                trades.append(t)
                i = j + 1
                continue
        i += 1
    return trades


STRATS = {
    "A_MA100_200": strat_ma,
    "B1_SMC_strict": lambda d, t, s=None: strat_smc(d, t, s, strict=True),
    "B2_SMC_simple": lambda d, t, s=None: strat_smc(d, t, s, strict=False),
    "C_Trend_Pullback": strat_pullback,
    "Z_Random": strat_random,
}

LABELS = {
    "A_MA100_200": "اختراق متوسط 100 (هدف 200)",
    "B1_SMC_strict": "SMC بفلاتر فينفيز",
    "B2_SMC_simple": "SMC بدون فلاتر",
    "C_Trend_Pullback": "تصحيح داخل الترند",
    "Z_Random": "دخول عشوائي (ضابط)",
}


# ------------------------------------------------------------------
# الإحصاءات
def max_losing_streak(rs):
    best = cur = 0
    for r in rs:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def equity_curve(tr, risk=0.01):
    """1% مخاطرة لكل صفقة، تُحسب على رأس المال عند الدخول، وتتحقق عند الخروج."""
    if tr.empty:
        return 0.0, 0.0
    ev = []
    for _, t in tr.iterrows():
        ev.append((t.entry_date, 0, t.name))
        ev.append((t.exit_date, 1, t.name))
    ev.sort(key=lambda x: (x[0], -x[1]))      # الخروج قبل الدخول في نفس اليوم
    eq, alloc, peak, mdd = 1.0, {}, 1.0, 0.0
    for d, kind, k in ev:
        if kind == 0:
            if len(alloc) < MAX_OPEN:
                alloc[k] = eq * risk
        elif k in alloc:
            eq += alloc.pop(k) * tr.at[k, "r"]
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak)
    years = (pd.Timestamp(tr.exit_date.max()) - pd.Timestamp(tr.entry_date.min())).days / 365.25
    cagr = eq ** (1 / years) - 1 if years > 0 and eq > 0 else float("nan")
    return cagr, mdd


def stats(tr):
    tr = tr[tr.reason != "open"]
    if tr.empty:
        return dict(trades=0)
    rs = tr.r.values
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    years = (pd.Timestamp(tr.exit_date.max()) - pd.Timestamp(tr.entry_date.min())).days / 365.25
    cagr, mdd = equity_curve(tr.reset_index(drop=True))
    by_year = tr.assign(y=tr.entry_date.str[:4]).groupby("y").r.mean()
    return dict(
        trades=len(tr), per_year=len(tr) / years if years else float("nan"),
        win_rate=(rs > 0).mean() * 100, avg_win=wins.mean() if len(wins) else 0,
        avg_loss=losses.mean() if len(losses) else 0, expectancy=rs.mean(),
        median=np.median(rs), pf=pf, total_r=rs.sum(), max_lose_streak=max_losing_streak(rs),
        avg_days=tr.days.mean(), pct_target=(tr.reason == "target").mean() * 100,
        pct_stop=(tr.reason == "stop").mean() * 100, pct_time=(tr.reason == "time").mean() * 100,
        cagr=cagr * 100, max_dd=mdd * 100,
        pos_years=f"{(by_year > 0).sum()}/{len(by_year)}",
    )


# ------------------------------------------------------------------
# البيانات
def load_yf(tickers, years):
    import yfinance as yf
    out = {}
    for k in range(0, len(tickers), 40):
        batch = tickers[k:k + 40]
        raw = yf.download(batch, period=f"{years}y", interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
        for t in batch:
            try:
                d = raw[t].dropna() if len(batch) > 1 else raw.dropna()
            except (KeyError, TypeError):
                continue
            if len(d) > 260:
                out[t] = d[["Open", "High", "Low", "Close", "Volume"]]
        print(f"  تحميل {min(k + 40, len(tickers))}/{len(tickers)}", flush=True)
    return out


def synthetic(tickers, days=2600, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    idx = pd.bdate_range("2016-01-01", periods=days)
    for t in tickers:
        drift = rng.normal(0.0004, 0.0003)
        r = rng.normal(drift, 0.02, days)
        c = 50 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.005, days))
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.008, days)))
        l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.008, days)))
        v = rng.lognormal(14, 0.5, days)
        out[t] = pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=v), index=idx)
    return out


# ------------------------------------------------------------------
def fmt_table(res):
    cols = [("trades", "الصفقات", "{:.0f}"), ("per_year", "صفقة/سنة", "{:.0f}"),
            ("win_rate", "نسبة الربح %", "{:.0f}"), ("avg_win", "متوسط الربح R", "{:+.2f}"),
            ("avg_loss", "متوسط الخسارة R", "{:+.2f}"), ("expectancy", "التوقع R/صفقة", "{:+.3f}"),
            ("pf", "معامل الربح", "{:.2f}"), ("max_lose_streak", "أطول سلسلة خسائر", "{:.0f}"),
            ("avg_days", "متوسط الأيام", "{:.0f}"), ("cagr", "نمو سنوي % (1% مخاطرة، 10 صفقات كحد)", "{:+.1f}"),
            ("max_dd", "أقصى تراجع %", "{:.1f}"), ("pos_years", "سنوات رابحة", "{}")]
    head = "| المقياس | " + " | ".join(LABELS[k] for k in res) + " |"
    sep = "|---|" + "---|" * len(res)
    rows = [head, sep]
    for key, ar, f in cols:
        vals = []
        for k in res:
            v = res[k].get(key, "")
            vals.append(f.format(v) if v != "" and res[k].get("trades", 0) else "—")
        rows.append(f"| {ar} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--out", default="backtest/results")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    tickers = UNIVERSE
    data = synthetic(tickers + ["SPY"]) if a.synthetic else load_yf(tickers + ["SPY"], a.years)
    spy = data.pop("SPY", None)
    print(f"أسهم محمّلة: {len(data)}", flush=True)

    all_tr = []
    for name, fn in STRATS.items():
        for t, d in data.items():
            try:
                all_tr += fn(d, t, spy)
            except Exception as e:  # noqa: BLE001
                print(f"{name} {t}: {e}", file=sys.stderr)
        print(f"{name}: {sum(1 for x in all_tr if x.strategy == name)} صفقة", flush=True)

    tr = pd.DataFrame([asdict(x) for x in all_tr])
    os.makedirs(a.out, exist_ok=True)
    tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)

    res = {k: stats(tr[tr.strategy == k]) if not tr.empty else dict(trades=0) for k in STRATS}
    first, last = (tr.entry_date.min(), tr.exit_date.max()) if not tr.empty else ("", "")

    # أداء كل سنة
    yearly = ""
    if not tr.empty:
        yt = (tr[tr.reason != "open"].assign(y=tr.entry_date.str[:4])
              .pivot_table(index="y", columns="strategy", values="r", aggfunc="mean"))
        yt = yt.rename(columns=LABELS)
        yearly = "| السنة | " + " | ".join(yt.columns) + " |\n|---|" + "---|" * len(yt.columns) + "\n"
        for y, row in yt.iterrows():
            yearly += f"| {y} | " + " | ".join("—" if pd.isna(v) else f"{v:+.2f}" for v in row) + " |\n"

    md = [f"# مقارنة الاستراتيجيات — {len(data)} سهم، {first} → {last}",
          "", "القياس بوحدة R (الربح ÷ المخاطرة). التوقع R/صفقة > 0 = ميزة. "
          "تكلفة 0.1% لكل جهة. الدخول على افتتاح اليوم التالي.", "",
          fmt_table(res), "", "## متوسط R لكل سنة", "", yearly,
          "", "## ملاحظات", "",
          "- SMC هنا على شموع يومية بدل 4 ساعات، وبدون فلتر الفلوت (غير متاح تاريخياً) — تقريب للمنطق، لا نسخة مطابقة.",
          "- القائمة أسهم موجودة اليوم (انحياز النجاة) — يرفع النتائج قليلاً لكل الاستراتيجيات بالتساوي.",
          "- النتائج على السهم نفسه. الأوبشن يضخّم الربح والخسارة ويضيف تكلفة الوقت."]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))

    # تيليجرام مختصر
    token, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if token and chat and not a.synthetic:
        import requests
        lines = ["📊 <b>مقارنة الاستراتيجيات — اختبار تاريخي</b>", f"{len(data)} سهم • {first} → {last}", ""]
        for k, s in res.items():
            if not s.get("trades"):
                lines.append(f"• {LABELS[k]}: لا صفقات")
                continue
            lines.append(f"• <b>{LABELS[k]}</b>: {s['trades']} صفقة • ربح {s['win_rate']:.0f}% • "
                         f"توقع {s['expectancy']:+.2f}R • PF {s['pf']:.2f} • تراجع {s['max_dd']:.0f}%")
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": "\n".join(lines), "parse_mode": "HTML"}, timeout=30)


if __name__ == "__main__":
    main()
