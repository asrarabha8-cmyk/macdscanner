#!/usr/bin/env python3
"""
صائد الفرص — الطبقة الآلية من "معيار اختيار الشركات".

يُشغَّل شهرياً (أول كل شهر). يمسح السوق الأمريكي بفلتر سيولة، ثم يطبّق
الجزء القابل للأتمتة من المعيار فقط:

  بوابة الإقصاء : تخفيف ملكية، تجزئة عكسية، حقوق مساهمين سالبة، بيانات ناقصة
  السؤال ٣      : نمو الإيراد (٤ أرباع) + الهامش الإجمالي
  السؤال ٤      : صافي النقد / المدرج بالأرباع / إصدار أسهم جديد
  السؤال ٦      : فجوة القيمة العادلة = EPS الآجل × متوسط مضاعف السهم نفسه (٣ سنوات)

الأسئلة ١ و٢ و٥ (المشكلة، لماذا الآن، السور) لا تُؤتمت — تُترك فارغة
عمداً لتُجاب في طبقة الحكم (المهمة المجدولة). السكربت لا يخمّنها.

المخرج: reports/hunter_latest.json + reports/hunter_<YYYY-MM>.json
وملخص تيليجرام قصير.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------ إعدادات
MIN_PRICE          = 5.0          # استبعاد أسهم القروش
MIN_DOLLAR_VOL     = 5_000_000    # فلتر السيولة اليومي
MIN_MARKET_CAP     = 300_000_000  # أقل من ٣٠٠ مليون = ضجيج
DRAWDOWN_MIN       = 0.20         # ≥ ٢٠٪ تحت قمة ٥٢ أسبوعاً (بوابة التسعير الخاطئ)

GROSS_MARGIN_PASS  = 0.40         # السؤال ٣: هامش إجمالي
GROSS_MARGIN_FAIL  = 0.20
REV_GROWTH_PASS    = 0.15         # نمو إيراد سنوي لآخر ٤ أرباع
RUNWAY_PASS        = 8            # السؤال ٤: مدرج بالأرباع
RUNWAY_FAIL        = 6
DILUTION_FAIL      = 0.03         # نمو عدد الأسهم > ٣٪ سنوياً

TOP_N              = 25           # حجم القائمة القصيرة المسلّمة لطبقة الحكم
REPORT_DIR         = "reports"

HEADERS = {"User-Agent": "Mozilla/5.0"}


# ------------------------------------------------------------------ الكون
def _wiki_table(url: str, col: str) -> set[str]:
    try:
        html = requests.get(url, headers=HEADERS, timeout=30).text
        for tbl in pd.read_html(io.StringIO(html)):
            if col in tbl.columns:
                return set(tbl[col].astype(str).str.replace(".", "-", regex=False))
    except Exception as e:  # noqa: BLE001
        print(f"  تعذّر جلب {url}: {e}", file=sys.stderr)
    return set()


def universe() -> list[str]:
    """السوق الأمريكي الواسع: S&P 1500 كأساس، ثم قائمة ناسداك الكاملة إن توفرت."""
    names: set[str] = set()
    names |= _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    names |= _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol")
    names |= _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol")

    # قائمة كل الأسهم المدرجة — تُوسّع الكون للشركات الصغيرة خارج المؤشرات
    try:
        r = requests.get(
            "https://api.nasdaq.com/api/screener/stocks",
            params={"tableonly": "true", "limit": "10000", "download": "true"},
            headers=HEADERS, timeout=45,
        )
        rows = r.json()["data"]["rows"]
        for row in rows:
            sym = row.get("symbol", "").strip()
            cap = str(row.get("marketCap", "")).replace(",", "").strip()
            if not sym or not sym.isascii() or any(c in sym for c in "^/$"):
                continue
            try:
                if float(cap or 0) < MIN_MARKET_CAP:
                    continue
            except ValueError:
                continue
            names.add(sym.replace(".", "-"))
        print(f"  قائمة ناسداك: {len(rows)} صف")
    except Exception as e:  # noqa: BLE001
        print(f"  قائمة ناسداك غير متاحة ({e}) — الاكتفاء بـ S&P 1500", file=sys.stderr)

    names = {n for n in names if n and len(n) <= 5}
    return sorted(names)


# ------------------------------------------------------- المرحلة ١: السعر
def stage1(tickers: list[str]) -> pd.DataFrame:
    """مسح رخيص بالجملة: سيولة + هبوط عن قمة ٥٢ أسبوعاً."""
    out = []
    CHUNK = 200
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        try:
            px = yf.download(batch, period="1y", interval="1d", progress=False,
                             auto_adjust=True, threads=True, group_by="ticker")
        except Exception as e:  # noqa: BLE001
            print(f"  فشل تحميل الدفعة {i}: {e}", file=sys.stderr)
            continue

        for t in batch:
            try:
                df = px[t] if isinstance(px.columns, pd.MultiIndex) else px
                df = df.dropna()
                if len(df) < 120:
                    continue
                close = df["Close"]
                price = float(close.iloc[-1])
                if price < MIN_PRICE:
                    continue
                dvol = float((close * df["Volume"]).tail(60).mean())
                if dvol < MIN_DOLLAR_VOL:
                    continue
                high52 = float(close.max())
                drawdown = (high52 - price) / high52
                if drawdown < DRAWDOWN_MIN:
                    continue
                out.append({"ticker": t, "price": round(price, 2),
                            "high52": round(high52, 2),
                            "drawdown": round(drawdown, 3),
                            "dollar_vol": int(dvol)})
            except Exception:  # noqa: BLE001
                continue
        print(f"  المرحلة ١: {i + len(batch)}/{len(tickers)} — مرشحون حتى الآن {len(out)}")
    return pd.DataFrame(out)


# ------------------------------------------- المرحلة ٢: الأساسيات والمعيار
def _safe(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def _row(df: pd.DataFrame, *labels):
    """يسحب صفاً من قائمة مالية بأي من التسميات الممكنة."""
    if df is None or df.empty:
        return None
    for lab in labels:
        if lab in df.index:
            s = df.loc[lab].dropna()
            if not s.empty:
                return s
    return None


def evaluate(t: str, base: dict) -> dict | None:
    """يطبّق بوابة الإقصاء والأسئلة ٣ و٤ و٦ على سهم واحد."""
    tk = yf.Ticker(t)
    try:
        info = tk.info or {}
    except Exception:  # noqa: BLE001
        info = {}

    rec: dict = dict(base)
    rec["name"] = _safe(info, "shortName", "longName", default=t)
    rec["sector"] = _safe(info, "sector", default="—")
    rec["market_cap"] = _safe(info, "marketCap", default=None)
    gate: list[str] = []
    notes: list[str] = []

    # ---------- بوابة الإقصاء ----------
    try:
        splits = tk.splits
        if splits is not None and len(splits):
            idx = pd.to_datetime(splits.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365)
            recent = [float(v) for ts, v in zip(idx, splits.values) if ts > cutoff]
            if recent and min(recent) < 1:
                gate.append("تجزئة عكسية خلال ١٢ شهراً")
    except Exception:  # noqa: BLE001
        pass

    try:
        bs = tk.quarterly_balance_sheet
    except Exception:  # noqa: BLE001
        bs = pd.DataFrame()
    try:
        inc = tk.quarterly_income_stmt
    except Exception:  # noqa: BLE001
        inc = pd.DataFrame()
    try:
        cf = tk.quarterly_cashflow
    except Exception:  # noqa: BLE001
        cf = pd.DataFrame()

    equity = _row(bs, "Stockholders Equity", "Total Stockholder Equity")
    if equity is not None and float(equity.iloc[0]) < 0:
        gate.append("حقوق مساهمين سالبة")

    shares = _row(bs, "Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding")
    dilution = None
    if shares is not None and len(shares) >= 4:
        now_sh, past_sh = float(shares.iloc[0]), float(shares.iloc[3])
        if past_sh > 0:
            dilution = (now_sh - past_sh) / past_sh
            if dilution > DILUTION_FAIL:
                gate.append(f"تخفيف ملكية {dilution:+.1%} سنوياً")
    rec["dilution"] = round(dilution, 4) if dilution is not None else None

    if bs.empty or inc.empty:
        gate.append("قوائم مالية ناقصة")

    rec["gate_flags"] = gate
    rec["gate_pass"] = not gate

    # ---------- السؤال ٣: هل الإيرادات تثبت الطلب ----------
    q3, q3_why = 0.0, []
    revenue = _row(inc, "Total Revenue", "Operating Revenue")
    gp = _row(inc, "Gross Profit")

    growths = []
    if revenue is not None and len(revenue) >= 5:
        for i in range(min(4, len(revenue) - 4)):
            cur, yr_ago = float(revenue.iloc[i]), float(revenue.iloc[i + 4])
            if yr_ago > 0:
                growths.append((cur - yr_ago) / yr_ago)
    elif revenue is not None and len(revenue) >= 2:
        cur, prev = float(revenue.iloc[0]), float(revenue.iloc[1])
        if prev > 0:
            growths.append((cur - prev) / prev)

    rec["rev_growth_last"] = round(growths[0], 4) if growths else None
    rec["rev_growth_streak"] = len([g for g in growths if g >= REV_GROWTH_PASS])

    margin = None
    if gp is not None and revenue is not None and float(revenue.iloc[0]) > 0:
        margin = float(gp.iloc[0]) / float(revenue.iloc[0])
    elif _safe(info, "grossMargins") is not None:
        margin = float(info["grossMargins"])
    rec["gross_margin"] = round(margin, 4) if margin is not None else None

    if growths and margin is not None:
        strong_growth = len([g for g in growths if g >= REV_GROWTH_PASS]) >= max(2, len(growths) - 1)
        if float(revenue.iloc[0]) <= 0:
            q3, q3_why = 0.0, ["إيراد صفر"]
        elif margin < GROSS_MARGIN_FAIL:
            q3, q3_why = 0.0, [f"هامش إجمالي {margin:.0%} — تحت العتبة"]
        elif growths[0] < 0:
            q3, q3_why = 0.0, [f"الإيراد منكمش {growths[0]:+.0%}"]
        elif strong_growth and margin >= GROSS_MARGIN_PASS:
            q3, q3_why = 1.0, [f"نمو {growths[0]:+.0%} بهامش {margin:.0%}"]
        else:
            q3, q3_why = 0.5, [f"نمو {growths[0]:+.0%} بهامش {margin:.0%} — دون العتبة الكاملة"]
    else:
        q3_why = ["بيانات إيراد غير كافية"]

    rec["q3_score"], rec["q3_why"] = q3, q3_why

    # ---------- السؤال ٤: السيولة ----------
    q4, q4_why = 0.0, []
    cash = _row(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    debt = _row(bs, "Total Debt")
    cash_v = float(cash.iloc[0]) if cash is not None else _safe(info, "totalCash", default=None)
    debt_v = float(debt.iloc[0]) if debt is not None else _safe(info, "totalDebt", default=0) or 0
    net_cash = (cash_v - debt_v) if cash_v is not None else None
    rec["net_cash"] = int(net_cash) if net_cash is not None else None

    ocf = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = _row(cf, "Capital Expenditure")
    fcf_q = None
    if ocf is not None:
        fcf_q = float(ocf.iloc[0]) + (float(capex.iloc[0]) if capex is not None else 0.0)
    rec["fcf_quarter"] = int(fcf_q) if fcf_q is not None else None

    runway = None
    if fcf_q is not None and cash_v is not None:
        if fcf_q >= 0:
            runway = 99
        elif fcf_q < 0:
            runway = cash_v / abs(fcf_q)
    rec["runway_quarters"] = round(runway, 1) if runway is not None else None

    if net_cash is not None and runway is not None:
        issued = dilution is not None and dilution > DILUTION_FAIL
        if runway < RUNWAY_FAIL or issued:
            q4 = 0.0
            q4_why = [f"مدرج {runway:.0f} أرباع" if runway < RUNWAY_FAIL else "إصدار أسهم جديد"]
        elif net_cash > 0 and runway >= RUNWAY_PASS:
            q4 = 1.0
            q4_why = ["صافي نقد موجب" + (f"، تدفق حر موجب" if fcf_q and fcf_q >= 0 else f"، مدرج {runway:.0f} أرباع")]
        else:
            q4 = 0.5
            q4_why = [f"صافي دَين" if net_cash <= 0 else f"مدرج {runway:.0f} أرباع"]
    else:
        q4_why = ["بيانات ميزانية غير كافية"]

    rec["q4_score"], rec["q4_why"] = q4, q4_why

    # ---------- السؤال ٦: ماذا يفترض السعر ----------
    fwd_eps = _safe(info, "forwardEps", default=None)
    hist_pe = None
    try:
        ann_inc = tk.income_stmt
        ann_rev = _row(ann_inc, "Net Income", "Net Income Common Stockholders")
        sh_out = _safe(info, "sharesOutstanding", default=None)
        if ann_rev is not None and sh_out:
            hist = tk.history(period="4y", interval="1mo", auto_adjust=True)["Close"]
            pes = []
            for ts in list(ann_rev.index)[:3]:
                eps_y = float(ann_rev.loc[ts]) / float(sh_out)
                if eps_y <= 0:
                    continue
                yr = pd.Timestamp(ts).year
                px_y = hist[hist.index.year == yr]
                if len(px_y):
                    pes.append(float(px_y.mean()) / eps_y)
            pes = [p for p in pes if 3 < p < 120]
            if pes:
                hist_pe = float(np.mean(pes))
    except Exception:  # noqa: BLE001
        pass

    fv, gap = None, None
    if fwd_eps and fwd_eps > 0 and hist_pe:
        fv = fwd_eps * hist_pe
        gap = (fv - rec["price"]) / rec["price"]
    rec["fwd_eps"] = round(fwd_eps, 2) if fwd_eps else None
    rec["hist_pe_3y"] = round(hist_pe, 1) if hist_pe else None
    rec["fair_value"] = round(fv, 2) if fv else None
    rec["gap_pct"] = round(gap, 3) if gap is not None else None

    if fv is None:
        notes.append("لا مرساة تقييم — EPS آجل سالب أو مضاعف تاريخي غير مستقر")

    # ---------- التجميع ----------
    rec["auto_score"] = q3 + q4                      # من ٢ (الأسئلة الآلية فقط)
    rec["manual_pending"] = ["١ المشكلة", "٢ لماذا الآن", "٥ السور"]
    rec["needs_web_check"] = ["تأخير إيداع", "ضعف رقابي", "استمرارية المنشأة"]
    rec["notes"] = notes
    return rec


# ------------------------------------------------------------------ تيليجرام
def telegram(text: str) -> None:
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if not tok or not chat:
        print("(بلا تيليجرام — المتغيرات غير مضبوطة)")
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"فشل تيليجرام: {e}", file=sys.stderr)


# ------------------------------------------------------------------ main
def main() -> None:
    t0 = time.time()
    print("صائد الفرص — الطبقة الآلية")
    print("جلب الكون…")
    tickers = universe()
    print(f"  {len(tickers)} رمزاً")

    print("المرحلة ١: السيولة والهبوط عن القمة…")
    cand = stage1(tickers)
    if cand.empty:
        print("لا مرشحين في المرحلة ١")
        return
    cand = cand.sort_values("drawdown", ascending=False)
    print(f"  {len(cand)} مرشحاً اجتازوا فلتر السيولة والهبوط")

    # نحدّ المرحلة ٢ بأعمق ٣٠٠ هبوطاً حتى لا تطول الجلسة
    short = cand.head(300).to_dict("records")

    print("المرحلة ٢: بوابة الإقصاء + الأسئلة ٣ و٤ و٦…")
    rows = []
    for n, base in enumerate(short, 1):
        try:
            r = evaluate(base["ticker"], base)
            if r:
                rows.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  {base['ticker']}: {e}", file=sys.stderr)
        if n % 25 == 0:
            print(f"  {n}/{len(short)}")

    # ترتيب: من يجتاز البوابة أولاً، ثم أعلى درجة آلية، ثم أكبر فجوة
    rows.sort(key=lambda r: (r["gate_pass"], r["auto_score"], r["gap_pct"] or -9), reverse=True)
    passed = [r for r in rows if r["gate_pass"] and r["auto_score"] >= 1.5]
    shortlist = (passed or rows)[:TOP_N]

    stamp = datetime.now(timezone.utc)
    payload = {
        "generated_at": stamp.isoformat(timespec="seconds"),
        "cycle": stamp.strftime("%Y-%m"),
        "universe_size": len(tickers),
        "stage1_candidates": len(cand),
        "stage2_evaluated": len(rows),
        "shortlist_size": len(shortlist),
        "thresholds": {
            "min_dollar_volume": MIN_DOLLAR_VOL, "min_market_cap": MIN_MARKET_CAP,
            "drawdown_min": DRAWDOWN_MIN, "gross_margin_pass": GROSS_MARGIN_PASS,
            "rev_growth_pass": REV_GROWTH_PASS, "runway_pass_quarters": RUNWAY_PASS,
        },
        "automated_questions": [3, 4, 6],
        "manual_questions": [1, 2, 5],
        "shortlist": shortlist,
        "rejected_by_gate": [
            {"ticker": r["ticker"], "name": r["name"], "flags": r["gate_flags"]}
            for r in rows if not r["gate_pass"]
        ][:40],
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    for path in (f"{REPORT_DIR}/hunter_latest.json",
                 f"{REPORT_DIR}/hunter_{payload['cycle']}.json"):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"حُفظ المخرج — {len(shortlist)} في القائمة القصيرة")

    top = shortlist[:8]
    lines = [f"<b>صائد الفرص — {payload['cycle']}</b>",
             f"الكون {len(tickers)} · اجتاز السيولة والهبوط {len(cand)} · القائمة القصيرة {len(shortlist)}", ""]
    for r in top:
        gap = f"{r['gap_pct']:+.0%}" if r["gap_pct"] is not None else "—"
        lines.append(f"<b>{r['ticker']}</b> {r['price']} · هبوط {r['drawdown']:.0%} · "
                     f"فجوة {gap} · آلي {r['auto_score']:.1f}/2")
    lines += ["", "الأسئلة ١ و٢ و٥ تُجاب في طبقة الحكم — هذه قائمة مرشحين لا توصيات."]
    telegram("\n".join(lines))
    print(f"تم في {time.time() - t0:.0f} ثانية")


if __name__ == "__main__":
    main()
