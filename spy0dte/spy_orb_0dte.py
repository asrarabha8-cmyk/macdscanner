"""
SPY 0DTE — Opening Range Breakout + Compression Filter (live alert on Telegram)

The rules (validated on real contract prices Oct 2024 – Oct 2026; research lives in phscanner/backtests/spy_0dte):
  1) Opening range = high/low of the first 15 minutes (9:30–9:45 New York)
  2) Trade only if the range (% of the open) < 0.8 × its average over the previous 20 days
  3) First 5-min close above the range → ATM Call, below it → ATM Put (signal bar no later than 12:00 NY)
  4) Stop −35% of the contract price, no fixed target, exit by 13:00 NY at the latest. One trade per day.

What the script does in one run (starts before 9:45 NY):
  - Scores the previous day's paper trade (if there was one) on real contract prices from Polygon and sends the result
  - 9:45 NY: is today qualified or not
  - At breakout: entry alert (contract + approx price + stop)
  - 13:00 NY: exit reminder
Replay test mode: REPLAY_DATE=YYYY-MM-DD runs an old day instantly (messages are marked as a test).
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

NY = ZoneInfo("America/New_York")
RIYADH = ZoneInfo("Asia/Riyadh")
LOG_DIR = os.getenv("LOG_DIR", "logs/spy0dte")
RATIO = float(os.getenv("RATIO", "0.8"))
LOOKBACK = int(os.getenv("LOOKBACK", "20"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.35"))
LAST_SIGNAL = (12, 0)
EXIT_AT = (13, 0)
BUDGET = float(os.getenv("BUDGET", "200"))
SLIP = 0.02
COMM = 0.65
REPLAY = os.getenv("REPLAY_DATE", "").strip()
FORCE = os.getenv("FORCE_RUN", "0") == "1"
POLY = os.getenv("POLYGON_API_KEY", "").strip()
os.makedirs(LOG_DIR, exist_ok=True)
TAG = "🧪 <b>تجربة (يوم سابق)</b>\n" if REPLAY else ""


# ───────────── عام ─────────────
def telegram(text):
    text = TAG + text
    print(text.replace("<b>", "").replace("</b>", ""), "\n", flush=True)
    token = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("(no telegram secrets)")
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML",
                                   "disable_web_page_preview": "true"}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15)
    except Exception as e:
        print("telegram error:", e)


def now():
    if REPLAY:
        return datetime.combine(date.fromisoformat(REPLAY), datetime.min.time(), NY).replace(hour=13, minute=6)
    return datetime.now(NY)


def at(d, hm):
    return datetime.combine(d, datetime.min.time(), NY).replace(hour=hm[0], minute=hm[1])


def sleep_until(t):
    while not REPLAY and now() < t:
        time.sleep(min(30, max(1, (t - now()).total_seconds())))


def riy(t):
    return t.astimezone(RIYADH).strftime("%I:%M %p").replace("AM", "ص").replace("PM", "م")


def state_path(d):
    return f"{LOG_DIR}/{d}.json"


def load_state(d):
    p = state_path(d)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(d, s):
    json.dump(s, open(state_path(d), "w"), ensure_ascii=False, indent=1, default=str)


# ───────────── البيانات ─────────────
def spy_bars():
    for _ in range(5):
        try:
            df = yf.download("SPY", interval="5m", period="60d", progress=False, auto_adjust=False, prepost=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.index = df.index.tz_convert(NY)
            df = df.between_time("09:30", "15:55")[["open", "high", "low", "close", "volume"]].dropna()
            if len(df):
                return df
        except Exception as e:
            print("yf error", e)
        time.sleep(20)
    return pd.DataFrame()


def completed(df, t_now):
    """الشموع المكتملة فقط (بداية الشمعة + 5 دقائق ≤ الآن)"""
    return df[df.index + pd.Timedelta(minutes=5) <= t_now]


def opening_range(day_df):
    first = day_df[day_df.index.time < datetime.strptime("09:45", "%H:%M").time()]
    if len(first) < 3:
        return None
    hi, lo, op = first.high.max(), first.low.min(), first.open.iloc[0]
    return hi, lo, (hi - lo) / op


def occ(d, cp, k):
    return f"O:SPY{d.strftime('%y%m%d')}{cp}{int(k) * 1000:08d}"


def polygon_bars(ticker, d):
    if not POLY:
        return None
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{d}/{d}"
           f"?adjusted=true&sort=asc&limit=500&apiKey={POLY}")
    for _ in range(4):
        try:
            j = json.load(urllib.request.urlopen(url, timeout=30))
            res = j.get("results") or []
            if not res:
                return None
            x = pd.DataFrame(res)
            x.index = pd.to_datetime(x.t, unit="ms", utc=True).dt.tz_convert(NY)
            return x.rename(columns=dict(o="open", h="high", l="low", c="close", v="volume"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15)
                continue
            print("polygon", e)
            return None
        except Exception as e:
            print("polygon", e)
            time.sleep(5)
    return None


def live_quote(d, cp, k):
    """سعر العقد من سلسلة ياهو (قد يتأخر قليلًا)"""
    try:
        tk = yf.Ticker("SPY")
        exp = d.strftime("%Y-%m-%d")
        if exp not in tk.options:
            return None
        ch = tk.option_chain(exp)
        tab = ch.calls if cp == "C" else ch.puts
        r = tab[tab.strike == float(k)]
        if r.empty:
            return None
        r = r.iloc[0]
        return dict(bid=float(r.bid or 0), ask=float(r.ask or 0), last=float(r.lastPrice or 0))
    except Exception as e:
        print("chain error", e)
        return None


# ───────────── تقييم الصفقة الورقية على الأسعار الحقيقية ─────────────
FWD = f"{LOG_DIR}/forward.csv"
FIELDS = ["date", "dir", "strike", "entry_time", "entry", "exit", "exit_time", "why", "ret_pct", "qty", "pnl"]


def scored_dates():
    if not os.path.exists(FWD):
        return set()
    return {r["date"] for r in csv.DictReader(open(FWD))}


def score_day(d):
    s = load_state(d)
    if s.get("status") != "signal":
        return None
    cp, k = s["cp"], s["strike"]
    x = polygon_bars(occ(d, cp, k), d)
    if x is None or x.empty:
        return "nodata"
    e_t = pd.Timestamp(s["entry_bar"]).tz_convert(NY)        # بداية شمعة الدخول
    x = x[x.index >= e_t]
    x = x[x.index < at(d, EXIT_AT) + timedelta(minutes=5)]
    if x.empty:
        return "nodata"
    prem = float(x.open.iloc[0]) + SLIP
    stop = prem * (1 - STOP_PCT)
    why, px, t_out = "TIME", float(x.close.iloc[-1]), x.index[-1]
    for i, (ts, b) in enumerate(x.iterrows()):
        if b.low <= stop:
            why, px, t_out = "SL", (min(stop, float(b.open)) if i > 0 else stop), ts
            break
    px = max(px - SLIP, 0)
    qty = max(1, int(BUDGET // (prem * 100)))
    pnl = (px - prem) * 100 * qty - 2 * COMM * qty
    row = dict(date=str(d), dir="Call" if cp == "C" else "Put", strike=k, entry_time=e_t.strftime("%H:%M"),
               entry=round(prem, 2), exit=round(px, 2), exit_time=t_out.strftime("%H:%M"), why=why,
               ret_pct=round((px / prem - 1) * 100, 1), qty=qty, pnl=round(pnl, 2))
    new = not os.path.exists(FWD)
    with open(FWD, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def report_scores():
    done = scored_dates()
    pend = sorted(f[:-5] for f in os.listdir(LOG_DIR) if f.endswith(".json"))
    today = now().date()
    for ds in pend:
        d = date.fromisoformat(ds)
        if ds in done or (d >= today and not REPLAY) or (REPLAY and ds != REPLAY):
            continue
        r = score_day(d)
        if r is None or r == "nodata":
            continue
        allr = pd.read_csv(FWD)
        wins = (allr.pnl > 0).sum()
        icon = "✅" if r["pnl"] > 0 else "❌"
        telegram(
            f"{icon} <b>نتيجة الصفقة الورقية — {ds}</b>\n"
            f"{r['dir']} {r['strike']} | دخول {r['entry']} ← خروج {r['exit']} ({'وقف' if r['why']=='SL' else 'خروج 13:00'})\n"
            f"العائد {r['ret_pct']:+.1f}% | {r['pnl']:+.0f}$ ({r['qty']} عقد)\n\n"
            f"📒 <b>الاختبار الأمامي حتى الآن</b>: {len(allr)} صفقة | فوز {wins} ({wins/len(allr)*100:.0f}%) | "
            f"الصافي {allr.pnl.sum():+.0f}$")


# ───────────── الجلسة ─────────────
def session():
    today = now().date()
    st = load_state(today)
    if st.get("final") and not FORCE:
        print("already handled", today, st.get("status"))
        return
    if not REPLAY and now() > at(today, (13, 5)) and not FORCE:
        print("too late for today's session")
        return

    # 1) نطاق الافتتاح والفلتر
    sleep_until(at(today, (9, 45)) + timedelta(seconds=40))
    df, day = None, None
    for _ in range(12):
        df = spy_bars()
        day = df[df.index.date == today] if len(df) else pd.DataFrame()
        if len(completed(day, now())) >= 3:
            break
        if REPLAY or now() > at(today, (10, 15)):
            break
        time.sleep(45)
    if day is None or len(day) < 3:
        save_state(today, dict(status="closed", final=True))
        print("no market data today (holiday?)")
        return
    hi, lo, rng = opening_range(day)
    hist = []
    for d, g in df[df.index.date < today].groupby(df[df.index.date < today].index.date):
        o = opening_range(g)
        if o and len(g) >= 70:
            hist.append(o[2])
    avg = sum(hist[-LOOKBACK:]) / len(hist[-LOOKBACK:])
    ratio = rng / avg
    mid = (hi + lo) / 2
    base = dict(date=str(today), hi=round(hi, 2), lo=round(lo, 2), range_pct=round(rng * 100, 3),
                avg_pct=round(avg * 100, 3), ratio=round(ratio, 2))
    if ratio >= RATIO:
        save_state(today, dict(base, status="skip", final=True))
        telegram(f"⏸ <b>SPY 0DTE — {today}</b>\nاليوم <b>غير مؤهل</b> — نطاق الافتتاح {rng*100:.2f}% "
                 f"= {ratio:.2f}× المتوسط (الشرط أقل من {RATIO}).\nلا تداول اليوم.")
        return
    telegram(f"🟢 <b>SPY 0DTE — {today}: اليوم مؤهل</b>\n"
             f"نطاق الافتتاح: {lo:.2f} — {hi:.2f} ({rng*100:.2f}% = {ratio:.2f}× المتوسط)\n"
             f"📈 إغلاق شمعة 5د فوق <b>{hi:.2f}</b> = Call\n📉 إغلاق تحت <b>{lo:.2f}</b> = Put\n"
             f"آخر وقت للإشارة: {riy(at(today, LAST_SIGNAL))} بتوقيتك. سأرسل لك لحظة الاختراق.")
    save_state(today, dict(base, status="qualified"))

    # 2) انتظار الاختراق
    sig = None
    while True:
        df = spy_bars()
        day = completed(df[df.index.date == today], now())
        after = day[(day.index >= at(today, (9, 45))) & (day.index <= at(today, LAST_SIGNAL))]
        for ts, b in after.iterrows():
            if b.close > hi or b.close < lo:
                sig = (ts, 1 if b.close > hi else -1, float(b.close))
                break
        if sig or REPLAY or now() >= at(today, LAST_SIGNAL) + timedelta(minutes=6):
            break
        time.sleep(40)
    if not sig:
        save_state(today, dict(base, status="none", final=True))
        telegram(f"⚪ <b>SPY 0DTE — {today}</b>\nلا اختراق قبل الموعد. لا صفقة اليوم.")
        return

    ts, dr, px = sig
    cp = "C" if dr == 1 else "P"
    k = int(round(px))
    entry_bar = ts + timedelta(minutes=5)
    late = (now() - (entry_bar + timedelta(minutes=0))).total_seconds() / 60 if not REPLAY else 0
    if REPLAY:
        ob = polygon_bars(occ(today, cp, k), today)
        q = None
        if ob is not None and len(ob[ob.index >= entry_bar]):
            q = dict(ask=float(ob[ob.index >= entry_bar].open.iloc[0]), bid=0, last=0)
    else:
        q = live_quote(today, cp, k)
    price = (q["ask"] or q["last"]) if q else None
    st = dict(base, status="signal", cp=cp, strike=k, signal_bar=str(ts), entry_bar=str(entry_bar),
              spy_px=round(px, 2), est_entry=price)
    save_state(today, st)
    side = "📈 <b>CALL</b>" if dr == 1 else "📉 <b>PUT</b>"
    ptxt = (f"السعر التقريبي: <b>{price:.2f}</b> (≈{price*100:.0f}$ للعقد)\n"
            f"🛑 الوقف: <b>{price*(1-STOP_PCT):.2f}</b> (−35%)\n") if price else \
        "السعر: شوفه في منصتك، والوقف = سعر دخولك × 0.65\n"
    telegram(f"🚨 <b>إشارة دخول — SPY 0DTE</b>\n{side} سترايك <b>{k}</b> انتهاء اليوم\n"
             f"SPY أغلق {px:.2f} {'فوق' if dr==1 else 'تحت'} النطاق ({hi:.2f} / {lo:.2f})\n" + ptxt +
             f"🎯 بدون هدف ثابت — اخرج قبل <b>{riy(at(today, EXIT_AT))}</b> بتوقيتك\n"
             f"الحجم المقترح: {BUDGET:.0f}$ تقريبًا"
             + (f"\n⚠️ الإشارة متأخرة {late:.0f} دقيقة — تأكد أن السعر ما ابتعد كثير" if late > 10 else ""))

    # 3) تذكير الخروج
    sleep_until(at(today, EXIT_AT))
    q2 = None if REPLAY else live_quote(today, cp, k)
    extra = ""
    if q2 and price and (q2["bid"] or q2["last"]):
        p2 = q2["bid"] or q2["last"]
        extra = f"\nالسعر الآن تقريبًا {p2:.2f} ({(p2/price-1)*100:+.0f}%) — إن لم يضرب الوقف."
    st["final"] = True
    save_state(today, st)
    telegram(f"⏰ <b>وقت الخروج — SPY {k} {'Call' if cp=='C' else 'Put'}</b>\n"
             f"إذا الصفقة لسا مفتوحة أغلقها الآن.{extra}\n(النتيجة الورقية الدقيقة توصلك بكرة)")


if __name__ == "__main__":
    report_scores()
    session()
    if REPLAY:
        report_scores()
