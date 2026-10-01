#!/usr/bin/env python3
"""
OptiTrader-style options SIMULATION bot  (paper only — not investment advice)
=============================================================================
Mirrors the workflow in the OptiTrader AI clip, with one fix the clip itself
called out (profit protection):

  1. Pre-open (09:25 ET): rank the watchlist by gap % and pre-market volume,
     keep the top 3, each with a bias (gap up -> CALL, gap down -> PUT).
  2. Opening range: the first OR_MINUTES of the session.
  3. Signal (09:35 -> SIGNAL_END_ET), on a *completed* 5-minute bar:
       CALL: close > OR high  and close > VWAP
       PUT : close < OR low   and close < VWAP
     Market confirmation: QQQ last completed close on the same side of its VWAP.
     Signals without confirmation are logged as rejected (no forced entries).
  4. Contract selection (nearest expiry with MIN_DTE..MAX_DTE days):
       Ask in [ASK_MIN, ASK_MAX], |delta| in [DELTA_MIN, DELTA_MAX],
       spread <= MAX_SPREAD, Ask*100 <= BUDGET.  Best = tightest spread,
       then delta closest to 0.40, then highest volume. Entry at the Ask.
  5. Entry is "locked" in SIM_POSITION.json with a SHA-256 hash before any result.
  6. Management (every minute), exits at the Bid:
       - stock target (trigger +/- TARGET_R x risk)  -> take profit
       - stock invalidation (5m close beyond the opposite OR side)
       - option hard stop (-HARD_STOP_PCT)
       - PROFIT PROTECTION: once the gain reaches +PROTECT_ARM_PCT, the stop
         moves up to lock PROTECT_LOCK_PCT of the best gain seen
       - time exit at EXIT_ET
  7. Telegram alerts at every step + CSV logs under logs/.

Data: yfinance (Yahoo). Quotes can be delayed — this is a simulation.
Run:  python optitrader_sim.py            (live paper session)
      python optitrader_sim.py --selftest (synthetic day, no network)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dtime, date
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
RIYADH = ZoneInfo("Asia/Riyadh")

# ───────────────────────── settings (env overrides) ─────────────────────────
def _env(name, default, cast=float):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default

BUDGET = _env("BUDGET", 350.0)
ASK_MIN = _env("ASK_MIN", 2.00)
ASK_MAX = _env("ASK_MAX", 3.50)
DELTA_MIN = _env("DELTA_MIN", 0.35)
DELTA_MAX = _env("DELTA_MAX", 0.65)
MAX_SPREAD = _env("MAX_SPREAD", 0.10)
MIN_DTE = _env("MIN_DTE", 2, int)
MAX_DTE = _env("MAX_DTE", 7, int)
OR_MINUTES = _env("OR_MINUTES", 15, int)
MIN_RISK_PCT = _env("MIN_RISK_PCT", 0.006)   # stop at least 0.6% away from trigger
MAX_CHASE_R = _env("MAX_CHASE_R", 0.5)       # skip if price already ran > 0.5R past trigger
TARGET_R = _env("TARGET_R", 2.0)
HARD_STOP_PCT = _env("HARD_STOP_PCT", 0.45)
PROTECT_ARM_PCT = _env("PROTECT_ARM_PCT", 0.25)
PROTECT_LOCK_PCT = _env("PROTECT_LOCK_PCT", 0.50)
MAX_TRADES = _env("MAX_TRADES", 2, int)
TOP_N = _env("TOP_N", 3, int)
SIGNAL_END_ET = _env("SIGNAL_END_ET", "11:30", str)
EXIT_ET = _env("EXIT_ET", "13:45", str)          # 20:45 Riyadh in US summer time
RISK_FREE = _env("RISK_FREE", 0.045)
MARKET_ETF = _env("MARKET_ETF", "QQQ", str)
POLL_SECONDS = _env("POLL_SECONDS", 60, int)

DEFAULT_WATCHLIST = ["NVDA", "AAPL", "TSLA", "AMZN", "MSFT", "META",
                     "AMD", "INTC", "XOM", "GOOGL", "AVGO", "PLTR"]

LOG_DIR = os.getenv("LOG_DIR", "logs")


def _hm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


# ───────────────────────── helpers ─────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, iv, r, kind):
    if T <= 0 or iv <= 0:
        return max(0.0, (S - K) if kind == "CALL" else (K - S))
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "CALL":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S, K, T, iv, r, kind):
    if T <= 0 or iv <= 0 or S <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    return norm_cdf(d1) if kind == "CALL" else norm_cdf(d1) - 1


def riyadh(ts: datetime) -> str:
    return ts.astimezone(RIYADH).strftime("%H:%M")


def session_bars(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df[df.index.date == day]
    return d[(d.index.time >= dtime(9, 30)) & (d.index.time < dtime(16, 0))]


def completed(df: pd.DataFrame, now: datetime, minutes=5) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[df.index + timedelta(minutes=minutes) <= now]


def vwap(df: pd.DataFrame) -> float:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].replace(0, pd.NA).fillna(1)
    return float((tp * v).sum() / v.sum())


# ───────────────────────── data provider (yfinance) ─────────────────────────
class YFProvider:
    def __init__(self):
        import yfinance as yf  # imported lazily so --selftest needs no network
        self.yf = yf
        self._t = {}

    def _tk(self, s):
        if s not in self._t:
            self._t[s] = self.yf.Ticker(s)
        return self._t[s]

    def now(self) -> datetime:
        return datetime.now(NY)

    def sleep(self, sec):
        time.sleep(sec)

    def bars(self, s) -> pd.DataFrame:
        df = self._tk(s).history(period="5d", interval="5m", prepost=True)
        if df.empty:
            return df
        df.index = df.index.tz_convert(NY)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def prev_close(self, s) -> float:
        d = self._tk(s).history(period="10d", interval="1d")
        d.index = d.index.tz_convert(NY)
        today = self.now().date()
        d = d[d.index.date < today]
        return float(d["Close"].iloc[-1])

    def expiries(self, s) -> list[str]:
        return list(self._tk(s).options)

    def chain(self, s, exp, kind) -> pd.DataFrame:
        oc = self._tk(s).option_chain(exp)
        return (oc.calls if kind == "CALL" else oc.puts).copy()


# ───────────────────────── state ─────────────────────────
@dataclass
class Candidate:
    symbol: str
    bias: str            # CALL / PUT
    gap_pct: float
    pm_vol_ratio: float
    score: float
    traded: bool = False
    rejects: int = 0


@dataclass
class Position:
    symbol: str
    kind: str
    expiry: str
    strike: float
    entry_ask: float
    delta: float
    spread: float
    entry_time: str
    stock_entry: float
    trigger: float
    invalidation: float
    target: float
    peak_bid: float = 0.0
    last_bid: float = 0.0
    contract: str = ""
    lock_hash: str = ""


@dataclass
class Sim:
    provider: object
    notify: object
    watchlist: list = field(default_factory=lambda: list(DEFAULT_WATCHLIST))
    candidates: list = field(default_factory=list)
    position: Position | None = None
    trades: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    seen_bar: dict = field(default_factory=dict)
    day: date | None = None

    # ── 1. pre-open ranking
    def rank(self):
        now = self.provider.now()
        self.day = now.date()
        rows = []
        for s in self.watchlist:
            try:
                df = self.provider.bars(s)
                pc = self.provider.prev_close(s)
                today = df[df.index.date == self.day]
                pm = today[today.index.time < dtime(9, 30)]
                if pm.empty:
                    continue
                last = float(pm["Close"].iloc[-1])
                gap = (last / pc - 1) * 100
                pm_vol = float(pm["Volume"].sum())
                past = df[(df.index.date < self.day) & (df.index.time < dtime(9, 30))]
                n_days = max(1, len(set(past.index.date)))
                avg_pm = float(past["Volume"].sum()) / n_days if not past.empty else 0
                ratio = pm_vol / avg_pm if avg_pm > 0 else 1.0
                score = abs(gap) * (1 + math.log1p(min(ratio, 10)))
                rows.append(Candidate(s, "CALL" if gap > 0 else "PUT", gap, ratio, score))
            except Exception as e:  # keep going on bad tickers
                print(f"rank {s}: {e}")
        rows.sort(key=lambda c: c.score, reverse=True)
        self.candidates = rows[:TOP_N]
        if not self.candidates:
            self.notify("⚠️ ما قدرت أرتب فرص اليوم (لا توجد بيانات ما قبل الافتتاح).")
            return
        lines = [f"🧭 <b>قبل الافتتاح — أفضل {len(self.candidates)} فرص</b>"]
        for i, c in enumerate(self.candidates, 1):
            lines.append(f"{i}. <b>{c.symbol}</b> {c.bias} | فجوة {c.gap_pct:+.2f}% | حجم ما قبل الافتتاح ×{c.pm_vol_ratio:.1f}")
        lines.append(f"التأكيد: {MARKET_ETF} مع VWAP • الافتتاح {riyadh(now.replace(hour=9, minute=30))} الرياض")
        self.notify("\n".join(lines))

    # ── 2-3. opening range + signal
    def levels(self, s, now):
        sess = completed(session_bars(self.provider.bars(s), self.day), now)
        if sess is None or sess.empty:
            return None
        n_or = max(1, OR_MINUTES // 5)
        if len(sess) < n_or + 1:
            return None
        orb = sess.iloc[:n_or]
        return {
            "or_high": float(orb["High"].max()),
            "or_low": float(orb["Low"].min()),
            "vwap": vwap(sess),
            "close": float(sess["Close"].iloc[-1]),
            "bar": sess.index[-1],
        }

    def market_ok(self, bias, now):
        lv = self.levels(MARKET_ETF, now)
        if lv is None:
            return False, "لا بيانات للسوق"
        above = lv["close"] > lv["vwap"]
        ok = above if bias == "CALL" else not above
        side = "فوق" if above else "تحت"
        return ok, f"{MARKET_ETF} {side} VWAP ({lv['close']:.2f} / {lv['vwap']:.2f})"

    def scan(self, now):
        for c in self.candidates:
            if c.traded or self.position:
                continue
            lv = self.levels(c.symbol, now)
            if lv is None or self.seen_bar.get(c.symbol) == lv["bar"]:
                continue
            self.seen_bar[c.symbol] = lv["bar"]
            if c.bias == "CALL":
                sig = lv["close"] > lv["or_high"] and lv["close"] > lv["vwap"]
                trigger = lv["or_high"]
                inval = min(lv["or_low"], lv["vwap"], trigger * (1 - MIN_RISK_PCT))
            else:
                sig = lv["close"] < lv["or_low"] and lv["close"] < lv["vwap"]
                trigger = lv["or_low"]
                inval = max(lv["or_high"], lv["vwap"], trigger * (1 + MIN_RISK_PCT))
            if not sig:
                continue
            if abs(lv["close"] - trigger) > MAX_CHASE_R * abs(trigger - inval):
                self.rejected.append((riyadh(now), c.symbol, c.bias, "السعر ابتعد عن التريقر — ما نلاحق"))
                continue
            ok, why = self.market_ok(c.bias, now)
            if not ok:
                c.rejects += 1
                self.rejected.append((riyadh(now), c.symbol, c.bias, f"شرط السوق ما تحقق: {why}"))
                if c.rejects == 1:
                    self.notify(f"⛔ إشارة {c.symbol} {c.bias} مرفوضة — {why}")
                continue
            risk = abs(trigger - inval)
            target = trigger + TARGET_R * risk if c.bias == "CALL" else trigger - TARGET_R * risk
            self.enter(c, lv, trigger, inval, target, why, now)

    # ── 4. contract selection
    def pick_contract(self, s, kind, S, now):
        today = now.date()
        exps = []
        for e in self.provider.expiries(s):
            dte = (date.fromisoformat(e) - today).days
            if MIN_DTE <= dte <= MAX_DTE:
                exps.append(e)
        for e in exps[:2]:
            ch = self.provider.chain(s, e, kind)
            if ch is None or ch.empty:
                continue
            exp_dt = datetime.combine(date.fromisoformat(e), dtime(16, 0), NY)
            T = max((exp_dt - now).total_seconds(), 60) / (365 * 24 * 3600)
            ch = ch[(ch["bid"] > 0) & (ch["ask"] > 0)].copy()
            ch["spread"] = (ch["ask"] - ch["bid"]).round(2)
            ch["delta"] = [bs_delta(S, k, T, iv, RISK_FREE, kind)
                           for k, iv in zip(ch["strike"], ch["impliedVolatility"])]
            ch["absd"] = ch["delta"].abs()
            ok = ch[(ch["ask"].between(ASK_MIN, ASK_MAX)) &
                    (ch["absd"].between(DELTA_MIN, DELTA_MAX)) &
                    (ch["spread"] <= MAX_SPREAD + 1e-9) &
                    (ch["ask"] * 100 <= BUDGET)]
            if ok.empty:
                continue
            ok = ok.assign(dd=(ok["absd"] - 0.40).abs(),
                           vol=ok.get("volume", pd.Series(0, index=ok.index)).fillna(0))
            best = ok.sort_values(["spread", "dd", "vol"], ascending=[True, True, False]).iloc[0]
            return e, best, ch
        return None, None, None

    # ── 5. entry + lock
    def enter(self, c, lv, trigger, inval, target, why, now):
        exp, row, ch = self.pick_contract(c.symbol, c.bias, lv["close"], now)
        if row is None:
            self.rejected.append((riyadh(now), c.symbol, c.bias, "لا يوجد عقد يطابق الشروط"))
            if sum(1 for r in self.rejected if r[1] == c.symbol and "عقد" in r[3]) == 1:
                    self.notify(f"⚠️ إشارة {c.symbol} {c.bias} تحققت لكن ما فيه عقد يطابق القواعد (Ask {ASK_MIN}-{ASK_MAX}، دلتا {DELTA_MIN}-{DELTA_MAX}، سبريد ≤ {MAX_SPREAD}).")
            return
        c.traded = True
        p = Position(
            symbol=c.symbol, kind=c.bias, expiry=exp, strike=float(row["strike"]),
            entry_ask=float(row["ask"]), delta=round(float(row["delta"]), 2),
            spread=float(row["spread"]), entry_time=now.isoformat(),
            stock_entry=lv["close"], trigger=round(trigger, 2),
            invalidation=round(inval, 2), target=round(target, 2),
            peak_bid=float(row["bid"]), last_bid=float(row["bid"]),
            contract=str(row.get("contractSymbol", "")),
        )
        blob = json.dumps(asdict(p), sort_keys=True, ensure_ascii=False)
        p.lock_hash = hashlib.sha256(blob.encode()).hexdigest()
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "SIM_POSITION.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(p), f, ensure_ascii=False, indent=2)
        self.position = p
        cost = p.entry_ask * 100
        letter = "C" if p.kind == "CALL" else "P"
        self.notify(
            f"🎯 <b>دخول (محاكاة): {p.symbol} {p.strike:g}{letter}</b> انتهاء {p.expiry}\n"
            f"Ask {p.entry_ask:.2f} • سبريد {p.spread:.2f} • دلتا {p.delta:+.2f}\n"
            f"التكلفة ${cost:.0f} (الميزانية ${BUDGET:.0f}) • الوقت {riyadh(now)} الرياض\n"
            f"السهم {p.stock_entry:.2f} | تريقر {p.trigger} | إبطال {p.invalidation} | هدف {p.target}\n"
            f"تأكيد: {why}\n🔒 مقفول قبل النتيجة SHA-256 {p.lock_hash[:12]}…"
        )

    # ── 6. management
    def manage(self, now):
        p = self.position
        ch = self.provider.chain(p.symbol, p.expiry, p.kind)
        row = ch[ch["strike"] == p.strike]
        bid = float(row["bid"].iloc[0]) if not row.empty else 0.0
        if bid > 0:
            p.last_bid = bid
            p.peak_bid = max(p.peak_bid, bid)
        bid = p.last_bid
        sess = session_bars(self.provider.bars(p.symbol), self.day)
        price = float(sess["Close"].iloc[-1])
        done = completed(sess, now)
        last_close = float(done["Close"].iloc[-1]) if not done.empty else price
        pnl = (bid - p.entry_ask) * 100
        self.log_tick(now, price, bid, pnl)

        reason = exit_reason(p, price, last_close, bid, now.time())
        if reason:
            self.exit(now, bid, reason)

    def exit(self, now, bid, reason):
        p = self.position
        pnl = (bid - p.entry_ask) * 100
        pct = (bid / p.entry_ask - 1) * 100
        best = (p.peak_bid - p.entry_ask) * 100
        t = dict(date=str(self.day), symbol=p.symbol, kind=p.kind, strike=p.strike, expiry=p.expiry,
                 entry_riyadh=riyadh(datetime.fromisoformat(p.entry_time)), exit_riyadh=riyadh(now),
                 entry=p.entry_ask, exit=bid, pnl=round(pnl, 2), pnl_pct=round(pct, 2),
                 best_unrealised=round(best, 2), reason=reason, lock_hash=p.lock_hash)
        self.trades.append(t)
        self._append_csv("trades.csv", t)
        self.notify(
            f"🏁 <b>خروج: {p.symbol} {p.strike:g}{'C' if p.kind=='CALL' else 'P'}</b> — {reason}\n"
            f"الدخول ${p.entry_ask*100:.0f} → الخروج ${bid*100:.0f} = <b>{pnl:+.0f}$ ({pct:+.2f}%)</b>\n"
            f"أفضل ربح غير محقق كان {best:+.0f}$ • {riyadh(now)} الرياض"
        )
        self.position = None

    # ── logging
    def _append_csv(self, name, row):
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, name)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def log_tick(self, now, price, bid, pnl):
        p = self.position
        self._append_csv(f"position_log_{self.day}.csv", dict(
            riyadh=now.astimezone(RIYADH).strftime("%H:%M:%S"), symbol=p.symbol,
            stock=round(price, 2), option_bid=round(bid, 2), value=round(bid * 100, 2),
            pnl=round(pnl, 2), peak_value=round(p.peak_bid * 100, 2)))

    def summary(self):
        lines = [f"📒 <b>ملخص المحاكاة {self.day}</b>"]
        if not self.trades:
            lines.append("0 دخول — ما أجبرنا الدخول بدون شروط.")
        for t in self.trades:
            lines.append(f"• {t['symbol']} {t['strike']:g}{t['kind'][0]}: {t['pnl']:+.0f}$ ({t['pnl_pct']:+.2f}%) — {t['reason']}")
        if self.rejected:
            lines.append(f"إشارات مرفوضة: {len(self.rejected)}")
            for r in self.rejected[:10]:
                lines.append(f"  {r[0]} {r[1]} {r[2]} — {r[3]}")
        for r in self.rejected:
            self._append_csv("rejected.csv", dict(date=str(self.day), riyadh=r[0], symbol=r[1], bias=r[2], reason=r[3]))
        self.notify("\n".join(lines))

    # ── main loop
    def run(self):
        pv = self.provider
        start = datetime.combine(pv.now().date(), dtime(9, 25), NY)
        while pv.now() < start:
            pv.sleep(min(60, (start - pv.now()).total_seconds()))
        if pv.now().weekday() >= 5:
            print("weekend"); return
        # a late/duplicate scheduled run must not spam; manual runs (FORCE_RUN=1) still go
        if pv.now().time() >= _hm(SIGNAL_END_ET) and os.getenv("FORCE_RUN") != "1":
            print("started after the signal window — nothing to do"); return
        self.rank()
        if not self.candidates:
            return
        sig_end, exit_t = _hm(SIGNAL_END_ET), _hm(EXIT_ET)
        while True:
            now = pv.now()
            try:
                if self.position:
                    self.manage(now)
                elif dtime(9, 30) <= now.time() < sig_end and len(self.trades) < MAX_TRADES:
                    self.scan(now)
            except Exception as e:
                print(f"{now:%H:%M} error: {e}")
            if not self.position and (now.time() >= sig_end or len(self.trades) >= MAX_TRADES):
                break
            if now.time() >= exit_t and not self.position:
                break
            if now.time() >= dtime(15, 55):
                if self.position:
                    self.exit(now, self.position.last_bid, "⏰ نهاية الجلسة")
                break
            pv.sleep(POLL_SECONDS)
        self.summary()



def exit_reason(p, price, last_close, bid, now_t):
    """Exit rules, in priority order. Returns an Arabic reason or None."""
    floor = None
    if p.peak_bid >= p.entry_ask * (1 + PROTECT_ARM_PCT):
        floor = p.entry_ask + PROTECT_LOCK_PCT * (p.peak_bid - p.entry_ask)
    if (p.kind == "CALL" and price >= p.target) or (p.kind == "PUT" and price <= p.target):
        return f"🎯 السهم وصل الهدف {p.target}"
    if (p.kind == "CALL" and last_close < p.invalidation) or (p.kind == "PUT" and last_close > p.invalidation):
        return f"❌ إغلاق 5 دقائق تجاوز مستوى الإبطال {p.invalidation}"
    if bid <= p.entry_ask * (1 - HARD_STOP_PCT):
        return f"🛑 وقف العقد −{HARD_STOP_PCT:.0%}"
    if floor is not None and bid <= floor:
        return f"🛡️ حماية الربح: أفضل قيمة ${p.peak_bid*100:.0f}، ثبتنا ${floor*100:.0f}"
    if now_t >= _hm(EXIT_ET):
        return "⏰ وقت الخروج المحدد"
    return None

# ───────────────────────── telegram ─────────────────────────
def telegram(text: str):
    print(text.replace("<b>", "").replace("</b>", ""), "\n")
    token = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML",
                                   "disable_web_page_preview": "true"}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15)
    except Exception as e:
        print("telegram error:", e)


def load_watchlist():
    if os.path.exists("watchlist.txt"):
        syms = [l.strip().upper() for l in open("watchlist.txt", encoding="utf-8")
                if l.strip() and not l.startswith("#")]
        if syms:
            return syms
    return list(DEFAULT_WATCHLIST)


# ───────────────────────── self-test (synthetic day, no network) ─────────────────────────
class FakeProvider:
    """A synthetic day: INTC gaps down and breaks the OR low with QQQ under VWAP,
    rallies against us first, then drops (like the clip); NVDA gaps up but gets
    rejected because QQQ is below VWAP."""

    def __init__(self, day=date(2026, 9, 28)):
        import random
        self.rng = random.Random(7)
        self.day = day
        self.t = datetime.combine(day, dtime(9, 20), NY)
        self.paths = {}
        specs = {"INTC": (120.0, -0.025, "intc"), "NVDA": (226.0, 0.03, "flat"),
                 "QQQ": (745.0, -0.004, "down"), "XOM": (118.0, 0.004, "flat"),
                 "AAPL": (255.0, 0.001, "flat")}
        for s, (pc, gap, shape) in specs.items():
            self.paths[s] = (pc, self._path(pc * (1 + gap), shape))

    def _path(self, open_px, shape):
        mins = {}
        px = open_px
        t = datetime.combine(self.day, dtime(4, 0), NY)
        end = datetime.combine(self.day, dtime(16, 0), NY)
        while t < end:
            m = (t - datetime.combine(self.day, dtime(9, 30), NY)).total_seconds() / 60
            drift = 0.0
            if shape == "intc" and m >= 0:
                if m < 10: drift = -0.0006
                elif m < 40: drift = +0.0004
                elif m < 150: drift = -0.00045
                else: drift = +0.00025
            elif shape == "down" and m >= 0:
                drift = -0.00012
            px *= 1 + drift + self.rng.gauss(0, 0.0004 if m >= 0 else 0.0001)
            mins[t] = px
            t += timedelta(minutes=1)
        return mins

    def now(self):
        return self.t

    def sleep(self, sec):
        self.t += timedelta(seconds=max(sec, 1))

    def bars(self, s):
        pc, mins = self.paths[s]
        rows = []
        days = [self.day - timedelta(days=k) for k in (3, 2, 1)]
        for d in days:   # some history for pre-market volume baseline
            for h in range(4, 9):
                rows.append((datetime.combine(d, dtime(h, 0), NY), pc, pc, pc, pc, 2000))
        bucket = {}
        for t, p in mins.items():
            if t > self.t:
                break
            b = t - timedelta(minutes=t.minute % 5)
            o, h, l, c, v = bucket.get(b, (p, p, p, p, 0))
            bucket[b] = (o, max(h, p), min(l, p), p, v + (5000 if t.time() >= dtime(9, 30) else 800))
        rows += [(b, *v) for b, v in bucket.items()]
        df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume"]).set_index("t")
        return df.sort_index()

    def prev_close(self, s):
        return self.paths[s][0]

    def expiries(self, s):
        return [str(self.day + timedelta(days=4)), str(self.day + timedelta(days=11))]

    def chain(self, s, exp, kind):
        S = float(self.bars(s)["Close"].iloc[-1])
        exp_dt = datetime.combine(date.fromisoformat(exp), dtime(16, 0), NY)
        T = (exp_dt - self.t).total_seconds() / (365 * 24 * 3600)
        rows = []
        step = 1.0 if S < 200 else 2.5
        k0 = round(S / step) * step
        for i in range(-8, 9):
            K = k0 + i * step
            iv = 0.55
            mid = bs_price(S, K, T, iv, RISK_FREE, kind)
            half = 0.025 if 1 < mid < 5 else 0.10
            rows.append(dict(contractSymbol=f"{s}{exp}{kind[0]}{K}", strike=K,
                             bid=round(max(mid - half, 0.01), 2), ask=round(mid + half, 2),
                             impliedVolatility=iv, volume=100 + abs(i) * 10))
        return pd.DataFrame(rows)


def selftest():
    global LOG_DIR
    import tempfile
    LOG_DIR = tempfile.mkdtemp()
    fp = FakeProvider()
    sim = Sim(provider=fp, notify=telegram, watchlist=["INTC", "NVDA", "XOM", "AAPL"])
    sim.run()
    assert sim.trades, "expected at least one simulated trade"
    t = sim.trades[0]
    assert t["symbol"] == "INTC" and t["kind"] == "PUT", t
    assert ASK_MIN <= t["entry"] <= ASK_MAX
    assert any(r[1] == "NVDA" for r in sim.rejected), "NVDA call should be rejected by QQQ filter"
    # profit protection with the clip's numbers: entry $320, peak $410, bid fades
    p = Position("INTC", "PUT", "2026-10-02", 116, 3.20, -0.38, 0.05, "", 117.01,
                 117.67, 120.0, 113.0, peak_bid=4.10)
    assert exit_reason(p, 115.0, 115.0, 3.80, dtime(12, 0)) is None          # still above floor
    r = exit_reason(p, 115.2, 115.2, 3.65, dtime(12, 0))                      # floor = 3.20+0.45 = 3.65
    assert r and "حماية" in r, r
    print("protection check: exit at $365 (+$45) instead of the clip's +$25 →", r)
    print("SELFTEST OK →", LOG_DIR, os.listdir(LOG_DIR))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        Sim(provider=YFProvider(), notify=telegram, watchlist=load_watchlist()).run()
