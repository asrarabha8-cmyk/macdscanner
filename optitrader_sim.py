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
import urllib.error
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
SPREAD_MODE = _env("SPREAD_MODE", "auto", str)     # auto: spread only when the single contract doesn't fit • off • always
SHORT_DELTA_MIN = _env("SHORT_DELTA_MIN", 0.15)
SHORT_DELTA_MAX = _env("SHORT_DELTA_MAX", 0.30)
LEG_MAX_SPREAD = _env("LEG_MAX_SPREAD", 0.15)        # bid/ask width allowed on each spread leg
SPREAD_DEBIT_MIN = _env("SPREAD_DEBIT_MIN", 0.80)
MIN_REWARD_RISK = _env("MIN_REWARD_RISK", 0.8)        # max profit must be ≥ 0.8 × cost
MAX_REWARD_RISK = _env("MAX_REWARD_RISK", 4.0)        # above this the quotes are almost surely stale
MIN_DTE = _env("MIN_DTE", 2, int)   # TRADING days to expiry (Fri→Mon = 1, so it's skipped)
MAX_DTE = _env("MAX_DTE", 7, int)
OR_MINUTES = _env("OR_MINUTES", 15, int)
MIN_RISK_PCT = _env("MIN_RISK_PCT", 0.006)   # stop at least 0.6% away from trigger
MAX_CHASE_R = _env("MAX_CHASE_R", 0.5)       # skip if price already ran > 0.5R past trigger
TARGET_R = _env("TARGET_R", 2.0)
HARD_STOP_PCT = _env("HARD_STOP_PCT", 0.45)
SPREAD_STOP_PCT = _env("SPREAD_STOP_PCT", 0.60)   # spreads swing hard; the stock invalidation is the real stop
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


AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو",
             "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]


def tdays(today: date, exp: date) -> int:
    """Trading days left until expiry (weekends excluded; exchange holidays not modelled)."""
    import numpy as np
    return int(np.busday_count(today, exp))


def exp_label(exp: str, today: date) -> str:
    d = date.fromisoformat(exp)
    n = tdays(today, d)
    days = "اليوم" if n == 0 else "يوم تداول" if n == 1 else "يومين تداول" if n == 2 else f"{n} أيام تداول"
    return f"{d.day} {AR_MONTHS[d.month-1]} ({days})"


def contract_name(sym, strike, kind, exp, short=0.0):
    d = date.fromisoformat(exp)
    k = "C" if kind == "CALL" else "P"
    if short:
        return f"{sym} {strike:g}/{short:g}{k} {d:%d/%m} (سبريد)"
    return f"{sym} {strike:g}{k} {d:%d/%m}"


def row_short(row):
    v = row.get("short_strike", 0) if hasattr(row, "get") else 0
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def implied_vol(price, S, K, T, r, kind):
    """IV from the option's own mid price (bisection) — Yahoo's IV field is often garbage."""
    intrinsic = max(0.0, (S - K) if kind == "CALL" else (K - S))
    if price <= intrinsic + 1e-4 or T <= 0:
        return float("nan")
    lo, hi = 0.01, 5.0
    if bs_price(S, K, T, hi, r, kind) < price:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, mid, r, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


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

    def last_rvol(self, s) -> float:
        """Yesterday's volume vs its 20-day average (Yahoo often reports 0 pre-market volume)."""
        d = self._tk(s).history(period="2mo", interval="1d")
        d.index = d.index.tz_convert(NY)
        d = d[d.index.date < self.now().date()]
        if len(d) < 5:
            return 1.0
        avg = float(d["Volume"].iloc[-21:-1].mean())
        return float(d["Volume"].iloc[-1]) / avg if avg > 0 else 1.0

    def expiries(self, s) -> list[str]:
        return list(self._tk(s).options)

    def chain(self, s, exp, kind) -> pd.DataFrame:
        oc = self._tk(s).option_chain(exp)
        return (oc.calls if kind == "CALL" else oc.puts).copy()



# ───────────────────────── Polygon / Massive provider (falls back to Yahoo) ─────────────────────────
class PolygonProvider:
    """Uses Polygon/Massive when the key's plan allows it; any endpoint that is not
    in the plan (403) or rate-limited (429) silently falls back to Yahoo."""

    def __init__(self, key, base=None, fallback=None):
        self.key = key
        self.base = (base or os.getenv("POLYGON_BASE_URL") or "https://api.polygon.io").rstrip("/")
        self.fb = fallback
        self.blocked = set()          # endpoint groups the plan does not include
        self.used = {}                # group -> "polygon"/"yahoo" (for the daily report)

    # plumbing
    def now(self):
        return datetime.now(NY)

    def sleep(self, sec):
        time.sleep(sec)

    def _get(self, path_or_url, params=None):
        url = path_or_url if path_or_url.startswith("http") else self.base + path_or_url
        q = dict(params or {})
        q["apiKey"] = self.key
        sep = "&" if "?" in url else "?"
        req = urllib.request.Request(url + sep + urllib.parse.urlencode(q),
                                     headers={"User-Agent": "optitrader-sim"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())

    def _try(self, group, fn, fb_fn):
        if group not in self.blocked:
            try:
                out = fn()
                self.used[group] = "Polygon"
                return out
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    self.blocked.add(group)
                    print(f"polygon: '{group}' not in plan ({e.code}) → Yahoo")
                else:
                    print(f"polygon {group} HTTP {e.code} → Yahoo this time")
            except Exception as e:
                print(f"polygon {group} error {e} → Yahoo this time")
        if self.fb is None:
            raise RuntimeError(f"no data for {group}")
        self.used[group] = "Yahoo"
        return fb_fn()

    def _aggs(self, s, mult, span, start, end):
        out, url = [], f"/v2/aggs/ticker/{s}/range/{mult}/{span}/{start}/{end}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        while url:
            j = self._get(url, params)
            out += j.get("results") or []
            url, params = j.get("next_url"), None
        if not out:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(out)
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(NY)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # data API used by Sim
    def bars(self, s):
        end = self.now().date()
        start = end - timedelta(days=7)
        return self._try("stock bars", lambda: self._aggs(s, 5, "minute", start, end),
                         lambda: self.fb.bars(s))

    def _daily(self, s, days):
        end = self.now().date() - timedelta(days=1)
        d = self._aggs(s, 1, "day", end - timedelta(days=days), end)
        return d[d.index.date < self.now().date()]

    def prev_close(self, s):
        return self._try("daily", lambda: float(self._daily(s, 10)["Close"].iloc[-1]),
                         lambda: self.fb.prev_close(s))

    def last_rvol(self, s):
        def poly():
            d = self._daily(s, 45)
            avg = float(d["Volume"].iloc[-21:-1].mean())
            return float(d["Volume"].iloc[-1]) / avg if avg > 0 else 1.0
        return self._try("daily", poly, lambda: self.fb.last_rvol(s))

    def expiries(self, s):
        def poly():
            today = self.now().date()
            j = self._get("/v3/reference/options/contracts", {
                "underlying_ticker": s, "expired": "false", "limit": 1000,
                "expiration_date.gte": str(today),
                "expiration_date.lte": str(today + timedelta(days=MAX_DTE * 2 + 7))})
            return sorted({r["expiration_date"] for r in j.get("results", [])})
        return self._try("contracts", poly, lambda: self.fb.expiries(s))

    def chain(self, s, exp, kind):
        def poly():
            rows, url = [], f"/v3/snapshot/options/{s}"
            params = {"expiration_date": exp, "contract_type": kind.lower(), "limit": 250}
            while url:
                j = self._get(url, params)
                for r in j.get("results") or []:
                    q = r.get("last_quote") or {}
                    d = r.get("details") or {}
                    rows.append(dict(
                        contractSymbol=d.get("ticker", ""), strike=float(d.get("strike_price", 0)),
                        bid=float(q.get("bid") or 0), ask=float(q.get("ask") or 0),
                        delta=(r.get("greeks") or {}).get("delta"),
                        impliedVolatility=float(r.get("implied_volatility") or 0),
                        volume=float((r.get("day") or {}).get("volume") or 0),
                        openInterest=float(r.get("open_interest") or 0)))
                url, params = j.get("next_url"), None
            df = pd.DataFrame(rows)
            if df.empty or (df["ask"] <= 0).all():
                raise RuntimeError("snapshot has no quotes on this plan")
            return df

        def yahoo_with_poly_greeks():
            ch = self.fb.chain(s, exp, kind)
            return ch
        return self._try("options snapshot", poly, yahoo_with_poly_greeks)

    def source_note(self):
        if not self.used:
            return ""
        return " • ".join(f"{k}: {v}" for k, v in self.used.items())


def make_provider():
    key = os.getenv("POLYGON_API_KEY", "").strip()
    yahoo = None
    try:
        yahoo = YFProvider()
    except Exception as e:
        print("yfinance unavailable:", e)
    if key:
        print("data: Polygon/Massive (Yahoo as fallback)")
        return PolygonProvider(key, fallback=yahoo)
    print("data: Yahoo")
    return yahoo

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
    vol_label: str = "حجم ما قبل الافتتاح"
    last: float = 0.0
    prev_close: float = 0.0
    pm_high: float = 0.0
    pm_low: float = 0.0
    now_price: float = 0.0


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
    short_strike: float = 0.0       # >0 → debit spread (long strike / short strike)


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
                if pm_vol > 0 and avg_pm > 0:
                    ratio, vlabel = pm_vol / avg_pm, "حجم ما قبل الافتتاح"
                else:   # Yahoo gave no pre-market volume → use yesterday's relative volume
                    ratio, vlabel = self.provider.last_rvol(s), "حجم أمس"
                score = abs(gap) * (1 + math.log1p(min(ratio, 10)))
                rows.append(Candidate(s, "CALL" if gap > 0 else "PUT", gap, ratio, score, vol_label=vlabel,
                                     last=last, prev_close=pc, pm_high=float(pm["High"].max()),
                                     pm_low=float(pm["Low"].min()),
                                     now_price=float(df["Close"].iloc[-1])))
            except Exception as e:  # keep going on bad tickers
                print(f"rank {s}: {e}")
        rows.sort(key=lambda c: c.score, reverse=True)
        self.candidates = rows[:TOP_N]
        if not self.candidates:
            self.notify("⚠️ ما قدرت أرتب فرص اليوم (لا توجد بيانات ما قبل الافتتاح).")
            return
        lines = [f"🧭 <b>قبل الافتتاح — أفضل {len(self.candidates)} فرص</b>", ""]
        for i, c in enumerate(self.candidates, 1):
            arrow = "📈" if c.bias == "CALL" else "📉"
            lines.append(f"{i}. <b>{c.symbol} — {c.bias}</b> {arrow}")
            lines.append(f"   السعر {c.last:.2f} | إغلاق أمس {c.prev_close:.2f} | فجوة {c.gap_pct:+.2f}%")
            if now.time() >= dtime(9, 30) and c.now_price:
                lines.append(f"   ⏱️ السوق مفتوح — السعر الآن {c.now_price:.2f} ({(c.now_price / c.prev_close - 1) * 100:+.2f}%)")
            lines.append(f"   ما قبل الافتتاح: أعلى {c.pm_high:.2f} / أدنى {c.pm_low:.2f} | {c.vol_label} ×{c.pm_vol_ratio:.1f}")
            side = "فوق أعلى" if c.bias == "CALL" else "تحت أدنى"
            lines.append(f"   الدخول: إغلاق 5د {side} أول {OR_MINUTES} دقيقة + {'فوق' if c.bias=='CALL' else 'تحت'} VWAP + {MARKET_ETF} معه")
            try:
                exp, row, _ = self.pick_contract(c.symbol, c.bias, c.now_price or c.last, now)
            except Exception as e:
                exp, row = None, None
                print(f"preview {c.symbol}: {e}")
            self._append_csv("picks.csv", dict(
                date=str(now.date()), rank=i, symbol=c.symbol, bias=c.bias, gap_pct=round(c.gap_pct, 2),
                vol=round(c.pm_vol_ratio, 2), vol_source=c.vol_label, last=round(c.last, 2),
                prev_close=round(c.prev_close, 2), pm_high=round(c.pm_high, 2), pm_low=round(c.pm_low, 2),
                preview=(contract_name(c.symbol, float(row["strike"]), c.bias, exp, row_short(row)) if row is not None else ""),
                preview_cost=(round(float(row["ask"]) * 100) if row is not None else ""),
                note=(getattr(self, "last_miss", "") or "") if row is None else ""))
            if row is not None:
                lines.append(f"   العقد المبدئي: <b>{contract_name(c.symbol, float(row['strike']), c.bias, exp, row_short(row))}</b> — ينتهي {exp_label(exp, now.date())}")
                if row_short(row):
                    lines.append(f"   تكلفة ${row['ask']*100:.0f} • أقصى ربح ${row['max_profit']*100:.0f} • دلتا صافية {row['delta']:+.2f}")
                else:
                    lines.append(f"   Bid {row['bid']:.2f} / Ask {row['ask']:.2f} • سبريد {row['spread']:.2f} • دلتا {row['delta']:+.2f} • التكلفة ${row['ask']*100:.0f}")
            elif getattr(self, "last_miss", None):
                lines.append(f"   ⚠️ ما فيه عقد يطابق القواعد الحين: {self.last_miss}")
            else:
                lines.append("   العقد المبدئي: أسعار الأوبشن ما تحدّثت قبل الافتتاح — يتحدد عند الإشارة")
            lines.append("")
        lines.append(f"⚙️ القواعد: Ask {ASK_MIN:.2f}–{ASK_MAX:.2f} • دلتا {DELTA_MIN}–{DELTA_MAX} • سبريد ≤ {MAX_SPREAD:.2f} • ميزانية ${BUDGET:.0f} • انتهاء {MIN_DTE}–{MAX_DTE} أيام تداول")
        lines.append(f"ℹ️ العقد النهائي يتحدد لحظة الإشارة (الأسعار تتغير بعد الافتتاح) • الافتتاح {riyadh(now.replace(hour=9, minute=30))} الرياض")
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
            dte = tdays(today, date.fromisoformat(e))
            if MIN_DTE <= dte <= MAX_DTE:
                exps.append(e)
        self.last_miss = None   # why nothing matched (shown in Telegram)
        near = []
        for e in exps[:2]:
            ch = self.provider.chain(s, e, kind)
            if ch is None or ch.empty:
                continue
            exp_dt = datetime.combine(date.fromisoformat(e), dtime(16, 0), NY)
            T = max((exp_dt - now).total_seconds(), 60) / (365 * 24 * 3600)
            ch = ch[(ch["bid"] > 0) & (ch["ask"] > 0)].copy()
            ch["spread"] = (ch["ask"] - ch["bid"]).round(2)
            # our own IV from each quote's mid price, then drop quotes that don't make sense
            mid = (ch["bid"] + ch["ask"]) / 2
            ch["impliedVolatility"] = [implied_vol(m, S, k, T, RISK_FREE, kind)
                                       for m, k in zip(mid, ch["strike"])]
            near_atm = ch[(ch["strike"] / S - 1).abs() < 0.10]["impliedVolatility"].dropna()
            med = float(near_atm.median()) if not near_atm.empty else float("nan")
            bad = ch["impliedVolatility"].isna()
            if med == med:   # not NaN
                bad |= (ch["impliedVolatility"] < 0.4 * med) | (ch["impliedVolatility"] > 2.5 * med)
            if bad.any():
                print(f"{s} {e}: dropped {int(bad.sum())} stale/broken quotes (ATM IV {med:.0%})")
            ch = ch[~bad].copy()
            if ch.empty:
                continue
            bs = [bs_delta(S, k, T, iv, RISK_FREE, kind)
                  for k, iv in zip(ch["strike"], ch["impliedVolatility"])]
            if "delta" in ch.columns:   # real greeks from Polygon; ours only where missing
                ch["delta"] = pd.to_numeric(ch["delta"], errors="coerce").fillna(pd.Series(bs, index=ch.index))
            else:
                ch["delta"] = bs
            ch["absd"] = ch["delta"].abs()
            ok = ch[(ch["ask"].between(ASK_MIN, ASK_MAX)) &
                    (ch["absd"].between(DELTA_MIN, DELTA_MAX)) &
                    (ch["spread"] <= MAX_SPREAD + 1e-9) &
                    (ch["ask"] * 100 <= BUDGET)]
            if SPREAD_MODE == "always":
                ok = ok.iloc[0:0]
            if ok.empty and SPREAD_MODE in ("auto", "always"):
                sp = self._debit_spread(ch, kind)
                if sp is not None:
                    return e, sp, ch
            if ok.empty:
                # closest contract by delta, to explain what failed
                dz = ch[ch["absd"].notna()].copy()
                if not dz.empty:
                    dz["gap"] = (dz["absd"] - 0.40).abs()
                    r = dz.sort_values("gap").iloc[0]
                    near.append((e, r))
                continue
            ok = ok.assign(dd=(ok["absd"] - 0.40).abs(),
                           vol=ok.get("volume", pd.Series(0, index=ok.index)).fillna(0))
            best = ok.sort_values(["spread", "dd", "vol"], ascending=[True, True, False]).iloc[0]
            return e, best, ch
        if not exps:
            self.last_miss = f"ما فيه انتهاء بين {MIN_DTE} و{MAX_DTE} أيام تداول"
        elif near:
            e, r = near[0]
            why = []
            if not (ASK_MIN <= r["ask"] <= ASK_MAX):
                why.append(f"Ask {r['ask']:.2f} {'أغلى' if r['ask'] > ASK_MAX else 'أرخص'} من الحد")
            if r["spread"] > MAX_SPREAD + 1e-9:
                why.append(f"سبريد {r['spread']:.2f} أوسع من {MAX_SPREAD:.2f}")
            if r["ask"] * 100 > BUDGET:
                why.append(f"التكلفة ${r['ask']*100:.0f} فوق الميزانية")
            self.last_miss = (f"أقرب عقد {contract_name(s, float(r['strike']), kind, e)} — "
                              f"Ask {r['ask']:.2f} • دلتا {r['delta']:+.2f} • سبريد {r['spread']:.2f}"
                              + (f" ← {'، '.join(why)}" if why else ""))
        return None, None, None

    def _debit_spread(self, ch, kind):
        """Buy the ~0.40-delta leg, sell a further OTM ~0.20-delta leg, same expiry.
        Prices are conservative: pay long Ask, receive short Bid."""
        longs = ch[ch["absd"].between(DELTA_MIN, DELTA_MAX) & (ch["spread"] <= LEG_MAX_SPREAD + 1e-9)]
        shorts = ch[ch["absd"].between(SHORT_DELTA_MIN, SHORT_DELTA_MAX) & (ch["spread"] <= LEG_MAX_SPREAD + 1e-9)]
        best = None
        for _, L in longs.iterrows():
            for _, S_ in shorts.iterrows():
                further = S_["strike"] > L["strike"] if kind == "CALL" else S_["strike"] < L["strike"]
                if not further:
                    continue
                if L["bid"] < S_["ask"]:      # the nearer leg must always be worth more
                    continue
                debit = round(float(L["ask"] - S_["bid"]), 2)
                width = abs(float(S_["strike"] - L["strike"]))
                if not (SPREAD_DEBIT_MIN <= debit <= ASK_MAX) or debit * 100 > BUDGET or debit >= width:
                    continue
                rr = (width - debit) / debit
                if not (MIN_REWARD_RISK <= rr <= MAX_REWARD_RISK):
                    continue
                key = (abs(abs(L["delta"]) - 0.40), -rr)
                if best is None or key < best[0]:
                    exit_val = round(float(L["bid"] - S_["ask"]), 2)
                    best = (key, pd.Series(dict(
                        strike=float(L["strike"]), short_strike=float(S_["strike"]),
                        ask=debit, bid=exit_val, spread=round(debit - exit_val, 2),
                        delta=float(L["delta"] - S_["delta"]), long_delta=float(L["delta"]),
                        impliedVolatility=float(L.get("impliedVolatility", float("nan"))),
                        width=width, max_profit=round(width - debit, 2),
                        contractSymbol=f"{L.get('contractSymbol','')} / {S_.get('contractSymbol','')}")))
        return best[1] if best else None

    # ── 5. entry + lock
    def enter(self, c, lv, trigger, inval, target, why, now):
        exp, row, ch = self.pick_contract(c.symbol, c.bias, lv["close"], now)
        if row is None:
            self.rejected.append((riyadh(now), c.symbol, c.bias, "لا يوجد عقد يطابق الشروط"))
            if sum(1 for r in self.rejected if r[1] == c.symbol and "عقد" in r[3]) == 1:
                    self.notify(f"⚠️ إشارة {c.symbol} {c.bias} تحققت لكن ما فيه عقد يطابق القواعد\n{getattr(self, 'last_miss', '') or ''}")
            return
        c.traded = True
        p = Position(
            symbol=c.symbol, kind=c.bias, expiry=exp, strike=float(row["strike"]),
            entry_ask=float(row["ask"]), delta=round(float(row["delta"]), 2),
            spread=float(row["spread"]), entry_time=now.isoformat(),
            stock_entry=lv["close"], trigger=round(trigger, 2),
            invalidation=round(inval, 2), target=round(target, 2),
            peak_bid=float(row["bid"]), last_bid=float(row["bid"]),
            contract=str(row.get("contractSymbol", "")), short_strike=row_short(row),
        )
        blob = json.dumps(asdict(p), sort_keys=True, ensure_ascii=False)
        p.lock_hash = hashlib.sha256(blob.encode()).hexdigest()
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "SIM_POSITION.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(p), f, ensure_ascii=False, indent=2)
        self.position = p
        cost = p.entry_ask * 100
        letter = "C" if p.kind == "CALL" else "P"
        T = max((datetime.combine(date.fromisoformat(exp), dtime(16, 0), NY) - now).total_seconds(), 60) / (365*24*3600)
        iv = float(row.get("impliedVolatility", float("nan")))
        be = p.strike + p.entry_ask if p.kind == "CALL" else p.strike - p.entry_ask
        stop_pct = SPREAD_STOP_PCT if p.short_strike else HARD_STOP_PCT
        stop_val = p.entry_ask * (1 - stop_pct) * 100
        arm_val = p.entry_ask * (1 + PROTECT_ARM_PCT) * 100
        name = contract_name(p.symbol, p.strike, p.kind, p.expiry, p.short_strike)
        if p.short_strike:
            k = "C" if p.kind == "CALL" else "P"
            maxp = float(row["max_profit"]) * 100
            be = p.strike + p.entry_ask if p.kind == "CALL" else p.strike - p.entry_ask
            body = (f"النوع: سبريد شرائي {p.kind} {'📈' if p.kind=='CALL' else '📉'}\n"
                    f"   شراء <b>{p.strike:g}{k}</b> (دلتا {float(row['long_delta']):+.2f}) + بيع <b>{p.short_strike:g}{k}</b>\n"
                    f"الانتهاء: {exp_label(p.expiry, now.date())}\n"
                    f"صافي الدخول {p.entry_ask:.2f} • قيمة الخروج الحالية {float(row['bid']):.2f} • دلتا صافية {p.delta:+.2f}\n"
                    f"💵 التكلفة ${cost:.0f} = أقصى خسارة • أقصى ربح ${maxp:.0f} (لو السهم {'فوق' if p.kind=='CALL' else 'تحت'} {p.short_strike:g} عند الانتهاء)\n"
                    f"التعادل عند الانتهاء {be:.2f}\n")
        else:
            be = p.strike + p.entry_ask if p.kind == "CALL" else p.strike - p.entry_ask
            body = (f"النوع: {'شراء CALL — رهان على الصعود 📈' if p.kind=='CALL' else 'شراء PUT — رهان على النزول 📉'} • السترايك <b>{p.strike:g}</b>\n"
                    f"الانتهاء: {exp_label(p.expiry, now.date())}\n"
                    f"Bid {float(row['bid']):.2f} / Ask {p.entry_ask:.2f} • سبريد {p.spread:.2f}\n"
                    f"دلتا {p.delta:+.2f} • IV {iv*100:.0f}% • التعادل عند الانتهاء {be:.2f}\n"
                    f"💵 الدخول ${cost:.0f} (عقد واحد = 100 سهم) • أقصى خسارة ${cost:.0f}\n")
        self.notify(
            f"🎯 <b>دخول (محاكاة): {name}</b>\n" + body +
            f"\n📊 السهم {p.stock_entry:.2f}\n"
            f"   تريقر {p.trigger} | إبطال {p.invalidation} | هدف {p.target}\n"
            f"🛑 الوقف الأساسي: إغلاق 5د بعد الإبطال {p.invalidation} • وقف أخير عند قيمة ${stop_val:.0f} (−{stop_pct:.0%})\n"
            f"🛡️ حماية الربح تشتغل عند ${arm_val:.0f} (+{PROTECT_ARM_PCT:.0%}) وتثبت {PROTECT_LOCK_PCT:.0%} من أفضل ربح\n"
            f"⏰ خروج بالوقت {riyadh(datetime.combine(now.date(), _hm(EXIT_ET), NY))} الرياض\n"
            f"✅ تأكيد: {why}\n"
            f"🕒 {riyadh(now)} الرياض • 🔒 SHA-256 {p.lock_hash[:12]}…"
        )

    # ── 6. management
    def manage(self, now):
        p = self.position
        ch = self.provider.chain(p.symbol, p.expiry, p.kind)
        row = ch[ch["strike"] == p.strike]
        bid = float(row["bid"].iloc[0]) if not row.empty else 0.0
        if p.short_strike and bid > 0:      # spread exit value = long Bid − short Ask
            srow = ch[ch["strike"] == p.short_strike]
            bid = bid - float(srow["ask"].iloc[0]) if not srow.empty and float(srow["ask"].iloc[0]) > 0 else 0.0
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
        t = dict(date=str(self.day), symbol=p.symbol, kind=p.kind, strike=p.strike, short_strike=p.short_strike, expiry=p.expiry,
                 entry_riyadh=riyadh(datetime.fromisoformat(p.entry_time)), exit_riyadh=riyadh(now),
                 entry=p.entry_ask, exit=bid, pnl=round(pnl, 2), pnl_pct=round(pct, 2),
                 best_unrealised=round(best, 2), reason=reason, lock_hash=p.lock_hash)
        self.trades.append(t)
        self._append_csv("trades.csv", t)
        self.notify(
            f"🏁 <b>خروج: {contract_name(p.symbol, p.strike, p.kind, p.expiry, p.short_strike)}</b> — {reason}\n"
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
        note = getattr(self.provider, "source_note", lambda: "")()
        if note:
            lines.append(f"📡 مصدر البيانات — {note}")
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
        force = os.getenv("FORCE_RUN") == "1"
        done_flag = os.path.join(LOG_DIR, f"done_{pv.now().date()}")
        if not force and os.path.exists(done_flag):
            print("today's session already ran — exiting"); return
        if not force and pv.now() < start - timedelta(minutes=75):
            print("too early — a later scheduled run will take today's session"); return
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
        os.makedirs(LOG_DIR, exist_ok=True)
        open(done_flag, "w").write(datetime.now(NY).isoformat())



def exit_reason(p, price, last_close, bid, now_t):
    """Exit rules, in priority order. Returns an Arabic reason or None."""
    floor = None
    if p.peak_bid >= p.entry_ask * (1 + PROTECT_ARM_PCT):
        floor = p.entry_ask + PROTECT_LOCK_PCT * (p.peak_bid - p.entry_ask)
    if (p.kind == "CALL" and price >= p.target) or (p.kind == "PUT" and price <= p.target):
        return f"🎯 السهم وصل الهدف {p.target}"
    if (p.kind == "CALL" and last_close < p.invalidation) or (p.kind == "PUT" and last_close > p.invalidation):
        return f"❌ إغلاق 5 دقائق تجاوز مستوى الإبطال {p.invalidation}"
    stop_pct = SPREAD_STOP_PCT if p.short_strike else HARD_STOP_PCT
    if bid <= p.entry_ask * (1 - stop_pct):
        return f"🛑 وقف {'السبريد' if p.short_strike else 'العقد'} −{stop_pct:.0%}"
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

    def last_rvol(self, s):
        return 1.0

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
        Sim(provider=make_provider(), notify=telegram, watchlist=load_watchlist()).run()
