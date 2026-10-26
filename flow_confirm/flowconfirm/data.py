"""جلب البيانات: سلسلة الخيارات من yfinance، والشموع من Polygon (أو yfinance كبديل)."""
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from . import config as C

NY = ZoneInfo("America/New_York")
MIN_T = 30 / (365 * 24 * 60)  # أقل زمن متبقٍ = ٣٠ دقيقة (بالسنوات)
CHAIN_COLS = ["type", "expiry", "strike", "bid", "ask", "last", "volume",
              "oi", "iv", "last_trade", "dte", "T"]


@dataclass
class Snapshot:
    ticker: str
    spot: float
    chain: pd.DataFrame      # سلسلة الخيارات
    intraday: pd.DataFrame   # شموع ٥ دقائق لآخر جلسة
    daily: pd.DataFrame      # شموع يومية
    earnings: dt.date | None
    asof: dt.datetime
    ext_price: float | None = None   # آخر سعر خارج الجلسة (بعد/قبل) إن وُجد بعد آخر جلسة
    src: str = ""                    # مصدر الشموع + عدد شموع ما بعد الجلسة (للتشخيص)


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY)


def _yf():
    import yfinance as yf
    return yf


def _expiry_close(d: dt.date) -> dt.datetime:
    return dt.datetime.combine(d, dt.time(16, 0), NY)


# ───────────────────────── الخيارات ─────────────────────────
def finalize_chain(ch: pd.DataFrame, now: dt.datetime) -> pd.DataFrame:
    """يضيف DTE والزمن T وينظف التذبذب الضمني."""
    if ch.empty:
        return pd.DataFrame(columns=CHAIN_COLS)
    ch = ch.copy()
    ch["dte"] = [(e - now.date()).days for e in ch["expiry"]]
    ch["T"] = [max((_expiry_close(e) - now).total_seconds() / (365 * 24 * 3600), MIN_T)
               for e in ch["expiry"]]
    ok = (ch["iv"] > 0.02) & (ch["iv"] < 4)
    med = ch[ok].groupby(["expiry", "type"])["iv"].median()
    overall = ch.loc[ok, "iv"].median() if ok.any() else 0.5
    fix = [med.get((e, t), overall) for e, t in zip(ch["expiry"], ch["type"])]
    ch["iv"] = np.where(ok, ch["iv"], fix)
    return ch


def fetch_chain(ticker: str, cfg: C.ModeCfg, now: dt.datetime) -> pd.DataFrame:
    t = _yf().Ticker(ticker)
    exps = []
    for s in t.options:
        d = dt.date.fromisoformat(s)
        dte = (d - now.date()).days
        if cfg.dte_min <= dte <= cfg.dte_max and _expiry_close(d) > now:
            exps.append(d)
    frames = []
    for d in exps[: cfg.max_expiries]:
        oc = t.option_chain(d.isoformat())
        for typ, df in (("C", oc.calls), ("P", oc.puts)):
            if df is None or df.empty:
                continue
            frames.append(pd.DataFrame({
                "type": typ,
                "expiry": d,
                "strike": df["strike"].astype(float),
                "bid": df["bid"].fillna(0).astype(float),
                "ask": df["ask"].fillna(0).astype(float),
                "last": df["lastPrice"].fillna(0).astype(float),
                "volume": df["volume"].fillna(0).astype(float),
                "oi": df["openInterest"].fillna(0).astype(float),
                "iv": df["impliedVolatility"].fillna(0).astype(float),
                "last_trade": pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce"),
            }))
    if not frames:
        return pd.DataFrame(columns=CHAIN_COLS)
    return finalize_chain(pd.concat(frames, ignore_index=True), now)


# ───────────────────────── الشموع ─────────────────────────
def _polygon_aggs(ticker, mult, span, start, end) -> pd.DataFrame:
    url = f"{C.POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start}/{end}"
    r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000,
                                  "apiKey": C.POLYGON_KEY}, timeout=20)
    r.raise_for_status()
    res = r.json().get("results") or []
    if not res:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(res)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["t"], unit="ms", utc=True)).tz_convert(NY)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _yf_hist(ticker, period, interval) -> pd.DataFrame:
    df = _yf().Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY)
    return df.tz_convert(NY).astype(float)


def last_session(df: pd.DataFrame) -> pd.DataFrame:
    """يبقي شموع الجلسة الرسمية لآخر يوم تداول فقط."""
    if df.empty:
        return df
    t = df.index.time
    df = df[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))]
    if df.empty:
        return df
    return df[df.index.date == df.index[-1].date()]


def extended_price(raw: pd.DataFrame, session: pd.DataFrame, need_volume: bool = True):
    """آخر سعر تداول بعد نهاية آخر جلسة رسمية (بعد الإغلاق أو قبل الافتتاح)."""
    if raw.empty or session.empty:
        return None, 0
    after = raw[raw.index > session.index[-1]]
    if need_volume:  # yfinance يعرض حجم صفر لشموع ما بعد الجلسة، فلا نفلتر هناك
        after = after[after["volume"] > 0]
    return (float(after["close"].iloc[-1]) if not after.empty else None), len(after)


def fetch_bars(ticker: str, now: dt.datetime):
    """يرجع: شموع آخر جلسة، الشموع اليومية، وسعر خارج الجلسة إن وُجد."""
    use_polygon = bool(C.POLYGON_KEY) and not ticker.startswith("^")
    note = "" if use_polygon else "|no_key"
    if use_polygon:
        try:
            raw = _polygon_aggs(ticker, 5, "minute", (now - dt.timedelta(days=7)).date(), now.date())
            daily = _polygon_aggs(ticker, 1, "day", (now - dt.timedelta(days=200)).date(), now.date())
            if not daily.empty:
                daily.index = daily.index.normalize()
                sess = last_session(raw)
                ext, n = extended_price(raw, sess)
                return sess, daily, ext, f"polygon:{n}"
            note = "|pg_empty"
        except Exception as e:  # نرجع لـ yfinance
            print(f"[{ticker}] Polygon فشل: {e} — استخدام yfinance")
            code = getattr(getattr(e, "response", None), "status_code", "")
            note = f"|pg_err:{type(e).__name__}:{code}"  # بدون نص الخطأ حتى لا يتسرب المفتاح
    try:
        raw = _yf().Ticker(ticker).history(period="5d", interval="5m", prepost=True, auto_adjust=False)
        raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize(NY)
        raw = raw.tz_convert(NY)
    except Exception:
        raw = _yf_hist(ticker, "5d", "5m")
    sess = last_session(raw)
    ext, n = extended_price(raw, sess, need_volume=False)
    return sess, _yf_hist(ticker, "1y", "1d"), ext, f"yf:{n}{note}"



def bars_since(ticker: str, start: dt.datetime) -> pd.DataFrame:
    """شموع ٥ دقائق للجلسة الرسمية من وقت معيّن حتى الآن (لمتابعة الإشارات)."""
    now = now_ny()
    raw = pd.DataFrame()
    if C.POLYGON_KEY and not ticker.startswith("^"):
        try:
            raw = _polygon_aggs(ticker, 5, "minute", start.astimezone(NY).date(), now.date())
        except Exception as e:
            print(f"[{ticker}] Polygon فشل في المتابعة: {e}")
    if raw.empty:
        days = (now.date() - start.astimezone(NY).date()).days
        raw = _yf_hist(ticker, "5d" if days < 5 else "60d", "5m")
    if raw.empty:
        return raw
    t = raw.index.time
    return raw[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))]


def market_state(now: dt.datetime) -> dict | None:
    """اتجاه السوق داخل اليوم: آخر سعر SPY مقابل VWAP الجلسة."""
    bars = fetch_bars(C.MARKET_TICKER, now)[0]
    if bars.empty or bars["volume"].sum() <= 0:
        return None
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap = float((tp * bars["volume"]).sum() / bars["volume"].sum())
    last = float(bars["close"].iloc[-1])
    return {"ticker": C.MARKET_TICKER, "last": last, "vwap": vwap, "up": last > vwap,
            "bar": bars.index[-1].isoformat()}


def fetch_earnings(ticker: str, today: dt.date) -> dt.date | None:
    try:
        cal = _yf().Ticker(ticker).calendar
        vals = []
        if isinstance(cal, dict):
            v = cal.get("Earnings Date")
            vals = v if isinstance(v, (list, tuple)) else [v]
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            vals = list(cal.loc["Earnings Date"].values)
        dates = [pd.Timestamp(x).date() for x in vals if x is not None]
        future = [d for d in dates if d >= today]
        return min(future) if future else None
    except Exception:
        return None


def get_snapshot(ticker: str, mode: str) -> Snapshot:
    cfg = C.MODES[mode]
    now = now_ny()
    intra, daily, ext, src = fetch_bars(ticker, now)
    if not intra.empty:
        spot = float(intra["close"].iloc[-1])
    elif not daily.empty:
        spot = float(daily["close"].iloc[-1])
    else:
        spot = float(_yf().Ticker(ticker).fast_info["last_price"])
    chain = fetch_chain(ticker, cfg, now)
    return Snapshot(ticker.upper(), spot, chain, intra, daily,
                    fetch_earnings(ticker, now.date()), now, ext, src)
