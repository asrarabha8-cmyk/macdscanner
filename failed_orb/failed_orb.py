"""Logs SPY "failed opening range break" paper trades for the GitHub Pages dashboard.

Same rules as the Pine script: 5-minute regular-hours bars, a 09:30-10:00 range, one signal a day,
entry on the next open, stop first when stop and target fall in one bar, exit at the last bar otherwise.
"""
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
SYMBOL = "SPY"
ATR_LEN = 14
STOP_BUFFER_ATR = 0.25
COST_PCT = 0.007
DATA_FILE = Path(os.getenv("DATA_FILE", "docs/failed_orb/data/trades.json"))


def http_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return json.load(response)
        except Exception as error:
            print("request failed:", error)
            time.sleep(15 * (attempt + 1))
    return None


def polygon_bars(start, end):
    key = os.getenv("POLYGON_API_KEY", "").strip()
    if not key:
        return None
    rows = []
    url = (f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/5/minute/{start}/{end}"
           f"?adjusted=false&sort=asc&limit=50000&apiKey={key}")
    while url:
        payload = http_json(url)
        if not payload:
            return None
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        url = f"{next_url}&apiKey={key}" if next_url else None
    if not rows:
        return None
    frame = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    frame.index = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(NY)
    return frame[["open", "high", "low", "close"]]


def yahoo_bars():
    import yfinance as yf
    frame = yf.download(SYMBOL, period="59d", interval="5m", auto_adjust=False, progress=False, prepost=False)
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame = frame.rename(columns=str.lower)[["open", "high", "low", "close"]]
    frame.index = frame.index.tz_convert(NY)
    return frame


def regular_hours(frame):
    minutes = frame.index.hour * 60 + frame.index.minute
    return frame[(minutes >= 570) & (minutes < 960) & (frame.index.dayofweek < 5)].copy()


def add_atr(frame):
    previous = frame["close"].shift(1)
    true_range = pd.concat([frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs()], axis=1).max(axis=1)
    true_range.iloc[0] = frame["high"].iloc[0] - frame["low"].iloc[0]
    values = [float("nan")] * len(frame)
    if len(frame) >= ATR_LEN:
        current = true_range.iloc[:ATR_LEN].mean()
        values[ATR_LEN - 1] = current
        for index in range(ATR_LEN, len(frame)):
            current = (current * (ATR_LEN - 1) + true_range.iloc[index]) / ATR_LEN
            values[index] = current
    frame["atr"] = values
    return frame


def simulate_day(day):
    """Returns one closed trade for the day, or None when there was no valid trade."""
    if len(day) < 2 or day.index[-1].hour * 60 + day.index[-1].minute != 955:
        return None
    minutes = day.index.hour * 60 + day.index.minute
    opening = day[minutes < 600]
    if opening.empty:
        return None
    range_high, range_low = float(opening["high"].max()), float(opening["low"].min())

    signal_index = None
    direction = 0
    stop = target = None
    for index in range(len(day) - 1):
        minute = day.index[index].hour * 60 + day.index[index].minute
        if not 600 <= minute < 930:
            continue
        bar = day.iloc[index]
        if pd.isna(bar["atr"]):
            continue
        so_far = day.iloc[: index + 1]
        if bar["low"] < range_low and bar["close"] > range_low:
            direction, stop, target = 1, float(so_far["low"].min() - bar["atr"] * STOP_BUFFER_ATR), range_high
        elif bar["high"] > range_high and bar["close"] < range_high:
            direction, stop, target = -1, float(so_far["high"].max() + bar["atr"] * STOP_BUFFER_ATR), range_low
        else:
            continue
        signal_index = index
        break
    if signal_index is None:
        return None

    entry = float(day.iloc[signal_index + 1]["open"])
    if not (direction * (entry - stop) > 0 and direction * (target - entry) > 0):
        return None
    risk = abs(entry - stop)
    cost_r = entry * COST_PCT / 100.0 / risk

    exit_price, reason, exit_index = None, None, None
    for index in range(signal_index + 1, len(day)):
        bar = day.iloc[index]
        stop_hit = bar["low"] <= stop if direction == 1 else bar["high"] >= stop
        target_hit = bar["high"] >= target if direction == 1 else bar["low"] <= target
        if stop_hit:
            exit_price, reason, exit_index = stop, "stop", index
            break
        if target_hit:
            exit_price, reason, exit_index = target, "target", index
            break
        if index == len(day) - 1:
            exit_price, reason, exit_index = float(bar["close"]), "session", index

    gross_r = -1.0 if reason == "stop" else direction * (exit_price - entry) / risk
    return {
        "date": day.index[0].strftime("%Y-%m-%d"),
        "side": "buy" if direction == 1 else "sell",
        "signal_time": day.index[signal_index].strftime("%H:%M"),
        "exit_time": day.index[exit_index].strftime("%H:%M"),
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "exit": round(float(exit_price), 2),
        "reason": reason,
        "r": round(gross_r - cost_r, 3),
    }


def process(frame, store, kind, first_day=None):
    """Adds every finished, unprocessed day from first_day on. Returns the new trades."""
    now = dt.datetime.now(NY)
    new_trades = []
    frame = add_atr(regular_hours(frame))
    for date, day in frame.groupby(frame.index.date):
        text = date.isoformat()
        if text in store["days"] or (first_day and text < first_day):
            continue
        if date == now.date() and now.hour * 60 + now.minute < 965:
            continue
        if date > now.date():
            continue
        if day.index[-1].hour * 60 + day.index[-1].minute != 955 and date != now.date():
            continue
        trade = simulate_day(day)
        store["days"][text] = "trade" if trade else "none"
        if trade:
            trade["kind"] = kind
            store["trades"].append(trade)
            new_trades.append(trade)
    return new_trades


def summary(trades):
    if not trades:
        return "لا توجد صفقات بعد"
    wins = sum(1 for trade in trades if trade["r"] > 0)
    total = sum(trade["r"] for trade in trades)
    return f"{len(trades)} صفقة | نجاح {100 * wins / len(trades):.1f}% | متوسط {total / len(trades):+.2f}R | إجمالي {total:+.1f}R"


def telegram(text):
    token = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(text)
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20)
    except Exception as error:
        print("telegram failed:", error)


def load_store():
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return {"symbol": SYMBOL, "days": {}, "trades": []}


def save_store(store):
    store["trades"].sort(key=lambda trade: trade["date"])
    store["updated"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    store = load_store()
    today = dt.datetime.now(NY).date()
    backfill_from = os.getenv("BACKFILL_FROM", "").strip()
    if backfill_from:
        frame = polygon_bars((dt.date.fromisoformat(backfill_from) - dt.timedelta(days=10)).isoformat(), today.isoformat())
        if frame is None:
            raise SystemExit("backfill needs Polygon data")
        added = process(frame, store, "backfill", backfill_from)
        store["forward_start"] = store.get("forward_start") or (today + dt.timedelta(days=1)).isoformat()
        save_store(store)
        telegram(f"كلاود العظيم: اختبار تاريخي SPY من {backfill_from}\n{summary([t for t in store['trades'] if t['kind'] == 'backfill'])}")
        print("backfill trades:", len(added))
        return

    frame = polygon_bars((today - dt.timedelta(days=12)).isoformat(), today.isoformat())
    store["source"] = "Polygon"
    if frame is None:
        frame = yahoo_bars()
        store["source"] = "Yahoo"
    if frame is None:
        raise SystemExit("no market data")
    store["forward_start"] = store.get("forward_start") or today.isoformat()
    added = process(frame, store, "forward", store["forward_start"])
    save_store(store)
    forward = [trade for trade in store["trades"] if trade["kind"] == "forward"]
    for trade in added:
        side = "شراء" if trade["side"] == "buy" else "بيع"
        reason = {"stop": "الوقف", "target": "الهدف", "session": "نهاية الجلسة"}[trade["reason"]]
        telegram(f"كلاود العظيم | صفقة ورقية SPY {trade['date']}\n{side} {trade['entry']} ← خروج {trade['exit']} ({reason})\nالنتيجة {trade['r']:+.2f}R\nالإجمالي: {summary(forward)}")
    print("forward trades added:", len(added))


if __name__ == "__main__":
    main()
