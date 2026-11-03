"""Paper log for the fixed dip-buy rules on SPY and QQQ, for the GitHub Pages dashboard.

Buy at the close when price is above its 200-day average and RSI(2) is under 10.
Sell at the first close above the 5-day average, or after 10 trading days.
"""
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
SYMBOLS = ["SPY", "QQQ"]
HISTORY_START = "2019-01-01"
COST = 0.00007
DATA_FILE = Path(os.getenv("DATA_FILE", "docs/dip/data/trades.json"))


def daily_bars(symbol):
    import yfinance as yf
    frame = yf.download(symbol, start="2017-01-01", auto_adjust=True, progress=False)
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.rename(columns=str.lower)[["close"]]


def rsi(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_trades(symbol, frame, forward_start):
    close = frame["close"]
    entries = ((close > close.rolling(200).mean()) & (rsi(close, 2) < 10)).to_numpy()
    exits = (close > close.rolling(5).mean()).to_numpy()
    prices, dates = close.to_numpy(), frame.index
    trades, index, last = [], 0, len(frame) - 1
    while index <= last:
        if not entries[index] or dates[index].strftime("%Y-%m-%d") < HISTORY_START:
            index += 1
            continue
        limit = index + 10
        exit_index = None
        for forward in range(index + 1, min(limit, last) + 1):
            if exits[forward] or forward == limit:
                exit_index = forward
                break
        entry_date = dates[index].strftime("%Y-%m-%d")
        trade = {
            "symbol": symbol,
            "entry_date": entry_date,
            "entry": round(float(prices[index]), 2),
            "kind": "forward" if entry_date >= forward_start else "backfill",
        }
        if exit_index is None:
            trade.update({"status": "open", "exit_date": None, "exit": None, "pct": round(float((prices[last] / prices[index] - 1) * 100), 2), "days": last - index})
            trades.append(trade)
            break
        trade.update({"status": "closed", "exit_date": dates[exit_index].strftime("%Y-%m-%d"), "exit": round(float(prices[exit_index]), 2), "pct": round(float((prices[exit_index] / prices[index] - 1 - COST) * 100), 2), "days": exit_index - index})
        trades.append(trade)
        index = exit_index
    return trades


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


def main():
    old = json.loads(DATA_FILE.read_text(encoding="utf-8")) if DATA_FILE.exists() else {}
    now = dt.datetime.now(NY)
    forward_start = old.get("forward_start") or (now.date() + dt.timedelta(days=1)).isoformat()
    previous = {(t["symbol"], t["entry_date"]): t for t in old.get("trades", [])}

    trades = []
    for symbol in SYMBOLS:
        frame = daily_bars(symbol)
        if frame is None:
            raise SystemExit(f"no data for {symbol}")
        if frame.index[-1].date() == now.date() and now.hour * 60 + now.minute < 970:
            frame = frame.iloc[:-1]
        trades.extend(build_trades(symbol, frame, forward_start))

    trades.sort(key=lambda trade: trade["entry_date"])
    if old:
        for trade in trades:
            if trade["kind"] != "forward":
                continue
            before = previous.get((trade["symbol"], trade["entry_date"]))
            if before is None:
                telegram(f"شراء الهبوط | إشارة ورقية {trade['symbol']}\nدخول عند إغلاق {trade['entry_date']}: {trade['entry']}")
            elif before["status"] == "open" and trade["status"] == "closed":
                telegram(f"شراء الهبوط | خروج ورقي {trade['symbol']}\n{trade['entry']} ← {trade['exit']} بعد {trade['days']} يوم\nالنتيجة {trade['pct']:+.2f}%")

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps({"forward_start": forward_start, "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "trades": trades}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("trades:", len(trades))


if __name__ == "__main__":
    main()
