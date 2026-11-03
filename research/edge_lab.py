"""Tests a short, fixed list of market effects with one rule set per idea.

Each idea is judged on a build period and an untouched holdout period. Results are saved to research/results.md.
"""
import json
import os
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
COST = 0.00007  # round-trip cost as a fraction of price
INTRADAY_SPLIT = pd.Timestamp("2024-07-01")
DAILY_SPLIT = pd.Timestamp("2019-01-01")
OUT = Path(os.getenv("OUT_DIR", "research"))


def http_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as error:
            print("request failed:", error)
            time.sleep(15 * (attempt + 1))
    return None


def polygon_5m(start, end):
    key = os.environ["POLYGON_API_KEY"].strip()
    rows, url = [], (f"https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/{start}/{end}"
                     f"?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    while url:
        payload = http_json(url)
        if not payload:
            break
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        url = f"{next_url}&apiKey={key}" if next_url else None
    frame = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    frame.index = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(NY)
    minutes = frame.index.hour * 60 + frame.index.minute
    return frame.loc[(minutes >= 570) & (minutes < 960), ["open", "high", "low", "close"]]


def yahoo_daily(symbol):
    import yfinance as yf
    frame = yf.download(symbol, start="1993-01-01", auto_adjust=True, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.rename(columns=str.lower)[["open", "high", "low", "close"]]


def stats(series):
    series = series.dropna()
    if len(series) < 2:
        return {"trades": int(len(series))}
    curve = (1 + series).cumprod()
    return {
        "trades": int(len(series)),
        "win_pct": round(float((series > 0).mean() * 100), 1),
        "avg_pct": round(float(series.mean() * 100), 4),
        "t_stat": round(float(series.mean() / series.std() * np.sqrt(len(series))), 2),
        "total_pct": round(float((curve.iloc[-1] - 1) * 100), 1),
        "max_dd_pct": round(float((curve / curve.cummax() - 1).min() * 100), 1),
    }


def split_report(name, series, split, days_held=None):
    series = series.dropna().sort_index()
    return {"name": name, "build": stats(series[series.index < split]), "holdout": stats(series[series.index >= split]), "days_in_market": days_held}


def intraday_table(bars):
    rows = {}
    for date, day in bars.groupby(bars.index.date):
        if len(day) < 70:
            continue
        minute = day.index.hour * 60 + day.index.minute
        def at(value, column):
            selected = day.loc[minute == value, column]
            return float(selected.iloc[0]) if len(selected) else np.nan
        first = day[minute < 600]
        rows[pd.Timestamp(date)] = {
            "open": at(570, "open"), "close": at(955, "close"), "c955": at(595, "close"), "c1455": at(895, "close"),
            "c1525": at(925, "close"), "o1530": at(930, "open"), "or_high": first["high"].max(), "or_low": first["low"].min(),
        }
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    table["prev_close"] = table["close"].shift(1)
    return table


def orb_returns(bars, table):
    results = {}
    for date, day in bars.groupby(bars.index.date):
        stamp = pd.Timestamp(date)
        if stamp not in table.index or pd.isna(table.at[stamp, "or_high"]):
            continue
        high, low = table.at[stamp, "or_high"], table.at[stamp, "or_low"]
        minute = day.index.hour * 60 + day.index.minute
        direction, start = 0, None
        for index in np.where((minute >= 600) & (minute < 930))[0]:
            price = day["close"].iloc[index]
            if price > high or price < low:
                direction, start = (1 if price > high else -1), index + 1
                break
        if not direction or start >= len(day):
            continue
        entry, stop = day["open"].iloc[start], (high + low) / 2
        exit_price = day["close"].iloc[-1]
        for index in range(start, len(day)):
            if (direction == 1 and day["low"].iloc[index] <= stop) or (direction == -1 and day["high"].iloc[index] >= stop):
                exit_price = stop
                break
        results[stamp] = direction * (exit_price / entry - 1) - COST
    return pd.Series(results, dtype=float)


def hold_trades(daily, entries, exit_rule, max_days):
    """Enters at the close of entry days and exits on exit_rule or after max_days. One trade at a time."""
    close = daily["close"].to_numpy()
    index, results, held = 0, {}, 0
    while index < len(daily):
        if entries.iloc[index]:
            exit_index = min(index + max_days, len(daily) - 1)
            for forward in range(index + 1, exit_index + 1):
                if exit_rule.iloc[forward]:
                    exit_index = forward
                    break
            if exit_index > index:
                results[daily.index[exit_index]] = close[exit_index] / close[index] - 1 - COST
                held += exit_index - index
            index = max(exit_index, index + 1)
            continue
        index += 1
    return pd.Series(results, dtype=float), held


def rsi(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def daily_ideas(daily, vix):
    daily = daily.copy()
    daily["sma200"] = daily["close"].rolling(200).mean()
    daily["sma5"] = daily["close"].rolling(5).mean()
    daily["rsi2"] = rsi(daily["close"], 2)
    daily["vix"] = vix["close"].reindex(daily.index).ffill()
    daily["vix10"] = daily["vix"].rolling(10).mean()
    uptrend = daily["close"] > daily["sma200"]
    reports = []

    overnight = daily["open"].shift(-1) / daily["close"] - 1 - COST
    overnight.index = daily.index
    reports.append(split_report("Overnight: every day", overnight, DAILY_SPLIT))
    reports.append(split_report("Overnight: uptrend only", overnight[uptrend], DAILY_SPLIT))

    day_session = daily["close"] / daily["open"] - 1 - COST
    reports.append(split_report("Open to close: every day", day_session, DAILY_SPLIT))

    panic, held = hold_trades(daily, (daily["rsi2"] < 10) & uptrend, daily["close"] > daily["sma5"], 10)
    reports.append(split_report("Dip buy: RSI2 under 10 in uptrend", panic, DAILY_SPLIT, held))

    spike, held = hold_trades(daily, (daily["vix"] > daily["vix10"] * 1.2) & uptrend, pd.Series(False, index=daily.index), 5)
    reports.append(split_report("VIX spike in uptrend: hold 5 days", spike, DAILY_SPLIT, held))

    month = daily.index.to_period("M")
    from_end = pd.Series(range(len(daily)), index=daily.index).groupby(month).rank(ascending=False)
    from_start = pd.Series(range(len(daily)), index=daily.index).groupby(month).rank()
    turn, held = hold_trades(daily, from_end == 2, from_start == 3, 7)
    reports.append(split_report("Turn of month: 2nd last day to 3rd day", turn, DAILY_SPLIT, held))

    hold = daily["close"].pct_change()
    reports.append(split_report("Benchmark: buy and hold, daily", hold, DAILY_SPLIT, int(hold.notna().sum())))
    return reports


def intraday_ideas(table, bars, vix):
    first = np.sign(table["c955"] / table["prev_close"] - 1)
    late = np.sign(table["c1525"] / table["c1455"] - 1)
    last = table["close"] / table["o1530"] - 1
    reports = [
        split_report("Last 30 min follows first 30 min", first * last - COST, INTRADAY_SPLIT),
        split_report("Last 30 min when first and 15:00 agree", (first * last - COST)[first == late], INTRADAY_SPLIT),
    ]
    orb = orb_returns(bars, table)
    prior_vix = vix["close"].shift(1).reindex(orb.index).ffill()
    reports.append(split_report("ORB 30 to close: every day", orb, INTRADAY_SPLIT))
    reports.append(split_report("ORB 30 to close: VIX under 20", orb[prior_vix < 20], INTRADAY_SPLIT))
    reports.append(split_report("ORB 30 to close: VIX 20 or more", orb[prior_vix >= 20], INTRADAY_SPLIT))
    return reports


def verdict(row):
    build, holdout = row["build"], row["holdout"]
    if "t_stat" not in build or "t_stat" not in holdout:
        return "little data"
    if build["t_stat"] >= 2 and holdout["t_stat"] >= 1 and holdout["avg_pct"] > 0:
        return "PASS"
    if build["avg_pct"] > 0 and holdout["avg_pct"] > 0:
        return "weak"
    return "fail"


def write_report(rows):
    lines = ["| Idea | Build trades | Build avg % | Build t | Hold trades | Hold win % | Hold avg % | Hold t | Hold total % | Max DD % | Verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        build, holdout = row["build"], row["holdout"]
        lines.append(f"| {row['name']} | {build.get('trades')} | {build.get('avg_pct')} | {build.get('t_stat')} | {holdout.get('trades')} | {holdout.get('win_pct')} | {holdout.get('avg_pct')} | {holdout.get('t_stat')} | {holdout.get('total_pct')} | {holdout.get('max_dd_pct')} | {row['verdict']} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


def main():
    daily, vix = yahoo_daily("SPY"), yahoo_daily("^VIX")
    rows = daily_ideas(daily, vix)
    bars = polygon_5m(os.getenv("INTRADAY_START", "2021-11-08"), pd.Timestamp.now().strftime("%Y-%m-%d"))
    print("5m bars:", len(bars), bars.index[0], bars.index[-1])
    rows += intraday_ideas(intraday_table(bars), bars, vix)
    for row in rows:
        row["verdict"] = verdict(row)
    write_report(rows)


if __name__ == "__main__":
    main()
