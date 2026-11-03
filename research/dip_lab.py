"""Runs the fixed dip-buy rules on other index funds. The rules are not changed for any symbol."""
import json

import pandas as pd

from edge_lab import DAILY_SPLIT, OUT, hold_trades, rsi, split_report, verdict, yahoo_daily

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE"]


def dip_report(symbol):
    daily = yahoo_daily(symbol)
    close = daily["close"]
    entries = (close > close.rolling(200).mean()) & (rsi(close, 2) < 10)
    exits = close > close.rolling(5).mean()
    trades, held = hold_trades(daily, entries, exits, 10)
    hold = close.pct_change()
    hold = hold[hold.index >= DAILY_SPLIT]
    row = split_report(symbol, trades, DAILY_SPLIT, held)
    row["hold_total_pct"] = round(float(((1 + hold).prod() - 1) * 100), 1)
    row["first_date"] = daily.index[0].strftime("%Y-%m-%d")
    row["verdict"] = verdict(row)
    return row


def main():
    rows = [dip_report(symbol) for symbol in SYMBOLS]
    lines = ["| Symbol | Start | Build trades | Build avg % | Build t | Hold trades | Hold win % | Hold avg % | Hold t | Hold total % | Hold max DD % | Buy and hold % | Verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        build, holdout = row["build"], row["holdout"]
        lines.append(f"| {row['name']} | {row['first_date']} | {build.get('trades')} | {build.get('avg_pct')} | {build.get('t_stat')} | {holdout.get('trades')} | {holdout.get('win_pct')} | {holdout.get('avg_pct')} | {holdout.get('t_stat')} | {holdout.get('total_pct')} | {holdout.get('max_dd_pct')} | {row['hold_total_pct']} | {row['verdict']} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dip_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "dip_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
