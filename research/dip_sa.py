"""Runs the fixed dip-buy rules on large Saudi stocks and the TASI index, with a higher trading cost."""
import json

import pandas as pd

import edge_lab
from edge_lab import DAILY_SPLIT, OUT, hold_trades, rsi, split_report, verdict, yahoo_daily

edge_lab.COST = 0.003  # estimated round-trip cost on Tadawul

SYMBOLS = {
    "^TASI.SR": "TASI index", "2222.SR": "Aramco", "1120.SR": "Al Rajhi", "1180.SR": "SNB", "2010.SR": "SABIC",
    "7010.SR": "stc", "1211.SR": "Maaden", "1150.SR": "Alinma", "1010.SR": "Riyad Bank", "2280.SR": "Almarai",
    "5110.SR": "Saudi Electricity", "2082.SR": "ACWA Power",
}


def report(symbol, name):
    daily = yahoo_daily(symbol)
    close = daily["close"]
    sma200, value = close.rolling(200).mean(), rsi(close, 2)
    entries = (close > sma200) & (value < 10)
    exits = close > close.rolling(5).mean()
    trades, held = hold_trades(daily, entries, exits, 10)
    row = split_report(name, trades, DAILY_SPLIT, held)
    row["symbol"] = symbol
    row["first_date"] = daily.index[0].strftime("%Y-%m-%d")
    row["last_date"] = daily.index[-1].strftime("%Y-%m-%d")
    row["last_close"] = round(float(close.iloc[-1]), 2)
    row["rsi2"] = round(float(value.iloc[-1]), 1)
    row["above_200"] = bool(close.iloc[-1] > sma200.iloc[-1])
    row["signal_now"] = bool(entries.iloc[-1])
    row["verdict"] = verdict(row)
    return row


def main():
    rows = []
    for symbol, name in SYMBOLS.items():
        try:
            rows.append(report(symbol, name))
        except Exception as error:
            print("failed", symbol, error)
    lines = ["| Name | Start | Build trades | Build avg % | Build t | Hold trades | Hold win % | Hold avg % | Hold t | Hold max DD % | Verdict | Above 200 | RSI2 | Signal now |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        build, holdout = row["build"], row["holdout"]
        lines.append(f"| {row['name']} | {row['first_date']} | {build.get('trades')} | {build.get('avg_pct')} | {build.get('t_stat')} | {holdout.get('trades')} | {holdout.get('win_pct')} | {holdout.get('avg_pct')} | {holdout.get('t_stat')} | {holdout.get('max_dd_pct')} | {row['verdict']} | {row['above_200']} | {row['rsi2']} | {row['signal_now']} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dip_sa_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "dip_sa_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
