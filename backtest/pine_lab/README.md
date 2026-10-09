# Pine Lab

Backtest of the entry rules from Arun K Bhaskar's Pine Script collection
(https://github.com/ArunKBhaskar/PineScript, Mozilla Public License 2.0), ported to Python
with the scripts' default settings, on S&P 500 daily bars (10 years).

Every strategy uses the same exits (5/10/20-bar holds and a 1.5×/3× ATR bracket) and is
compared with the same-day average return of all S&P 500 stocks. Results land in `results/`.
