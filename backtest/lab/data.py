"""
تحميل بيانات الأسهم بشكل "عريض": جدول لكل من Open/High/Low/Close/Volume،
الصفوف = الأيام، والأعمدة = الأسهم. مع كاش محلي حتى لا نعيد التحميل كل مرة.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
FIELDS = ("Open", "High", "Low", "Close", "Volume")


def load_universe(name: str = "sp500") -> list[str]:
    path = HERE / f"universe_{name}.txt"
    return [s.strip() for s in path.read_text().split() if s.strip() and not s.startswith("#")]


def _download(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    parts = {f: [] for f in FIELDS}
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                                  group_by="column", progress=False, threads=True)
                break
            except Exception as e:  # noqa: BLE001
                print("download retry", attempt, e, flush=True)
                time.sleep(5)
        for f in FIELDS:
            block = raw[f]
            if isinstance(block, pd.Series):
                block = block.to_frame(chunk[0])
            parts[f].append(block)
        print(f"downloaded {min(k + 100, len(symbols))}/{len(symbols)}", flush=True)
    data = {f: pd.concat(parts[f], axis=1).sort_index() for f in FIELDS}
    keep = data["Close"].count() > 300                       # تاريخ كافٍ فقط
    return {f: df.loc[:, keep] for f, df in data.items()}


def get_data(universe: str = "sp500", years: int = 10, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """يرجع {Open, High, Low, Close, Volume}. الكاش صالح لنفس اليوم."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{universe}_{years}y.pkl"
    if path.exists() and not refresh:
        cached = pd.read_pickle(path)
        if cached.get("_date") == dt.date.today().isoformat():
            print(f"cache hit: {path.name}", flush=True)
            return {f: cached[f] for f in FIELDS}
    data = _download(load_universe(universe), years)
    pd.to_pickle({**data, "_date": dt.date.today().isoformat()}, path)
    return data
