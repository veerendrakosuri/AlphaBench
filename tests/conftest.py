from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SYMBOLS = ["AAA.NS", "BBB.NS", "CCC.NS"]
N_DAYS = 10  # 3 symbols * 10 days = 30 rows

LONG_SYMBOLS = ["AAA.NS", "BBB.NS", "CCC.NS"]
LONG_BENCHMARK = "^TEST"
N_LONG_DAYS = 320  # clears every rolling window used in the feature pipeline (252d)


def _make_clean_panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    for sym in SYMBOLS:
        price = 100.0
        for d in dates:
            o, h, lo, c = price, price * 1.01, price * 0.99, price * 1.002
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "adj_close": c,
                    "volume": 1_000_000,
                }
            )
            price = c
    return pd.DataFrame(rows)


@pytest.fixture
def clean_panel() -> pd.DataFrame:
    return _make_clean_panel()


@pytest.fixture
def dirty_panel() -> pd.DataFrame:
    """Clean panel plus one duplicated (symbol, date) row and one high < low row."""
    df = _make_clean_panel()

    # Duplicate the first row of AAA.NS.
    dup = df.iloc[[0]]
    df = pd.concat([df, dup], ignore_index=True)

    # Corrupt one row so high < low.
    bad_idx = df[df["symbol"] == "BBB.NS"].index[3]
    df.loc[bad_idx, "high"] = df.loc[bad_idx, "low"] - 1.0

    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


@pytest.fixture
def jumpy_panel() -> pd.DataFrame:
    """Clean panel with one single-day move >50% for a single symbol."""
    df = _make_clean_panel()
    idx = df[df["symbol"] == "CCC.NS"].index[5]
    df.loc[idx, ["open", "high", "low", "close", "adj_close"]] *= 3.0
    return df


def _walk(seed: int, n: int, start: float = 100.0) -> pd.DataFrame:
    """A plausible-looking daily OHLCV random walk, deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, size=n)
    close = start * np.exp(np.cumsum(rets))
    prev_close = np.empty(n)
    prev_close[0] = start
    prev_close[1:] = close[:-1]
    open_ = prev_close * (1 + rng.normal(0.0, 0.002, size=n))
    intraday = np.abs(rng.normal(0.0, 0.006, size=n))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


@pytest.fixture
def long_panel() -> pd.DataFrame:
    """320 trading days for 3 symbols plus a benchmark — clears every rolling window
    used by the feature pipeline (the longest is 252d)."""
    dates = pd.bdate_range("2015-01-01", periods=N_LONG_DAYS)
    frames = []
    for i, sym in enumerate([*LONG_SYMBOLS, LONG_BENCHMARK]):
        ohlcv = _walk(seed=100 + i, n=N_LONG_DAYS)
        ohlcv["date"] = dates
        ohlcv["symbol"] = sym
        frames.append(ohlcv)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


@pytest.fixture
def long_panel_benchmark() -> str:
    return LONG_BENCHMARK


@pytest.fixture
def deadband_panel() -> pd.DataFrame:
    """3 symbols, flat prices except one engineered forward move each, for exact
    zero-sigma deadband testing in build_targets. Each symbol's price is flat
    (zero daily return) through the second-to-last day, so the rolling-std
    volatility estimate is exactly 0 there and the deadband collapses to 0 —
    making the expected y_dir label unambiguous."""
    n = 10
    dates = pd.bdate_range("2021-01-01", periods=n)
    specs = {
        "UP.NS": np.exp(0.05),  # comfortably above a zero deadband
        "DOWN.NS": np.exp(-0.05),  # comfortably below
        "FLAT.NS": 1.0,  # exactly on the deadband boundary
    }
    rows = []
    for sym, jump_mult in specs.items():
        closes = [100.0] * (n - 1) + [100.0 * jump_mult]
        for d, c in zip(dates, closes, strict=True):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": c,
                    "high": c,
                    "low": c,
                    "close": c,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
