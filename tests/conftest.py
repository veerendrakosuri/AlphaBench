from __future__ import annotations

import pandas as pd
import pytest

SYMBOLS = ["AAA.NS", "BBB.NS", "CCC.NS"]
N_DAYS = 10  # 3 symbols * 10 days = 30 rows


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
