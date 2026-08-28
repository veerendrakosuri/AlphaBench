from __future__ import annotations

import numpy as np
import pandas as pd

from alphabench.features.base import safe_shift
from alphabench.features.pipeline import build_features


def test_safe_shift_lags_one_row_per_symbol_no_cross_leak():
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 6 + ["B"] * 6,
            "date": list(range(6)) * 2,
            "val": [10, 11, 12, 13, 14, 15, 100, 101, 102, 103, 104, 105],
        }
    )
    shifted = safe_shift(df, ["val"], by="symbol", lag=1)

    row5_a = shifted[(shifted["symbol"] == "A") & (shifted["date"] == 5)]
    row5_b = shifted[(shifted["symbol"] == "B") & (shifted["date"] == 5)]
    assert row5_a["val"].iloc[0] == 14  # original row 4 of A
    assert row5_b["val"].iloc[0] == 104  # original row 4 of B, not leaked from A

    row0_a = shifted[(shifted["symbol"] == "A") & (shifted["date"] == 0)]
    row0_b = shifted[(shifted["symbol"] == "B") & (shifted["date"] == 0)]
    assert pd.isna(row0_a["val"].iloc[0])
    assert pd.isna(row0_b["val"].iloc[0])  # not leaked from A's last row (15)


def test_build_features_mom_1d_has_no_lookahead(long_panel, long_panel_benchmark):
    feats = build_features(long_panel, long_panel_benchmark)

    sym = "AAA.NS"
    raw = long_panel[long_panel["symbol"] == sym].sort_values("date").reset_index(drop=True)
    pos = 300  # well past every rolling window, with history on both sides
    target_date = raw.loc[pos, "date"]

    expected = np.log(raw.loc[pos - 1, "close"] / raw.loc[pos - 2, "close"])
    leaked = np.log(raw.loc[pos, "close"] / raw.loc[pos - 1, "close"])

    actual = feats.loc[(feats["symbol"] == sym) & (feats["date"] == target_date), "mom_1d"].iloc[0]

    assert np.isclose(actual, expected)
    assert not np.isclose(actual, leaked)


def test_xs_rank_is_cross_sectional_not_time_series(long_panel, long_panel_benchmark):
    panel = long_panel.copy()
    stock_symbols = ["AAA.NS", "BBB.NS", "CCC.NS"]
    sym_dates = {
        sym: panel.loc[panel["symbol"] == sym, "date"].sort_values().reset_index(drop=True)
        for sym in stock_symbols
    }
    pos = 300
    target_date = sym_dates["AAA.NS"].iloc[pos]

    # Engineer a known cross-sectional ordering of 1-day returns on `target_date`:
    # AAA highest, CCC middle, BBB lowest.
    jumps = {"AAA.NS": 0.05, "CCC.NS": 0.01, "BBB.NS": -0.03}
    for sym, ret in jumps.items():
        prev_date = sym_dates[sym].iloc[pos - 1]
        prev_mask = (panel["symbol"] == sym) & (panel["date"] == prev_date)
        prev_close = panel.loc[prev_mask, "close"].iloc[0]
        cur_mask = (panel["symbol"] == sym) & (panel["date"] == target_date)
        panel.loc[cur_mask, "close"] = prev_close * np.exp(ret)

    feats = build_features(panel, long_panel_benchmark)

    next_date = sym_dates["AAA.NS"].iloc[pos + 1]
    ranks = {
        sym: feats.loc[
            (feats["symbol"] == sym) & (feats["date"] == next_date), "xs_rank_ret_1d"
        ].iloc[0]
        for sym in jumps
    }

    assert ranks["AAA.NS"] > ranks["CCC.NS"] > ranks["BBB.NS"]
    assert np.isclose(ranks["AAA.NS"], 1.0)
    assert np.isclose(ranks["BBB.NS"], 1 / 3)
    assert np.isclose(ranks["CCC.NS"], 2 / 3)
