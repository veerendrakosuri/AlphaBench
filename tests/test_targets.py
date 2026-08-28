from __future__ import annotations

import pandas as pd

from alphabench.targets.builder import build_targets


def test_build_targets_deadband_labels(deadband_panel):
    tgt = build_targets(deadband_panel, horizon=1, kappa=0.3, vol_window=5)
    tgt = tgt.set_index(["symbol", "date"])

    def second_last_date(sym: str) -> pd.Timestamp:
        dates = deadband_panel.loc[deadband_panel["symbol"] == sym, "date"].sort_values()
        return dates.iloc[-2]

    up_row = tgt.loc[("UP.NS", second_last_date("UP.NS"))]
    down_row = tgt.loc[("DOWN.NS", second_last_date("DOWN.NS"))]
    flat_row = tgt.loc[("FLAT.NS", second_last_date("FLAT.NS"))]

    assert up_row["sigma_1d"] == 0.0
    assert down_row["sigma_1d"] == 0.0
    assert up_row["y_dir_1d"] == 1.0
    assert down_row["y_dir_1d"] == 0.0
    assert pd.isna(flat_row["y_dir_1d"])


def test_build_targets_last_h_rows_have_nan_forward_return(deadband_panel):
    horizon = 3
    tgt = build_targets(deadband_panel, horizon=horizon, kappa=0.3, vol_window=5)
    col = f"fwd_ret_{horizon}d"

    for _sym, g in tgt.groupby("symbol"):
        g = g.sort_values("date")
        assert g[col].iloc[-horizon:].isna().all()
        assert g[col].iloc[:-horizon].notna().all()
