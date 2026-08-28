from __future__ import annotations

import numpy as np
import pandas as pd


def build_targets(
    panel: pd.DataFrame, horizon: int = 1, kappa: float = 0.3, vol_window: int = 20
) -> pd.DataFrame:
    """Forward returns + volatility-scaled deadband direction labels.

    y = 1  if fwd_ret >  kappa * sigma_t
    y = 0  if fwd_ret < -kappa * sigma_t
    y = NaN otherwise (dropped at training time — those days are noise)
    """
    out = panel[["date", "symbol", "close"]].sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)["close"]

    out[f"fwd_ret_{horizon}d"] = np.log(g.shift(-horizon) / out["close"])

    daily = np.log(g.transform(lambda s: s / s.shift(1)))
    sigma = daily.groupby(out["symbol"]).transform(lambda s: s.rolling(vol_window).std()) * np.sqrt(
        horizon
    )

    fwd = out[f"fwd_ret_{horizon}d"]
    band = kappa * sigma
    y = pd.Series(np.nan, index=out.index)
    y[fwd > band] = 1.0
    y[fwd < -band] = 0.0

    out[f"y_dir_{horizon}d"] = y
    out[f"sigma_{horizon}d"] = sigma
    return out.drop(columns=["close"])
