from __future__ import annotations

import numpy as np
import pandas as pd

from alphabench.features import technical as tech
from alphabench.features.base import safe_shift

MOM_LAGS = [1, 2, 3, 5, 10, 21, 63, 126, 252]
VOL_WINDOWS = [5, 10, 21, 63]


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, low, v = g["close"], g["high"], g["low"], g["volume"]
    logret = np.log(c).diff()
    g["ret_1d"] = logret

    # --- momentum -----------------------------------------------------------
    for k in MOM_LAGS:
        g[f"mom_{k}d"] = np.log(c / c.shift(k))
    g["mom_21d_skip5"] = np.log(c.shift(5) / c.shift(26))  # reversal-cleaned momentum

    # --- volatility ---------------------------------------------------------
    for w in VOL_WINDOWS:
        g[f"vol_{w}d"] = logret.rolling(w).std() * np.sqrt(252)
    g["vol_ratio"] = g["vol_21d"] / g["vol_21d"].rolling(252).median()
    g["vol_of_vol"] = g["vol_21d"].rolling(63).std()
    # Parkinson high-low estimator: uses intraday range, lower variance than close-to-close
    g["parkinson_21d"] = np.sqrt(
        (np.log(h / low) ** 2).rolling(21).mean() / (4 * np.log(2))
    ) * np.sqrt(252)

    # --- technical ----------------------------------------------------------
    g["rsi_14"] = tech.rsi(c, 14)
    g = g.join(tech.macd(c))
    g["atr_14"] = tech.atr(h, low, c, 14)
    g = g.join(tech.bollinger(c))
    g = g.join(tech.stochastic(h, low, c))
    g["obv_slope_20"] = tech.obv_slope(c, v, 20)
    for w in (20, 50, 200):
        g[f"px_to_sma{w}"] = c / c.rolling(w).mean() - 1.0

    # --- volume -------------------------------------------------------------
    lv = np.log1p(v)
    g["vol_z_21"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
    g["dollar_vol_z"] = np.log1p(c * v).pipe(
        lambda s: (s - s.rolling(21).mean()) / s.rolling(21).std()
    )
    g["amihud"] = (logret.abs() / (c * v).replace(0.0, np.nan)).rolling(21).mean() * 1e9

    return g


def build_features(panel: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    bench = panel.loc[panel["symbol"] == benchmark, ["date", "close"]].copy()
    bench["mkt_ret_1d"] = np.log(bench["close"]).diff()
    bench["mkt_ret_5d"] = np.log(bench["close"] / bench["close"].shift(5))
    bench["mkt_vol_21d"] = bench["mkt_ret_1d"].rolling(21).std() * np.sqrt(252)
    bench = bench.drop(columns=["close"])

    stocks = panel[panel["symbol"] != benchmark].copy()

    # Explicit concat over groups rather than groupby().apply(): apply's return
    # shape and index behaviour shifted across pandas 2.x, and reset_index(drop=True)
    # on its output can SILENTLY MISALIGN rows. Explicit is safe.
    feats = pd.concat(
        [_per_symbol(g) for _, g in stocks.groupby("symbol", sort=False)],
        ignore_index=True,
    )
    feats = feats.merge(bench, on="date", how="left")
    feats = feats.sort_values(["symbol", "date"]).reset_index(drop=True)

    # rolling 60-day beta to the benchmark
    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["ret_1d"].rolling(60).cov(g["mkt_ret_1d"])
        var = g["mkt_ret_1d"].rolling(60).var()
        return cov / var.replace(0.0, np.nan)

    # sort_index() realigns each group's Series back onto feats' own index.
    feats["beta_60d"] = pd.concat(
        [_beta(g) for _, g in feats.groupby("symbol", sort=False)]
    ).sort_index()

    # --- cross-sectional ranks: only possible because this is a panel --------
    for col in ["ret_1d", "mom_21d", "vol_21d", "vol_z_21", "rsi_14"]:
        feats[f"xs_rank_{col}"] = feats.groupby("date")[col].rank(pct=True)

    # --- calendar -----------------------------------------------------------
    d = feats["date"]
    feats["dow"] = d.dt.dayofweek
    feats["month"] = d.dt.month
    feats["is_month_end"] = d.dt.is_month_end.astype(int)
    feats["is_quarter_end"] = d.dt.is_quarter_end.astype(int)

    # --- THE CRITICAL STEP: lag everything by one day -----------------------
    meta = ["date", "symbol", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in feats.columns if c not in meta]
    feats = safe_shift(feats, feature_cols, by="symbol", lag=1)

    return feats.sort_values(["symbol", "date"]).reset_index(drop=True)
