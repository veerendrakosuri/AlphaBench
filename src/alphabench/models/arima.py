from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.stattools import pacf as _pacf


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test. H0: the series has a unit root (non-stationary)."""
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "stationary_at_5pct": bool(pvalue < 0.05),
    }


def kpss_test(series: pd.Series) -> dict:
    """KPSS test. H0: the series is (level-)stationary — the opposite null of ADF, so
    agreement between the two (ADF rejects a unit root AND KPSS fails to reject
    stationarity) is the strongest evidence a series is actually stationary."""
    stat, pvalue, lags, crit = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "lags": int(lags),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "stationary_at_5pct": bool(pvalue > 0.05),
    }


def acf_pacf(series: pd.Series, nlags: int = 20) -> dict:
    s = series.dropna()
    return {
        "acf": [float(x) for x in acf(s, nlags=nlags, fft=True)],
        "pacf": [float(x) for x in _pacf(s, nlags=nlags)],
    }


def diagnose_symbol(returns: pd.Series) -> dict:
    """ADF + KPSS on one symbol's daily log-return series."""
    return {
        "n_obs": int(returns.dropna().shape[0]),
        "adf": adf_test(returns),
        "kpss": kpss_test(returns),
    }


def rolling_arima_forecast(
    train_returns: pd.Series,
    known_returns: pd.Series,
    val_dates: pd.DatetimeIndex,
    horizon: int,
    order: tuple[int, int, int] = (1, 0, 1),
) -> pd.Series:
    """Fit ARIMA(order) once on `train_returns`, then walk forward one day at a time
    through `val_dates`.

    Each iteration forecasts `horizon` steps ahead from the model's current state
    *before* that day's return is revealed (no look-ahead), then appends the day's
    now-known return via `.append(refit=False)` — a cheap Kalman-filter state update
    that uses the already-fitted ARIMA parameters rather than re-optimising the
    likelihood on every step, which is what makes a day-by-day walk-forward loop
    computationally feasible across 29 symbols x 7 folds.

    The cumulative h-day forecast mean is the sum of the `horizon` per-step means; its
    standard error is approximated as sqrt(sum of per-step variances), which assumes
    roughly uncorrelated step-ahead errors — a standard simplification, noted here
    because this is a reporting baseline, not the primary model.

    Returns a probability per `val_dates` entry that the cumulative forward return is
    positive: `norm.cdf(cumulative_mean / cumulative_se)`.
    """
    model = ARIMA(train_returns.to_numpy(), order=order)
    results = model.fit()

    probs: dict[pd.Timestamp, float] = {}
    for date in val_dates:
        fc = results.get_forecast(steps=horizon)
        mean = float(np.sum(fc.predicted_mean))
        se = float(np.sqrt(np.sum(np.asarray(fc.se_mean) ** 2)))
        probs[date] = float(norm.cdf(mean / se)) if se > 0 else 0.5

        if date in known_returns.index and not pd.isna(known_returns.loc[date]):
            results = results.append([known_returns.loc[date]], refit=False)

    return pd.Series(probs)
