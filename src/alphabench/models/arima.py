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


def run_diagnostics_report(panel: pd.DataFrame, symbols: list[str]) -> dict:
    """ADF/KPSS on every symbol's own daily log-return series (never on price levels —
    the whole point of section 4.1 is that prices are non-stationary and returns are),
    aggregated into pass rates. Returns per-symbol results plus the ACF/PACF of one
    representative symbol (the first, alphabetically, for reproducibility) for plotting.
    """
    per_symbol: dict[str, dict] = {}
    for symbol in symbols:
        close = panel.loc[panel["symbol"] == symbol].sort_values("date")["close"]
        returns = np.log(close).diff()
        if returns.dropna().shape[0] < 100:
            continue
        per_symbol[symbol] = diagnose_symbol(returns)

    n = len(per_symbol)
    adf_stationary = sum(1 for d in per_symbol.values() if d["adf"]["stationary_at_5pct"])
    kpss_stationary = sum(1 for d in per_symbol.values() if d["kpss"]["stationary_at_5pct"])
    both_agree = sum(
        1
        for d in per_symbol.values()
        if d["adf"]["stationary_at_5pct"] and d["kpss"]["stationary_at_5pct"]
    )

    rep_symbol = sorted(per_symbol)[0] if per_symbol else None
    rep_acf_pacf = None
    if rep_symbol is not None:
        close = panel.loc[panel["symbol"] == rep_symbol].sort_values("date")["close"]
        rep_acf_pacf = acf_pacf(np.log(close).diff(), nlags=20)

    return {
        "n_symbols": n,
        "adf_stationary_pct": adf_stationary / n if n else None,
        "kpss_stationary_pct": kpss_stationary / n if n else None,
        "both_agree_stationary_pct": both_agree / n if n else None,
        "representative_symbol": rep_symbol,
        "representative_acf_pacf": rep_acf_pacf,
        "per_symbol": per_symbol,
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
