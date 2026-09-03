from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

TRADING_DAYS = 252


def block_bootstrap_auc(
    y_true: np.ndarray,
    proba: np.ndarray,
    dates: pd.Series,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for ROC-AUC using contiguous date-blocks, resampled the same
    way as `block_bootstrap_sharpe` — preserves the serial correlation an iid bootstrap
    would destroy. Used for the sealed holdout's SC-4 requirement (AUC + bootstrap CI).
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(np.asarray(dates))
    y = np.asarray(y_true)[order]
    p = np.asarray(proba)[order]
    n = len(y)

    n_blocks = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        y_s, p_s = y[idx], p[idx]
        if len(np.unique(y_s)) < 2:
            continue
        out.append(roc_auc_score(y_s, p_s))
    out_arr = np.array(out)
    obs = float(roc_auc_score(y, p))
    return {
        "auc": obs,
        "ci_lower": float(np.percentile(out_arr, 2.5)),
        "ci_upper": float(np.percentile(out_arr, 97.5)),
        "n_boot_used": len(out_arr),
    }


def by_ticker(trades: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """Per-ticker breakdown, mirroring `by_period` but grouped by symbol. Reveals a
    strategy whose entire return is carried by a handful of names rather than
    generalising across the universe (PROPOSAL section 7.3)."""
    rows = []
    for symbol, g in trades.groupby(symbol_col):
        r = g.sort_values("date")["net_ret"]
        sd = r.std()
        rows.append(
            {
                "symbol": symbol,
                "ann_return": float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
                if len(r)
                else 0.0,
                "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
                "hit_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.0,
                "n_days": len(r),
                "n_trades": int((g["position"] != g["prev_position"]).sum())
                if "position" in g and "prev_position" in g
                else int((r != 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def block_bootstrap_sharpe(
    returns: pd.Series, block: int = 21, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap CI for Sharpe using contiguous blocks, which preserves the
    autocorrelation that an iid bootstrap would destroy."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    n_blocks = int(np.ceil(len(r) / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(len(r) - block, 1), n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[: len(r)]
        sd = sample.std()
        out.append(sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
    out = np.array(out)  # type: ignore[assignment]
    obs = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
    return {
        "sharpe": float(obs),
        "ci_lower": float(np.percentile(out, 2.5)),
        "ci_upper": float(np.percentile(out, 97.5)),
        "p_gt_zero": float((out > 0).mean()),  # type: ignore[operator]
    }


EULER = 0.5772156649015329


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    *,
    periods_per_year: int = 252,
    annualized: bool = True,
    trial_sharpe_std: float | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Corrects an observed Sharpe for (a) selection bias from trying `n_trials`
    configurations and (b) non-normal returns.

    UNITS MATTER: the observed Sharpe and the null bar must be in the same
    units. Annualised Sharpes are converted to per-period internally. Getting
    this wrong makes the function return 1.0 for every input, which looks like
    it works and silently tells you every result is real.
    """
    sr = observed_sharpe / np.sqrt(periods_per_year) if annualized else observed_sharpe

    # Expected maximum Sharpe under the null of zero skill (extreme-value approx)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        v = trial_sharpe_std if trial_sharpe_std is not None else 1.0 / np.sqrt(n_obs)
        sr0 = v * (
            (1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

    denom = np.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    prob = float(stats.norm.cdf(z))

    return {
        "observed_sharpe_ann": float(observed_sharpe),
        "null_bar_ann": float(sr0 * np.sqrt(periods_per_year)),
        "deflated_sharpe_prob": prob,
        "n_trials": int(n_trials),
        "is_significant_at_95": prob > 0.95,
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Test whether two forecasts have significantly different accuracy.
    e1, e2 are squared (or absolute) forecast errors."""
    d = np.asarray(e1) - np.asarray(e2)
    n = len(d)
    dbar = d.mean()
    gamma0 = d.var(ddof=0)
    gammas = [np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)] if h > 1 else []
    var = (gamma0 + 2 * sum(gammas)) / n
    stat = dbar / np.sqrt(max(var, 1e-12))
    return {
        "dm_stat": float(stat),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))),
        "favours": "model_2" if dbar > 0 else "model_1",
    }


def by_period(trades: pd.DataFrame, freq: str = "YE") -> pd.DataFrame:
    """Per-year breakdown. Reveals a strategy whose whole return is one event."""
    daily = trades.groupby("date")["net_ret"].mean()
    g = daily.groupby(pd.Grouper(freq=freq))
    return pd.DataFrame(
        {
            "ann_return": g.apply(lambda r: (1 + r).prod() - 1),
            "sharpe": g.apply(
                lambda r: r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
            ),
            "n_days": g.size(),
        }
    )
