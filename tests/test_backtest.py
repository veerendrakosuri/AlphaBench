from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphabench.evaluation.backtest import cost_sensitivity, run_backtest
from alphabench.evaluation.robustness import (
    block_bootstrap_sharpe,
    deflated_sharpe,
    diebold_mariano,
)
from alphabench.targets.builder import build_targets


@pytest.fixture
def toy_backtest_df() -> pd.DataFrame:
    """5 days, 1 symbol, hand-computable turnover/cost/net-return sequence.
    threshold=0.55, commission=slippage=5bps (cost_rate = 0.001 per unit turnover).

    day  proba  fwd_ret  ->  position  prev  turnover  gross    cost    net
    1    0.60   0.02          1.0      0.0   1.0       0.02     0.001   0.019
    2    0.60  -0.01          1.0      1.0   0.0      -0.01     0.000  -0.010
    3    0.40   0.03          0.0      1.0   1.0       0.00     0.001  -0.001
    4    0.70   0.01          1.0      0.0   1.0       0.01     0.001   0.009
    5    0.70  -0.02          1.0      1.0   0.0      -0.02     0.000  -0.020
    """
    dates = pd.bdate_range("2021-01-04", periods=5)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * 5,
            "proba": [0.60, 0.60, 0.40, 0.70, 0.70],
            "fwd_ret_1d": [0.02, -0.01, 0.03, 0.01, -0.02],
        }
    )


def test_run_backtest_hand_computed_toy_case(toy_backtest_df):
    result = run_backtest(
        toy_backtest_df,
        proba_col="proba",
        threshold=0.55,
        commission_bps=5.0,
        slippage_bps=5.0,
        allow_short=False,
    )
    trades = result["trades"].reset_index(drop=True)

    expected_position = [1.0, 1.0, 0.0, 1.0, 1.0]
    expected_turnover = [1.0, 0.0, 1.0, 1.0, 0.0]
    expected_gross = [0.02, -0.01, 0.0, 0.01, -0.02]
    expected_cost = [0.001, 0.0, 0.001, 0.001, 0.0]
    expected_net = [0.019, -0.01, -0.001, 0.009, -0.02]

    assert trades["position"].tolist() == pytest.approx(expected_position)
    assert trades["turnover"].tolist() == pytest.approx(expected_turnover)
    assert trades["gross_ret"].tolist() == pytest.approx(expected_gross)
    assert trades["cost"].tolist() == pytest.approx(expected_cost)
    assert trades["net_ret"].tolist() == pytest.approx(expected_net)


@pytest.fixture
def random_signal_backtest_df(synthetic_panel) -> pd.DataFrame:
    """Real fwd_ret_1d (from a random-walk panel) paired with a proba column
    that is genuinely uniform-random and by construction uncorrelated with it
    — reproduces the BUILD_PLAN §5.3 calibration setup."""
    t = build_targets(synthetic_panel, horizon=1)
    df = t.dropna(subset=["fwd_ret_1d"]).copy()
    rng = np.random.default_rng(7)
    df["proba"] = rng.uniform(size=len(df))
    return df[["date", "symbol", "proba", "fwd_ret_1d"]]


def test_cost_drag_zero_cost_sharpe_is_near_zero(random_signal_backtest_df):
    m = run_backtest(random_signal_backtest_df, commission_bps=0, slippage_bps=0)["metrics"]
    assert abs(m["sharpe"]) < 0.5


def test_cost_drag_sharpe_decays_monotonically_with_bps(random_signal_backtest_df):
    cost_df = cost_sensitivity(random_signal_backtest_df, bps_levels=(0, 2, 5, 10, 20))
    sharpes = cost_df["sharpe"].tolist()
    assert all(sharpes[i] >= sharpes[i + 1] - 1e-9 for i in range(len(sharpes) - 1))


def test_deflated_sharpe_prob_strictly_decreases_with_n_trials():
    n_trials_list = [1, 10, 50, 100, 200, 1000]
    probs = [
        deflated_sharpe(observed_sharpe=1.0, n_trials=nt, n_obs=2000)["deflated_sharpe_prob"]
        for nt in n_trials_list
    ]
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))


def test_block_bootstrap_sharpe_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    result = block_bootstrap_sharpe(returns, block=21, n_boot=500, seed=1)
    assert result["ci_lower"] <= result["sharpe"] <= result["ci_upper"]


def test_diebold_mariano_smoke():
    rng = np.random.default_rng(4)
    e1 = rng.normal(0.5, 0.1, 200) ** 2
    e2 = rng.normal(0.4, 0.1, 200) ** 2
    result = diebold_mariano(e1, e2)
    assert set(result.keys()) == {"dm_stat", "p_value", "favours"}
    assert np.isfinite(result["dm_stat"])
    assert np.isfinite(result["p_value"])
    assert result["favours"] in {"model_1", "model_2"}
