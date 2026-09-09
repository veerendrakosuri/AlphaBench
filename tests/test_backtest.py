from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphabench.evaluation.backtest import cost_sensitivity, run_backtest
from alphabench.evaluation.robustness import (
    block_bootstrap_auc,
    block_bootstrap_sharpe,
    by_ticker,
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


def test_by_ticker_breaks_out_each_symbol(random_signal_backtest_df):
    trades = run_backtest(random_signal_backtest_df)["trades"]
    result = by_ticker(trades)
    assert set(result["symbol"]) == set(random_signal_backtest_df["symbol"].unique())
    assert result["n_days"].sum() == len(trades)
    # sorted descending by sharpe
    assert result["sharpe"].is_monotonic_decreasing


def test_block_bootstrap_auc_ci_brackets_point_estimate():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 2, n)
    proba = np.clip(y * 0.1 + rng.normal(0.5, 0.15, n), 0, 1)  # weak real signal
    dates = pd.bdate_range("2022-01-01", periods=n)
    result = block_bootstrap_auc(y, proba, pd.Series(dates), block=21, n_boot=500, seed=1)
    assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]
    assert result["n_boot_used"] > 0


def test_run_backtest_horizon_gt_1_drops_overlapping_windows():
    """Regression test: an h=5 run once produced an annualised Sharpe above 3 and a
    >700% single-year return because consecutive rows' fwd_ret_5d windows overlap by
    4 of 5 days, and compounding them daily multiplies the same underlying days'
    moves together several times over. At horizon=5, run_backtest must restrict the
    panel to one shared calendar date every 5 trading days (dates[0], dates[5], ...,
    the same dates for every symbol) before compounding, and annualise on 252/5
    periods/year rather than 252."""
    dates = pd.bdate_range("2021-01-04", periods=10)
    aaa_rets = [0.10, 0.99, 0.99, 0.99, 0.99, 0.20, 0.99, 0.99, 0.99, 0.99]
    bbb_rets = [0.05, 0.99, 0.99, 0.99, 0.99, 0.15, 0.99, 0.99, 0.99, 0.99]
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["AAA"] * 10 + ["BBB"] * 10,
            "proba": [0.60] * 20,  # always above threshold -> position=1 throughout
            "fwd_ret_5d": [*aaa_rets, *bbb_rets],
        }
    )

    result = run_backtest(
        df,
        proba_col="proba",
        threshold=0.55,
        commission_bps=0,
        slippage_bps=0,
        fwd_ret_col="fwd_ret_5d",
        horizon=5,
    )
    trades = result["trades"]

    # Only dates[0] and dates[5] survive -> 2 rows per symbol, the huge 0.99 filler
    # values on every other day must never be counted.
    assert len(trades) == 4
    aaa = trades[trades["symbol"] == "AAA"].sort_values("date")
    assert aaa["date"].tolist() == [dates[0], dates[5]]
    assert aaa["net_ret"].tolist() == pytest.approx([0.10, 0.20])

    assert result["periods_per_year"] == pytest.approx(252 / 5)

    # Portfolio-level net return per surviving period is the equal-weight mean across
    # symbols: (0.10+0.05)/2, (0.20+0.15)/2. ann_return must compound exactly these two
    # points at 252/5 periods/year -- not the 10-row overlapping series at 252.
    expected_net = np.array([0.075, 0.175])
    expected_ann_return = (1 + expected_net).prod() ** ((252 / 5) / 2) - 1
    assert result["metrics"]["ann_return"] == pytest.approx(expected_ann_return)


def test_run_backtest_horizon_gt_1_synchronises_rebalance_dates_across_symbols():
    """Regression test for a second, subtler version of the same bug: downsampling
    each symbol independently (e.g. every 5th row of ITS OWN history) does not
    synchronise across symbols that start on different dates or have different rows
    missing -- each symbol then lands on a different subset of calendar dates, and the
    portfolio-level (date-grouped) series stays almost daily even though every
    individual symbol's own series is correctly 5x sparser. The whole panel must share
    one rebalance-date schedule, computed from the full set of calendar dates."""
    all_dates = pd.bdate_range("2021-01-04", periods=10)
    # BBB is missing its first row, so a naive per-symbol cumcount() would put BBB's
    # kept rows on a different phase (offset by 1) than AAA's.
    aaa = pd.DataFrame({"date": all_dates, "symbol": "AAA", "proba": 0.60, "fwd_ret_5d": 0.01})
    bbb = pd.DataFrame({"date": all_dates[1:], "symbol": "BBB", "proba": 0.60, "fwd_ret_5d": 0.01})
    df = pd.concat([aaa, bbb], ignore_index=True)

    result = run_backtest(
        df,
        proba_col="proba",
        threshold=0.55,
        commission_bps=0,
        slippage_bps=0,
        fwd_ret_col="fwd_ret_5d",
        horizon=5,
    )
    # 10 shared calendar dates / horizon 5 -> exactly 2 rebalance dates, for both
    # symbols -- not close to 10, which is what the unsynchronised per-symbol version
    # of this fix produced.
    assert result["trades"]["date"].nunique() == 2
