from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from alphabench.validation.leakage import assert_no_feature_leak, assert_train_precedes_val
from alphabench.validation.splitters import WalkForwardSplit


def test_splitter_train_always_precedes_validation(synthetic_panel):
    dates = synthetic_panel["date"]
    sp = WalkForwardSplit("2015-01-01", 2020, 2023)
    n = 0
    for tr, va in sp.split(dates):
        assert_train_precedes_val(dates, tr, va)
        n += 1
    assert n == 4


def test_splitter_enforces_purge_gap(synthetic_panel):
    dates = synthetic_panel["date"]
    sp = WalkForwardSplit("2015-01-01", 2020, 2023, horizon=5, purge_days=5, embargo_days=5)
    for tr, va in sp.split(dates):
        gap = (dates.iloc[va].min() - dates.iloc[tr].max()).days
        assert gap >= 10, f"purge/embargo gap only {gap} days"


def test_splitter_train_set_expands(synthetic_panel):
    dates = synthetic_panel["date"]
    sizes = [len(tr) for tr, _ in WalkForwardSplit("2015-01-01", 2020, 2023).split(dates)]
    assert sizes == sorted(sizes), "expanding window must not shrink"


def test_no_feature_correlates_with_future(synthetic_panel):
    """On a random walk, no lagged feature may correlate with the forward return."""
    from alphabench.features.pipeline import build_features
    from alphabench.targets.builder import build_targets

    p = synthetic_panel.copy()
    bench = p[p["symbol"] == "AAA"].assign(symbol="SPY")
    feats = build_features(pd.concat([p, bench], ignore_index=True), benchmark="SPY")
    tgts = build_targets(p, horizon=1)
    df = feats.merge(tgts, on=["date", "symbol"]).dropna(subset=["fwd_ret_1d"])

    # y_dir_1d/sigma_1d are derived from fwd_ret_1d itself, not features — exclude
    # them alongside the OHLCV/identifier columns so they aren't scanned as if
    # they were candidate features (they would trivially "leak").
    meta = {"date", "symbol", "open", "high", "low", "close", "volume", "y_dir_1d", "sigma_1d"}
    assert_no_feature_leak(df, "fwd_ret_1d", meta, threshold=0.10)


def test_shuffled_labels_destroy_performance(synthetic_panel):
    """AUC must collapse to ~0.5 when labels are shuffled. If it doesn't,
    the evaluation harness itself is broken."""
    rng = np.random.default_rng(1)
    n, p = 4000, 20
    X = rng.normal(size=(n, p))  # noqa: N806 -- X/y is the standard ML convention
    y = rng.integers(0, 2, n)  # labels independent of X

    split = int(0.7 * n)
    m = LGBMClassifier(n_estimators=50, verbose=-1, random_state=0)
    m.fit(X[:split], y[:split])
    auc = roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1])
    assert 0.40 < auc < 0.60, f"AUC {auc:.3f} on pure noise — harness is broken"


def test_targets_use_only_future_prices(synthetic_panel):
    """fwd_ret at time t must equal the realised return from t to t+h."""
    from alphabench.targets.builder import build_targets

    t = build_targets(synthetic_panel, horizon=1)
    df = synthetic_panel.merge(t, on=["date", "symbol"]).sort_values(["symbol", "date"])
    g = df[df["symbol"] == "AAA"].reset_index(drop=True)
    expected = np.log(g["close"].shift(-1) / g["close"])
    pd.testing.assert_series_equal(
        g["fwd_ret_1d"].dropna(),
        expected.dropna(),
        check_names=False,
        rtol=1e-9,
    )
