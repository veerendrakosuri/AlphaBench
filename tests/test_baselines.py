from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from alphabench.models.baselines import b0_majority, b0_persistence, b1_logistic


def test_b0_majority_accuracy_equals_base_rate():
    df = pd.DataFrame({"y_dir_1d": [1] * 7 + [0] * 3})
    res = b0_majority(df, "y_dir_1d")
    assert res["accuracy"] == pytest.approx(0.7)
    assert res["base_rate"] == pytest.approx(0.7)


@pytest.fixture
def baseline_frame() -> pd.DataFrame:
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ret_1d": rng.normal(0, 0.01, n),
            "mom_5d": rng.normal(0, 0.02, n),
            "vol_21d": rng.uniform(0.1, 0.3, n),
            "rsi_14": rng.uniform(20, 80, n),
            "vol_z_21": rng.normal(0, 1, n),
            "y_dir_1d": [i % 2 for i in range(n)],  # alternating, so every
            # contiguous slice has both classes present
        }
    )


def test_b0_persistence_auc_in_range(baseline_frame):
    res = b0_persistence(baseline_frame, "y_dir_1d")
    assert 0.0 <= res["auc"] <= 1.0


def test_b1_logistic_auc_in_range(baseline_frame):
    train, val = baseline_frame.iloc[:40], baseline_frame.iloc[40:]
    res = b1_logistic(train, val, "y_dir_1d")
    assert 0.0 <= res["auc"] <= 1.0


def test_b1_logistic_scaler_is_fit_only_on_train(monkeypatch, baseline_frame):
    """Fit two b1_logistic calls with the same `train` but a `val` that gains
    extra rows between calls, and confirm the pipeline's fitted StandardScaler
    statistics are identical both times — proof `val` never touched .fit()."""
    captured: list[np.ndarray] = []
    original_fit = StandardScaler.fit

    def spy_fit(self, x, y=None, **kwargs):
        result = original_fit(self, x, y, **kwargs)
        captured.append(np.array(self.mean_))
        return result

    monkeypatch.setattr(StandardScaler, "fit", spy_fit)

    train = baseline_frame.iloc[:40]
    val_small = baseline_frame.iloc[40:45]
    val_large = baseline_frame.iloc[40:60]

    b1_logistic(train, val_small, "y_dir_1d")
    b1_logistic(train, val_large, "y_dir_1d")

    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], captured[1])
