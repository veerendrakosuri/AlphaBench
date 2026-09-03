from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphabench.features.pipeline import build_features
from alphabench.models.arima import adf_test, diagnose_symbol, kpss_test, rolling_arima_forecast
from alphabench.targets.builder import build_targets
from alphabench.training.train_arima import train_arima_walkforward


def test_adf_rejects_unit_root_on_iid_returns():
    """A pure iid-normal return series is stationary by construction — ADF should
    reject the unit-root null comfortably."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    res = adf_test(returns)
    assert res["stationary_at_5pct"] is True


def test_kpss_fails_to_reject_stationarity_on_iid_returns():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    res = kpss_test(returns)
    assert res["stationary_at_5pct"] is True


def test_diagnose_symbol_shape():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    out = diagnose_symbol(returns)
    assert out["n_obs"] == 500
    assert "adf" in out and "kpss" in out


def test_rolling_arima_forecast_returns_probabilities_in_range():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=300)
    train = pd.Series(rng.normal(0, 0.01, 200), index=dates[:200])
    known = pd.Series(rng.normal(0, 0.01, 100), index=dates[200:])
    val_dates = dates[200:]

    probs = rolling_arima_forecast(train, known, val_dates, horizon=1, order=(1, 0, 1))
    assert len(probs) == len(val_dates)
    assert probs.between(0.0, 1.0).all()


@dataclass
class _FakeValidationCfg:
    train_start: str
    first_val_year: int
    last_val_year: int
    holdout_start: str
    purge_days: int = 5
    embargo_days: int = 5


@dataclass
class _FakeDataCfg:
    processed_dir: str = "processed"


@dataclass
class _FakeCfg:
    validation: _FakeValidationCfg
    data: _FakeDataCfg = field(default_factory=_FakeDataCfg)


def test_train_arima_walkforward_smoke(tmp_path, monkeypatch, synthetic_panel):
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    p = synthetic_panel.copy()
    bench = p[p["symbol"] == "AAA"].assign(symbol="SPY")
    feats = build_features(pd.concat([p, bench], ignore_index=True), benchmark="SPY")
    tgts = build_targets(p, horizon=1)
    df = feats.merge(tgts, on=["date", "symbol"])

    cfg = _FakeCfg(
        _FakeValidationCfg(
            train_start="2015-01-01",
            first_val_year=2023,
            last_val_year=2024,
            holdout_start="2030-01-01",
        )
    )

    out_dir = tmp_path / "model_out"
    res = train_arima_walkforward(df, horizon=1, cfg=cfg, out_dir=out_dir)

    assert (out_dir / "metadata.json").exists()
    if len(res):
        assert res["auc"].between(0.0, 1.0).all()
        mean_auc = res["auc"].mean()
        assert 0.30 <= mean_auc <= 0.70, f"mean AUC {mean_auc:.3f} — should be noise-level"
