from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import alphabench.api.main as api_main
from alphabench.config import Config


def _make_cfg() -> Config:
    return Config(
        project={"name": "test", "seed": 0},
        data={
            "start_date": "2010-01-01",
            "end_date": None,
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
        },
        universe={"file": "config/universe_in.yaml", "benchmark": "^NSEI"},
        target={"horizons": [1], "deadband_kappa": 0.3, "vol_window": 20},
        validation={
            "train_start": "2010-01-01",
            "first_val_year": 2017,
            "last_val_year": 2023,
            "holdout_start": "2025-01-01",
            "purge_days": 5,
            "embargo_days": 5,
        },
        backtest={
            "commission_bps": 5.0,
            "slippage_bps": 5.0,
            "prob_threshold": 0.55,
            "allow_short": False,
            "execution": "next_open",
        },
    )


def _make_model() -> LogisticRegression:
    """Strongly separable on feat1 so a large-positive latest row lands well
    above 0.55 and a large-negative one lands well below 0.45."""
    x_train = pd.DataFrame({"feat1": [-10, -9, -8, 8, 9, 10], "feat2": [0, 0, 0, 0, 0, 0]})
    y_train = [0, 0, 0, 1, 1, 1]
    return LogisticRegression().fit(x_train, y_train)


def _make_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=2)
    return pd.DataFrame(
        {
            "date": [dates[0], dates[1], dates[0], dates[1]],
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "feat1": [5.0, 10.0, -5.0, -10.0],
            "feat2": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _make_oof() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=6)
    aaa = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "proba": [0.60, 0.60, 0.40, 0.70, 0.50, 0.60],
            "fwd_ret_1d": [0.01, -0.02, 0.01, 0.02, -0.01, 0.01],
        }
    )
    # BBB only spans the middle 3 dates, so filtering by symbol should
    # narrow the response's date range relative to the unfiltered backtest.
    bbb = pd.DataFrame(
        {
            "date": dates[1:4],
            "symbol": "BBB",
            "proba": [0.55, 0.55, 0.55],
            "fwd_ret_1d": [0.005, -0.005, 0.01],
        }
    )
    return pd.concat([aaa, bbb], ignore_index=True)


@pytest.fixture
def api_state(monkeypatch):
    state = {
        "cfg": _make_cfg(),
        "model": _make_model(),
        "meta": {"features": ["feat1", "feat2"], "horizon": 1, "train_end": "2023-12-21"},
        "data": _make_data(),
        "oof": _make_oof(),
    }
    monkeypatch.setattr(api_main, "STATE", state)
    yield state
    state.clear()


@pytest.fixture
def client(api_state):
    # Plain instantiation (no `with`) deliberately skips the lifespan startup
    # handler, so our monkeypatched STATE above is never overwritten.
    return TestClient(api_main.app)


def test_health_reflects_synthetic_state(client, api_state):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_symbols"] == api_state["data"]["symbol"].nunique()
    assert body["data_last_updated"] == api_state["data"]["date"].max().date().isoformat()
    assert body["disclaimer"]


def test_tickers_returns_sorted_synthetic_symbols(client, api_state):
    resp = client.get("/tickers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tickers"] == sorted(api_state["data"]["symbol"].unique().tolist())
    assert body["disclaimer"]


def test_predict_known_symbol_long_side(client):
    resp = client.get("/predict/AAA")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["probability_up"] > 0.55
    assert body["signal"] == "LONG"
    assert body["confidence_band"] == "high"
    assert body["disclaimer"]


def test_predict_known_symbol_short_side(client):
    resp = client.get("/predict/BBB")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["probability_up"] < 0.45
    assert body["signal"] == "SHORT"
    assert body["confidence_band"] == "high"


def test_predict_unknown_symbol_404(client):
    resp = client.get("/predict/UNKNOWN")
    assert resp.status_code == 404


def test_metrics_404_when_files_missing(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp = client.get("/metrics")
    assert resp.status_code == 404


def test_metrics_200_with_walkforward_and_backtest(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = tmp_path / "reports" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "walkforward_results.json").write_text('[{"fold": 1, "auc": 0.51}]')
    (metrics_dir / "backtest_results.json").write_text('{"metrics": {"sharpe": 0.1}}')

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "walkforward" in body
    assert "backtest" in body
    assert "model" in body
    assert body["disclaimer"]


def test_backtest_unfiltered_covers_full_oof_range(client, api_state):
    resp = client.get("/backtest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] is None
    assert len(body["equity_curve"]) == api_state["oof"]["date"].nunique()
    assert body["disclaimer"]


def test_backtest_symbol_filter_narrows_date_range(client):
    full = client.get("/backtest").json()
    resp = client.get("/backtest", params={"symbol": "BBB"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BBB"
    assert body["start"] >= full["start"]
    assert body["end"] <= full["end"]
    assert (body["start"], body["end"]) != (full["start"], full["end"])


def test_backtest_unknown_symbol_404(client):
    resp = client.get("/backtest", params={"symbol": "ZZZ"})
    assert resp.status_code == 404
