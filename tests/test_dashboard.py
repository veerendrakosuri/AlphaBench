from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "alphabench" / "dashboard" / "app.py"


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _fake_httpx_get(url: str, params=None, timeout=None):
    if url.endswith("/health"):
        return _FakeResponse(
            {
                "status": "ok",
                "model_loaded": True,
                "data_last_updated": "2024-01-02",
                "n_symbols": 2,
            }
        )
    if url.endswith("/tickers"):
        return _FakeResponse({"tickers": ["AAA", "BBB"]})
    if "/predict/" in url:
        return _FakeResponse(
            {
                "symbol": "AAA",
                "as_of": "2024-01-02",
                "horizon_days": 1,
                "probability_up": 0.52,
                "signal": "FLAT",
                "confidence_band": "low",
                "model_version": "2023-12-31",
                "disclaimer": "Educational research output. Not investment advice.",
            }
        )
    if url.endswith("/backtest"):
        return _FakeResponse(
            {
                "symbol": "AAA",
                "start": "2022-01-01",
                "end": "2022-01-05",
                "metrics": {
                    "sharpe": 0.108,
                    "excess_sharpe": -1.691,
                    "ann_return": 0.01,
                    "max_drawdown": -0.29,
                    "hit_rate": 0.427,
                },
                "equity_curve": [
                    {"date": "2022-01-03", "equity": 1.0, "benchmark": 1.0},
                    {"date": "2022-01-04", "equity": 1.01, "benchmark": 1.02},
                    {"date": "2022-01-05", "equity": 1.02, "benchmark": 1.01},
                ],
                "disclaimer": "Historical simulation. Past performance does not predict future results.",
            }
        )
    if url.endswith("/metrics"):
        return _FakeResponse(
            {
                "walkforward": [
                    {"fold": 1, "val_year": 2022, "auc": 0.51},
                    {"fold": 2, "val_year": 2023, "auc": 0.52},
                ],
                "model": {},
            }
        )
    raise AssertionError(f"unexpected URL requested by dashboard: {url}")


def test_dashboard_app_renders_without_exception(monkeypatch):
    monkeypatch.setattr("httpx.get", _fake_httpx_get)

    at = AppTest.from_file(str(APP_PATH))
    at.run()

    assert not at.exception
    assert at.title[0].value == "AlphaBench"
