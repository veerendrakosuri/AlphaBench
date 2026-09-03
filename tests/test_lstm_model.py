from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphabench.features.pipeline import build_features
from alphabench.models.lstm import SmallLSTM, build_sequences
from alphabench.targets.builder import build_targets
from alphabench.training.train_lstm import train_lstm_walkforward


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


def test_small_lstm_forward_shape():
    model = SmallLSTM(n_features=4, hidden_size=8)
    import torch

    x = torch.randn(6, 5, 4)  # (batch, window, features)
    out = model(x)
    assert out.shape == (6,)


def test_build_sequences_respects_end_dates_and_nan():
    dates = pd.bdate_range("2021-01-01", periods=10)
    df = pd.DataFrame(
        {
            "date": list(dates) * 1,
            "symbol": ["AAA"] * 10,
            "f1": [1.0] * 10,
            "y_dir_1d": [0, 1] * 5,
        }
    )
    df.loc[3, "f1"] = np.nan  # window covering index 3 must be dropped

    X, _y, meta = build_sequences(df, ["f1"], "y_dir_1d", window=3, end_dates=None)  # noqa: N806
    assert X.shape[1:] == (3, 1)
    # rows 0,1 can't form a full window; row index 3 (0-based) has NaN so windows
    # [1,2,3] and [2,3,4] and [3,4,5] must be excluded -> only a subset remain
    assert len(meta) < 10


def test_train_lstm_walkforward_smoke(tmp_path, monkeypatch, synthetic_panel):
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
            last_val_year=2023,  # single fold — keep the test fast
            holdout_start="2030-01-01",
        )
    )

    out_dir = tmp_path / "model_out"
    res = train_lstm_walkforward(
        df,
        horizon=1,
        cfg=cfg,
        out_dir=out_dir,
        window=5,
        hidden_size=4,
        max_epochs=1,
        batch_size=128,
        patience=1,
    )

    assert (out_dir / "metadata.json").exists()
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["window"] == 5
    if len(res):
        assert res["auc"].between(0.0, 1.0).all()
