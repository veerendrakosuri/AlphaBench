from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from alphabench.features.pipeline import build_features
from alphabench.targets.builder import build_targets
from alphabench.training.train_xgboost import train_xgboost_walkforward


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


def test_train_xgboost_walkforward_smoke(tmp_path, monkeypatch, synthetic_panel):
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
    res = train_xgboost_walkforward(
        df, horizon=1, cfg=cfg, params={"n_estimators": 20}, out_dir=out_dir
    )

    assert len(res) == 2  # 2023 and 2024
    assert (out_dir / "final.joblib").exists()
    assert (out_dir / "metadata.json").exists()
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["horizon"] == 1

    mean_auc = res["auc"].mean()
    assert 0.35 <= mean_auc <= 0.65, f"mean AUC {mean_auc:.3f} — should be noise on a random walk"

    oof_path = tmp_path / "processed" / "oof_predictions_xgboost_h1.parquet"
    assert oof_path.exists()
    oof = pd.read_parquet(oof_path)
    assert oof["proba"].between(0.0, 1.0).all()
