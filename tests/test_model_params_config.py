from __future__ import annotations

import json

from alphabench.config import ROOT, load_model_params
from alphabench.training.train import DEFAULT_PARAMS as LIGHTGBM_DEFAULT_PARAMS
from alphabench.training.train_xgboost import DEFAULT_PARAMS as XGBOOST_DEFAULT_PARAMS


def _metadata(model_dir: str) -> dict:
    with open(ROOT / "models" / model_dir / "metadata.json") as f:
        return json.load(f)


def test_lightgbm_yaml_matches_shipped_model_metadata():
    """config/models/lightgbm.yaml must equal the params actually used to train the
    shipped model — not a fresh set of "reasonable" values. Drifting apart here
    would silently misdocument what the API serves without changing its behaviour."""
    assert load_model_params("lightgbm") == _metadata("lightgbm_h1")["params"]


def test_xgboost_yaml_matches_shipped_model_metadata():
    assert load_model_params("xgboost") == _metadata("xgboost_h1")["params"]


def test_lstm_yaml_matches_shipped_model_metadata():
    yaml_params = load_model_params("lstm")
    meta = _metadata("lstm_h1")
    for key in ("features", "window", "hidden_size", "max_epochs"):
        assert yaml_params[key] == meta[key], key


def test_training_modules_actually_read_from_yaml():
    """Guards against a future edit hardcoding a literal dict back into the training
    module — DEFAULT_PARAMS must be identically the object load_model_params returns,
    not a copy that happens to match today."""
    assert load_model_params("lightgbm") == LIGHTGBM_DEFAULT_PARAMS
    assert load_model_params("xgboost") == XGBOOST_DEFAULT_PARAMS
