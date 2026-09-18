from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]


class DataCfg(BaseModel):
    start_date: str
    end_date: str | None = None
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


class TargetCfg(BaseModel):
    horizons: list[int]
    deadband_kappa: float
    vol_window: int


class ValidationCfg(BaseModel):
    train_start: str
    first_val_year: int
    last_val_year: int
    holdout_start: str
    purge_days: int
    embargo_days: int


class BacktestCfg(BaseModel):
    commission_bps: float
    slippage_bps: float
    prob_threshold: float
    allow_short: bool
    execution: str


class Config(BaseModel):
    project: dict
    data: DataCfg
    universe: dict
    target: TargetCfg
    validation: ValidationCfg
    backtest: BacktestCfg

    @property
    def seed(self) -> int:
        return int(self.project.get("seed", 42))


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(ROOT / path) as f:
        return Config(**yaml.safe_load(f))


def load_universe(path: str | Path) -> tuple[list[str], str]:
    with open(ROOT / path) as f:
        u = yaml.safe_load(f)
    return [t["symbol"] for t in u["tickers"]], u["benchmark"]


def load_model_params(model: str) -> dict:
    """Load a model's hyperparameters from config/models/<model>.yaml.

    The YAML is the single source of truth for a model's DEFAULT_PARAMS — the training
    module for `model` reads it at import time rather than hardcoding a literal dict, so
    the two cannot silently drift apart. tests/test_model_params_config.py additionally
    asserts these values equal what the already-trained model's metadata.json recorded,
    so this file documents what was actually trained rather than a live tuning knob.
    """
    with open(ROOT / "config" / "models" / f"{model}.yaml") as f:
        return yaml.safe_load(f)
