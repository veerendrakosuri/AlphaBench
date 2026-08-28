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
