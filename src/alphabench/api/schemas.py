from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

DISCLAIMER = "Educational research artifact. Not investment advice. All results are historical simulations with no guarantee of future performance."


class PredictionOut(BaseModel):
    symbol: str
    as_of: date
    horizon_days: int
    probability_up: float = Field(..., ge=0.0, le=1.0)
    signal: str = Field(..., description="LONG | FLAT | SHORT")
    confidence_band: str
    model_version: str
    disclaimer: str = "Educational research output. Not investment advice."


class BacktestOut(BaseModel):
    symbol: str | None
    start: date
    end: date
    metrics: dict
    equity_curve: list[dict]
    disclaimer: str = "Historical simulation. Past performance does not predict future results."


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    data_last_updated: date | None
    n_symbols: int
    disclaimer: str = DISCLAIMER
