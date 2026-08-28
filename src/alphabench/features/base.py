from __future__ import annotations

from collections.abc import Callable

import pandas as pd

FEATURE_REGISTRY: dict[str, Callable] = {}


def feature(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        FEATURE_REGISTRY[name] = fn
        return fn

    return deco


def safe_shift(df: pd.DataFrame, cols: list[str], by: str = "symbol", lag: int = 1) -> pd.DataFrame:
    """Lag feature columns by `lag` within each symbol.

    Applied to EVERY feature before it reaches a model. A feature computed from
    the close of day t is only usable for a decision made after that close, so
    it is aligned to t+1. This one call prevents the most common fatal bug.
    """
    out = df.copy()
    out[cols] = out.groupby(by, sort=False)[cols].shift(lag)
    return out
