from __future__ import annotations

from typing import Protocol

import pandas as pd


class Provider(Protocol):
    def fetch(self, symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame: ...
