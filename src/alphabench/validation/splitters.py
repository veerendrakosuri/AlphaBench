from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    """Expanding-window walk-forward splitter with purging and embargo.

    purge_days:   drop training samples whose label window (t .. t+horizon)
                  overlaps the validation block. Without this the last
                  `horizon` training rows leak validation information.
    embargo_days: additional gap after training to break serial correlation
                  between adjacent feature windows.
    """

    train_start: str
    first_val_year: int
    last_val_year: int
    horizon: int = 1
    purge_days: int = 5
    embargo_days: int = 5

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        t0 = pd.Timestamp(self.train_start)
        gap = pd.Timedelta(days=self.purge_days + self.embargo_days + self.horizon)

        for year in range(self.first_val_year, self.last_val_year + 1):
            val_start = pd.Timestamp(f"{year}-01-01")
            val_end = pd.Timestamp(f"{year}-12-31")
            train_end = val_start - gap

            train_idx = np.where((dates >= t0) & (dates <= train_end))[0]
            val_idx = np.where((dates >= val_start) & (dates <= val_end))[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, *_args) -> int:
        return self.last_val_year - self.first_val_year + 1
