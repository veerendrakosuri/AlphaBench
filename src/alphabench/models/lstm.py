from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

# A small, deliberately narrow feature subset — the point of M3 is a like-for-like
# sequence-model comparison arm, not a second attempt to beat M1 on features.
LSTM_FEATURES = ["ret_1d", "mom_5d", "vol_21d", "rsi_14", "macd", "vol_z_21", "xs_rank_ret_1d"]
WINDOW = 20


class SmallLSTM(nn.Module):
    """One LSTM layer + a linear head emitting a logit. Deliberately small (per
    PROPOSAL §4.2: CPU-only, expected to lose to the GBM at this data scale) — the
    point is an honest comparison, not a bid to win."""

    def __init__(self, n_features: int, hidden_size: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    window: int = WINDOW,
    end_dates: set | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Build (n_samples, window, n_features) arrays, one sample per (symbol, date) whose
    date is in `end_dates` (or every eligible date if `end_dates` is None).

    `df` should already contain each symbol's full available history in chronological
    order (e.g. the fold's train+val rows concatenated) so that samples near the start of
    a restricted `end_dates` window can still look back into earlier rows — the lookback
    itself is never leakage (it is all information strictly before the sample's own
    date), only the label is restricted to `end_dates`.
    """
    X_list, y_list, meta = [], [], []  # noqa: N806 -- X/y convention
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date")
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g[y_col].to_numpy()
        dates = g["date"].to_numpy()
        valid_feat = ~np.isnan(feats).any(axis=1)

        for i in range(window - 1, len(g)):
            if end_dates is not None and dates[i] not in end_dates:
                continue
            if not valid_feat[i - window + 1 : i + 1].all() or np.isnan(labels[i]):
                continue
            X_list.append(feats[i - window + 1 : i + 1])
            y_list.append(labels[i])
            meta.append((dates[i], symbol))

    if not X_list:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty(0), []
    return np.stack(X_list), np.array(y_list, dtype=np.float32), meta
