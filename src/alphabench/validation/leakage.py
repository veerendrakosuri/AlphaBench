from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd


def assert_train_precedes_val(dates: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Raise if any training date is not strictly before every validation date."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    train_max = dates.iloc[train_idx].max()
    val_min = dates.iloc[val_idx].min()
    if not train_max < val_min:
        raise AssertionError(
            f"TRAIN OVERLAPS VALIDATION: train max date {train_max} "
            f"is not before val min date {val_min}"
        )


def assert_no_feature_leak(
    feats_df: pd.DataFrame,
    target_col: str,
    meta_cols: Collection[str],
    threshold: float = 0.10,
) -> pd.Series:
    """Raise if any feature column correlates with `target_col` above `threshold`.

    `feats_df` must already have `target_col` merged in (e.g. features merged
    with targets). `meta_cols` lists identifier/OHLCV columns to exclude from
    the correlation scan — if `feats_df` also carries sibling target columns
    (e.g. a multi-horizon target frame's other `fwd_ret_*`/`y_dir_*`/`sigma_*`
    columns), include those in `meta_cols` too, or they'll be scanned as if
    they were features and trivially "leak" against `target_col`. Returns the
    |corr| series (feature -> value) on success, for callers that want to log
    or inspect it further.
    """
    cols = [c for c in feats_df.columns if c not in meta_cols and c != target_col]
    corr = feats_df[cols].corrwith(feats_df[target_col]).abs()
    worst = corr.idxmax()
    if corr.max() >= threshold:
        raise AssertionError(
            f"LEAK: '{worst}' correlates {corr.max():.3f} with {target_col} (threshold {threshold})"
        )
    return corr
