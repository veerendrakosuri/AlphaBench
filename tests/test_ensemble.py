from __future__ import annotations

import numpy as np
import pandas as pd

from alphabench.models.ensemble import rank_average_ensemble, score_by_fold


def _oof(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * n,
            "proba": rng.uniform(0, 1, n),
            "y_dir_1d": rng.integers(0, 2, n),
            "fwd_ret_1d": rng.normal(0, 0.01, n),
        }
    )


def test_rank_average_ensemble_merges_on_date_symbol_and_ranks_within_fold():
    a = _oof(0)
    b = _oof(1)
    # b's label/target columns must match a's (same underlying truth) for a fair merge —
    # overwrite b's label columns to match a's, as if they scored the same rows.
    b["y_dir_1d"] = a["y_dir_1d"]
    b["fwd_ret_1d"] = a["fwd_ret_1d"]

    ens = rank_average_ensemble({"m1": a, "m2": b}, "y_dir_1d", "fwd_ret_1d")

    assert set(ens.columns) == {"date", "symbol", "fold_year", "proba", "y_dir_1d", "fwd_ret_1d"}
    assert len(ens) == len(a)
    assert ens["proba"].between(0.0, 1.0).all()
    assert (ens["fold_year"] == 2023).all()


def test_rank_average_ensemble_only_keeps_common_rows():
    a = _oof(0, n=100)
    b = _oof(1, n=50)
    b["y_dir_1d"] = a["y_dir_1d"].iloc[:50].to_numpy()
    b["fwd_ret_1d"] = a["fwd_ret_1d"].iloc[:50].to_numpy()
    b["date"] = a["date"].iloc[:50].to_numpy()

    ens = rank_average_ensemble({"m1": a, "m2": b}, "y_dir_1d", "fwd_ret_1d")
    assert len(ens) == 50


def test_score_by_fold_matches_manual_auc():
    from sklearn.metrics import roc_auc_score

    df = _oof(2)
    df["fold_year"] = pd.to_datetime(df["date"]).dt.year
    scores = score_by_fold(df, "y_dir_1d")
    assert len(scores) == 1
    expected_auc = roc_auc_score(df["y_dir_1d"], df["proba"])
    assert scores.loc[0, "auc"] == expected_auc
    assert scores.loc[0, "fold"] == 1
