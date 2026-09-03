from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def rank_average_ensemble(
    oof_frames: dict[str, pd.DataFrame], y_col: str, fwd_ret_col: str
) -> pd.DataFrame:
    """M4: combine several models' out-of-fold probabilities via a rank average.

    Within each fold's validation year, every model's `proba` is converted to a
    percentile rank (so models with different probability scales/calibration are
    combined fairly), then the ranks are averaged across models. Only (date, symbol)
    rows present in *every* input model's OOF set are kept, so the comparison is
    like-for-like.
    """
    base: pd.DataFrame | None = None
    proba_cols: list[str] = []
    for name, frame in oof_frames.items():
        col = f"proba_{name}"
        f = frame[["date", "symbol", "proba", y_col, fwd_ret_col]].rename(columns={"proba": col})
        proba_cols.append(col)
        base = (
            f
            if base is None
            else base.merge(f[["date", "symbol", col]], on=["date", "symbol"], how="inner")
        )
    if base is None or base.empty:
        raise ValueError("no OOF frames supplied to rank_average_ensemble")

    base = base.dropna(subset=proba_cols).copy()
    base["fold_year"] = pd.to_datetime(base["date"]).dt.year

    for col in proba_cols:
        base[f"rank_{col}"] = base.groupby("fold_year")[col].rank(pct=True)
    base["proba"] = base[[f"rank_{c}" for c in proba_cols]].mean(axis=1)

    return base[["date", "symbol", "fold_year", "proba", y_col, fwd_ret_col]].reset_index(drop=True)


def score_by_fold(df: pd.DataFrame, y_col: str, fold_col: str = "fold_year") -> pd.DataFrame:
    """Same metric set/shape as the other walk-forward trainers' per-fold results table,
    computed from an already-scored (proba, label) frame grouped by validation year."""
    rows = []
    for year, g in sorted(df.groupby(fold_col), key=lambda kv: kv[0]):
        y = g[y_col].to_numpy()
        p = g["proba"].to_numpy()
        rows.append(
            {
                "val_year": int(year),
                "n_val": len(g),
                "base_rate": float(y.mean()),
                "auc": float(roc_auc_score(y, p)),
                "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
                "brier": float(brier_score_loss(y, p)),
                "logloss": float(log_loss(y, p, labels=[0, 1])),
            }
        )
    res = pd.DataFrame(rows).sort_values("val_year").reset_index(drop=True)
    res.insert(0, "fold", range(1, len(res) + 1))
    return res
