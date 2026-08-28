from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def b0_persistence(df: pd.DataFrame, y_col: str) -> dict:
    """Predict tomorrow's direction = today's direction."""
    d = df.dropna(subset=[y_col, "ret_1d"])
    pred = (d["ret_1d"] > 0).astype(int)
    return {
        "model": "B0_persistence",
        "accuracy": float(accuracy_score(d[y_col], pred)),
        "auc": float(roc_auc_score(d[y_col], pred)),
        "base_rate": float(d[y_col].mean()),
    }


def b0_majority(df: pd.DataFrame, y_col: str) -> dict:
    d = df.dropna(subset=[y_col])
    maj = int(d[y_col].mode().iloc[0])
    return {
        "model": "B0_majority",
        "accuracy": float(accuracy_score(d[y_col], np.full(len(d), maj))),
        "auc": 0.5,
        "base_rate": float(d[y_col].mean()),
    }


def b1_logistic(
    train: pd.DataFrame, val: pd.DataFrame, y_col: str, cols: list[str] | None = None
) -> dict:
    cols = cols or ["ret_1d", "mom_5d", "vol_21d", "rsi_14", "vol_z_21"]
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1),
    )
    tr = train.dropna(subset=[y_col])
    va = val.dropna(subset=[y_col])
    pipe.fit(tr[cols], tr[y_col])  # scaler fitted on TRAIN ONLY
    proba = pipe.predict_proba(va[cols])[:, 1]
    return {
        "model": "B1_logistic",
        "accuracy": float(accuracy_score(va[y_col], (proba > 0.5).astype(int))),
        "auc": float(roc_auc_score(va[y_col], proba)),
        "base_rate": float(va[y_col].mean()),
    }
