from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from alphabench.data.repository import Repository
from alphabench.training.train import _feature_cols
from alphabench.validation.splitters import WalkForwardSplit

log = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=50,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    early_stopping_rounds=100,
    eval_metric="auc",
    n_jobs=-1,
    random_state=42,
)
# Same conservative philosophy as LightGBM's DEFAULT_PARAMS: shallow trees, heavy
# regularisation, small learning rate. XGBoost is the diversification arm in the
# ensemble (M4), not a bid to beat M1 outright.


def train_xgboost_walkforward(
    df: pd.DataFrame,
    horizon: int,
    cfg,
    params: dict | None = None,
    out_dir: Path = Path("models/xgboost_h1"),
    tag: str = "",
) -> pd.DataFrame:
    params = {**DEFAULT_PARAMS, **(params or {})}
    out_dir.mkdir(parents=True, exist_ok=True)

    y_col = f"y_dir_{horizon}d"
    df = df[df["date"] < cfg.validation.holdout_start]
    df = df.dropna(subset=[y_col]).reset_index(drop=True)
    cols = _feature_cols(df)

    sp = WalkForwardSplit(
        cfg.validation.train_start,
        cfg.validation.first_val_year,
        cfg.validation.last_val_year,
        horizon=horizon,
        purge_days=cfg.validation.purge_days,
        embargo_days=cfg.validation.embargo_days,
    )

    fwd_ret_col = f"fwd_ret_{horizon}d"

    mlflow.set_experiment("alphabench")
    rows = []
    oof_rows = []
    with mlflow.start_run(run_name=f"xgboost_h{horizon}{tag}"):
        mlflow.log_params({**params, "horizon": horizon, "n_features": len(cols)})

        for i, (tr, va) in enumerate(sp.split(df["date"]), 1):
            Xtr, ytr = df.loc[tr, cols], df.loc[tr, y_col]  # noqa: N806
            Xva, yva = df.loc[va, cols], df.loc[va, y_col]  # noqa: N806
            year = df.loc[va, "date"].dt.year.iloc[0]

            model = XGBClassifier(**params)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            proba = model.predict_proba(Xva)[:, 1]

            oof_rows.append(
                pd.DataFrame(
                    {
                        "date": df.loc[va, "date"].to_numpy(),
                        "symbol": df.loc[va, "symbol"].to_numpy(),
                        "proba": proba,
                        y_col: yva.to_numpy(),
                        fwd_ret_col: df.loc[va, fwd_ret_col].to_numpy(),
                    }
                )
            )

            m = {
                "fold": i,
                "val_year": int(year),
                "n_train": len(tr),
                "n_val": len(va),
                "base_rate": float(yva.mean()),
                "auc": float(roc_auc_score(yva, proba)),
                "accuracy": float(accuracy_score(yva, (proba > 0.5).astype(int))),
                "brier": float(brier_score_loss(yva, proba)),
                "logloss": float(log_loss(yva, proba)),
                "best_iter": int(model.best_iteration or params["n_estimators"]),
            }
            rows.append(m)
            for k, v in m.items():
                if k != "fold":
                    mlflow.log_metric(f"fold{i}_{k}", v)
            log.info(
                "XGBoost fold %d (%d): AUC=%.4f acc=%.4f base=%.4f",
                i,
                year,
                m["auc"],
                m["accuracy"],
                m["base_rate"],
            )

            joblib.dump(model, out_dir / f"fold_{year}.joblib")

        res = pd.DataFrame(rows)
        mlflow.log_metric("mean_auc", res["auc"].mean())
        mlflow.log_metric("std_auc", res["auc"].std())

        final_params = {**params, "n_estimators": int(res["best_iter"].median())}
        final_params.pop("early_stopping_rounds", None)
        final = XGBClassifier(**final_params)
        final.fit(df[cols], df[y_col])
        joblib.dump(final, out_dir / "final.joblib")

        (out_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "features": cols,
                    "params": params,
                    "horizon": horizon,
                    "train_start": str(df["date"].min().date()),
                    "train_end": str(df["date"].max().date()),
                    "cv_mean_auc": float(res["auc"].mean()),
                    "n_rows": len(df),
                },
                indent=2,
            )
        )

    oof = pd.concat(oof_rows, ignore_index=True)
    Repository(cfg.data.processed_dir).write(oof, f"oof_predictions_xgboost_h{horizon}{tag}")

    res.to_json(
        f"reports/metrics/walkforward_results_xgboost_h{horizon}{tag}.json",
        orient="records",
        indent=2,
    )
    print(res.to_string(index=False))
    print(f"\nMean AUC {res['auc'].mean():.4f} +/- {res['auc'].std():.4f}")
    return res
