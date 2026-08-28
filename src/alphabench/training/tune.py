from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

from alphabench.validation.splitters import WalkForwardSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(
    df: pd.DataFrame, cols: list[str], y_col: str, cfg, horizon: int = 1, n_trials: int = 50
) -> optuna.Study:
    sp = WalkForwardSplit(
        cfg.validation.train_start,
        cfg.validation.first_val_year,
        cfg.validation.last_val_year,
        horizon=horizon,
        purge_days=cfg.validation.purge_days,
        embargo_days=cfg.validation.embargo_days,
    )
    folds = list(sp.split(df["date"]))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 500),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            subsample_freq=1,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
        )
        aucs = []
        for tr, va in folds:
            m = LGBMClassifier(**params)  # type: ignore[arg-type]
            m.fit(
                df.loc[tr, cols],
                df.loc[tr, y_col],
                eval_set=[(df.loc[va, cols], df.loc[va, y_col])],
                eval_metric="auc",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            proba = m.predict_proba(df.loc[va, cols])[:, 1]  # type: ignore[call-overload]
            aucs.append(roc_auc_score(df.loc[va, y_col], proba))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"best mean AUC: {study.best_value:.4f}")
    print(f"trials run: {len(study.trials)}  <-- RECORD THIS for the deflated Sharpe")
    return study
