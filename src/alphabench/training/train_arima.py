from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from alphabench.data.repository import Repository
from alphabench.models.arima import rolling_arima_forecast
from alphabench.validation.splitters import WalkForwardSplit

log = logging.getLogger(__name__)

DEFAULT_ORDER = (1, 0, 1)
MIN_TRAIN_OBS = 100  # skip a (symbol, fold) if too little history to fit ARIMA sensibly


def train_arima_walkforward(
    df: pd.DataFrame,
    horizon: int,
    cfg,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    out_dir: Path = Path("models/arima_h1"),
    tag: str = "",
) -> pd.DataFrame:
    """B2 baseline: per-symbol ARIMA on log returns, scored walk-forward.

    Unlike the tree models, ARIMA is univariate per symbol — it never sees the engineered
    feature set, only that symbol's own historical log-return series (`ret_1d`, which
    features/pipeline.py already lag-shifts by one day, i.e. it is exactly "the most
    recently realised daily return, known as of this row's date" — the same information
    a forecaster standing at that date would have). No model artifacts are pickled here
    (an ARIMA fit is just a handful of coefficients); `metadata.json` records the order
    and mean CV AUC.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_col = f"y_dir_{horizon}d"
    fwd_ret_col = f"fwd_ret_{horizon}d"
    df = df[df["date"] < cfg.validation.holdout_start]
    df = df.dropna(subset=[y_col, "ret_1d"]).reset_index(drop=True)

    sp = WalkForwardSplit(
        cfg.validation.train_start,
        cfg.validation.first_val_year,
        cfg.validation.last_val_year,
        horizon=horizon,
        purge_days=cfg.validation.purge_days,
        embargo_days=cfg.validation.embargo_days,
    )

    rows = []
    oof_rows = []

    for i, (tr, va) in enumerate(sp.split(df["date"]), 1):
        train_fold = df.loc[tr]
        val_fold = df.loc[va]
        year = int(val_fold["date"].dt.year.iloc[0])

        fold_frames = []
        for symbol, val_g in val_fold.groupby("symbol", sort=False):
            train_g = train_fold[train_fold["symbol"] == symbol]
            if len(train_g) < MIN_TRAIN_OBS or len(val_g) == 0:
                continue

            train_series = pd.Series(train_g["ret_1d"].to_numpy(), index=train_g["date"])
            known_series = pd.Series(val_g["ret_1d"].to_numpy(), index=val_g["date"])
            val_dates = pd.DatetimeIndex(val_g["date"].sort_values().unique())

            try:
                probs = rolling_arima_forecast(
                    train_series, known_series, val_dates, horizon, order
                )
            except Exception as exc:
                log.warning("ARIMA failed for %s fold %d: %s", symbol, i, exc)
                continue

            sub = val_g.set_index("date").loc[probs.index]
            fold_frames.append(
                pd.DataFrame(
                    {
                        "date": probs.index,
                        "symbol": symbol,
                        "proba": probs.to_numpy(),
                        y_col: sub[y_col].to_numpy(),
                        fwd_ret_col: sub[fwd_ret_col].to_numpy(),
                    }
                )
            )

        if not fold_frames:
            log.warning("ARIMA fold %d (%d): no symbols converged, skipping", i, year)
            continue

        fold_oof = pd.concat(fold_frames, ignore_index=True)
        oof_rows.append(fold_oof)

        proba = fold_oof["proba"].to_numpy()
        yva = fold_oof[y_col].to_numpy()
        m = {
            "fold": i,
            "val_year": year,
            "n_train": len(tr),
            "n_val": len(fold_oof),
            "base_rate": float(yva.mean()),
            "auc": float(roc_auc_score(yva, proba)),
            "accuracy": float(accuracy_score(yva, (proba > 0.5).astype(int))),
            "brier": float(brier_score_loss(yva, proba)),
            "logloss": float(log_loss(yva, proba, labels=[0, 1])),
        }
        rows.append(m)
        log.info(
            "ARIMA fold %d (%d): AUC=%.4f acc=%.4f base=%.4f",
            i,
            year,
            m["auc"],
            m["accuracy"],
            m["base_rate"],
        )

    res = pd.DataFrame(rows)

    if oof_rows:
        oof = pd.concat(oof_rows, ignore_index=True)
        Repository(cfg.data.processed_dir).write(oof, f"oof_predictions_arima_h{horizon}{tag}")

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "order": list(order),
                "horizon": horizon,
                "cv_mean_auc": float(res["auc"].mean()) if len(res) else None,
            },
            indent=2,
        )
    )

    res.to_json(
        f"reports/metrics/walkforward_results_arima_h{horizon}{tag}.json",
        orient="records",
        indent=2,
    )
    print(res.to_string(index=False))
    if len(res):
        print(f"\nMean AUC {res['auc'].mean():.4f} +/- {res['auc'].std():.4f}")
        print(">>> Expect near-zero predictive power from ARIMA on daily returns — that")
        print(">>> is the textbook result, and demonstrating it is the point of B2.")
    return res
