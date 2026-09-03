from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from alphabench.data.repository import Repository
from alphabench.models.lstm import LSTM_FEATURES, WINDOW, SmallLSTM, build_sequences
from alphabench.validation.splitters import WalkForwardSplit

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 500


def _run_epochs(
    model: nn.Module,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_dev: torch.Tensor,
    y_dev: torch.Tensor,
    max_epochs: int,
    batch_size: int,
    patience: int,
) -> None:
    """Train with Adam/BCE, early-stopping on an internal dev split carved out of this
    fold's own training window — never the fold's real validation set, so the epoch
    count is never chosen by peeking at what we're about to score."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_dev_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    n = x_fit.shape[0]

    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(x_fit[idx])
            loss = loss_fn(logits, y_fit[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_loss = loss_fn(model(x_dev), y_dev).item() if len(x_dev) else 0.0

        if dev_loss < best_dev_loss - 1e-4:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)


def train_lstm_walkforward(
    df: pd.DataFrame,
    horizon: int,
    cfg,
    out_dir: Path = Path("models/lstm_h1"),
    tag: str = "",
    feature_cols: list[str] | None = None,
    window: int = WINDOW,
    hidden_size: int = 16,
    max_epochs: int = 8,
    batch_size: int = 256,
    patience: int = 2,
    dev_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_col = f"y_dir_{horizon}d"
    fwd_ret_col = f"fwd_ret_{horizon}d"
    feature_cols = feature_cols or [c for c in LSTM_FEATURES if c in df.columns]

    df = df[df["date"] < cfg.validation.holdout_start]
    df = df.dropna(subset=[y_col]).reset_index(drop=True)

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

        train_valid = train_fold.dropna(subset=feature_cols)
        if len(train_valid) < MIN_TRAIN_ROWS:
            log.warning("LSTM fold %d (%d): too few clean rows, skipping", i, year)
            continue

        # Scaler fit on this fold's TRAIN rows only — never on val.
        scaler = StandardScaler()
        scaler.fit(train_valid[feature_cols])

        combined = pd.concat([train_fold, val_fold], ignore_index=True).sort_values(
            ["symbol", "date"]
        )
        scaled = combined.copy()
        scaled[feature_cols] = scaler.transform(combined[feature_cols])

        train_dates_sorted = np.sort(train_fold["date"].unique())
        cutoff = int(len(train_dates_sorted) * (1 - dev_frac))
        fit_dates = set(train_dates_sorted[:cutoff])
        dev_dates = set(train_dates_sorted[cutoff:])
        val_dates = set(val_fold["date"].unique())

        x_fit_np, y_fit_np, _ = build_sequences(scaled, feature_cols, y_col, window, fit_dates)
        x_dev_np, y_dev_np, _ = build_sequences(scaled, feature_cols, y_col, window, dev_dates)
        x_val_np, y_val_np, meta_val = build_sequences(
            scaled, feature_cols, y_col, window, val_dates
        )

        if len(x_fit_np) < MIN_TRAIN_ROWS or len(x_val_np) == 0:
            log.warning("LSTM fold %d (%d): too few sequences, skipping", i, year)
            continue

        x_fit = torch.from_numpy(x_fit_np)
        y_fit = torch.from_numpy(y_fit_np)
        x_dev = torch.from_numpy(x_dev_np)
        y_dev = torch.from_numpy(y_dev_np)
        x_val = torch.from_numpy(x_val_np)

        model = SmallLSTM(n_features=len(feature_cols), hidden_size=hidden_size)
        _run_epochs(model, x_fit, y_fit, x_dev, y_dev, max_epochs, batch_size, patience)

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(x_val)).numpy()

        fold_oof = pd.DataFrame(
            {
                "date": [d for d, _ in meta_val],
                "symbol": [s for _, s in meta_val],
                "proba": proba,
                y_col: y_val_np,
            }
        )
        val_lookup = val_fold.set_index(["date", "symbol"])[fwd_ret_col]
        fold_oof[fwd_ret_col] = fold_oof.set_index(["date", "symbol"]).index.map(val_lookup)
        oof_rows.append(fold_oof)

        m = {
            "fold": i,
            "val_year": year,
            "n_train": len(x_fit_np),
            "n_val": len(x_val_np),
            "base_rate": float(y_val_np.mean()),
            "auc": float(roc_auc_score(y_val_np, proba)),
            "accuracy": float(accuracy_score(y_val_np, (proba > 0.5).astype(int))),
            "brier": float(brier_score_loss(y_val_np, proba)),
            "logloss": float(log_loss(y_val_np, proba, labels=[0, 1])),
        }
        rows.append(m)
        log.info(
            "LSTM fold %d (%d): AUC=%.4f acc=%.4f base=%.4f",
            i,
            year,
            m["auc"],
            m["accuracy"],
            m["base_rate"],
        )
        torch.save(model.state_dict(), out_dir / f"fold_{year}.pt")

    res = pd.DataFrame(rows)

    if oof_rows:
        oof = pd.concat(oof_rows, ignore_index=True)
        Repository(cfg.data.processed_dir).write(oof, f"oof_predictions_lstm_h{horizon}{tag}")

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "features": feature_cols,
                "window": window,
                "hidden_size": hidden_size,
                "max_epochs": max_epochs,
                "horizon": horizon,
                "cv_mean_auc": float(res["auc"].mean()) if len(res) else None,
            },
            indent=2,
        )
    )

    res.to_json(
        f"reports/metrics/walkforward_results_lstm_h{horizon}{tag}.json",
        orient="records",
        indent=2,
    )
    print(res.to_string(index=False))
    if len(res):
        print(f"\nMean AUC {res['auc'].mean():.4f} +/- {res['auc'].std():.4f}")
        print(">>> M3 is a deliberately small, CPU-only comparison arm — expected to")
        print(">>> underperform M1 at this data scale (PROPOSAL section 4.2).")
    return res
