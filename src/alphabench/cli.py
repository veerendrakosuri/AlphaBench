from __future__ import annotations

import json
from functools import reduce
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; this is a CLI, not a notebook

import matplotlib.pyplot as plt
import pandas as pd
import typer
from rich.console import Console

from alphabench import config as config_mod
from alphabench import logging_conf
from alphabench.data import ingest
from alphabench.data.repository import Repository
from alphabench.evaluation.backtest import cost_sensitivity, run_backtest
from alphabench.evaluation.robustness import block_bootstrap_sharpe, by_period
from alphabench.features import pipeline
from alphabench.models.baselines import b0_majority, b0_persistence, b1_logistic
from alphabench.targets import builder as targets
from alphabench.training.train import _feature_cols, train_walkforward
from alphabench.training.tune import tune as tune_fn
from alphabench.validation.leakage import assert_no_feature_leak
from alphabench.validation.splitters import WalkForwardSplit

app = typer.Typer()
console = Console()


def _print_report(report: dict) -> None:
    console.print(f"rows:      {report['rows']}")
    console.print(f"symbols:   {report['symbols']}")
    console.print(f"date_min:  {report['date_min']}")
    console.print(f"date_max:  {report['date_max']}")
    if report["issues"]:
        console.print("[bold red]issues:[/bold red]")
        for issue in report["issues"]:
            console.print(f"  - {issue}")
    if report["warnings"]:
        console.print("[bold yellow]warnings:[/bold yellow]")
        for warning in report["warnings"]:
            console.print(f"  - {warning}")


@app.command(name="ingest")
def ingest_cmd(
    universe: str = typer.Option(..., "--universe", help="Path to a universe YAML file"),
    incremental: bool = typer.Option(False, "--incremental", help="Only fetch new bars"),
) -> None:
    """Fetch OHLCV data for a universe and cache it to data/raw."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config()
    panel = ingest.run_ingest(cfg, universe_file=universe, incremental=incremental)
    console.print(
        f"[green]ingested[/green] {len(panel)} rows for {panel['symbol'].nunique()} symbols"
    )


@app.command()
def validate() -> None:
    """Validate the raw OHLCV cache and write the adjusted panel to data/interim."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config()
    report = ingest.run_validate(cfg)
    _print_report(report)


@app.command(name="build-features")
def build_features_cmd() -> None:
    """Build features and targets from the validated panel."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config()
    _, benchmark = config_mod.load_universe(cfg.universe["file"])

    interim = Repository(cfg.data.interim_dir)
    panel = interim.read("panel")

    feats = pipeline.build_features(panel, benchmark)
    processed = Repository(cfg.data.processed_dir)
    processed.write(feats, "features")

    horizon_frames = [
        targets.build_targets(
            panel,
            horizon=h,
            kappa=cfg.target.deadband_kappa,
            vol_window=cfg.target.vol_window,
        )
        for h in cfg.target.horizons
    ]
    tgt = reduce(
        lambda left, right: left.merge(right, on=["date", "symbol"], how="outer"), horizon_frames
    )
    processed.write(tgt, "targets")

    meta = ["date", "symbol", "open", "high", "low", "close", "volume"]
    feature_count = len([c for c in feats.columns if c not in meta])
    console.print(f"rows:     {len(feats)}")
    console.print(f"features: {feature_count}")
    for h in cfg.target.horizons:
        col = f"y_dir_{h}d"
        balance = tgt[col].value_counts(normalize=True).round(4).to_dict()
        dropped = round(tgt[col].isna().mean(), 4)
        console.print(f"horizon {h}d — label balance: {balance}  deadband dropped: {dropped}")


def _load_merged_processed(cfg) -> pd.DataFrame:
    processed = Repository(cfg.data.processed_dir)
    feats = processed.read("features")
    tgts = processed.read("targets")
    return feats.merge(tgts, on=["date", "symbol"])


def _prepare_fold_frame(merged: pd.DataFrame, cfg, y_col: str) -> pd.DataFrame:
    """Same preprocessing train_walkforward applies internally, replicated here
    so the CLI's baselines and leak check see the exact same fold boundaries."""
    df = merged[merged["date"] < cfg.validation.holdout_start]
    return df.dropna(subset=[y_col]).reset_index(drop=True)


@app.command(name="train")
def train_cmd(
    model: str = typer.Option("lightgbm", "--model", help="Model to train"),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
) -> None:
    """Train B0/B1 baselines and LightGBM in identical walk-forward folds."""
    if model != "lightgbm":
        raise NotImplementedError(
            f"model={model!r} is not implemented; only 'lightgbm' is supported"
        )

    logging_conf.setup_logging()
    cfg = config_mod.load_config()

    merged = _load_merged_processed(cfg)
    y_col = f"y_dir_{horizon}d"
    df_pre = _prepare_fold_frame(merged, cfg, y_col)

    sp = WalkForwardSplit(
        cfg.validation.train_start,
        cfg.validation.first_val_year,
        cfg.validation.last_val_year,
        horizon=horizon,
        purge_days=cfg.validation.purge_days,
        embargo_days=cfg.validation.embargo_days,
    )

    baseline_rows: list[dict] = []
    for i, (tr, va) in enumerate(sp.split(df_pre["date"]), 1):
        train_fold = df_pre.loc[tr]
        val_fold = df_pre.loc[va]
        year = int(df_pre.loc[va, "date"].dt.year.iloc[0])
        for m in (
            b0_persistence(val_fold, y_col),
            b0_majority(val_fold, y_col),
            b1_logistic(train_fold, val_fold, y_col),
        ):
            m["fold"] = i
            m["val_year"] = year
            baseline_rows.append(m)

    baselines_df = pd.DataFrame(baseline_rows)[
        ["model", "fold", "val_year", "accuracy", "auc", "base_rate"]
    ]
    console.print("[bold]Baselines (per fold)[/bold]")
    console.print(baselines_df.to_string(index=False))
    means = baselines_df.groupby("model")[["accuracy", "auc"]].mean().round(4)
    console.print("\n[bold]Baselines (mean across folds)[/bold]")
    console.print(means.to_string())

    target_col = f"fwd_ret_{horizon}d"
    meta_cols = {"date", "symbol", "open", "high", "low", "close", "volume"} | {
        c for c in df_pre.columns if c.startswith(("y_", "fwd_", "sigma_")) and c != target_col
    }
    leak_df = df_pre.dropna(subset=[target_col])
    assert_no_feature_leak(leak_df, target_col, meta_cols)
    console.print(f"\n[green]pre-flight leak check passed[/green] (target={target_col})")

    console.print("\n[bold]LightGBM walk-forward[/bold]")
    train_walkforward(merged, horizon, cfg, out_dir=Path(f"models/lightgbm_h{horizon}"))


@app.command(name="tune")
def tune_cmd(
    model: str = typer.Option("lightgbm", "--model", help="Model to tune"),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    trials: int = typer.Option(50, "--trials", help="Number of Optuna trials"),
) -> None:
    """Optuna hyperparameter search inside the walk-forward loop."""
    if model != "lightgbm":
        raise NotImplementedError(
            f"model={model!r} is not implemented; only 'lightgbm' is supported"
        )

    logging_conf.setup_logging()
    cfg = config_mod.load_config()

    merged = _load_merged_processed(cfg)
    y_col = f"y_dir_{horizon}d"
    df_pre = _prepare_fold_frame(merged, cfg, y_col)
    cols = _feature_cols(df_pre)

    study = tune_fn(df_pre, cols, y_col, cfg, horizon=horizon, n_trials=trials)
    console.print(f"best value: {study.best_value:.4f}")
    console.print(f"trials run: {len(study.trials)}")
    console.print(f"best params: {study.best_params}")


@app.command(name="backtest")
def backtest_cmd(
    model: str = typer.Option("lightgbm", "--model", help="Model to backtest"),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
) -> None:
    """Cost-aware backtest + robustness checks on out-of-fold predictions."""
    if model != "lightgbm":
        raise NotImplementedError(
            f"model={model!r} is not implemented; only 'lightgbm' is supported"
        )

    logging_conf.setup_logging()
    cfg = config_mod.load_config()

    processed = Repository(cfg.data.processed_dir)
    oof_name = f"oof_predictions_h{horizon}"
    if not processed.exists(oof_name):
        raise FileNotFoundError(
            f"{oof_name}.parquet not found in {cfg.data.processed_dir} — "
            f"run `alphabench train --model lightgbm --horizon {horizon}` first."
        )
    oof = processed.read(oof_name)

    result = run_backtest(
        oof,
        proba_col="proba",
        threshold=cfg.backtest.prob_threshold,
        commission_bps=cfg.backtest.commission_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        allow_short=cfg.backtest.allow_short,
    )
    cost_df = cost_sensitivity(oof, proba_col="proba", threshold=cfg.backtest.prob_threshold)
    boot = block_bootstrap_sharpe(result["daily"]["net"])
    period_df = by_period(result["trades"])

    console.print("[bold]Backtest metrics[/bold]")
    metrics_df = pd.DataFrame(result["metrics"].items(), columns=["metric", "value"])
    console.print(metrics_df.to_string(index=False))

    console.print("\n[bold]Cost sensitivity[/bold]")
    console.print(cost_df.to_string(index=False))

    console.print("\n[bold]Block-bootstrap Sharpe CI[/bold]")
    console.print(
        f"sharpe={boot['sharpe']:.4f}  "
        f"CI=[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]  "
        f"p(sharpe>0)={boot['p_gt_zero']:.4f}"
    )

    console.print("\n[bold]Per-year breakdown[/bold]")
    console.print(period_df.to_string())

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # by_period's index is a Timestamp (period end) and isn't JSON-serialisable directly.
    by_period_records = period_df.reset_index()
    by_period_records[by_period_records.columns[0]] = by_period_records[
        by_period_records.columns[0]
    ].astype(str)

    combined = {
        "metrics": result["metrics"],
        "cost_sensitivity": cost_df.to_dict(orient="records"),
        "bootstrap_sharpe": boot,
        "by_period": by_period_records.to_dict(orient="records"),
    }
    (reports_dir / "backtest_results.json").write_text(json.dumps(combined, indent=2))

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result["equity"].index, result["equity"].to_numpy(), label="Strategy (net of costs)")
    ax.plot(
        result["equity_bench"].index,
        result["equity_bench"].to_numpy(),
        label="Benchmark (buy & hold)",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (starting at 1.0)")
    ax.set_title(f"AlphaBench LightGBM h{horizon} — Equity Curve (net of costs)")
    ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / f"equity_curve_h{horizon}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    console.print(f"\n[green]wrote[/green] {reports_dir / 'backtest_results.json'}")
    console.print(f"[green]wrote[/green] {fig_path}")


if __name__ == "__main__":
    app()
