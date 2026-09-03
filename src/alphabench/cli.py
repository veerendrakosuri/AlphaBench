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
from alphabench.evaluation.backtest import cost_sensitivity, run_backtest, threshold_sensitivity
from alphabench.evaluation.robustness import (
    block_bootstrap_auc,
    block_bootstrap_sharpe,
    by_period,
)
from alphabench.features import pipeline
from alphabench.models.arima import run_diagnostics_report
from alphabench.models.baselines import b0_majority, b0_persistence, b1_logistic
from alphabench.models.ensemble import rank_average_ensemble, score_by_fold
from alphabench.targets import builder as targets
from alphabench.training.train import _feature_cols, train_walkforward
from alphabench.training.train_arima import train_arima_walkforward
from alphabench.training.train_lstm import train_lstm_walkforward
from alphabench.training.train_xgboost import train_xgboost_walkforward
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


CONFIG_OPTION = typer.Option("config/config.yaml", "--config", help="Path to a config YAML file")


@app.command(name="ingest")
def ingest_cmd(
    universe: str = typer.Option(..., "--universe", help="Path to a universe YAML file"),
    incremental: bool = typer.Option(False, "--incremental", help="Only fetch new bars"),
    config: str = CONFIG_OPTION,
) -> None:
    """Fetch OHLCV data for a universe and cache it to data/raw."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)
    panel = ingest.run_ingest(cfg, universe_file=universe, incremental=incremental)
    console.print(
        f"[green]ingested[/green] {len(panel)} rows for {panel['symbol'].nunique()} symbols"
    )


@app.command()
def validate(config: str = CONFIG_OPTION) -> None:
    """Validate the raw OHLCV cache and write the adjusted panel to data/interim."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)
    report = ingest.run_validate(cfg)
    _print_report(report)


@app.command(name="build-features")
def build_features_cmd(config: str = CONFIG_OPTION) -> None:
    """Build features and targets from the validated panel."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)
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
    model: str = typer.Option("lightgbm", "--model", help="lightgbm | xgboost | arima | lstm"),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    config: str = CONFIG_OPTION,
    tag: str = typer.Option(
        "", "--tag", help="Suffix for output dirs/files, e.g. '_us' for a second universe"
    ),
) -> None:
    """Train B0/B1 baselines plus the requested model in identical walk-forward folds."""
    if model not in {"lightgbm", "xgboost", "arima", "lstm"}:
        raise NotImplementedError(
            f"model={model!r} is not implemented; choose lightgbm, xgboost, arima or lstm"
        )

    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)

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

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    baselines_df.to_json(
        reports_dir / f"baseline_results_h{horizon}{tag}.json", orient="records", indent=2
    )

    target_col = f"fwd_ret_{horizon}d"
    meta_cols = {"date", "symbol", "open", "high", "low", "close", "volume"} | {
        c for c in df_pre.columns if c.startswith(("y_", "fwd_", "sigma_")) and c != target_col
    }
    leak_df = df_pre.dropna(subset=[target_col])
    assert_no_feature_leak(leak_df, target_col, meta_cols)
    console.print(f"\n[green]pre-flight leak check passed[/green] (target={target_col})")

    out_dir = Path(f"models/{model}_h{horizon}{tag}")
    if model == "lightgbm":
        console.print("\n[bold]LightGBM walk-forward[/bold]")
        train_walkforward(merged, horizon, cfg, out_dir=out_dir, tag=tag)
    elif model == "xgboost":
        console.print("\n[bold]XGBoost walk-forward[/bold]")
        train_xgboost_walkforward(merged, horizon, cfg, out_dir=out_dir, tag=tag)
    elif model == "arima":
        console.print("\n[bold]ARIMA walk-forward[/bold]")
        train_arima_walkforward(merged, horizon, cfg, out_dir=out_dir, tag=tag)
    elif model == "lstm":
        console.print("\n[bold]LSTM walk-forward[/bold]")
        train_lstm_walkforward(merged, horizon, cfg, out_dir=out_dir, tag=tag)


@app.command(name="tune")
def tune_cmd(
    model: str = typer.Option("lightgbm", "--model", help="Model to tune"),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    trials: int = typer.Option(50, "--trials", help="Number of Optuna trials"),
    config: str = CONFIG_OPTION,
) -> None:
    """Optuna hyperparameter search inside the walk-forward loop.

    Written to reports/metrics/optuna_study_h{horizon}.json for the record: best params,
    trial count, and every trial's value (the deflated Sharpe ratio needs the trial count
    and the spread of trial outcomes to correct for selection bias — see BUILD_PLAN
    Stage 4.3 / PROPOSAL §7.3). This never re-fits the shipped model on the "best" params
    found here — tuning is a reporting exercise on top of the frozen run, not a way to
    quietly improve the headline number after the fact.
    """
    if model != "lightgbm":
        raise NotImplementedError(
            f"model={model!r} is not implemented; only 'lightgbm' is supported"
        )

    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)

    merged = _load_merged_processed(cfg)
    y_col = f"y_dir_{horizon}d"
    df_pre = _prepare_fold_frame(merged, cfg, y_col)
    cols = _feature_cols(df_pre)

    study = tune_fn(df_pre, cols, y_col, cfg, horizon=horizon, n_trials=trials)
    console.print(f"best value: {study.best_value:.4f}")
    console.print(f"trials run: {len(study.trials)}")
    console.print(f"best params: {study.best_params}")

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    trial_values = [t.value for t in study.trials if t.value is not None]
    (reports_dir / f"optuna_study_h{horizon}.json").write_text(
        json.dumps(
            {
                "horizon": horizon,
                "n_trials": len(study.trials),
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "trial_values": trial_values,
                "trial_value_std": float(pd.Series(trial_values).std()),
            },
            indent=2,
        )
    )
    console.print(f"[green]wrote[/green] reports/metrics/optuna_study_h{horizon}.json")


@app.command(name="backtest")
def backtest_cmd(
    model: str = typer.Option(
        "lightgbm", "--model", help="lightgbm | xgboost | arima | lstm | ensemble"
    ),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    config: str = CONFIG_OPTION,
    tag: str = typer.Option(
        "", "--tag", help="Suffix for output dirs/files, e.g. '_us' for a second universe"
    ),
) -> None:
    """Cost-aware backtest + robustness checks on out-of-fold predictions."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)

    processed = Repository(cfg.data.processed_dir)
    model_suffix = "" if model == "lightgbm" else f"_{model}"
    oof_name = f"oof_predictions{model_suffix}_h{horizon}{tag}"
    if not processed.exists(oof_name):
        raise FileNotFoundError(
            f"{oof_name}.parquet not found in {cfg.data.processed_dir} — "
            f"run `alphabench train --model {model} --horizon {horizon}` first."
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
    threshold_df = threshold_sensitivity(
        oof,
        proba_col="proba",
        commission_bps=cfg.backtest.commission_bps,
        slippage_bps=cfg.backtest.slippage_bps,
    )
    boot = block_bootstrap_sharpe(result["daily"]["net"])
    period_df = by_period(result["trades"])

    console.print("[bold]Backtest metrics[/bold]")
    metrics_df = pd.DataFrame(result["metrics"].items(), columns=["metric", "value"])
    console.print(metrics_df.to_string(index=False))

    console.print("\n[bold]Cost sensitivity[/bold]")
    console.print(cost_df.to_string(index=False))

    console.print("\n[bold]Threshold sensitivity[/bold]")
    console.print(threshold_df.to_string(index=False))

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
        "threshold_sensitivity": threshold_df.to_dict(orient="records"),
        "bootstrap_sharpe": boot,
        "by_period": by_period_records.to_dict(orient="records"),
    }
    # Preserve the exact existing filename for the default lightgbm/h=1/untagged run (the
    # API and dashboard read it by that literal name); any other model/horizon/tag gets
    # its own file so a second run never clobbers a previously-committed result.
    is_default_run = model == "lightgbm" and horizon == 1 and not tag
    results_name = (
        "backtest_results.json"
        if is_default_run
        else f"backtest_results{model_suffix}_h{horizon}{tag}.json"
    )
    (reports_dir / results_name).write_text(json.dumps(combined, indent=2))

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
    ax.set_title(f"AlphaBench {model} h{horizon}{tag} — Equity Curve (net of costs)")
    ax.legend()
    fig.tight_layout()
    fig_name = (
        "equity_curve_h1.png"
        if is_default_run
        else f"equity_curve{model_suffix}_h{horizon}{tag}.png"
    )
    fig_path = figures_dir / fig_name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    console.print(f"\n[green]wrote[/green] {reports_dir / results_name}")
    console.print(f"[green]wrote[/green] {fig_path}")


MODEL_LADDER_ORDER = {
    "B0_persistence": 0,
    "B0_majority": 1,
    "B1_logistic": 2,
    "B2_arima": 3,
    "M1_lightgbm": 4,
    "M2_xgboost": 5,
    "M3_lstm": 6,
    "M4_ensemble": 7,
}


@app.command(name="compare-models")
def compare_models_cmd(
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    config: str = CONFIG_OPTION,
    tag: str = typer.Option("", "--tag", help="Suffix matching the --tag used at train time"),
) -> None:
    """Assemble the full baseline ladder (B0 first) from each model's already-written
    walk-forward results, computing the M4 rank-average ensemble along the way. Run this
    after training whichever subset of {lightgbm, xgboost, arima, lstm} you have."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)
    reports_dir = Path("reports/metrics")

    def _load(name: str) -> pd.DataFrame | None:
        p = reports_dir / name
        return pd.read_json(p) if p.exists() else None

    lgbm_name = (
        "walkforward_results.json"
        if horizon == 1 and not tag
        else f"walkforward_results_h{horizon}{tag}.json"
    )
    baseline = _load(f"baseline_results_h{horizon}{tag}.json")
    lgbm = _load(lgbm_name)
    xgb = _load(f"walkforward_results_xgboost_h{horizon}{tag}.json")
    arima = _load(f"walkforward_results_arima_h{horizon}{tag}.json")
    lstm = _load(f"walkforward_results_lstm_h{horizon}{tag}.json")

    def _summary(model_name: str, res: pd.DataFrame) -> dict:
        return {
            "model": model_name,
            "auc": float(res["auc"].mean()),
            "accuracy": float(res["accuracy"].mean()),
            "base_rate": float(res["base_rate"].mean()),
            "n_folds": len(res),
        }

    rows: list[dict] = []
    if baseline is not None:
        for model_name, g in baseline.groupby("model"):
            rows.append(_summary(model_name, g))
    if arima is not None and len(arima):
        rows.append(_summary("B2_arima", arima))
    if lgbm is not None:
        rows.append(_summary("M1_lightgbm", lgbm))
    if xgb is not None:
        rows.append(_summary("M2_xgboost", xgb))
    if lstm is not None:
        rows.append(_summary("M3_lstm", lstm))

    processed = Repository(cfg.data.processed_dir)
    oof_names = {
        "lightgbm": f"oof_predictions_h{horizon}{tag}",
        "xgboost": f"oof_predictions_xgboost_h{horizon}{tag}",
        "lstm": f"oof_predictions_lstm_h{horizon}{tag}",
    }
    available = {k: processed.read(v) for k, v in oof_names.items() if processed.exists(v)}
    if len(available) >= 2:
        y_col = f"y_dir_{horizon}d"
        fwd_ret_col = f"fwd_ret_{horizon}d"
        ens = rank_average_ensemble(available, y_col, fwd_ret_col)
        ens_scores = score_by_fold(ens, y_col)
        ens_scores.to_json(
            reports_dir / f"walkforward_results_ensemble_h{horizon}{tag}.json",
            orient="records",
            indent=2,
        )
        rows.append(_summary("M4_ensemble", ens_scores))
    else:
        console.print(
            "[yellow]Skipping M4 ensemble — need at least 2 of "
            f"{sorted(oof_names)} OOF files, found {sorted(available)}[/yellow]"
        )

    table = pd.DataFrame(rows)
    table["_order"] = table["model"].map(MODEL_LADDER_ORDER).fillna(99)
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    table[["auc", "accuracy", "base_rate"]] = table[["auc", "accuracy", "base_rate"]].round(4)

    console.print("[bold]Model ladder (mean across folds)[/bold]")
    console.print(table.to_string(index=False))

    out_path = reports_dir / f"model_comparison_h{horizon}{tag}.json"
    out_path.write_text(json.dumps(table.to_dict(orient="records"), indent=2))
    console.print(f"\n[green]wrote[/green] {out_path}")


@app.command(name="explain")
def explain_cmd(
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    config: str = CONFIG_OPTION,
    sample_size: int = typer.Option(5000, "--sample-size", help="Rows to run SHAP over"),
) -> None:
    """SHAP interpretability for the final LightGBM model: global importance, a
    beeswarm plot, and top-10-feature stability across walk-forward folds."""
    import joblib

    from alphabench.evaluation.explain import (
        compute_shap_values,
        fold_feature_stability,
        plot_beeswarm,
        plot_global_importance,
        top_n_features,
    )

    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)

    model_dir = Path(f"models/lightgbm_h{horizon}")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    cols = metadata["features"]

    merged = _load_merged_processed(cfg)
    y_col = f"y_dir_{horizon}d"
    df_pre = _prepare_fold_frame(merged, cfg, y_col)
    sample = df_pre.dropna(subset=cols).sample(
        n=min(sample_size, len(df_pre)), random_state=cfg.seed
    )
    X = sample[cols]  # noqa: N806

    final_model = joblib.load(model_dir / "final.joblib")
    shap_values = compute_shap_values(final_model, X)

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_global_importance(shap_values, cols, figures_dir / f"shap_importance_h{horizon}.png")
    plot_beeswarm(shap_values, X, figures_dir / f"shap_beeswarm_h{horizon}.png")
    console.print(f"[green]wrote[/green] {figures_dir / f'shap_importance_h{horizon}.png'}")
    console.print(f"[green]wrote[/green] {figures_dir / f'shap_beeswarm_h{horizon}.png'}")

    fold_top: dict[int, list[str]] = {}
    for fold_file in sorted(model_dir.glob("fold_*.joblib")):
        year = int(fold_file.stem.removeprefix("fold_"))
        fold_model = joblib.load(fold_file)
        fold_sv = compute_shap_values(fold_model, X)
        fold_top[year] = top_n_features(fold_sv, cols, n=10)

    stability = fold_feature_stability(fold_top)
    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"shap_fold_stability_h{horizon}.json"
    out_path.write_text(json.dumps(stability, indent=2))
    console.print(f"[green]wrote[/green] {out_path}")
    if stability["mean_jaccard"] is not None:
        console.print(
            f"mean pairwise top-10 Jaccard overlap across folds: {stability['mean_jaccard']:.3f}"
        )


@app.command(name="evaluate-holdout")
def evaluate_holdout_cmd(
    model: str = typer.Option("lightgbm", "--model", help="Model to evaluate"),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days"),
    config: str = CONFIG_OPTION,
) -> None:
    """Score the sealed holdout period. Run ONCE. Do not tune afterwards.

    Refuses to run if reports/metrics/holdout_results.json already exists — the holdout
    is single-use by design. The temptation to "just try one more feature" after a
    disappointing holdout is exactly the failure mode the holdout exists to prevent.
    """
    import joblib

    out_path = Path("reports/metrics/holdout_results.json")
    if out_path.exists():
        raise SystemExit(
            "holdout_results.json already exists. The holdout is single-use by design — "
            "re-running it after seeing results is how backtest overfitting happens. "
            "Delete the file deliberately if you truly must."
        )

    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)

    model_dir = Path(f"models/{model}_h{horizon}")
    fitted = joblib.load(model_dir / "final.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    cols = metadata["features"]

    merged = _load_merged_processed(cfg)
    y_col = f"y_dir_{horizon}d"
    fwd_ret_col = f"fwd_ret_{horizon}d"
    holdout = merged[merged["date"] >= cfg.validation.holdout_start].dropna(subset=[y_col, *cols])
    if holdout.empty:
        raise SystemExit(
            f"no rows on/after holdout_start={cfg.validation.holdout_start} — "
            "is data/processed up to date?"
        )

    proba = fitted.predict_proba(holdout[cols])[:, 1]
    y_true = holdout[y_col].to_numpy()

    auc_ci = block_bootstrap_auc(y_true, proba, holdout["date"])
    console.print("[bold]Holdout ROC-AUC[/bold]")
    console.print(
        f"AUC={auc_ci['auc']:.4f}  95% CI=[{auc_ci['ci_lower']:.4f}, {auc_ci['ci_upper']:.4f}]"
    )

    holdout_oof = holdout[["date", "symbol"]].copy()
    holdout_oof["proba"] = proba
    holdout_oof[fwd_ret_col] = holdout[fwd_ret_col].to_numpy()
    # run_backtest expects fwd_ret_1d specifically; alias it for horizons other than 1.
    if fwd_ret_col != "fwd_ret_1d":
        holdout_oof["fwd_ret_1d"] = holdout_oof[fwd_ret_col]

    bt = run_backtest(
        holdout_oof,
        proba_col="proba",
        threshold=cfg.backtest.prob_threshold,
        commission_bps=cfg.backtest.commission_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        allow_short=cfg.backtest.allow_short,
    )
    console.print("\n[bold]Holdout backtest metrics (net of costs)[/bold]")
    console.print(
        pd.DataFrame(bt["metrics"].items(), columns=["metric", "value"]).to_string(index=False)
    )

    combined = {
        "model": model,
        "horizon": horizon,
        "holdout_start": cfg.validation.holdout_start,
        "n_rows": len(holdout),
        "auc": auc_ci,
        "backtest_metrics": bt["metrics"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2))
    console.print(f"\n[green]wrote[/green] {out_path} — holdout is now sealed; do not re-run.")


@app.command(name="diagnose-arima")
def diagnose_arima_cmd(config: str = CONFIG_OPTION) -> None:
    """ADF/KPSS stationarity tests + ACF/PACF on every symbol's own daily log-return
    series (never on price levels — PROPOSAL section 4.1's point is that prices are
    non-stationary and returns are). Justifies the ARIMA order used by B2."""
    logging_conf.setup_logging()
    cfg = config_mod.load_config(config)
    symbols, _benchmark = config_mod.load_universe(cfg.universe["file"])

    panel = Repository(cfg.data.interim_dir).read("panel")
    report = run_diagnostics_report(panel, symbols)

    console.print(
        f"ADF stationary: {report['adf_stationary_pct']:.1%}  "
        f"KPSS stationary: {report['kpss_stationary_pct']:.1%}  "
        f"both agree: {report['both_agree_stationary_pct']:.1%}  "
        f"(n={report['n_symbols']} symbols)"
    )

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "arima_diagnostics.json"
    out_path.write_text(json.dumps(report, indent=2))
    console.print(f"[green]wrote[/green] {out_path}")

    ap = report["representative_acf_pacf"]
    if ap is not None:
        figures_dir = Path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        lags = range(len(ap["acf"]))
        ax1.stem(lags, ap["acf"])
        ax1.set_title(f"ACF — {report['representative_symbol']} daily log returns")
        ax2.stem(lags, ap["pacf"])
        ax2.set_title("PACF")
        ax2.set_xlabel("lag (days)")
        fig.tight_layout()
        fig_path = figures_dir / "arima_acf_pacf.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        console.print(f"[green]wrote[/green] {fig_path}")


if __name__ == "__main__":
    app()
