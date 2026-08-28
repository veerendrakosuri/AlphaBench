from __future__ import annotations

from functools import reduce

import typer
from rich.console import Console

from alphabench import config as config_mod
from alphabench import logging_conf
from alphabench.data import ingest
from alphabench.data.repository import Repository
from alphabench.features import pipeline
from alphabench.targets import builder as targets

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


if __name__ == "__main__":
    app()
