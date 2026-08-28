from __future__ import annotations

import typer
from rich.console import Console

from alphabench import config as config_mod
from alphabench import logging_conf
from alphabench.data import ingest

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


if __name__ == "__main__":
    app()
