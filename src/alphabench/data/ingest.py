from __future__ import annotations

import logging

import pandas as pd

from alphabench.config import Config, load_universe
from alphabench.data.providers import stooq, yahoo
from alphabench.data.repository import Repository
from alphabench.data.validate import validate_panel

log = logging.getLogger(__name__)


def run_ingest(
    cfg: Config, universe_file: str | None = None, incremental: bool = False
) -> pd.DataFrame:
    symbols, benchmark = load_universe(universe_file or cfg.universe["file"])
    all_symbols = [*symbols, benchmark]
    raw = Repository(cfg.data.raw_dir)

    start = cfg.data.start_date
    if incremental and raw.exists("ohlcv"):
        existing = raw.read("ohlcv")
        start = (existing["date"].max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        log.info("incremental ingest from %s", start)

    try:
        panel = yahoo.fetch(all_symbols, start, cfg.data.end_date)
    except Exception as exc:
        log.error("yahoo failed (%s) — trying stooq failover", exc)
        panel = stooq.fetch(all_symbols, start, cfg.data.end_date)

    if incremental and raw.exists("ohlcv"):
        panel = (
            pd.concat([raw.read("ohlcv"), panel], ignore_index=True)
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    raw.write(panel, "ohlcv")
    log.info("wrote %d rows for %d symbols", len(panel), panel["symbol"].nunique())
    return panel


def run_validate(cfg: Config) -> dict:
    panel = Repository(cfg.data.raw_dir).read("ohlcv")

    # Prefer adjusted prices everywhere downstream.
    if "adj_close" in panel.columns:
        ratio = (panel["adj_close"] / panel["close"]).fillna(1.0)
        for col in ["open", "high", "low"]:
            panel[col] = panel[col] * ratio
        panel["close"] = panel["adj_close"]
        panel = panel.drop(columns=["adj_close"])

    report = validate_panel(panel, strict=True)
    Repository(cfg.data.interim_dir).write(panel, "panel")
    return report
