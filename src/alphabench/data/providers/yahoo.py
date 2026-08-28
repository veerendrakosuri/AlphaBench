from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)
COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, ValueError)),
    reraise=True,
)
def _download_one(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"empty response for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
    df.index.name = "date"
    return df.reset_index()


def fetch(
    symbols: list[str], start: str, end: str | None = None, pause: float = 1.5
) -> pd.DataFrame:
    """Fetch OHLCV for symbols. Returns a long panel. Failures are logged, not raised."""
    frames, failed = [], []
    for i, sym in enumerate(symbols, 1):
        try:
            df = _download_one(sym, start, end)
            df["symbol"] = sym
            frames.append(df)
            log.info("fetched %s (%d/%d) rows=%d", sym, i, len(symbols), len(df))
        except Exception as exc:
            log.warning("FAILED %s: %s", sym, exc)
            failed.append(sym)
        time.sleep(pause)  # be a good citizen; avoids throttling

    if failed:
        log.warning("failed symbols (%d): %s", len(failed), failed)
    if not frames:
        raise RuntimeError("all symbols failed — check connectivity or try the Stooq provider")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    keep = ["date", "symbol", *[c for c in COLUMNS if c in panel.columns]]
    return panel[keep].sort_values(["symbol", "date"]).reset_index(drop=True)
