from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)
REQUIRED = ["date", "symbol", "open", "high", "low", "close", "volume"]


class DataQualityError(Exception):
    pass


def validate_panel(df: pd.DataFrame, *, strict: bool = True) -> dict:
    issues: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing required columns: {missing}")

    dupes = df.duplicated(subset=["symbol", "date"]).sum()
    if dupes:
        issues.append(f"{dupes} duplicate (symbol, date) rows")

    for sym, g in df.groupby("symbol", sort=False):
        if not g["date"].is_monotonic_increasing:
            issues.append(f"{sym}: dates not sorted")

    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        issues.append(f"{bad_hl} rows with high < low")

    bad_range = (
        (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    ).sum()
    if bad_range:
        issues.append(f"{bad_range} rows with open/close outside [low, high]")

    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if nonpos:
        issues.append(f"{nonpos} rows with non-positive prices")

    negvol = (df["volume"] < 0).sum()
    if negvol:
        issues.append(f"{negvol} rows with negative volume")

    # Likely unadjusted splits — warn, don't fail; needs human eyes.
    px = "adj_close" if "adj_close" in df.columns else "close"
    rets = df.groupby("symbol")[px].pct_change()
    jumps = (rets.abs() > 0.50).sum()
    if jumps:
        warnings.append(f"{jumps} single-day moves >50% — inspect for unadjusted actions")

    nulls = df[REQUIRED].isna().sum()
    if nulls.any():
        warnings.append(f"nulls: {nulls[nulls > 0].to_dict()}")

    report = {
        "rows": len(df),
        "symbols": df["symbol"].nunique(),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "issues": issues,
        "warnings": warnings,
    }
    for w in warnings:
        log.warning("data quality: %s", w)
    if issues and strict:
        raise DataQualityError(f"validation failed: {issues}")
    return report
