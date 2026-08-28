from __future__ import annotations

import pandas as pd
import pandas_datareader.data as web


def fetch(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        try:
            df = web.DataReader(sym, "stooq", start, end).sort_index()
            df = df.rename(columns=str.lower).reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = sym
            df["adj_close"] = df["close"]  # Stooq daily is already adjusted
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("stooq failover returned nothing")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)
