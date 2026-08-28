from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class Repository:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def write(self, df: pd.DataFrame, name: str) -> Path:
        path = self._p(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        return path

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self._p(name), engine="pyarrow")

    def exists(self, name: str) -> bool:
        return self._p(name).exists()

    def query(self, sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
        """Run DuckDB SQL over parquet files or in-memory frames.

        Example:
            repo.query("SELECT symbol, count(*) FROM panel GROUP BY 1",
                       panel=repo.read("panel"))
        """
        con = duckdb.connect()
        try:
            for alias, frame in frames.items():
                con.register(alias, frame)
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def query_file(self, name: str, sql_template: str) -> pd.DataFrame:
        """SQL directly against a parquet file without loading it. Use {src} placeholder."""
        con = duckdb.connect()
        try:
            return con.execute(
                sql_template.format(src=f"read_parquet('{self._p(name)}')")
            ).fetchdf()
        finally:
            con.close()
