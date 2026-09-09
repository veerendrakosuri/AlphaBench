# AlphaBench — Build Plan (Zero to Deployed)

Copy-paste-ready. Assumes Python 3.12, git, and Docker installed. Commands are bash (macOS/Linux/WSL); Windows PowerShell differences are flagged.

Stages map to proposal weeks, but each stage is independently runnable.

---

## STAGE 0 — Scaffold (Week 1, Day 1)

```bash
mkdir alphabench && cd alphabench
git init

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

mkdir -p config/models docker .github/workflows
mkdir -p data/{raw,interim,processed,external}
mkdir -p models mlruns reports/{figures,metrics} notebooks tests
mkdir -p src/alphabench/{data/providers,features,targets,validation,models,training,evaluation,api/routes,dashboard/components}

find src -type d -exec touch {}/__init__.py \;
find data models reports/figures reports/metrics -type d -exec touch {}/.gitkeep \;
```

Create `requirements.txt` and `requirements-dev.txt` from the companion file, then:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

**`pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "alphabench"
version = "0.1.0"
description = "Walk-forward equity return forecasting and backtesting platform"
requires-python = ">=3.12"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}

[tool.setuptools.packages.find]
where = ["src"]

[project.scripts]
alphabench = "alphabench.cli:app"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "PD", "NPY", "RUF"]
ignore = ["E501", "PD901"]

# Hard guardrail: random CV on time series is a fatal bug in this project.
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"sklearn.model_selection.KFold".msg = "Time-series data: use alphabench.validation.splitters.WalkForwardSplit"
"sklearn.model_selection.train_test_split".msg = "Time-series data: use WalkForwardSplit"
"sklearn.model_selection.StratifiedKFold".msg = "Time-series data: use WalkForwardSplit"

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

```bash
pip install -e .
```

**`.gitignore`**

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
data/**
!data/**/.gitkeep
models/**
!models/.gitkeep
mlruns/
reports/figures/*
!reports/figures/.gitkeep
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.DS_Store
```

**`.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=5000]
  - repo: local
    hooks:
      - id: no-notebook-output
        name: strip notebook outputs
        entry: jupyter nbconvert --clear-output --inplace
        language: system
        files: \.ipynb$
```

```bash
pre-commit install
```

**`Makefile`** — tabs, not spaces, for indentation.

```makefile
.PHONY: setup data features train backtest api dashboard test lint all clean

setup:
	pip install -r requirements.txt -r requirements-dev.txt && pip install -e .

data:
	python -m alphabench.cli ingest --universe config/universe_us.yaml
	python -m alphabench.cli validate

features:
	python -m alphabench.cli build-features

train:
	python -m alphabench.cli train --model lightgbm --horizon 1

backtest:
	python -m alphabench.cli backtest --model lightgbm --horizon 1

holdout:
	python -m alphabench.cli evaluate-holdout --model lightgbm --horizon 1

api:
	uvicorn alphabench.api.main:app --reload --port 8000

dashboard:
	streamlit run src/alphabench/dashboard/app.py

test:
	pytest tests/ -v

lint:
	ruff check src tests && ruff format --check src tests && mypy src

all: data features train backtest

clean:
	rm -rf data/interim/* data/processed/* models/* reports/figures/*
```

**`config/config.yaml`**

```yaml
project:
  name: alphabench
  seed: 42

data:
  start_date: "2010-01-01"
  end_date: null            # null = today
  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed

universe:
  file: config/universe_us.yaml
  benchmark: SPY

target:
  horizons: [1, 5]
  deadband_kappa: 0.3
  vol_window: 20

validation:
  train_start: "2010-01-01"
  first_val_year: 2017
  last_val_year: 2023
  holdout_start: "2025-01-01"   # DO NOT TOUCH UNTIL WEEK 9
  purge_days: 5
  embargo_days: 5

backtest:
  commission_bps: 5.0
  slippage_bps: 5.0
  prob_threshold: 0.55
  allow_short: false
  execution: next_open
```

**`config/universe_us.yaml`** — spread across sectors. Do not pick 30 tech names.

```yaml
tickers:
  - {symbol: AAPL,  sector: Technology}
  - {symbol: MSFT,  sector: Technology}
  - {symbol: NVDA,  sector: Technology}
  - {symbol: AVGO,  sector: Technology}
  - {symbol: JPM,   sector: Financials}
  - {symbol: BAC,   sector: Financials}
  - {symbol: GS,    sector: Financials}
  - {symbol: BRK-B, sector: Financials}
  - {symbol: JNJ,   sector: Healthcare}
  - {symbol: UNH,   sector: Healthcare}
  - {symbol: PFE,   sector: Healthcare}
  - {symbol: ABBV,  sector: Healthcare}
  - {symbol: XOM,   sector: Energy}
  - {symbol: CVX,   sector: Energy}
  - {symbol: COP,   sector: Energy}
  - {symbol: PG,    sector: ConsumerStaples}
  - {symbol: KO,    sector: ConsumerStaples}
  - {symbol: WMT,   sector: ConsumerStaples}
  - {symbol: COST,  sector: ConsumerStaples}
  - {symbol: HD,    sector: ConsumerDiscretionary}
  - {symbol: MCD,   sector: ConsumerDiscretionary}
  - {symbol: NKE,   sector: ConsumerDiscretionary}
  - {symbol: CAT,   sector: Industrials}
  - {symbol: BA,    sector: Industrials}
  - {symbol: UNP,   sector: Industrials}
  - {symbol: HON,   sector: Industrials}
  - {symbol: LIN,   sector: Materials}
  - {symbol: NEE,   sector: Utilities}
  - {symbol: AMT,   sector: RealEstate}
  - {symbol: VZ,    sector: CommunicationServices}
benchmark: SPY
```

For the India variant, `config/universe_in.yaml` uses `.NS` symbols (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`, …) with `benchmark: ^NSEI`.

**Verify Stage 0:**

```bash
python -c "import alphabench; print('package importable')"
make lint
```

---

## STAGE 1 — Data Ingestion (Week 1)

### 1.1 Config loader — `src/alphabench/config.py`

```python
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class DataCfg(BaseModel):
    start_date: str
    end_date: str | None = None
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


class TargetCfg(BaseModel):
    horizons: list[int]
    deadband_kappa: float
    vol_window: int


class ValidationCfg(BaseModel):
    train_start: str
    first_val_year: int
    last_val_year: int
    holdout_start: str
    purge_days: int
    embargo_days: int


class BacktestCfg(BaseModel):
    commission_bps: float
    slippage_bps: float
    prob_threshold: float
    allow_short: bool
    execution: str


class Config(BaseModel):
    project: dict
    data: DataCfg
    universe: dict
    target: TargetCfg
    validation: ValidationCfg
    backtest: BacktestCfg

    @property
    def seed(self) -> int:
        return int(self.project.get("seed", 42))


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(ROOT / path) as f:
        return Config(**yaml.safe_load(f))


def load_universe(path: str | Path) -> tuple[list[str], str]:
    with open(ROOT / path) as f:
        u = yaml.safe_load(f)
    return [t["symbol"] for t in u["tickers"]], u["benchmark"]
```

### 1.2 Provider with backoff — `src/alphabench/data/providers/yahoo.py`

The backoff is not optional. Yahoo rate-limits aggressively and a naive loop over 30 tickers will get throttled.

```python
from __future__ import annotations
import logging
import time
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
        symbol, start=start, end=end,
        auto_adjust=False, actions=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"empty response for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
    df.index.name = "date"
    return df.reset_index()


def fetch(symbols: list[str], start: str, end: str | None = None,
          pause: float = 1.5) -> pd.DataFrame:
    """Fetch OHLCV for symbols. Returns a long panel. Failures are logged, not raised."""
    frames, failed = [], []
    for i, sym in enumerate(symbols, 1):
        try:
            df = _download_one(sym, start, end)
            df["symbol"] = sym
            frames.append(df)
            log.info("fetched %s (%d/%d) rows=%d", sym, i, len(symbols), len(df))
        except Exception as exc:              # noqa: BLE001
            log.warning("FAILED %s: %s", sym, exc)
            failed.append(sym)
        time.sleep(pause)                     # be a good citizen; avoids throttling

    if failed:
        log.warning("failed symbols (%d): %s", len(failed), failed)
    if not frames:
        raise RuntimeError("all symbols failed — check connectivity or try the Stooq provider")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    keep = ["date", "symbol", *[c for c in COLUMNS if c in panel.columns]]
    return panel[keep].sort_values(["symbol", "date"]).reset_index(drop=True)
```

**Failover — `src/alphabench/data/providers/stooq.py`**

```python
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
            df["adj_close"] = df["close"]      # Stooq daily is already adjusted
            frames.append(df)
        except Exception:                      # noqa: BLE001
            continue
    if not frames:
        raise RuntimeError("stooq failover returned nothing")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)
```

### 1.3 Repository — `src/alphabench/data/repository.py`

The only module that touches disk. Everything else asks it for dataframes.

```python
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
```

### 1.4 Validation — `src/alphabench/data/validate.py`

```python
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
        (df["close"] > df["high"]) | (df["close"] < df["low"])
        | (df["open"] > df["high"]) | (df["open"] < df["low"])
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
```

### 1.5 Ingest orchestration — `src/alphabench/data/ingest.py`

```python
from __future__ import annotations
import logging
import pandas as pd
from alphabench.config import Config, load_universe
from alphabench.data.providers import yahoo, stooq
from alphabench.data.repository import Repository
from alphabench.data.validate import validate_panel

log = logging.getLogger(__name__)


def run_ingest(cfg: Config, universe_file: str | None = None,
               incremental: bool = False) -> pd.DataFrame:
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
    except Exception as exc:                       # noqa: BLE001
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
```

### 1.6 Run it

```bash
make data
```

Expect 3–5 minutes for 31 symbols with the 1.5 s pause. If you see rate-limit errors, raise `pause` to 3.0 and rerun — the incremental path means you won't refetch what you already have.

**Verify:**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/interim/panel.parquet')
print(df.shape)
print(df.groupby('symbol')['date'].agg(['min','max','count']).head())
"
```

---

## STAGE 2 — Features & Targets (Week 3)

### 2.1 Shift guard — `src/alphabench/features/base.py`

Every feature function is registered and forced through a lag. This decorator is the structural defence against leakage.

```python
from __future__ import annotations
from collections.abc import Callable
import pandas as pd

FEATURE_REGISTRY: dict[str, Callable] = {}


def feature(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        FEATURE_REGISTRY[name] = fn
        return fn
    return deco


def safe_shift(df: pd.DataFrame, cols: list[str], by: str = "symbol",
               lag: int = 1) -> pd.DataFrame:
    """Lag feature columns by `lag` within each symbol.

    Applied to EVERY feature before it reaches a model. A feature computed from
    the close of day t is only usable for a decision made after that close, so
    it is aligned to t+1. This one call prevents the most common fatal bug.
    """
    out = df.copy()
    out[cols] = out.groupby(by, sort=False)[cols].shift(lag)
    return out
```

### 2.2 Technical indicators — `src/alphabench/features/technical.py`

Hand-rolled rather than imported. Three reasons: no `ta-lib` C-compilation pain in Docker, no beta-version dependency risk, and you can explain every line in a viva. (`pandas-ta==0.4.71b0` does work on NumPy 2.x if you prefer it — it is verified — but it is a beta.)

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    # Normalise by price: a raw MACD of 2.0 means something different at $10 vs $500.
    return pd.DataFrame({
        "macd": line / close,
        "macd_signal": sig / close,
        "macd_hist": (line - sig) / close,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / close


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper, lower = ma + k * sd, ma - k * sd
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({
        "bb_pctb": (close - lower) / width,
        "bb_width": (upper - lower) / ma,
    })


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               window: int = 14, smooth: int = 3) -> pd.DataFrame:
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    rng = (hh - ll).replace(0.0, np.nan)
    k = 100.0 * (close - ll) / rng
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(smooth).mean()})


def obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    return obv.diff(window) / volume.rolling(window).sum().replace(0.0, np.nan)
```

### 2.3 Feature pipeline — `src/alphabench/features/pipeline.py`

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from alphabench.features import technical as tech
from alphabench.features.base import safe_shift

MOM_LAGS = [1, 2, 3, 5, 10, 21, 63, 126, 252]
VOL_WINDOWS = [5, 10, 21, 63]


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, low, v = g["close"], g["high"], g["low"], g["volume"]
    logret = np.log(c).diff()
    g["ret_1d"] = logret

    # --- momentum -----------------------------------------------------------
    for k in MOM_LAGS:
        g[f"mom_{k}d"] = np.log(c / c.shift(k))
    g["mom_21d_skip5"] = np.log(c.shift(5) / c.shift(26))   # reversal-cleaned momentum

    # --- volatility ---------------------------------------------------------
    for w in VOL_WINDOWS:
        g[f"vol_{w}d"] = logret.rolling(w).std() * np.sqrt(252)
    g["vol_ratio"] = g["vol_21d"] / g["vol_21d"].rolling(252).median()
    g["vol_of_vol"] = g["vol_21d"].rolling(63).std()
    # Parkinson high-low estimator: uses intraday range, lower variance than close-to-close
    g["parkinson_21d"] = np.sqrt(
        (np.log(h / low) ** 2).rolling(21).mean() / (4 * np.log(2))
    ) * np.sqrt(252)

    # --- technical ----------------------------------------------------------
    g["rsi_14"] = tech.rsi(c, 14)
    g = g.join(tech.macd(c))
    g["atr_14"] = tech.atr(h, low, c, 14)
    g = g.join(tech.bollinger(c))
    g = g.join(tech.stochastic(h, low, c))
    g["obv_slope_20"] = tech.obv_slope(c, v, 20)
    for w in (20, 50, 200):
        g[f"px_to_sma{w}"] = c / c.rolling(w).mean() - 1.0

    # --- volume -------------------------------------------------------------
    lv = np.log1p(v)
    g["vol_z_21"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
    g["dollar_vol_z"] = np.log1p(c * v).pipe(
        lambda s: (s - s.rolling(21).mean()) / s.rolling(21).std()
    )
    g["amihud"] = (logret.abs() / (c * v).replace(0.0, np.nan)).rolling(21).mean() * 1e9

    return g


def build_features(panel: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    bench = panel.loc[panel["symbol"] == benchmark, ["date", "close"]].copy()
    bench["mkt_ret_1d"] = np.log(bench["close"]).diff()
    bench["mkt_ret_5d"] = np.log(bench["close"] / bench["close"].shift(5))
    bench["mkt_vol_21d"] = bench["mkt_ret_1d"].rolling(21).std() * np.sqrt(252)
    bench = bench.drop(columns=["close"])

    stocks = panel[panel["symbol"] != benchmark].copy()

    # Explicit concat over groups rather than groupby().apply(): apply's return
    # shape and index behaviour shifted across pandas 2.x, and reset_index(drop=True)
    # on its output can SILENTLY MISALIGN rows. Explicit is safe.
    feats = pd.concat(
        [_per_symbol(g) for _, g in stocks.groupby("symbol", sort=False)],
        ignore_index=True,
    )
    feats = feats.merge(bench, on="date", how="left")
    feats = feats.sort_values(["symbol", "date"]).reset_index(drop=True)

    # rolling 60-day beta to the benchmark
    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["ret_1d"].rolling(60).cov(g["mkt_ret_1d"])
        var = g["mkt_ret_1d"].rolling(60).var()
        return cov / var.replace(0.0, np.nan)

    # sort_index() realigns each group's Series back onto feats' own index.
    feats["beta_60d"] = pd.concat(
        [_beta(g) for _, g in feats.groupby("symbol", sort=False)]
    ).sort_index()

    # --- cross-sectional ranks: only possible because this is a panel --------
    for col in ["ret_1d", "mom_21d", "vol_21d", "vol_z_21", "rsi_14"]:
        feats[f"xs_rank_{col}"] = feats.groupby("date")[col].rank(pct=True)

    # --- calendar -----------------------------------------------------------
    d = feats["date"]
    feats["dow"] = d.dt.dayofweek
    feats["month"] = d.dt.month
    feats["is_month_end"] = d.dt.is_month_end.astype(int)
    feats["is_quarter_end"] = d.dt.is_quarter_end.astype(int)

    # --- THE CRITICAL STEP: lag everything by one day -----------------------
    meta = ["date", "symbol", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in feats.columns if c not in meta]
    feats = safe_shift(feats, feature_cols, by="symbol", lag=1)

    return feats.sort_values(["symbol", "date"]).reset_index(drop=True)
```

> **Why lag the calendar features too?** `dow` for day *t* is genuinely known in advance, so lagging it costs a little information. It is lagged anyway because a blanket rule you cannot forget is safer than a per-feature exemption list you will eventually get wrong. The cost is one day of calendar signal; the benefit is that leakage cannot enter through this door.

### 2.4 Targets — `src/alphabench/targets/builder.py`

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def build_targets(panel: pd.DataFrame, horizon: int = 1, kappa: float = 0.3,
                  vol_window: int = 20) -> pd.DataFrame:
    """Forward returns + volatility-scaled deadband direction labels.

    y = 1  if fwd_ret >  kappa * sigma_t
    y = 0  if fwd_ret < -kappa * sigma_t
    y = NaN otherwise (dropped at training time — those days are noise)
    """
    out = panel[["date", "symbol", "close"]].sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)["close"]

    out[f"fwd_ret_{horizon}d"] = np.log(g.shift(-horizon) / out["close"])

    daily = np.log(g.transform(lambda s: s / s.shift(1)))
    sigma = daily.groupby(out["symbol"]).transform(
        lambda s: s.rolling(vol_window).std()
    ) * np.sqrt(horizon)

    fwd = out[f"fwd_ret_{horizon}d"]
    band = kappa * sigma
    y = pd.Series(np.nan, index=out.index)
    y[fwd > band] = 1.0
    y[fwd < -band] = 0.0

    out[f"y_dir_{horizon}d"] = y
    out[f"sigma_{horizon}d"] = sigma
    return out.drop(columns=["close"])
```

### 2.5 Build and verify

```bash
make features
```

**Sanity checks that catch real bugs:**

```bash
python -c "
import pandas as pd, numpy as np
f = pd.read_parquet('data/processed/features.parquet')
t = pd.read_parquet('data/processed/targets.parquet')
df = f.merge(t, on=['date','symbol'])

print('rows:', len(df), '| features:', f.shape[1])
print('label balance:', df['y_dir_1d'].value_counts(normalize=True).round(4).to_dict())
print('deadband dropped:', df['y_dir_1d'].isna().mean().round(3))

# Any feature correlating strongly with the target is a leak, not a discovery.
feats = [c for c in f.columns if c not in ['date','symbol','open','high','low','close','volume']]
corr = df[feats].corrwith(df['fwd_ret_1d']).abs().sort_values(ascending=False)
print('\nTop |corr| with forward return:')
print(corr.head(8).round(4))
print('\n>>> Anything above ~0.10 here is almost certainly LEAKAGE. Investigate before proceeding.')
"
```

Realistic output: top correlations in the 0.01–0.06 range. If you see 0.4, you have a bug — find it now, not in Week 9.

---

## STAGE 3 — Walk-Forward Splitter & Leakage Tests (Week 4)

This stage is the intellectual core of the project. Write it before you have results you might be tempted to protect.

### 3.1 Splitter — `src/alphabench/validation/splitters.py`

```python
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    """Expanding-window walk-forward splitter with purging and embargo.

    purge_days:   drop training samples whose label window (t .. t+horizon)
                  overlaps the validation block. Without this the last
                  `horizon` training rows leak validation information.
    embargo_days: additional gap after training to break serial correlation
                  between adjacent feature windows.
    """
    train_start: str
    first_val_year: int
    last_val_year: int
    horizon: int = 1
    purge_days: int = 5
    embargo_days: int = 5

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        t0 = pd.Timestamp(self.train_start)
        gap = pd.Timedelta(days=self.purge_days + self.embargo_days + self.horizon)

        for year in range(self.first_val_year, self.last_val_year + 1):
            val_start = pd.Timestamp(f"{year}-01-01")
            val_end = pd.Timestamp(f"{year}-12-31")
            train_end = val_start - gap

            train_idx = np.where((dates >= t0) & (dates <= train_end))[0]
            val_idx = np.where((dates >= val_start) & (dates <= val_end))[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, *_args) -> int:
        return self.last_val_year - self.first_val_year + 1
```

### 3.2 Leakage tests — `tests/test_leakage.py`

The most important file in the repository.

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from alphabench.validation.splitters import WalkForwardSplit


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Pure random walk: by construction there is NO signal. Any model that
    finds one is finding a bug in our pipeline."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    frames = []
    for sym in ["AAA", "BBB", "CCC"]:
        r = rng.normal(0, 0.01, len(dates))
        close = 100 * np.exp(np.cumsum(r))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": sym, "close": close,
            "open": close * (1 + rng.normal(0, 0.001, len(dates))),
            "high": close * (1 + abs(rng.normal(0, 0.004, len(dates)))),
            "low": close * (1 - abs(rng.normal(0, 0.004, len(dates)))),
            "volume": rng.integers(1e6, 5e6, len(dates)),
        }))
    return pd.concat(frames, ignore_index=True)


def test_splitter_train_always_precedes_validation(synthetic_panel):
    dates = synthetic_panel["date"]
    sp = WalkForwardSplit("2015-01-01", 2020, 2023)
    n = 0
    for tr, va in sp.split(dates):
        assert dates.iloc[tr].max() < dates.iloc[va].min(), "TRAIN OVERLAPS VALIDATION"
        n += 1
    assert n == 4


def test_splitter_enforces_purge_gap(synthetic_panel):
    dates = synthetic_panel["date"]
    sp = WalkForwardSplit("2015-01-01", 2020, 2023, horizon=5,
                          purge_days=5, embargo_days=5)
    for tr, va in sp.split(dates):
        gap = (dates.iloc[va].min() - dates.iloc[tr].max()).days
        assert gap >= 10, f"purge/embargo gap only {gap} days"


def test_splitter_train_set_expands(synthetic_panel):
    dates = synthetic_panel["date"]
    sizes = [len(tr) for tr, _ in WalkForwardSplit("2015-01-01", 2020, 2023).split(dates)]
    assert sizes == sorted(sizes), "expanding window must not shrink"


def test_no_feature_correlates_with_future(synthetic_panel):
    """On a random walk, no lagged feature may correlate with the forward return."""
    from alphabench.features.pipeline import build_features
    from alphabench.targets.builder import build_targets

    p = synthetic_panel.copy()
    bench = p[p["symbol"] == "AAA"].assign(symbol="SPY")
    feats = build_features(pd.concat([p, bench], ignore_index=True), benchmark="SPY")
    tgts = build_targets(p, horizon=1)
    df = feats.merge(tgts, on=["date", "symbol"]).dropna(subset=["fwd_ret_1d"])

    meta = {"date", "symbol", "open", "high", "low", "close", "volume"}
    cols = [c for c in feats.columns if c not in meta]
    corr = df[cols].corrwith(df["fwd_ret_1d"]).abs()
    worst = corr.idxmax()
    assert corr.max() < 0.10, f"LEAK: '{worst}' correlates {corr.max():.3f} with the future"


def test_shuffled_labels_destroy_performance(synthetic_panel):
    """AUC must collapse to ~0.5 when labels are shuffled. If it doesn't,
    the evaluation harness itself is broken."""
    rng = np.random.default_rng(1)
    n, p = 4000, 20
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)          # labels independent of X

    split = int(0.7 * n)
    m = LGBMClassifier(n_estimators=50, verbose=-1, random_state=0)
    m.fit(X[:split], y[:split])
    auc = roc_auc_score(y[split:], m.predict_proba(X[split:])[:, 1])
    assert 0.40 < auc < 0.60, f"AUC {auc:.3f} on pure noise — harness is broken"


def test_targets_use_only_future_prices(synthetic_panel):
    """fwd_ret at time t must equal the realised return from t to t+h."""
    from alphabench.targets.builder import build_targets
    t = build_targets(synthetic_panel, horizon=1)
    df = synthetic_panel.merge(t, on=["date", "symbol"]).sort_values(["symbol", "date"])
    g = df[df["symbol"] == "AAA"].reset_index(drop=True)
    expected = np.log(g["close"].shift(-1) / g["close"])
    pd.testing.assert_series_equal(
        g["fwd_ret_1d"].dropna(), expected.dropna(),
        check_names=False, rtol=1e-9,
    )
```

```bash
pytest tests/test_leakage.py -v
```

**All of these must pass before you train anything you intend to report.**

---

## STAGE 4 — Training (Weeks 5–6)

### 4.1 Walk-forward training loop — `src/alphabench/training/train.py`

```python
from __future__ import annotations
import json
import logging
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score

from alphabench.validation.splitters import WalkForwardSplit

log = logging.getLogger(__name__)
META = {"date", "symbol", "open", "high", "low", "close", "volume"}

DEFAULT_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=15, max_depth=4,
    min_child_samples=200, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
    verbose=-1, n_jobs=-1, random_state=42,
)
# Deliberately conservative: shallow trees, heavy regularisation, small learning
# rate. In a 52%-signal regime the default LightGBM settings memorise noise.


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META and not c.startswith(("y_", "fwd_", "sigma_"))]


def train_walkforward(df: pd.DataFrame, horizon: int, cfg, params: dict | None = None,
                      out_dir: Path = Path("models/lightgbm_h1")) -> pd.DataFrame:
    params = {**DEFAULT_PARAMS, **(params or {})}
    out_dir.mkdir(parents=True, exist_ok=True)

    y_col = f"y_dir_{horizon}d"
    df = df[df["date"] < cfg.validation.holdout_start]        # holdout is sealed
    df = df.dropna(subset=[y_col]).reset_index(drop=True)     # drop deadband rows
    cols = _feature_cols(df)

    sp = WalkForwardSplit(
        cfg.validation.train_start, cfg.validation.first_val_year,
        cfg.validation.last_val_year, horizon=horizon,
        purge_days=cfg.validation.purge_days,
        embargo_days=cfg.validation.embargo_days,
    )

    mlflow.set_experiment("alphabench")
    rows = []
    with mlflow.start_run(run_name=f"lightgbm_h{horizon}"):
        mlflow.log_params({**params, "horizon": horizon, "n_features": len(cols)})

        for i, (tr, va) in enumerate(sp.split(df["date"]), 1):
            Xtr, ytr = df.loc[tr, cols], df.loc[tr, y_col]
            Xva, yva = df.loc[va, cols], df.loc[va, y_col]
            year = df.loc[va, "date"].dt.year.iloc[0]

            model = LGBMClassifier(**params)
            model.fit(
                Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            proba = model.predict_proba(Xva)[:, 1]

            m = {
                "fold": i, "val_year": int(year),
                "n_train": len(tr), "n_val": len(va),
                "base_rate": float(yva.mean()),
                "auc": float(roc_auc_score(yva, proba)),
                "accuracy": float(accuracy_score(yva, (proba > 0.5).astype(int))),
                "brier": float(brier_score_loss(yva, proba)),
                "logloss": float(log_loss(yva, proba)),
                "best_iter": int(model.best_iteration_ or params["n_estimators"]),
            }
            rows.append(m)
            for k, v in m.items():
                if k != "fold":
                    mlflow.log_metric(f"fold{i}_{k}", v)
            log.info("fold %d (%d): AUC=%.4f acc=%.4f base=%.4f",
                     i, year, m["auc"], m["accuracy"], m["base_rate"])

            joblib.dump(model, out_dir / f"fold_{year}.joblib")

        res = pd.DataFrame(rows)
        mlflow.log_metric("mean_auc", res["auc"].mean())
        mlflow.log_metric("std_auc", res["auc"].std())

        # Final model: refit on everything before the holdout, using the median
        # best-iteration from CV so we don't need an eval set.
        final = LGBMClassifier(**{**params, "n_estimators": int(res["best_iter"].median())})
        final.fit(df[cols], df[y_col])
        joblib.dump(final, out_dir / "final.joblib")

        (out_dir / "metadata.json").write_text(json.dumps({
            "features": cols, "params": params, "horizon": horizon,
            "train_start": str(df["date"].min().date()),
            "train_end": str(df["date"].max().date()),
            "cv_mean_auc": float(res["auc"].mean()),
            "n_rows": int(len(df)),
        }, indent=2))

    res.to_json("reports/metrics/walkforward_results.json", orient="records", indent=2)
    print(res.to_string(index=False))
    print(f"\nMean AUC {res['auc'].mean():.4f} ± {res['auc'].std():.4f}")
    print(">>> Compare against 0.5000. Anything above 0.60 warrants a leakage audit.")
    return res
```

### 4.2 Baselines — `src/alphabench/models/baselines.py`

Run these **first** and print B0 at the top of every results table.

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def b0_persistence(df: pd.DataFrame, y_col: str) -> dict:
    """Predict tomorrow's direction = today's direction."""
    d = df.dropna(subset=[y_col, "ret_1d"])
    pred = (d["ret_1d"] > 0).astype(int)
    return {
        "model": "B0_persistence",
        "accuracy": float(accuracy_score(d[y_col], pred)),
        "auc": float(roc_auc_score(d[y_col], pred)),
        "base_rate": float(d[y_col].mean()),
    }


def b0_majority(df: pd.DataFrame, y_col: str) -> dict:
    d = df.dropna(subset=[y_col])
    maj = int(d[y_col].mode().iloc[0])
    return {
        "model": "B0_majority",
        "accuracy": float(accuracy_score(d[y_col], np.full(len(d), maj))),
        "auc": 0.5,
        "base_rate": float(d[y_col].mean()),
    }


def b1_logistic(train: pd.DataFrame, val: pd.DataFrame, y_col: str,
                cols: list[str] | None = None) -> dict:
    cols = cols or ["ret_1d", "mom_5d", "vol_21d", "rsi_14", "vol_z_21"]
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.1),
    )
    tr = train.dropna(subset=[y_col])
    va = val.dropna(subset=[y_col])
    pipe.fit(tr[cols], tr[y_col])                # scaler fitted on TRAIN ONLY
    proba = pipe.predict_proba(va[cols])[:, 1]
    return {
        "model": "B1_logistic",
        "accuracy": float(accuracy_score(va[y_col], (proba > 0.5).astype(int))),
        "auc": float(roc_auc_score(va[y_col], proba)),
        "base_rate": float(va[y_col].mean()),
    }
```

### 4.3 Optuna tuning — `src/alphabench/training/tune.py`

```python
from __future__ import annotations
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

from alphabench.validation.splitters import WalkForwardSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(df: pd.DataFrame, cols: list[str], y_col: str, cfg,
         horizon: int = 1, n_trials: int = 50) -> optuna.Study:
    sp = WalkForwardSplit(
        cfg.validation.train_start, cfg.validation.first_val_year,
        cfg.validation.last_val_year, horizon=horizon,
        purge_days=cfg.validation.purge_days,
        embargo_days=cfg.validation.embargo_days,
    )
    folds = list(sp.split(df["date"]))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 500),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            subsample_freq=1, verbose=-1, n_jobs=-1, random_state=42,
        )
        aucs = []
        for tr, va in folds:
            m = LGBMClassifier(**params)
            m.fit(df.loc[tr, cols], df.loc[tr, y_col],
                  eval_set=[(df.loc[va, cols], df.loc[va, y_col])], eval_metric="auc",
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
            aucs.append(roc_auc_score(df.loc[va, y_col],
                                      m.predict_proba(df.loc[va, cols])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"best mean AUC: {study.best_value:.4f}")
    print(f"trials run: {len(study.trials)}  <-- RECORD THIS for the deflated Sharpe")
    return study
```

> **Record `len(study.trials)`.** The deflated Sharpe ratio in Stage 6 requires it. Reporting "best Sharpe 0.8 across 200 trials" is honest; reporting "Sharpe 0.8" alone is not.

```bash
make train
```

---

## STAGE 5 — Backtesting (Week 7)

### 5.1 Cost-aware engine — `src/alphabench/evaluation/backtest.py`

```python
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def generate_positions(proba: pd.Series, threshold: float = 0.55,
                       allow_short: bool = False) -> pd.Series:
    """Map probabilities to positions. The band around 0.5 means 'no trade'."""
    pos = pd.Series(0.0, index=proba.index)
    pos[proba > threshold] = 1.0
    if allow_short:
        pos[proba < (1 - threshold)] = -1.0
    return pos


def run_backtest(df: pd.DataFrame, proba_col: str = "proba",
                 threshold: float = 0.55, commission_bps: float = 5.0,
                 slippage_bps: float = 5.0, allow_short: bool = False) -> dict:
    """Equal-weight portfolio over the panel.

    Timing: signal from the close of day t; the return earned is the NEXT day's
    return. `fwd_ret_1d` is already t -> t+1, so no further shift is applied.
    Costs are charged on position CHANGES, per side.
    """
    d = df.dropna(subset=[proba_col, "fwd_ret_1d"]).sort_values(["date", "symbol"]).copy()
    d["position"] = generate_positions(d[proba_col], threshold, allow_short)

    d["prev_position"] = d.groupby("symbol")["position"].shift(1).fillna(0.0)
    d["turnover"] = (d["position"] - d["prev_position"]).abs()

    cost_rate = (commission_bps + slippage_bps) / 10_000.0
    d["gross_ret"] = d["position"] * d["fwd_ret_1d"]
    d["cost"] = d["turnover"] * cost_rate
    d["net_ret"] = d["gross_ret"] - d["cost"]

    daily = d.groupby("date").agg(
        gross=("gross_ret", "mean"),
        net=("net_ret", "mean"),
        turnover=("turnover", "mean"),
        n_positions=("position", lambda s: int((s != 0).sum())),
    )
    bench = d.groupby("date")["fwd_ret_1d"].mean()          # equal-weight buy & hold

    return {
        "daily": daily,
        "benchmark": bench,
        "equity": (1 + daily["net"]).cumprod(),
        "equity_gross": (1 + daily["gross"]).cumprod(),
        "equity_bench": (1 + bench).cumprod(),
        "metrics": _metrics(daily["net"], bench, daily["turnover"]),
        "trades": d,
    }


def _metrics(net: pd.Series, bench: pd.Series, turnover: pd.Series) -> dict:
    def ann_ret(r):  return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
    def ann_vol(r):  return float(r.std() * np.sqrt(TRADING_DAYS))
    def sharpe(r):
        v = ann_vol(r)
        return float(ann_ret(r) / v) if v > 0 else 0.0
    def sortino(r):
        dn = r[r < 0].std() * np.sqrt(TRADING_DAYS)
        return float(ann_ret(r) / dn) if dn > 0 else 0.0
    def max_dd(r):
        eq = (1 + r).cumprod()
        return float((eq / eq.cummax() - 1).min())

    mdd = max_dd(net)
    active = net[net != 0]
    wins = active[active > 0]
    losses = active[active < 0]

    return {
        "ann_return": ann_ret(net),
        "ann_vol": ann_vol(net),
        "sharpe": sharpe(net),
        "sortino": sortino(net),
        "max_drawdown": mdd,
        "calmar": float(ann_ret(net) / abs(mdd)) if mdd < 0 else 0.0,
        "hit_rate": float((active > 0).mean()) if len(active) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "avg_daily_turnover": float(turnover.mean()),
        "n_days": int(len(net)),
        "benchmark_ann_return": ann_ret(bench),
        "benchmark_sharpe": sharpe(bench),
        "benchmark_max_drawdown": max_dd(bench),
        "excess_sharpe": sharpe(net) - sharpe(bench),
    }


def cost_sensitivity(df: pd.DataFrame, proba_col: str = "proba",
                     threshold: float = 0.55,
                     bps_levels: tuple = (0, 2, 5, 10, 20)) -> pd.DataFrame:
    """At what cost level does the edge die? Often the most interesting result."""
    rows = []
    for bps in bps_levels:
        m = run_backtest(df, proba_col, threshold,
                         commission_bps=bps / 2, slippage_bps=bps / 2)["metrics"]
        rows.append({"total_bps": bps, "sharpe": m["sharpe"],
                     "ann_return": m["ann_return"], "turnover": m["avg_daily_turnover"]})
    return pd.DataFrame(rows)
```

### 5.2 Robustness — `src/alphabench/evaluation/robustness.py`

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def block_bootstrap_sharpe(returns: pd.Series, block: int = 21,
                           n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI for Sharpe using contiguous blocks, which preserves the
    autocorrelation that an iid bootstrap would destroy."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    n_blocks = int(np.ceil(len(r) / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(len(r) - block, 1), n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        sd = sample.std()
        out.append(sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
    out = np.array(out)
    obs = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
    return {
        "sharpe": float(obs),
        "ci_lower": float(np.percentile(out, 2.5)),
        "ci_upper": float(np.percentile(out, 97.5)),
        "p_gt_zero": float((out > 0).mean()),
    }


EULER = 0.5772156649015329


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int, *,
                    periods_per_year: int = 252, annualized: bool = True,
                    trial_sharpe_std: float | None = None,
                    skew: float = 0.0, kurtosis: float = 3.0) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Corrects an observed Sharpe for (a) selection bias from trying `n_trials`
    configurations and (b) non-normal returns.

    UNITS MATTER: the observed Sharpe and the null bar must be in the same
    units. Annualised Sharpes are converted to per-period internally. Getting
    this wrong makes the function return 1.0 for every input, which looks like
    it works and silently tells you every result is real.
    """
    sr = observed_sharpe / np.sqrt(periods_per_year) if annualized else observed_sharpe

    # Expected maximum Sharpe under the null of zero skill (extreme-value approx)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        v = trial_sharpe_std if trial_sharpe_std is not None else 1.0 / np.sqrt(n_obs)
        sr0 = v * (
            (1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

    denom = np.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    prob = float(stats.norm.cdf(z))

    return {
        "observed_sharpe_ann": float(observed_sharpe),
        "null_bar_ann": float(sr0 * np.sqrt(periods_per_year)),
        "deflated_sharpe_prob": prob,
        "n_trials": int(n_trials),
        "is_significant_at_95": prob > 0.95,
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Test whether two forecasts have significantly different accuracy.
    e1, e2 are squared (or absolute) forecast errors."""
    d = np.asarray(e1) - np.asarray(e2)
    n = len(d)
    dbar = d.mean()
    gamma0 = d.var(ddof=0)
    gammas = [np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)] if h > 1 else []
    var = (gamma0 + 2 * sum(gammas)) / n
    stat = dbar / np.sqrt(max(var, 1e-12))
    return {
        "dm_stat": float(stat),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))),
        "favours": "model_2" if dbar > 0 else "model_1",
    }


def by_period(trades: pd.DataFrame, freq: str = "YE") -> pd.DataFrame:
    """Per-year breakdown. Reveals a strategy whose whole return is one event."""
    daily = trades.groupby("date")["net_ret"].mean()
    g = daily.groupby(pd.Grouper(freq=freq))
    return pd.DataFrame({
        "ann_return": g.apply(lambda r: (1 + r).prod() - 1),
        "sharpe": g.apply(
            lambda r: r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0
        ),
        "n_days": g.size(),
    })
```

### 5.3 Two numbers worth putting in your report

Both tables below were produced by running the code above on a **synthetic random walk** — data with no signal in it by construction. Reproduce them yourself in Week 7; they are the most persuasive exhibits in the whole project because they show the machinery is calibrated.

**(a) Cost drag on a no-signal strategy.** Random probabilities, threshold 0.55, 3 symbols, 10 years:

| Total cost | Turnover/day | Implied annual drag | Ann. return | Sharpe |
|---|---|---|---|---|
| 0 bps | 0.502 | 0.00% | +0.89% | **+0.14** |
| 5 bps | 0.502 | 6.33% | −5.30% | −0.86 |
| 10 bps | 0.502 | 12.65% | −11.11% | −1.80 |
| 20 bps | 0.502 | 25.31% | −21.68% | −3.49 |

At zero cost the Sharpe is ~0, exactly as it should be on a random walk. The moment realistic costs are applied, a signal-free daily strategy loses roughly 12% a year. **This is why a 52% hit rate is not a strategy** — and stating it with your own numbers is far stronger than asserting it.

**(b) Deflated Sharpe: the selection-bias correction.** Observed annualised Sharpe of 1.0 over 2,000 daily observations:

| Trials run | Null bar (annualised) | P(skill is real) | Significant at 95%? |
|---|---|---|---|
| 1 | 0.000 | 0.998 | yes |
| 10 | 0.559 | 0.893 | no |
| 50 | 0.808 | 0.706 | no |
| 100 | 0.898 | 0.613 | no |
| 200 | 0.982 | **0.521** | no |
| 1000 | 1.155 | 0.331 | no |

The same Sharpe of 1.0 is near-certain evidence of skill if you tried one configuration, and a coin flip if you tried 200. This is why `len(study.trials)` must be recorded. If a reviewer asks "how do you know this isn't just the best of many tries?", this table is the answer.

```bash
make backtest
```

---

## STAGE 6 — API (Week 8)

### 6.1 Schemas — `src/alphabench/api/schemas.py`

```python
from __future__ import annotations
from datetime import date
from pydantic import BaseModel, Field


class PredictionOut(BaseModel):
    symbol: str
    as_of: date
    horizon_days: int
    probability_up: float = Field(..., ge=0.0, le=1.0)
    signal: str = Field(..., description="LONG | FLAT | SHORT")
    confidence_band: str
    model_version: str
    disclaimer: str = "Educational research output. Not investment advice."


class BacktestOut(BaseModel):
    symbol: str | None
    start: date
    end: date
    metrics: dict
    equity_curve: list[dict]
    disclaimer: str = "Historical simulation. Past performance does not predict future results."


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    data_last_updated: date | None
    n_symbols: int
```

### 6.2 App — `src/alphabench/api/main.py`

```python
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from alphabench.api.schemas import PredictionOut, BacktestOut, HealthOut

STATE: dict = {}
MODEL_DIR = Path("models/lightgbm_h1")
FEATURES = Path("data/processed/features_with_targets.parquet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request.
    if (MODEL_DIR / "final.joblib").exists():
        STATE["model"] = joblib.load(MODEL_DIR / "final.joblib")
        STATE["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    if FEATURES.exists():
        STATE["data"] = pd.read_parquet(FEATURES)
    yield
    STATE.clear()


app = FastAPI(
    title="AlphaBench API",
    description=(
        "Walk-forward equity return forecasting. "
        "**Educational research artifact — not investment advice.**"
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", response_model=HealthOut)
def health():
    df = STATE.get("data")
    return HealthOut(
        status="ok",
        model_loaded="model" in STATE,
        data_last_updated=df["date"].max().date() if df is not None else None,
        n_symbols=int(df["symbol"].nunique()) if df is not None else 0,
    )


@app.get("/tickers")
def tickers():
    df = STATE.get("data")
    if df is None:
        raise HTTPException(503, "data not loaded")
    return {"tickers": sorted(df["symbol"].unique().tolist())}


@app.get("/predict/{symbol}", response_model=PredictionOut)
def predict(symbol: str):
    if "model" not in STATE or "data" not in STATE:
        raise HTTPException(503, "model or data not loaded")

    df = STATE["data"]
    rows = df[df["symbol"] == symbol.upper()].sort_values("date")
    if rows.empty:
        raise HTTPException(404, f"unknown symbol: {symbol}")

    latest = rows.iloc[[-1]]
    cols = STATE["meta"]["features"]
    proba = float(STATE["model"].predict_proba(latest[cols])[0, 1])

    if proba > 0.55:
        sig, band = "LONG", "high"
    elif proba < 0.45:
        sig, band = "SHORT", "high"
    else:
        sig, band = "FLAT", "low"

    return PredictionOut(
        symbol=symbol.upper(),
        as_of=latest["date"].iloc[0].date(),
        horizon_days=STATE["meta"]["horizon"],
        probability_up=proba,
        signal=sig,
        confidence_band=band,
        model_version=STATE["meta"].get("train_end", "unknown"),
    )


@app.get("/metrics")
def metrics():
    p = Path("reports/metrics/walkforward_results.json")
    if not p.exists():
        raise HTTPException(404, "no metrics — run training first")
    return {"walkforward": json.loads(p.read_text()), "model": STATE.get("meta", {})}


@app.get("/backtest", response_model=BacktestOut)
def backtest(symbol: str | None = None, threshold: float = 0.55):
    from alphabench.evaluation.backtest import run_backtest
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")

    df = STATE["data"].copy()
    if symbol:
        df = df[df["symbol"] == symbol.upper()]
        if df.empty:
            raise HTTPException(404, f"unknown symbol: {symbol}")

    cols = STATE["meta"]["features"]
    df = df.dropna(subset=cols + ["fwd_ret_1d"])
    df["proba"] = STATE["model"].predict_proba(df[cols])[:, 1]
    res = run_backtest(df, threshold=threshold)

    eq = res["equity"].reset_index()
    eq.columns = ["date", "equity"]
    eq["benchmark"] = res["equity_bench"].to_numpy()

    return BacktestOut(
        symbol=symbol.upper() if symbol else None,
        start=df["date"].min().date(),
        end=df["date"].max().date(),
        metrics=res["metrics"],
        equity_curve=[
            {"date": str(r.date.date()), "equity": float(r.equity),
             "benchmark": float(r.benchmark)}
            for r in eq.itertuples()
        ],
    )
```

```bash
make api
# open http://localhost:8000/docs
curl http://localhost:8000/health
curl http://localhost:8000/predict/AAPL
```

---

## STAGE 7 — Dashboard (Week 8)

`src/alphabench/dashboard/app.py`

```python
from __future__ import annotations
import os
import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AlphaBench", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


st.title("AlphaBench")
st.caption("Walk-forward equity return forecasting & backtesting")

st.warning(
    "**Educational research artifact — not investment advice.** All results are "
    "historical simulations with no guarantee of future performance. Do not trade "
    "on these outputs.",
    icon="⚠️",
)

try:
    health = get("/health")
except Exception as exc:                            # noqa: BLE001
    st.error(f"API unreachable at {API}: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Model", "loaded" if health["model_loaded"] else "missing")
c2.metric("Symbols", health["n_symbols"])
c3.metric("Data through", health["data_last_updated"] or "—")

symbols = get("/tickers")["tickers"]
with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Ticker", symbols)
    threshold = st.slider("Probability threshold", 0.50, 0.75, 0.55, 0.01)
    st.caption("Higher threshold → fewer, more selective trades.")

tab1, tab2, tab3 = st.tabs(["Signal", "Backtest", "Validation"])

with tab1:
    p = get(f"/predict/{symbol}")
    a, b, c = st.columns(3)
    a.metric("P(up)", f"{p['probability_up']:.3f}")
    b.metric("Signal", p["signal"])
    c.metric("As of", p["as_of"])
    st.progress(p["probability_up"])
    st.caption(
        "A probability near 0.50 means the model sees no edge. That is the "
        "expected state on most days and is a feature, not a failure."
    )

with tab2:
    bt = get("/backtest", symbol=symbol, threshold=threshold)
    m = bt["metrics"]
    eq = pd.DataFrame(bt["equity_curve"])
    eq["date"] = pd.to_datetime(eq["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"], name="Strategy (net of costs)"))
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["benchmark"], name="Buy & hold",
                             line=dict(dash="dash")))
    fig.update_layout(height=440, yaxis_title="Growth of 1 unit", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    k = st.columns(4)
    k[0].metric("Sharpe (net)", f"{m['sharpe']:.3f}",
                delta=f"{m['excess_sharpe']:+.3f} vs B&H")
    k[1].metric("Ann. return", f"{m['ann_return']:.2%}")
    k[2].metric("Max drawdown", f"{m['max_drawdown']:.2%}")
    k[3].metric("Hit rate", f"{m['hit_rate']:.2%}")

    with st.expander("All metrics"):
        st.json(m)

with tab3:
    mt = get("/metrics")
    wf = pd.DataFrame(mt["walkforward"])
    st.subheader("Walk-forward results by fold")
    st.dataframe(wf, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=wf["val_year"], y=wf["auc"], name="AUC"))
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="chance (0.50)")
    fig.update_layout(height=340, yaxis_title="ROC-AUC", yaxis_range=[0.45, 0.65])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Mean AUC {wf['auc'].mean():.4f} across {len(wf)} folds. "
        "Values of 0.52–0.55 are realistic for daily equity direction. "
        "Anything above 0.60 should trigger a leakage audit before it is believed."
    )
```

```bash
make dashboard
```

---

## STAGE 8 — Containerisation (Week 7, not Week 10)

**`docker/api.Dockerfile`**

```dockerfile
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
# libgomp1 is required by LightGBM/XGBoost at runtime. Omitting it produces a
# confusing "cannot open shared object file" error only inside the container.

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
RUN pip install -e . --no-deps

COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY reports/metrics/ ./reports/metrics/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn alphabench.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

**`docker/dashboard.Dockerfile`**

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install \
    streamlit==1.62.0 plotly==7.0.0 httpx==0.28.1 pandas==2.3.3 pyyaml==6.0.3

COPY src/alphabench/dashboard/ ./src/alphabench/dashboard/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["sh", "-c", "streamlit run src/alphabench/dashboard/app.py \
     --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
```

**`compose.yaml`**

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    ports: ["8000:8000"]
    environment:
      - PORT=8000
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: docker/dashboard.Dockerfile
    ports: ["8501:8501"]
    environment:
      - API_URL=http://api:8000     # service name, not localhost
      - PORT=8501
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
```

**`.dockerignore`**

```
.venv/
.git/
__pycache__/
data/raw/
data/interim/
mlruns/
notebooks/
tests/
reports/figures/
*.ipynb
.pytest_cache/
.ruff_cache/
```

```bash
docker compose build
docker compose up
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

---

## STAGE 9 — CI & Automation (Week 10)

**`.github/workflows/ci.yaml`**

```yaml
name: CI
on:
  push: {branches: [main]}
  pull_request: {branches: [main]}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy src --ignore-missing-imports
      - run: pytest tests/ -v --tb=short
```

**`.github/workflows/refresh-data.yaml`**

```yaml
name: Nightly data refresh
on:
  schedule:
    - cron: "0 23 * * 1-5"      # 23:00 UTC, weekdays (after US close)
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -r requirements.txt && pip install -e .
      - name: Incremental ingest
        continue-on-error: true      # fail-soft: a bad night must not break the API
        run: |
          python -m alphabench.cli ingest --incremental
          python -m alphabench.cli validate
          python -m alphabench.cli build-features
      - uses: actions/upload-artifact@v4
        with:
          name: features
          path: data/processed/
          retention-days: 7
```

**Keep-alive (optional, prevents Render cold starts):**

```yaml
name: Keep alive
on:
  schedule: [{cron: "*/10 6-22 * * *"}]
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS ${{ secrets.API_URL }}/health || true
```

---

## STAGE 10 — Deployment (Week 10)

### 10.1 API → Render

1. Push the repository to GitHub (models and processed features must be committed or fetched at build time — see the note below).
2. Render dashboard → **New → Web Service** → connect the repo.
3. Settings:
   - Runtime: **Docker**
   - Dockerfile path: `docker/api.Dockerfile`
   - Instance type: **Free**
   - Health check path: `/health`
4. Deploy. You get `https://alphabench-api.onrender.com`.

> **Model artifacts in git.** A LightGBM model is a few MB, which is acceptable to commit. Do **not** commit raw data. If artifacts exceed ~50 MB, use Git LFS or download them from a GitHub Release in the Dockerfile.

> **Free tier sleeps** after ~15 minutes idle, with a 30–60 s cold start. Either add the keep-alive workflow above or note the behaviour in your README so a reviewer isn't confused by a slow first load.

### 10.2 Dashboard → Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli repo create alphabench --type space --space_sdk docker
git clone https://huggingface.co/spaces/<your-username>/alphabench hf-space
cd hf-space
```

Copy in `src/alphabench/dashboard/`, `requirements.txt`, and the dashboard Dockerfile renamed to `Dockerfile`. Add a Space README header:

```markdown
---
title: AlphaBench
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
---
```

Set `API_URL` to your Render URL under Space **Settings → Variables**, then:

```bash
git add . && git commit -m "deploy dashboard" && git push
```

### 10.3 Final verification

```bash
curl -fsS https://<your-api>.onrender.com/health
curl -fsS https://<your-api>.onrender.com/predict/AAPL
# then open the Space URL and confirm the disclaimer renders before anything else
```

---

## STAGE 11 — The Holdout (Week 9, exactly once)

Run this **once**, after everything else is frozen. Write the number down whatever it says.

```bash
python -m alphabench.cli evaluate-holdout --model lightgbm --horizon 1
```

```python
# src/alphabench/cli.py (excerpt)
def evaluate_holdout(model: str = "lightgbm", horizon: int = 1) -> None:
    """Score the sealed holdout period. Run ONCE. Do not tune afterwards."""
    from pathlib import Path
    out = Path("reports/metrics/holdout_results.json")
    if out.exists():
        raise SystemExit(
            "holdout_results.json already exists. The holdout is single-use by "
            "design — re-running it after seeing results is how backtest "
            "overfitting happens. Delete the file deliberately if you truly must."
        )
    ...
```

That guard is not paranoia. The temptation to "just try one more feature" after a disappointing holdout is exactly the failure mode the holdout exists to prevent, and a reviewer will ask whether you resisted it.

---

## Final checklist before you call it done

```
[ ] make test — all green, including test_leakage.py and test_splitters.py
[ ] B0 baseline number appears at the top of every results table
[ ] Holdout evaluated exactly once; holdout_results.json committed
[ ] Backtest reported NET of costs, with the buy-and-hold benchmark on the same axes
[ ] Cost sensitivity sweep included
[ ] Bootstrap CI on Sharpe and AUC reported
[ ] Optuna trial count recorded and deflated Sharpe computed
[ ] Per-year and per-ticker breakdowns in the report
[ ] Survivorship bias disclosed in README and report
[ ] Disclaimer visible on dashboard load and in every API response
[ ] README states the headline result honestly, including if it is null
[ ] Both public URLs live and reachable
[ ] Prophet rejection justified in one paragraph
[ ] LSTM result reported whatever it turned out to be
```
