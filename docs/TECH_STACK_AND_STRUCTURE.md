# AlphaBench — Tech Stack & Project Structure

Companion to `PROPOSAL.md`. Every version below was resolved against PyPI in August 2026 and installs cleanly together on Python 3.12.

---

# PART 1 — TECH STACK

## 1.1 At a glance

| Layer | Choice | One-line reason |
|---|---|---|
| Language | **Python 3.12** | Whole stack in one language; 3.13 still has patchy wheel coverage for ML libs |
| Dataframes | **pandas 2.3** | Universal in finance tutorials/literature; you already know it |
| Columnar I/O | **PyArrow + Parquet** | 5–10× smaller and faster than CSV; preserves dtypes |
| Query engine | **DuckDB 1.5** | SQL over Parquet, zero server, zero ops |
| Classical ML | **scikit-learn 1.9** | Pipelines, metrics, calibration, splitter API to subclass |
| Primary model | **LightGBM 4.7** | Fastest strong tabular learner; speed enables walk-forward CV |
| Secondary model | **XGBoost 3.4** | Diversification arm for the ensemble |
| Statistical | **statsmodels 0.14** | ARIMA baseline, ADF/KPSS, Diebold–Mariano |
| Deep learning | **PyTorch 2.13** | LSTM comparison arm; CPU-only is fine at this scale |
| Tuning | **Optuna 4.9** | TPE search with pruning; clean walk-forward integration |
| Tracking | **MLflow 3.15** | Local file backend; the audit trail behind the deflated Sharpe |
| Interpretability | **SHAP 0.52** | Native fast path for tree models |
| Backend | **FastAPI 0.141 + Uvicorn 0.52** | Auto OpenAPI docs; Pydantic validation |
| Schemas/config | **Pydantic 2.13 + pydantic-settings** | One validation library for API and config |
| Frontend | **Streamlit 1.62 + Plotly 7.0** | Dashboard in Python; Plotly for interactive financial charts |
| CLI | **Typer 0.27 + Rich 15.0** | Every stage runnable as `python -m alphabench.<stage>` |
| Resilience | **Tenacity 9.1** | Exponential backoff — mandatory for yfinance |
| Packaging | **Docker + compose** | Two services; identical local and cloud runtime |
| CI/automation | **GitHub Actions** | Free CI + free cron for nightly refresh |
| Quality | **pytest 9.1, ruff 0.16, mypy 2.3, pre-commit 4.6** | Ruff replaces black+isort+flake8 in one tool |
| Deploy (API) | **Render** free web service | No credit card, Docker-native, git-push deploy |
| Deploy (UI) | **Hugging Face Spaces** (Docker SDK) | Free, no cold-start sleep, ML-audience visibility |

## 1.2 Decisions that need explaining

**Parquet + DuckDB over PostgreSQL.** The dataset is ~50 MB of immutable, append-only, columnar time series. DuckDB runs full SQL directly against Parquet files with no server process, no connection pooling, no Docker service, and no backup story, and it reads one column out of a wide table without touching the others. Postgres would buy transactional writes and concurrent multi-user access — neither of which this project has. If a reviewer insists on a relational database, add it behind the existing `data/repository.py` interface; the swap is an afternoon because nothing above that layer knows where bytes come from.

**LightGBM over an LSTM as the primary model.** The deciding constraint is not accuracy, it is *iteration speed*. Walk-forward validation means refitting across 7+ folds, and Optuna means doing that 50–200 times. LightGBM fits this dataset in seconds on CPU, so the full search finishes in minutes. An LSTM turns the same search into days you do not have, and at 114k rows of low-SNR tabular data it is very unlikely to win anyway. The LSTM stays in the project as a comparison arm precisely so you can *show* that with evidence.

**Streamlit over React.** Streamlit costs ~150 lines and half a day. React + Recharts costs 1.5–2 weeks and buys visual polish plus frontend credibility. For a 10-week timeline whose scarce resource is validation rigour, Streamlit is correct. The FastAPI layer is deliberately frontend-agnostic — if you already know React, or you are targeting full-stack roles, swap the frontend without touching anything below it.

**Optuna over `GridSearchCV`.** Grid search scales exponentially and wastes budget on obviously-bad regions. Optuna's TPE sampler concentrates trials where results are promising and prunes hopeless ones early. It also *records the trial count*, which you need to compute the deflated Sharpe ratio honestly.

**MLflow even though you are solo.** Not for collaboration — for defensibility. When a reviewer asks "how many configurations did you try before this one?", the answer must be a number you can produce, not a shrug. MLflow's local file backend needs no server.

**Ruff over black + isort + flake8 + pylint.** One Rust binary, ~100× faster, one config block, formats and lints. There is no reason to run four tools in 2026.

**Typer for a CLI instead of argparse or bare scripts.** Every pipeline stage becomes a discoverable, documented command. This is what makes `make all` reproducible from a clean clone — and reproducibility is success criterion SC-1.

## 1.3 Deployment options compared

| Platform | Free tier | Docker | Best for | Watch out for |
|---|---|---|---|---|
| **Render** | Yes, no card required | Native | **The FastAPI service** | Free web services sleep after ~15 min idle; 30–60 s cold start |
| **Hugging Face Spaces** | Yes | Docker SDK | **The Streamlit dashboard** | Public by default; ML-recruiter visibility is a bonus |
| **Streamlit Community Cloud** | Yes | No (source deploy) | Simplest dashboard path | No Docker; less to talk about in an interview |
| **Google Cloud Run** | Generous request-based | Native | Scaling beyond a portfolio project | Requires a billing account |
| **Fly.io** | Effectively paid | Native | Global/low-latency needs | Free tier withdrawn; ~$2/mo minimum |
| **Railway** | Trial credits only | Native | Short bursts | No real free tier since 2023 |

**Recommendation:** Render for the API + HF Spaces for the dashboard. Both free, both Docker-native, no credit card.

**Cold-start mitigation:** Render free services sleep. Either (a) accept it and note it in the README, or (b) add a GitHub Actions cron that pings `/health` every 10 minutes during waking hours. Option (b) is two lines of YAML and prevents a reviewer hitting a dead link. Verify current free-tier terms before you rely on them — this market changes frequently.

## 1.4 Explicitly rejected

| Rejected | Why |
|---|---|
| **Prophet** | Trend + seasonality + holiday decomposition. Daily returns have no such structure; it fits smooth curves to noise. Rejecting it with a reason is worth a paragraph in your report. |
| **TensorFlow/Keras** | No advantage over PyTorch here; heavier install; PyTorch is the research default. |
| **Airflow / Prefect** | Real orchestrators for real DAG complexity. This pipeline is linear. `Makefile` + GitHub Actions cron does the job without a scheduler to babysit. |
| **Kubernetes** | Two containers. `docker compose` locally, PaaS in the cloud. K8s here is résumé theatre that eats a week. |
| **Kafka / Spark** | 50 MB of data. |
| **MongoDB** | Schema-less document storage for rigidly schema'd numeric time series is the wrong shape. |
| **`ta-lib`** | Excellent library, but needs a C library compiled before the pip install and reliably breaks Docker builds for beginners. Use hand-rolled indicators (better for the report anyway) or `pandas-ta`. |

---

# PART 2 — PROJECT STRUCTURE

```
alphabench/
│
├── README.md                       # architecture diagram, quickstart, HONEST headline result
├── LICENSE                         # MIT
├── Makefile                        # make setup | data | features | train | backtest | all
├── pyproject.toml                  # package metadata + ruff/mypy/pytest config
├── requirements.txt                # runtime deps (pinned)
├── requirements-dev.txt            # dev/test deps (pinned)
├── .env.example                    # API keys template — NEVER commit .env
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── compose.yaml                    # api + dashboard services
│
├── config/
│   ├── config.yaml                 # master config: paths, universe, dates, horizon
│   ├── universe_us.yaml            # 30 US tickers, sector-tagged
│   ├── universe_in.yaml            # 30 NSE tickers, sector-tagged
│   ├── features.yaml               # which feature groups + their lookbacks
│   └── models/
│       ├── lightgbm.yaml
│       ├── xgboost.yaml
│       └── lstm.yaml
│
├── docker/
│   ├── api.Dockerfile
│   ├── dashboard.Dockerfile
│   └── entrypoint.sh
│
├── .github/workflows/
│   ├── ci.yaml                     # ruff + mypy + pytest on every push
│   ├── refresh-data.yaml           # nightly cron: incremental ingest, fail-soft
│   └── deploy.yaml                 # build + push images on tag
│
├── data/                           # ALL gitignored except .gitkeep files
│   ├── raw/                        # immutable provider output. NEVER edit in place.
│   │   └── ohlcv/ticker=AAPL/year=2024/data.parquet
│   ├── interim/                    # validated, cleaned panel
│   │   └── panel.parquet
│   ├── processed/                  # model-ready features + targets
│   │   ├── features.parquet
│   │   └── targets.parquet
│   └── external/                   # macro series (FRED), trading calendars
│
├── models/                         # gitignored; artifacts + metadata
│   ├── lightgbm_h1/
│   │   ├── fold_2017.joblib ... fold_2023.joblib
│   │   ├── final.joblib
│   │   └── metadata.json           # feature list, params, git SHA, data hash, train dates
│   ├── xgboost_h1/
│   └── lstm_h1/
│
├── mlruns/                         # gitignored; MLflow local backend
│
├── reports/
│   ├── figures/                    # equity curves, calibration, SHAP, drawdowns
│   ├── metrics/
│   │   ├── walkforward_results.json
│   │   ├── holdout_results.json    # written EXACTLY ONCE
│   │   └── backtest_results.json
│   └── technical_report.md
│
├── notebooks/                      # EXPLORATION ONLY — never imported by src
│   ├── 01_data_quality.ipynb
│   ├── 02_eda_stationarity.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_model_diagnostics.ipynb
│   └── 05_results_figures.ipynb
│
├── src/alphabench/
│   ├── __init__.py
│   ├── cli.py                      # Typer app — the single entrypoint
│   ├── config.py                   # pydantic-settings loader for config/*.yaml
│   ├── logging_conf.py             # structured logging via Rich
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── base.py             # Provider protocol — makes sources swappable
│   │   │   ├── yahoo.py            # yfinance + tenacity backoff
│   │   │   ├── stooq.py            # automatic failover
│   │   │   └── fred.py             # macro series
│   │   ├── ingest.py               # bulk backfill + incremental append
│   │   ├── validate.py             # schema, bounds, calendar, anomaly checks
│   │   └── repository.py           # DuckDB read/write layer — the ONLY I/O path
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                 # shift()-guard decorator + registry
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   ├── technical.py            # RSI, MACD, ATR, Bollinger, Stochastic, OBV
│   │   ├── volume.py
│   │   ├── cross_sectional.py      # per-date ranks across the universe
│   │   ├── market.py               # index return, rolling beta, calendar flags
│   │   └── pipeline.py             # orchestrates all groups → features.parquet
│   │
│   ├── targets/
│   │   ├── __init__.py
│   │   └── builder.py              # forward returns + vol-scaled deadband labels
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── splitters.py            # WalkForwardSplit with purge + embargo
│   │   └── leakage.py              # runtime assertions used by tests/
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # Model protocol: fit/predict_proba/save/load
│   │   ├── baselines.py            # B0 persistence, B1 logistic
│   │   ├── arima.py                # B2
│   │   ├── gbm.py                  # M1 LightGBM, M2 XGBoost
│   │   ├── lstm.py                 # M3 PyTorch
│   │   ├── ensemble.py             # M4 rank-average
│   │   └── registry.py             # name → class lookup for config-driven runs
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                # walk-forward loop + MLflow logging
│   │   └── tune.py                 # Optuna study inside the walk-forward loop
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── statistical.py          # AUC, Brier, log-loss, calibration, DM test
│   │   ├── economic.py             # Sharpe, Sortino, maxDD, Calmar, turnover
│   │   ├── backtest.py             # cost-aware engine, next-open execution
│   │   ├── robustness.py           # block bootstrap, deflated Sharpe, per-year
│   │   └── explain.py              # SHAP + fold-stability of importances
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app + lifespan model loading
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   ├── backtest.py
│   │   │   └── metrics.py
│   │   └── deps.py                 # cached model/data loaders
│   │
│   └── dashboard/
│       ├── app.py                  # Streamlit entrypoint
│       ├── api_client.py           # thin httpx wrapper over the API
│       ├── components/
│       │   ├── price_chart.py
│       │   ├── equity_curve.py
│       │   ├── metrics_table.py
│       │   ├── shap_panel.py
│       │   └── disclaimer.py       # rendered on EVERY page
│       └── .streamlit/config.toml
│
└── tests/
    ├── conftest.py                 # synthetic OHLCV fixtures
    ├── test_ingest.py
    ├── test_validate.py
    ├── test_features.py
    ├── test_leakage.py             # ★ THE MOST IMPORTANT FILE IN THE REPO
    ├── test_splitters.py           # ★ purge/embargo correctness on synthetic dates
    ├── test_targets.py
    ├── test_models.py
    ├── test_backtest.py            # hand-checked toy case with known answer
    └── test_api.py
```

## 2.1 Structural rules worth stating

**`src/` layout, not a flat package.** Forces you to install the package (`pip install -e .`), which means imports behave identically in tests, notebooks, the API, and Docker. Flat layouts silently work locally and break in containers.

**Notebooks import from `src`, never the reverse.** Notebooks are for exploration and figure generation. Any logic that matters gets promoted into `src/` and tested. A project whose pipeline lives in notebooks is not reproducible and reviewers know it.

**`data/raw/` is immutable.** Write once, never edit in place. Every transformation produces a new artifact in `interim/` or `processed/`. When something looks wrong three weeks in, you can always go back to what the provider actually returned.

**`repository.py` is the only module that touches disk.** Everything else asks it for dataframes. This is what makes the Parquet→Postgres swap an afternoon rather than a rewrite.

**`metadata.json` beside every model.** Feature list, hyperparameters, git SHA, data hash, training date range. Without it, a model file six weeks old is unidentifiable and your results are unreproducible.

**The two starred test files carry the project.** `test_leakage.py` and `test_splitters.py` are what convert "I built a model" into "I built a model I can defend." Write them in Weeks 3 and 4, before you have results to be attached to.
