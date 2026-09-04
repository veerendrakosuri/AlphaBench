from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from alphabench.api.schemas import DISCLAIMER, BacktestOut, HealthOut, PredictionOut
from alphabench.config import load_config
from alphabench.data.repository import Repository

STATE: dict = {}
MODEL_DIR = Path("models/lightgbm_h1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request.
    cfg = load_config()
    STATE["cfg"] = cfg
    if (MODEL_DIR / "final.joblib").exists():
        STATE["model"] = joblib.load(MODEL_DIR / "final.joblib")
        STATE["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())

    repo = Repository(cfg.data.processed_dir)
    if repo.exists("features") and repo.exists("targets"):
        STATE["data"] = repo.read("features").merge(repo.read("targets"), on=["date", "symbol"])
    if repo.exists("oof_predictions_h1"):
        STATE["oof"] = repo.read("oof_predictions_h1")
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    return {"tickers": sorted(df["symbol"].unique().tolist()), "disclaimer": DISCLAIMER}


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
    wf = Path("reports/metrics/walkforward_results.json")
    bt = Path("reports/metrics/backtest_results.json")
    if not wf.exists():
        raise HTTPException(404, "no metrics — run training first")
    out = {
        "walkforward": json.loads(wf.read_text()),
        "model": STATE.get("meta", {}),
        "disclaimer": DISCLAIMER,
    }
    if bt.exists():
        out["backtest"] = json.loads(bt.read_text())
    return out


@app.get("/backtest", response_model=BacktestOut)
def backtest(symbol: str | None = None, threshold: float | None = None):
    from alphabench.evaluation.backtest import run_backtest

    if "oof" not in STATE:
        raise HTTPException(503, "out-of-fold predictions not loaded — run `train` first")

    df = STATE["oof"]
    if symbol:
        df = df[df["symbol"] == symbol.upper()]
        if df.empty:
            raise HTTPException(404, f"unknown symbol: {symbol}")

    thr = threshold if threshold is not None else STATE["cfg"].backtest.prob_threshold
    res = run_backtest(
        df,
        threshold=thr,
        commission_bps=STATE["cfg"].backtest.commission_bps,
        slippage_bps=STATE["cfg"].backtest.slippage_bps,
    )

    eq = res["equity"].reset_index()
    eq.columns = ["date", "equity"]
    eq["benchmark"] = res["equity_bench"].to_numpy()

    return BacktestOut(
        symbol=symbol.upper() if symbol else None,
        start=df["date"].min().date(),
        end=df["date"].max().date(),
        metrics=res["metrics"],
        equity_curve=[
            {
                "date": str(r.date.date()),
                "equity": float(r.equity),
                "benchmark": float(r.benchmark),
            }
            for r in eq.itertuples()
        ],
    )
