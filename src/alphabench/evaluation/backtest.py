from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def generate_positions(
    proba: pd.Series, threshold: float = 0.55, allow_short: bool = False
) -> pd.Series:
    """Map probabilities to positions. The band around 0.5 means 'no trade'."""
    pos = pd.Series(0.0, index=proba.index)
    pos[proba > threshold] = 1.0
    if allow_short:
        pos[proba < (1 - threshold)] = -1.0
    return pos


def run_backtest(
    df: pd.DataFrame,
    proba_col: str = "proba",
    threshold: float = 0.55,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    allow_short: bool = False,
) -> dict:
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
    bench = d.groupby("date")["fwd_ret_1d"].mean()  # equal-weight buy & hold

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
    def ann_ret(r):
        return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)

    def ann_vol(r):
        return float(r.std() * np.sqrt(TRADING_DAYS))

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
        "n_days": len(net),
        "benchmark_ann_return": ann_ret(bench),
        "benchmark_sharpe": sharpe(bench),
        "benchmark_max_drawdown": max_dd(bench),
        "excess_sharpe": sharpe(net) - sharpe(bench),
    }


def threshold_sensitivity(
    df: pd.DataFrame,
    proba_col: str = "proba",
    quantiles: tuple = (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99),
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    """Sweep the position threshold across quantiles of the model's own probability
    distribution, rather than assuming an absolute probability level is meaningful."""
    rows = []
    probs = df[proba_col].dropna()
    for q in quantiles:
        thr = float(probs.quantile(q))
        result = run_backtest(df, proba_col, thr, commission_bps, slippage_bps)
        m = result["metrics"]
        rows.append(
            {
                "quantile": q,
                "threshold": thr,
                "n_trades": int((df[proba_col] > thr).sum()),
                "trade_pct": float((df[proba_col] > thr).mean()),
                "sharpe": m["sharpe"],
                "ann_return": m["ann_return"],
                "hit_rate": m["hit_rate"],
                "avg_daily_turnover": m["avg_daily_turnover"],
            }
        )
    return pd.DataFrame(rows)


def cost_sensitivity(
    df: pd.DataFrame,
    proba_col: str = "proba",
    threshold: float = 0.55,
    bps_levels: tuple = (0, 2, 5, 10, 20),
) -> pd.DataFrame:
    """At what cost level does the edge die? Often the most interesting result."""
    rows = []
    for bps in bps_levels:
        m = run_backtest(df, proba_col, threshold, commission_bps=bps / 2, slippage_bps=bps / 2)[
            "metrics"
        ]
        rows.append(
            {
                "total_bps": bps,
                "sharpe": m["sharpe"],
                "ann_return": m["ann_return"],
                "turnover": m["avg_daily_turnover"],
            }
        )
    return pd.DataFrame(rows)
