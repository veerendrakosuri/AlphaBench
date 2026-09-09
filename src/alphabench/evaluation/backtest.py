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
    fwd_ret_col: str = "fwd_ret_1d",
    horizon: int = 1,
) -> dict:
    """Equal-weight portfolio over the panel.

    Timing: signal from the close of day t; the return earned is the forward return
    already computed by targets/builder.py for this horizon (`fwd_ret_col`, t -> t+h).

    For horizon > 1, consecutive rows' fwd_ret_{h}d windows overlap by h-1 days (day
    t's window is [t, t+h], day t+1's is [t+1, t+h+1], etc.). Compounding overlapping
    windows at daily frequency — as if each were a distinct realised day of return —
    would multiply the same underlying days' moves together several times over and
    produce a fabricated equity curve (this was caught precisely because it did: an
    h=5 run once produced an annualised Sharpe above 3 and a "confirmed" >700%
    single-year return, both impossible). To avoid that, when horizon > 1 the whole
    PANEL (not each symbol independently — an earlier version of this fix downsampled
    per symbol, which does not synchronise across symbols with different start dates
    or missing rows, so the portfolio-level series stayed almost daily) is restricted
    to one shared calendar date every `horizon` trading days, in date order across the
    full panel. Every symbol rebalances on the same dates, giving a genuinely
    non-overlapping, synchronised "decide, hold for h days, then decide again"
    portfolio schedule. Annualisation uses TRADING_DAYS / horizon periods per year.
    """
    d = df.dropna(subset=[proba_col, fwd_ret_col]).sort_values(["date", "symbol"]).copy()
    if horizon > 1:
        unique_dates = np.sort(d["date"].unique())
        keep_dates = unique_dates[np.arange(len(unique_dates)) % horizon == 0]
        d = d[d["date"].isin(keep_dates)]

    d["position"] = generate_positions(d[proba_col], threshold, allow_short)

    d["prev_position"] = d.groupby("symbol")["position"].shift(1).fillna(0.0)
    d["turnover"] = (d["position"] - d["prev_position"]).abs()

    cost_rate = (commission_bps + slippage_bps) / 10_000.0
    d["gross_ret"] = d["position"] * d[fwd_ret_col]
    d["cost"] = d["turnover"] * cost_rate
    d["net_ret"] = d["gross_ret"] - d["cost"]

    daily = d.groupby("date").agg(
        gross=("gross_ret", "mean"),
        net=("net_ret", "mean"),
        turnover=("turnover", "mean"),
        n_positions=("position", lambda s: int((s != 0).sum())),
    )
    bench = d.groupby("date")[fwd_ret_col].mean()  # equal-weight buy & hold
    periods_per_year = TRADING_DAYS / horizon

    return {
        "daily": daily,
        "benchmark": bench,
        "equity": (1 + daily["net"]).cumprod(),
        "equity_gross": (1 + daily["gross"]).cumprod(),
        "equity_bench": (1 + bench).cumprod(),
        "metrics": _metrics(daily["net"], bench, daily["turnover"], periods_per_year),
        "trades": d,
        "periods_per_year": periods_per_year,
    }


def _metrics(
    net: pd.Series, bench: pd.Series, turnover: pd.Series, periods_per_year: float = TRADING_DAYS
) -> dict:
    def ann_ret(r):
        return float((1 + r).prod() ** (periods_per_year / len(r)) - 1)

    def ann_vol(r):
        return float(r.std() * np.sqrt(periods_per_year))

    def sharpe(r):
        v = ann_vol(r)
        return float(ann_ret(r) / v) if v > 0 else 0.0

    def sortino(r):
        dn = r[r < 0].std() * np.sqrt(periods_per_year)
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
    fwd_ret_col: str = "fwd_ret_1d",
    horizon: int = 1,
) -> pd.DataFrame:
    """Sweep the position threshold across quantiles of the model's own probability
    distribution, rather than assuming an absolute probability level is meaningful."""
    rows = []
    probs = df[proba_col].dropna()
    for q in quantiles:
        thr = float(probs.quantile(q))
        result = run_backtest(
            df,
            proba_col,
            thr,
            commission_bps,
            slippage_bps,
            fwd_ret_col=fwd_ret_col,
            horizon=horizon,
        )
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
    fwd_ret_col: str = "fwd_ret_1d",
    horizon: int = 1,
) -> pd.DataFrame:
    """At what cost level does the edge die? Often the most interesting result."""
    rows = []
    for bps in bps_levels:
        m = run_backtest(
            df,
            proba_col,
            threshold,
            commission_bps=bps / 2,
            slippage_bps=bps / 2,
            fwd_ret_col=fwd_ret_col,
            horizon=horizon,
        )["metrics"]
        rows.append(
            {
                "total_bps": bps,
                "sharpe": m["sharpe"],
                "ann_return": m["ann_return"],
                "turnover": m["avg_daily_turnover"],
            }
        )
    return pd.DataFrame(rows)
