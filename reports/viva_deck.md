# AlphaBench

## Can a model predict the direction of daily equity returns well enough to beat buy-and-hold after costs?

**Answer: no — and the value is in how carefully that was established.**

Veerendra Kosuri · Walk-forward equity return forecasting & backtesting

> Educational research artifact — not investment advice. All results are historical
> simulations with no guarantee of future performance.

---

## 1 · The problem, reframed

**The naive version, which nearly every "stock prediction" project builds:**
predict tomorrow's *price*.

It fails for two reasons at once:

- Prices are non-stationary. Confirmed here: **0/29 symbols stationary** under both ADF
  and KPSS. Returns are stationary on both: **29/29**.
- A model trained on price levels wins on RMSE by predicting "roughly today's price."
  It scores beautifully and has learned nothing.

**The version actually built:** predict the *sign* of the forward return, on a
volatility-scaled deadband — is the next move large enough, in either direction, to be
worth acting on?

Labels: `y = 1` if `fwd_ret > κ·σ`, `0` if `fwd_ret < −κ·σ`, dropped otherwise. The
deadband stops the target being dominated by whichever volatility regime happens to
prevail.

---

## 2 · Methodology — the parts that decide whether a result is real

**Walk-forward, expanding window.** Seven folds, validating 2017 → 2023. Never a random
k-fold: shuffling a time series lets the model train on the future.

**Purge + embargo.** A 5-day purge drops training rows whose label window overlaps the
validation block; a 5-day embargo follows it. Without these, the last rows of training
leak the first days of validation.

**Leakage caught structurally, not by inspection.**

- Every feature is forced through one `safe_shift` before it reaches a model — a blanket
  rule that cannot be forgotten per-feature.
- The test suite trains on a synthetic random walk and *fails the build* if AUC departs
  from chance, or if any feature correlates > 0.10 with the forward return.
- Strongest observed correlation with tomorrow's return across ~50 features: **0.0196.**

**The holdout is sealed and single-use.** 2025-01-01 onward, never touched during
development; the CLI refuses to score it twice.

---

## 3 · The baseline ladder — B0 first

A number without its baseline is not a result.

| | Model | ROC-AUC | Accuracy |
|---|---|---|---|
| B0 | Persistence (today's direction → tomorrow's) | 0.487 | 0.488 |
| B0 | Majority class | 0.500 | 0.518 |
| B1 | Logistic regression (5 features) | 0.510 | 0.518 |
| B2 | ARIMA(1,0,1) on log returns | 0.503 | 0.511 |
| **M1** | **LightGBM — primary, and the model served** | **0.506** | **0.518** |
| M2 | XGBoost | 0.515 | 0.518 |
| M3 | LSTM (PyTorch, CPU, small) | 0.515 | 0.518 |
| M4 | Rank-average ensemble (M1+M2+M3) | 0.514 | 0.508 |

Every entry sits in **0.487 – 0.515**. Each model's own fold-to-fold standard deviation is
**0.01 – 0.02**.

**The gaps between models are smaller than the noise within any one model.** There is no
ladder to climb here.

---

## 4 · Two results reported against expectation

**The LSTM did not lose.** PROPOSAL §4.2 predicted it would underperform the GBM at this
data scale. It came in at 0.515 — tied with XGBoost, marginally *above* LightGBM's 0.506.

Reported as it happened rather than adjusted to match the prediction. It also doesn't
contradict the prediction in any way that matters: every confidence interval overlaps
every other, so "M3 beat M1" is not a claim this sample size supports.

**Prophet was rejected, not run.** It decomposes a series into trend + seasonality +
holiday components — structure daily equity returns demonstrably lack (the ACF is flat
beyond lag 0). It would fit smooth curves to noise, convincingly, because that is what a
smoothing model does regardless of whether structure is present. A model guaranteed to
produce plausible-looking output on data it has no business modelling invites false
confidence. B2 ARIMA fills the classical-baseline role on a specification that at least
matches the data-generating process.

---

## 5 · Economics — where the result actually dies

| Metric | Strategy | Buy-and-hold NIFTY |
|---|---|---|
| Sharpe (annualised, net of 10 bps) | **0.108** | 1.799 |
| Annual return | 1.00% | 34.96% |
| Max drawdown | −29.1% | −37.5% |

**Excess Sharpe: −1.69.**

**Cost sensitivity — the edge does not survive being traded:**

| Total cost | 0 bps | 2 bps | 5 bps | **10 bps** | 20 bps |
|---|---|---|---|---|---|
| Sharpe | 0.353 | 0.303 | 0.230 | **0.108** | −0.131 |

Even *gross* of costs the Sharpe is 0.35. It crosses zero between 10 and 20 bps — a
realistic commission-plus-slippage assumption is enough to erase it entirely.

---

## 6 · How confident are we in "no"?

**Block-bootstrap Sharpe** (21-day blocks, 2000 resamples):
0.155, 95% CI **[−0.45, 1.13]**, P(Sharpe > 0) = 0.699. The interval comfortably contains
zero.

**Deflated Sharpe ratio.** 50 Optuna trials were run during development. Correcting the
observed Sharpe for that search effort:

| | Value |
|---|---|
| Observed Sharpe | 0.108 |
| Null bar under 50 trials | **0.146** |
| P(skill is real) | **0.46** |

**The null bar is higher than the Sharpe actually achieved** — the best of 50 random
configurations with zero real skill would be expected to look *better* than this did.

---

## 7 · The sealed holdout — scored once

2025-01-01 onward, 5,128 rows, `models/lightgbm_h1/final.joblib`, run exactly once.

| Metric | Value |
|---|---|
| ROC-AUC | **0.5047** |
| 95% CI (block bootstrap) | **[0.4817, 0.5266]** |
| Backtest Sharpe, net | −0.583 |
| Buy-and-hold Sharpe | 0.726 |

The AUC interval **straddles 0.50**. The strategy is loss-making out of sample against a
benchmark that rose.

This is the number the whole design existed to protect: nothing was tuned after seeing it,
and it is reported unchanged.

---

## 8 · Does it replicate? A second market

The identical pipeline, same hyperparameters, no retuning, rerun on 30 US large-caps
(benchmark SPY) in fully separate data and model paths.

| | NSE (primary) | US (generalisation) |
|---|---|---|
| Walk-forward AUC | 0.506 | 0.519 |
| Backtest Sharpe, net | 0.108 | 0.349 |
| Buy-and-hold Sharpe | 1.799 | 1.053 |
| Excess Sharpe | −1.69 | −0.70 |

Same qualitative finding in both: a small, statistically unreliable edge that does not
survive costs against buy-and-hold.

That it replicates across two markets with different data-quality profiles and no shared
tuning argues against "this NSE universe was unlucky" and for "this feature set, at this
data scale, does not carry a robust directional signal."

---

## 9 · Interpretability — and what it says

SHAP on the final model: top features are **market-driven** (`mkt_ret_5d`, `ret_1d`,
`mkt_ret_1d`, `mom_5d`, `mkt_vol_21d`) rather than stock-specific — whatever is being
picked up leans toward broad market beta.

**Feature stability across folds: mean pairwise Jaccard overlap of the top-10 = 0.352.**

A feature that matters in one fold is roughly as likely to drop out of the top 10 next
fold as to stay. That is evidence *against* a stable exploitable signal, not for one — a
genuine edge should show more consistent importance through time.

The correlation matrix says the same thing from another angle: `ret_1d` and `mom_1d`
correlate at **1.000** (the same quantity computed two ways), with several other pairs
above 0.9. ~50 columns, far fewer independent things.

---

## 10 · Limitations — stated, not buried

**Survivorship bias.** The universe is today's constituents, back-tested through history —
firms that delisted or fell out of the index are absent. Point-in-time constituent data
isn't freely available, so this is inherited rather than fixed. It inflates the *benchmark*
more than the direction-classification task, but is not zero.

**Single-market primary result**, mitigated but not eliminated by the US replication.

**Daily bars only.** No intraday microstructure, no order book, no fundamentals, no news
or sentiment — all explicitly out of scope, all plausibly where remaining signal lives.

**Costs are modelled, not measured.** A flat 10 bps commission-plus-slippage; real
execution has market impact that scales with size.

**One flagged data row** (−76% one-day move in NESTLEIND.NS, Jan 2010 — an unadjusted
corporate action) is left in rather than hand-patched. It sits seven years before the first
validation fold.

---

## 11 · What I'd do next

**Accept the null result and change the question, rather than keep tuning this one.** The
deflated Sharpe already says the search effort spent exceeds the signal found; more search
on the same features would produce a better-looking number and a worse-supported one.

Ordered by expected value:

1. **Lower the frequency.** Daily direction is close to a martingale difference sequence.
   Weekly or monthly horizons have a better signal-to-noise ratio and — decisively —
   proportionally far lower cost drag, which is what actually killed this result.
2. **Add genuinely orthogonal data.** Fundamentals, analyst revisions, news sentiment.
   ~50 technical features derived from OHLCV are close to one view of the same thing.
3. **Predict something other than direction.** Volatility *is* forecastable — the ACF of
   |returns| decays slowly where the ACF of returns dies at lag 1. A vol-targeting or
   risk-model application uses the structure that is demonstrably there.
4. **Cross-sectional ranking rather than per-name timing.** Relative ordering within a
   universe is a better-posed problem than absolute direction, and it is what most
   working equity strategies actually do.

---

## What this project demonstrates

**Not** a profitable trading strategy. It says so, everywhere, in its own README.

**A validation harness rigorous enough that a negative result can be trusted:**

- Leakage prevented structurally and enforced by tests that fail the build
- Walk-forward with purge and embargo, never a random split
- A full baseline ladder, B0 first, with every model reported honestly — including the two
  that contradicted expectations
- Costs applied, swept, and shown to be decisive
- Distribution-free confidence intervals, and a deflated Sharpe that accounts for search
- A sealed holdout, scored once, reported unchanged
- Replicated on a second market with no retuning

Two bugs found by refusing to accept a good-looking number: an h=5 backtest reporting a
Sharpe above 3 and a >700% single year (overlapping h-day return windows compounded as if
daily), and — while fixing that — a second, subtler version where per-symbol downsampling
failed to synchronise across symbols. Both are now covered by regression tests. A result
that looks too good is treated as a defect until proven otherwise.

**Live:** https://alphabench.onrender.com · https://alphabench-dashboard.onrender.com
