# AlphaBench — Technical Report

> **Disclaimer.** AlphaBench is an academic research and software-engineering artifact.
> It is not investment advice, not a trading product, and must not be used to make
> financial decisions. All results below are historical simulations and carry no
> guarantee of future performance.

## 1. Question and methodology

**Question.** Given a universe of liquid equities and their daily OHLCV history, can a
supervised model predict the sign of the forward *h*-day return with accuracy
statistically distinguishable from chance, and does that edge survive realistic
transaction costs in a walk-forward backtest?

**Target reframing.** The naive formulation — predict tomorrow's *price* — is broken for
three reasons: prices are non-stationary (a model fit on ₹500 stocks won't transfer to
₹3,000 ones), the series is highly autocorrelated (`Close[t+1] ≈ Close[t]`, so a
persistence forecast trivially achieves low RMSE that looks impressive and means
nothing), and no real decision depends on the price level, only on the change. AlphaBench
predicts direction instead: $r_{t+h} = \log(C_{t+h}/C_t)$, labelled with a
volatility-scaled deadband ($\kappa = 0.3 \times \sigma_t$, $\sigma_t$ a 20-day trailing
realised volatility computed strictly from data up to $t$) so the model is not forced to
take a position on days that are pure noise.

**Feature set.** ~60 features (47 after excluding OHLCV pass-through columns) across six
groups — momentum/reversal, volatility, technical indicators, volume, cross-sectional
ranks (only possible because this is a 30-ticker panel, not a single name), and
market/calendar — every one lag-shifted by one day within its symbol before it can reach
a model (`features/base.py::safe_shift`), enforced structurally rather than by
convention.

**Validation.** Expanding-window walk-forward cross-validation with purging (drop
training rows whose label window overlaps the validation block) and a 5-day embargo
(additional gap to break serial correlation at the boundary) — `validation/splitters.py`.
Seven folds, validating 2017 through 2023 in turn on an expanding 2010-start training
window. `tests/test_leakage.py` asserts train always strictly precedes validation, the
purge/embargo gap is honoured, no feature correlates with the forward return above 0.10
on a synthetic random walk, and shuffled labels collapse AUC to ~0.50 — the harness itself
is calibrated, not just the model. A final holdout block (2025-01-01 onward) was frozen
from the start and touched exactly once, at the very end (section 6).

**Backtest.** Signal from day $t$'s close, execution at day $t+1$'s open (the OOF
probability is already aligned this way), equal-weight positions, 5 bps commission + 5
bps slippage per side applied to turnover, no leverage, no shorting. Position fires only
when the model's probability clears a threshold calibrated to that model's *own*
out-of-fold probability distribution (its 90th percentile — necessary because raw
probabilities cluster tightly around 0.5 at this signal-to-noise level; not a search for
the threshold that maximises backtest P&L).

## 2. The baseline ladder — B0 first

Primary market: NSE, 30 large-cap Indian equities (29 resolved), h=1 day, mean across the
7 walk-forward folds (2017-2023):

| Model | ROC-AUC | Accuracy | Base rate |
|---|---|---|---|
| B0 persistence (today's direction = tomorrow's) | 0.487 | 0.488 | 0.518 |
| B0 majority class | 0.500 | 0.518 | 0.518 |
| B1 logistic regression (5 features) | 0.510 | 0.518 | 0.518 |
| B2 ARIMA(1,0,1) on log returns | 0.503 | 0.511 | 0.518 |
| **M1 LightGBM (primary)** | 0.506 | 0.518 | 0.518 |
| M2 XGBoost | 0.515 | 0.518 | 0.518 |
| M3 LSTM (PyTorch, CPU, small) | 0.515 | 0.518 | 0.518 |
| M4 rank-average ensemble (M1+M2+M3) | 0.514 | 0.508 | 0.518 |

Full per-fold detail: `reports/metrics/baseline_results_h1.json`,
`walkforward_results.json`, `walkforward_results_{arima,xgboost,lstm,ensemble}_h1.json`.

Every entry sits in a 0.487-0.515 band, and each model's own fold-to-fold standard
deviation is roughly 0.01-0.02 (e.g. M1: 0.506 ± 0.020; M3: 0.515 ± 0.019 — see the
per-fold JSON). **The gaps between models are smaller than the noise within any one
model.** There is no ladder to climb here in any statistically meaningful sense — B0
majority (AUC exactly 0.500 by construction) is not reliably beaten by anything.

### B2 — ARIMA diagnostics

ADF and KPSS were run on every symbol's own daily log-return series (never on price
levels — `reports/metrics/arima_diagnostics.json`). Both tests agree the series is
stationary for **29/29 symbols** (ADF rejects the unit-root null; KPSS fails to reject
stationarity), which is the textbook result for daily returns and justifies fitting ARIMA
directly with $d=0$. The ACF/PACF of a representative symbol
(`reports/figures/arima_acf_pacf.png`) is indistinguishable from white noise beyond lag
0 — there is essentially no linear autocorrelation structure for any ARMA order to
exploit, which is consistent with B2's AUC landing at 0.503, statistically
indistinguishable from B0 majority's 0.500.

### M3 — the LSTM comparison, reported honestly

PROPOSAL.md section 4.2 predicted the LSTM would lose to the GBM at this data scale — the
standard, well-supported expectation for ~114k rows of low-SNR tabular-shaped data. It
did not lose here: M3 (0.515 ± 0.019) came in essentially tied with M2 XGBoost and
slightly *above* M1 LightGBM (0.506 ± 0.020). Two things are true at once: this is
reported as it happened rather than adjusted to match the prediction, and it does not
actually contradict the prediction in any way that matters — every model's confidence
interval overlaps every other model's, so "M3 beat M1" is not a claim this sample size can
support. The honest statement is that no model in the ladder is distinguishable from any
other, LSTM included.

### The Prophet rejection

Prophet decomposes a series into trend, seasonality, and holiday components — built for
business time series with real calendar structure (retail demand curves, web traffic
cycles, holiday spikes). Daily equity log returns have none of that: the ACF/PACF above
shows no periodic structure to decompose, and the mean daily return is close enough to
zero that "trend" is not a meaningful concept over a 1-day horizon. Fitting Prophet here
would mean handing it noise and asking it to find smooth curves in it — which it will do,
convincingly, because that is what a smoothing model does regardless of whether structure
is actually present. That combination (a model guaranteed to produce plausible-looking
output on data it has no business modelling) is worse than not including it: it invites
false confidence rather than testing a real hypothesis. B2 ARIMA already covers the
"classical statistical baseline" role properly, on a specification (autoregression on
returns) that at least matches the data-generating process being examined.

## 3. Economic evaluation

### Headline (NSE, M1 LightGBM, net of 10 bps)

| Metric | Strategy | Buy-and-hold NIFTY |
|---|---|---|
| Sharpe (annualised) | **0.108** | 1.797 |
| Ann. return | 1.00% | 34.96% |
| Max drawdown | −29.1% | −37.5% |

Excess Sharpe: **−1.69**. The strategy does not beat buy-and-hold; it substantially
underperforms it, both before and after accounting for the benchmark's own larger
drawdown.

### Cost-sensitivity sweep (NSE)

| Total bps | Sharpe | Ann. return |
|---|---|---|
| 0 | 0.353 | 3.27% |
| 2 | 0.303 | 2.81% |
| 5 | 0.230 | 2.13% |
| **10** | **0.108** | **1.00%** |
| 20 | −0.131 | −1.22% |

The edge is not robust even before costs (Sharpe 0.35 gross is modest on its own), and it
crosses zero between 10 and 20 bps — a realistic commission+slippage assumption is enough
to erase it. Full table: `reports/metrics/backtest_results.json`.

### Bootstrap confidence intervals

- **Sharpe** (block bootstrap, 21-day blocks, 2000 resamples): 0.155, 95% CI
  **[−0.45, 1.13]**, P(Sharpe > 0) = 0.699. The interval comfortably contains zero.
- **Holdout AUC** (block bootstrap over dates, 2000 resamples): 0.505, 95% CI
  **[0.482, 0.527]** (section 6). Straddles 0.50.

### Deflated Sharpe ratio

A 50-trial Optuna search over LightGBM hyperparameters (`reports/metrics/optuna_study_h1.json`)
found a best mean-CV-AUC of 0.5222 (vs. 0.5060 for the shipped `DEFAULT_PARAMS` model —
the model that is actually reported everywhere in this document; the tuned parameters were
never used to retrain the shipped model, precisely to avoid reporting a result inflated by
that search). Correcting the reported backtest Sharpe (0.108) for the 50 trials run and
the sample's own variability:

| | Value |
|---|---|
| Observed Sharpe (annualised) | 0.108 |
| Null bar under 50 trials (annualised) | 0.146 |
| P(skill is real) | **0.46** |
| Significant at 95%? | **No** |

The null bar — the Sharpe you would expect from the best of 50 random configurations with
zero real skill — is *higher* than the Sharpe actually observed. Even without invoking the
holdout, the walk-forward backtest result alone does not clear the bar a reviewer should
expect from chance plus search effort.

### Per-year breakdown (NSE)

| Year | Ann. return | Sharpe |
|---|---|---|
| 2017 | +5.7% | 1.89 |
| 2018 | +8.0% | 0.85 |
| 2019 | −0.0% | −1.02 |
| 2020 | −2.3% | 0.01 |
| 2021 | −2.7% | −1.25 |
| 2022 | −2.9% | −1.45 |
| 2023 | +1.5% | 1.18 |

Sign flips essentially every other year with no visible regime pattern (not, for example,
concentrated in the 2020 COVID crash) — the strategy has not found one event, it has found
noise that occasionally points the right way.

### Per-ticker breakdown (NSE)

Full table: `reports/metrics/backtest_results.json` (`by_ticker`). Per-symbol Sharpe
ranges from **+0.62 (DRREDDY.NS)** to **−0.80 (NTPC.NS)**, with roughly half the 29
symbols positive and half negative. This directly answers PROPOSAL.md section 7.3's
question — "does the edge generalise, or is it two names carrying twenty-eight?" — and
the answer is neither: there is no small subset of names driving the aggregate result:
the aggregate itself is unremarkable, and the dispersion across names is exactly what you
would expect from independent noise around a near-zero mean, not a real cross-sectional
signal that happens to concentrate somewhere.

## 4. Week 9 generalisation test — a second, independent market

The identical M1 LightGBM h=1 pipeline (same `DEFAULT_PARAMS`, no retuning) was rerun on
`config/universe_us.yaml` (30 US large-caps, benchmark SPY), in fully separate
`data/*_us/` and `models/*_us/` paths so the NSE run is never touched:

| Metric | NSE (primary) | US (generalisation) |
|---|---|---|
| Walk-forward AUC (mean, 7 folds) | 0.506 ± 0.020 | 0.519 ± 0.028 |
| Backtest Sharpe, net of 10 bps | 0.108 | 0.349 |
| Buy-and-hold Sharpe | 1.797 (NIFTY) | 1.053 (SPY) |
| Excess Sharpe | −1.691 | −0.704 |
| Bootstrap 95% CI, P(Sharpe>0) | [−0.45, 1.13], 0.699 | [−0.22, 1.30], 0.884 |

The US run shows a somewhat higher (still noise-level) AUC and a less negative excess
Sharpe than NSE, but the qualitative finding is identical: a small, statistically
unreliable edge that does not survive realistic costs against buy-and-hold. That the
result replicates in direction and rough magnitude across two markets with different
data-quality profiles, different benchmark dynamics, and no shared tuning is itself
informative — it argues against "this particular NSE universe happened to be unlucky" and
for "this feature set, at this data scale, does not carry a robust directional signal."

## 5. Interpretability

SHAP (`TreeExplainer`) on the final LightGBM model, 5000-row sample of pre-holdout data
(`reports/figures/shap_importance_h1.png`, `shap_beeswarm_h1.png`). The top features are
market/momentum-driven — `mkt_ret_5d`, `ret_1d`, `mkt_ret_1d`, `mom_5d`, `mkt_vol_21d` —
rather than idiosyncratic technical indicators, suggesting whatever the model is picking
up leans toward broad market beta rather than stock-specific structure.

Top-10-feature stability across the 7 folds' own models
(`reports/metrics/shap_fold_stability_h1.json`): mean pairwise Jaccard overlap **0.352**.
Moderate churn — a feature that matters in one fold is roughly as likely to drop out of
the top 10 next fold as to stay — which is itself evidence against a stable, exploitable
signal rather than for one. A genuinely robust edge should show more consistent feature
importance across time than this.

## 6. The sealed holdout (SC-3, SC-4)

`reports/metrics/holdout_results.json`, written exactly once, by
`alphabench evaluate-holdout`, which refuses to run a second time by design. Scored on
`models/lightgbm_h1/final.joblib` (the `DEFAULT_PARAMS` model, never retrained on the
Optuna search) against every row from **2025-01-01 onward** — 5,128 rows, 276 trading
days, never touched by any fold, feature-selection decision, or tuning run before this.

| Metric | Value |
|---|---|
| ROC-AUC | **0.5047** |
| 95% bootstrap CI (block, 2000 resamples) | **[0.4817, 0.5266]** |
| Backtest Sharpe, net of 10 bps | **−0.583** |
| Buy-and-hold Sharpe (same window) | 0.726 |
| Excess Sharpe | −1.309 |
| Ann. return (strategy / benchmark) | −1.3% / +10.2% |

The CI straddles 0.50. The strategy loses money in absolute terms on the holdout while
buy-and-hold gains 10%. This is the single most consequential number in the project, and
it is fully consistent with every walk-forward fold, both markets, the deflated Sharpe,
and the per-ticker dispersion reported above — nothing here comes as a surprise given the
rest of the evidence.

## 7. Limitations

**Survivorship bias (PROPOSAL section 5.4).** The universe is today's 30 constituents,
backfilled to 2010; companies delisted, acquired, or bankrupted over that window are
absent, which biases the observed return distribution upward. This is not eliminated,
only disclosed and bounded — point-in-time constituent membership is not available for
free. Because the target here is relative direction rather than absolute return level,
the effect on directional AUC is smaller than it would be on a long-only return backtest,
but it is not zero, since any symbol's long-run drift being survivorship-conditioned still
shapes the label distribution the model trains on.

**Costs modelled, market impact not.** 5+5 bps commission/slippage per side is applied
uniformly; no model of price impact from the strategy's own trading, which at the position
sizes implied by a 30-name equal-weight book is a reasonable simplification but would not
scale to larger capital.

**No point-in-time universe reconstruction; no intraday data; no leverage or shorting**
modelled beyond a flag. All explicitly out of scope per PROPOSAL section 3.2, not
oversights.

**Deflated Sharpe uses the Optuna search's trial count (50) applied to the reported
backtest Sharpe**, not a search directly over backtest configurations — a defensible but
not perfectly literal application of Bailey & López de Prado's correction, since the
Optuna objective was walk-forward AUC, not Sharpe. The shipped model was not selected by
that search at all (see section 3), which is the more important guardrail: whatever the
correction's precision, no configuration search informed which model's Sharpe is being
reported as the headline number.

## 8. Conclusion

Across a full model ladder (B0 through M4), two independent markets, a 50-trial
hyperparameter search with its selection bias corrected for, per-year and per-ticker
breakdowns, and a sealed holdout touched exactly once, **AlphaBench finds no directional
signal in daily equity returns that survives realistic transaction costs.** Walk-forward
AUC sits at 0.49-0.52 for every model tried, with fold-to-fold noise as large as the gaps
between models. The primary market's backtest underperforms simply buying and holding by
a wide margin (excess Sharpe −1.69), the deflated Sharpe correction shows the modest
observed edge doesn't clear the bar chance-plus-search-effort would produce on its own,
and the sealed holdout — the number that matters most, precisely because it could not be
p-hacked after the fact — lands at AUC 0.505 with a 95% CI straddling chance, and loses
money net of costs while the benchmark gains 10%.

This is not a failed project. It is what a rigorous treatment of this question is
expected to produce (PROPOSAL section 1, section 8.1): daily equity returns are close to
a martingale difference sequence, and a result indistinguishable from chance, arrived at
through a leakage-audited, walk-forward-validated, cost-aware pipeline with honest
uncertainty bounds throughout, is the credible outcome — not evidence the pipeline is
broken, and not a result to spin toward a more exciting conclusion than the data supports.
