# Demand time-series model catalog and algorithm-selection guide

## This is forecasting, not the inherited regression task

Demand predicts future periods in ordered SKU/channel/geography series at a
forecast origin, with weekly/monthly frequency and multiple horizons. Revenue
snapshot regression predicts a current month's scalar from partial-month
operational features. Their inputs, validation protocols, algorithm catalogs,
selection metrics and acceptance criteria differ.

The demand path does **not** call the revenue model factory or AutoML
regression builder. Historical regression docs remain explicitly scoped to
revenue. [ADR 0007](../architecture/adr/0007-demand-time-series-model-selection.md)
records this distinction.

## Implemented and discoverable

```bash
uv run revenue-prediction demand-models
uv run revenue-prediction compare-demand --metric mase --out outputs/demand-comparison
```

The CLI returns the same capability registry used by Python forecasting and
point-output validation. `create_demand_forecaster` constructs an independent
`Forecaster` with `fit`, `predict` and `capability`. `forecast_demand_models`
performs origin filtering, per-series fitting and strict point output validation.
The old `forecast_demand_baselines` API remains unchanged.

| ID | Family / algorithm | Pattern it tests | Minimum/eligibility |
| --- | --- | --- | --- |
| `naive` | Last-value baseline | Stable local level | At least one arrived observation; stale last value allowed and traced |
| `seasonal_naive` | Calendar-season baseline | Stable repeatable seasonality | Every required calendar seasonal source must have arrived |
| `moving_average` | Trailing-window baseline | Noisy level, no explicit trend/seasonality | Contiguous history; at least `average_periods` (default 3) |
| `seasonal_moving_average` | Average matching trailing seasonal positions | Noisy repeating seasonal profile | Contiguous history; at least `average_seasons * seasonal_periods` (default two seasons) |
| `drift` | Average first-to-last trend extrapolation | Persistent approximately linear trend | At least 2 contiguous observations; vulnerable to endpoint outliers |
| `ses` | Simple exponential smoothing | Evolving level, **no** trend or seasonal state | At least 2 contiguous observations |
| `holt_damped` | Additive damped level/trend smoothing | Trend that attenuates with lead | At least 3 contiguous observations; **no** seasonal state |
| `croston` | Intermittent size/interval smoothing | Sparse positive arrivals | At least 2 contiguous periods; all-zero training forecasts zero |
| `sba` | Syntetos-Boylan correction to Croston | Intermittent demand with Croston bias correction | Same eligibility; forecast multiplied by `1-alpha/2` |
| `tsb` | Teunter-Syntetos-Babai occurrence/size smoothing | Intermittency and declining occurrence/obsolescence | Same eligibility; occurrence probability decays during zero runs |
| `ets` | Optimized additive Holt-Winters | Level, optional trend and seasonality | `forecasting` extra; seasonal fits need at least two complete seasons and `m >= 2` |
| `arima` | Nonseasonal ARIMA | Integrated/autocorrelated dynamics | `forecasting` extra; at least `max(8, 2*(p+d+q+1))` contiguous periods |
| `sarima` | Seasonal ARIMA | Seasonal integrated/autocorrelated dynamics | `forecasting` extra; `m >= 2`, at least `max(8, 2*m, 2*(p+d+q+(P+D+Q)*m+1))` contiguous periods |

The ten baseline/smoothing/intermittent implementations are original NumPy work
under the repository's MIT license. The three optional statistical adapters
call BSD-3-Clause statsmodels APIs; no library source is copied. Install with
`uv sync --extra forecasting` (add `--extra api` to retain the API environment).
All support `MS`/`W-MON`, local univariate
point forecasting only. No covariates, probabilistic output, hierarchy
reconciliation, live Azure validation or serialization API is claimed.
Persisted experiment configs and metadata reproduce the local comparison;
registered production model artifacts remain future work.

### Fitting and horizon specifics

- Season length defaults to 12 monthly or 52 weekly periods. The weekly value
  is a cycle length, not ISO-year alignment; configure it deliberately.
- `TimeSeriesModelConfig`: `alpha=0.2`, `beta=0.1`, `damping=0.9`,
  `average_periods=3`, `average_seasons=2`. Parameters are fixed before
  comparison, validated and saved in the experiment config.
- SES initializes level with the first actual and updates
  `level = alpha * actual + (1-alpha) * previous_level`. Its forecasts are flat.
- Damped Holt initializes level with the first actual and trend with the first
  difference. Its level update uses the damped prior trend; the trend update
  smooths the observed level change and damped prior trend. A lead-h forecast
  adds `trend * sum(damping**j for j in 1..h)` to the last fitted level.
- SES/damped Holt are **fixed-coefficient** smoothing implementations, not automatic ETS,
  maximum-likelihood optimization or Holt-Winters seasonal smoothing.
- Drift/Holt use calendar lead from the last arrived observation, not just the
  request's horizon number. With a delayed final actual, a horizon-1 request
  may be two or more periods ahead of the last observation.
- Seasonal moving average cycles the mean of corresponding positions in the
  trailing configured seasons. It never feeds future actuals into long horizons.
- Projections below zero are clipped to zero, then evaluated. This policy fits
  nonnegative demand units; it is not valid for an unreviewed returns/net target.
- Internal arrival gaps fail statistical/window models; no row compression,
  forward fill, cross-series pooling or silent candidate substitution occurs.
- Croston/SBA initialize size from the first positive value and interval from
  its one-based position. Subsequent positive arrivals smooth size and elapsed
  interarrival interval using `alpha`; zeros advance the elapsed counter.
- TSB initializes occurrence probability to the reciprocal of that position,
  then updates it every period using `beta`; positive size uses `alpha`.
  No positive history gives an explicit zero state for all three methods.
- ETS defaults to additive trend/seasonality and damped trend, with estimated
  initialization and training-only optimization. It is not a full automatic
  ETS model-class search. `ets_trend`/`ets_seasonal` can be `None` or `"add"`;
  disabling trend also requires `ets_damped=False`. Multiplicative components
  are deliberately unavailable so zeros are valid.
- ARIMA/SARIMA use the statsmodels SARIMAX estimator with no exogenous variables,
  no deterministic trend/intercept, and enforced stationarity/invertibility.
  Defaults: `arima_order=(1,1,0)`, `sarima_order=(1,0,0)` and configured seasonal
  length. Order entries are strict integers 0-5. Optimization is limited by
  `statistical_maxiter` (default 200, allowed 1-2000).
- Nonconvergence, numeric fit failure and nonfinite output are explicit errors.
  All fitted adapters forecast through the actual calendar lead before slicing
  requested dates; nonnegative clipping matches the existing target contract.

## How to choose the candidate set

1. Fix target units/grain, frequency, seasonal calendar, origins and horizon.
2. Inspect trend, seasonal profiles, autocorrelation, missing/revised values,
   volume, zeros/intermittency, outliers and regime changes using training only.
3. Always establish naive/seasonal-naive performance at each required horizon.
4. Add eligible pattern-matching candidates from the table; do not treat a
   longer history or more complex model as evidence of superiority.
5. Fit/refit inside temporal validation origins; fix hyperparameters or tune
   them in nested temporal training/validation, never on final holdout.
6. Compare [scaled metrics and portfolio/horizon bias](time-series-metrics.md)
   on the same cohort, horizons and actual availability rules.
7. Apply predeclared business gates, freeze selection, then evaluate holdout.

The default comparison retains the original seven models. New IDs require
explicit `--models`; optional adapters never become implicit core dependencies.
The fixed comparison aborts on failed eligibility. The
[tuning workflow](tuning.md) records failed full-cohort candidate trials while
retaining mandatory baselines and never dropping hard series.

## Future adapters: not installed or executable yet

| Family | When to investigate | Checks before acceptance |
| --- | --- | --- |
| SARIMAX / dynamic regression | Demand with valid future external/planned predictors | Forecast-origin vintages; future covariate availability and separate scenario assumptions |
| Global lag-feature ML | Many related series and nonlinear available signals | Per-origin lag construction, train-only preprocessing, recursive/direct multi-step strategy; no random rows |
| LSTM / seq2seq / TFT | Sufficient history/series, nonlinear sequential/covariate structure | Optional dependencies, license/compatibility, donor adaptation tests, compute cost and baseline improvement |
| Time-series foundation models | Transfer learning with limited local history | License/provider, domain calibration, uncertainty, privacy and temporal holdout validation |

These are research/backlog choices, not registry capabilities. Classical
forecasters do not need an LLM; an LLM cannot select or invent official outputs.
Future Azure ML AutoML demand uses **SDK v2 forecasting**, not regression.
Azure/Fabric integrations remain opt-in and unvalidated in this slice.
