# Demand time-series metrics and selection specifics

R2 and MAPE are not the demand selection objective. A good in-sample regression
fit does not establish multi-step forecasting performance. Evaluate ordered
future periods from rolling origins, by horizon and series, with actual arrival
times and a separate final holdout.

## Seasonal training scales: per series and per origin

For seasonal length `m`, use only historical pairs `(y_t, y_(t-m))` where both
actuals have arrived by that origin and are inside the training window:

```text
absolute scale = mean(abs(y_t - y_(t-m)))
squared scale  = mean((y_t - y_(t-m)) ** 2)
```

Pairs are matched by **calendar date**, not row offsets after arrival filtering.
Monthly m defaults to 12, weekly m to 52; configure the season explicitly.
Neither validation nor holdout actuals participate in these denominators.

Row-level scales and pair counts are persisted with evaluated forecasts.
`insufficient_seasonal_pairs` and `zero_seasonal_scale` are explicit conditions.
When any evaluated row lacks a defined scale, aggregate MASE/RMSSE are `NaN`
with `scaled_metrics_status=undefined_training_scale`; never rank a model using
only its easier/scalable rows. `n_scaled_predictions` exposes coverage.
Seasonal length must be positive. Unrepresentable training scales raise an
explicit overflow error requiring demand-unit rescaling, rather than silently
producing infinite denominators and misleading zero scaled errors.

## Implemented metrics

Error is `prediction - actual`; positive means overforecasting.

| Metric | Definition / use | Caveat |
| --- | --- | --- |
| **MASE (default selection)** | Mean absolute forecast error divided by the corresponding origin/series absolute training scale | Scale-free comparison across series; undefined for insufficient/constant seasonal history |
| **RMSSE** | Square root of mean squared forecast error divided by corresponding squared training scales | Penalizes larger scaled misses; not the mean of per-series roots |
| **WAPE** | Sum absolute error / sum absolute actuals | Volume-weighted portfolio comparison; high-volume series dominate; undefined near zero actual total |
| MAE | Mean absolute error, demand units | Interpretable magnitude, not scale-free across products |
| RMSE | Root mean squared error, demand units | Sensitive to large misses/outliers |
| sMAPE | Mean `2 * abs(error) / (abs(actual)+abs(prediction))`, fraction [0,2] | Both-zero contributes zero; near-zero behavior is unstable, so not the default ranker |
| Mean error | Mean signed error, demand units | Shows systematic direction; offsetting series may hide bias |
| Bias fraction / percent | Sum error / sum actuals; percent = fraction * 100 | Undefined near zero actual total; positive is overforecast |
| Tracking signal | Cumulative signed error / mean absolute error | Zero for perfect predictions; depends on evaluation row count and pooling, so compare like windows |

The implemented multi-series MASE/RMSSE weight forecast rows equally after
per-series scaling. With a fixed balanced cohort this gives equal series/horizon
representation. WAPE instead weights total volume. Overlapping rolling horizons
evaluate a target at multiple origins; these are different forecasts, not
independent observations.

`horizon=0` is the portfolio aggregate. Positive values report individual
horizons. Row-level predictions permit additional series/segment diagnostics;
dedicated segment acceptance gates are not implemented yet. Backtests now
persist per-series and brand/family/product/package/channel/geography summaries
using the same formulas and per-horizon protocol.

**MASE < 1 is not proof of beating out-of-sample seasonal naive.** Its denominator
is in-sample seasonal variation. Compare the candidate directly to the
seasonal-naive validation forecast errors on the same rows.

For constant or perfectly seasonal training, MASE/RMSSE may be undefined even
when forecasts are perfect. The default selection fails with a corrective
message; an explicitly predeclared WAPE policy can handle nonzero stable demand.
There is no automatic metric fallback or artificially zero scale.

## Validation-only selection and configurable gates

`DemandSelectionConfig` fixes:

- Rolling validation and nonoverlapping later holdout.
- Same frequency/horizon, models, smoothing parameters, seasonal length,
  training strategy/window and explicit scope.
- Primary metric: `mase` (default), `rmsse` or `wape`.
- Minimum relative improvement over seasonal naive (default zero, with strict
  improvement still required for alternatives).
- Optional absolute bias-fraction tolerance.
- Optional maximum horizon-specific relative degradation versus seasonal naive.

No universal bias or improvement tolerance is imposed. Choose gates based on
planning cost, strategic segments, horizon and business risk.

Validation targets must close before holdout starts; their actuals must be
evaluated no later than the first holdout origin. Selection uses validation
metrics only. Undefined primary metrics cannot pass; no qualifying model raises
an error. Ties retain seasonal naive, with deterministic ID ordering among
strictly improving alternatives.

After selection, evaluate only the fixed winner plus naive/seasonal-naive
anchors on holdout. At successive holdout origins, refit the **same algorithm
with the same parameters** using newly arrived history. This mirrors
reforecasting, not a static never-updated model. Never choose a replacement from
holdout errors. Every decision remains local and unapproved.

The [parameter-grid tuner](tuning.md) applies these same gates to every
predeclared model/parameter trial using validation only, freezes the winning
configuration and then evaluates holdout. Training-only eligibility is not a
substitute for measured out-of-sample error or business acceptance.

## Not yet implemented

Probabilistic forecasters will require pinball loss, interval coverage/width and
calibration by horizon. Inventory/service decisions require supplied under/over
forecast costs and service-level metrics. Intermittent demand, revision vintages,
hierarchical reconciliation and segment gates need their own validation.

Reference for mathematical context:
[Forecasting: Principles and Practice - accuracy](https://otexts.com/fpp3/accuracy.html).
See [model catalog](model-catalog.md) and [runnable comparison](backtesting.md).
