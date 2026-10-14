# Validation-only tuning and demand diagnostics

This offline slice extends **time-series forecasting**, not the inherited
revenue regression factory. There are thirteen registry IDs: the original
seven, Croston/SBA/TSB, and optional ETS/ARIMA/SARIMA. No model is automatically
approved, and no complexity-based preference is imposed.

## Runnable synthetic demonstrations

The intermittent methods need only the existing core dependencies:

```bash
uv run revenue-prediction compare-demand \
  --pattern intermittent --models naive,seasonal_naive,croston,sba,tsb \
  --tuning-grid configs/examples/demand-tuning.json \
  --out outputs/demand-tuned-sparse
```

Use `--pattern long_zero_runs` or `--pattern obsolescence` to exercise zero-run
and retirement-like behavior. Configure `--occurrence-probability`,
`--zero-run-periods` and `--obsolete-after-fraction` deliberately. All-zero
training scales remain undefined; there is no automatic MASE-to-WAPE fallback.

Install guarded statistical adapters explicitly:

```bash
uv sync --extra api --extra forecasting
uv run revenue-prediction compare-demand \
  --models naive,seasonal_naive,ets,arima,sarima,ses,holt_damped \
  --tuning-grid configs/examples/demand-tuning.json \
  --skus 1 --channels 1 --geographies 1 \
  --out outputs/demand-tuned-monthly
uv run revenue-prediction compare-demand \
  --frequency W-MON --window 156 \
  --models naive,seasonal_naive,ets,ses,holt_damped \
  --tuning-grid configs/examples/demand-tuning.json \
  --skus 1 --channels 1 --geographies 1 \
  --out outputs/demand-tuned-weekly
```

Counts are intentionally small for statistical demonstrations. The same API
supports larger synthetic cohorts, but repeated seasonal state-space fits cost
more, especially at weekly seasonal length 52. Nothing runs in Azure/Fabric.
Without `--models`/`--tuning-grid`, the seven-model fixed comparison is preserved.

## Grid and temporal protocol

[Example grid](../../configs/examples/demand-tuning.json) is a JSON array of
unique `TimeSeriesModelConfig` objects. Omitted fields retain validated defaults.
The two supplied trials change smoothing, averaging and ARIMA orders; they
are illustrative, not calibrated optimal settings. Parameters irrelevant to a
model do not alter that model: ETS estimates coefficients within each training
window, while original SES/Holt/intermittent methods use fixed coefficients.

1. Predeclare the grid, candidate set, cohort, seasonal calendar, horizons,
   validation/holdout blocks, evaluation cutoffs, primary metric and gates.
2. Establish mandatory naive/seasonal-naive anchors and the fixed cohort.
3. For every grid row and non-anchor model, fit/refit each series separately
   at each rolling validation origin using arrived history only.
4. Require complete candidate coverage. A structural, numerical, convergence
   or evaluation failure rejects the **whole model trial**, not a hard series.
   Record its reason explicitly. Missing optional dependencies abort with setup
   instructions; they are not negative performance evidence.
5. Apply the same MASE/RMSSE/WAPE, bias and horizon gates to all successful
   trials. Alternatives must strictly improve seasonal naive. Select lowest
   eligible validation metric, then model ID, then earliest trial for ties.
6. Freeze model ID, parameters and successfully fitted candidate set. Baselines
   remain mandatory. Persist the original full design and rejected trials too.
7. Evaluate only the fixed winner and anchors on the separate final holdout.
   Later holdout origins can refit from newly arrived history, but cannot
   reselect parameters/model identity or replace a failing winner.

This is a predeclared grid over rolling validation, not random-row search,
per-series hyperparameter selection, a nested inner-CV implementation or a
full automatic ETS/ARIMA search. Validation scores are search-biased; final
holdout estimates must remain separately reported. After inspecting holdout,
changes require a new experiment/test period, not relabeling that holdout as
untouched evidence.

Python entry point: `run_demand_tuning(actuals, DemandTuningConfig(...))`.
Its `frozen_config` reproduces the selected experiment through
`run_demand_selection`. The immutable decision includes `selected_parameters`.
See [metrics and gates](time-series-metrics.md).

## Persisted evidence

In addition to [comparison artifacts](backtesting.md), tuning writes:

- `tuning_config.json`: original experiment and complete predeclared grid.
- `tuning_trials.csv`: every model/trial, metric/gate eligibility, parameters,
  failure reason and zero-based trial index.
- `tuning_metrics.csv`: successful validation portfolio/horizon metrics by trial.
- `tuning_selection.json`: selected trial index.
- `experiment_config.json`: frozen configuration for direct reproduction.

All comparisons also write series and segment/horizon metrics for validation
and holdout. Brand/family/product/package/channel/geography views use the same
origin-specific scales and expose undefined metrics; no difficult rows are
silently excluded. They are diagnostic views, not reconciliation or segment
acceptance gates.

`training_diagnostics.csv` describes the first validation origin only:
positive counts, zero fraction, average demand interval, positive-size squared
CV, longest/trailing calendar-contiguous zero runs, calendar continuity and
seasonal scale coverage. Interval/CV summaries describe arrived observations;
with arrival gaps they must not be interpreted as complete calendar estimates.
All-zero training has explicit status and undefined interval/CV.
`model_eligibility.csv` checks structural requirements for the requested models
with the selected trial's parameters. It does not claim convergence,
appropriateness or accuracy; baseline seasonal-source coverage is checked
during actual forecasting. These diagnostics cannot observe future targets.

CSV dates are strings; use explicit `pd.to_datetime` when reading them.
Parquet predictions preserve date dtypes. Canonical row ordering makes direct
and tuned reproductions exact. Successful reruns overwrite named artifacts;
failures write no new success artifacts but do not remove earlier runs.

## Limits and verification

Tests cover exact Croston/SBA/TSB recurrences, all-zero states, TSB decay,
provider-equivalent calendar leads, failed/nonfinite fits, dependency guards,
future-value invariance, complete-cohort trial rejection, business gates,
parameter selection and persisted configuration/metric/prediction roundtrips.
Statistical tests opt out if the optional package is absent; install the extra
to exercise them. Core-only import/forecast tests block statsmodels explicitly.

Not implemented: model serialization/production registry, probabilistic
forecasts, inventory/service-cost optimization, ACF/PACF/residual acceptance
gates, revised vintages, external covariates, reconciliation and live cloud
validation. See [ADR 0008](../architecture/adr/0008-demand-adapters-and-tuning.md).
