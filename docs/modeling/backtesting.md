# Offline demand baselines and rolling backtests

This path implements deterministic time-series **point forecasts**, not the
revenue regression task, probabilistic models or a production-approved system.
Existing revenue commands/API/UI are preserved.

## Run the synthetic-only demonstration

```bash
# Five years of monthly synthetic data, expanding windows, 3 origins x 3 horizons
uv run revenue-prediction backtest-demand --out outputs/demand-monthly

# Weekly synthetic data, sliding two-year calendar history
uv run revenue-prediction backtest-demand --frequency W-MON --window 104 \
  --out outputs/demand-weekly
```

PowerShell can use the same one-line commands; replace Bash line continuation
with a backtick when splitting a command. No cloud credentials are required.

Options: `--frequency MS|W-MON`, `--horizon`, `--origins`, `--step`, `--window`,
`--seed`, `--out`. All counts must be positive; seed must be nonnegative.
The CLI generates only synthetic data. It picks the trailing origins whose
full horizons fit the generated history and uses its latest actual arrival as
the explicitly persisted evaluation cutoff.

Both baselines always participate in this demonstration. Too little seasonal
history/window or invalid horizons produce a nonzero exit with a corrective
message, before any result files are written.

Artifacts in the selected directory:

- `actuals.parquet`: strict historical demand contract.
- `backtest_predictions.parquet`: forecast point contract plus actual,
  actual arrival, evaluation cutoff and signed `error = prediction - actual`.
- `metrics.csv`: model portfolio and per-horizon summaries.
- `data_config.json`, `backtest_config.json`: reproducible synthetic and
  evaluation configuration.

Rerunning overwrites these named generated files. Use a dedicated output
directory; the command never deploys, publishes, registers, approves or chooses
a champion. The directory is not a Fabric Gold table.

## Direct inference API

```python
from datetime import date

from revenue_prediction.core.data.demand_forecast import DemandForecastRequest
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.models.demand_baselines import forecast_demand_baselines

actuals = generate_synthetic_demand_dataset()
request = DemandForecastRequest(forecast_origin=date(2025, 1, 1), horizon=3)
forecasts = forecast_demand_baselines(actuals, request)
assert len(forecasts) == 12 * 3 * 2  # series x horizons x baseline models
assert forecasts["approval_status"].eq("unapproved").all()
```

Naive repeats the most recent origin-available value for every horizon.
Seasonal naive repeats the previous calendar cycle (12 months or 52 weeks by
default), including beyond one season. It never feeds future actuals back into
later horizons. A 52-week cycle does not align ISO-year week numbers perfectly.

If a seasonal source has not arrived, the default comparison fails explicitly.
For an intentional naive-only workflow, pass `models=("naive",)`. No silent
fallback, future pooling, forward fill or imputation occurs.

## Backtest API

```python
from datetime import date

from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    run_demand_backtest,
)

actuals = generate_synthetic_demand_dataset()
config = DemandBacktestConfig(
    first_origin=date(2024, 7, 1),
    evaluation_as_of=date(2025, 1, 1),
    horizon=3,
    n_origins=3,
    step_periods=1,
    strategy="expanding",
)
result = run_demand_backtest(actuals, config)
assert len(result.predictions) == 3 * 12 * 3 * 2
print(result.metrics)
```

For sliding evaluation, set `strategy="sliding"` and
`training_window_periods=24` (monthly) or an appropriate weekly window. Supplying
a window with expanding strategy or omitting it with sliding strategy is invalid.
`seasonal_periods` is configurable. `models` and explicit `series` scope can be
provided through configuration. `models` now accepts any implemented
[registry ID](model-catalog.md), and `model_parameters` fixes smoothing/window
parameters before fitting. The baseline-only defaults remain backward compatible.

### Temporal protocol

1. Validate frequency, calendar continuity, hierarchy and raw actuals.
2. Establish explicit series scope or discover the cohort at the first origin.
3. Advance aligned origins by `step_periods`, without random row splitting.
4. For each origin, filter training actuals by their actual arrival time.
5. For sliding windows, exclude dates outside the specified calendar window.
6. Generate all model/series/horizon predictions before accessing target truth.
7. Join exact target keys to actuals received by `evaluation_as_of`.
8. Require complete coverage; missing/unavailable truth aborts the run.
9. Calculate the same metrics for the same cohort across baseline models.

The fixed cohort excludes later launches from this comparison. Explicitly
requesting a series with no first-origin history fails; a future cold-start
workflow needs its own model and evaluation policy.

### Metrics

`horizon=0` denotes the model's portfolio aggregate; positive values denote
individual horizons. Each record includes prediction/origin counts:

- MAE and RMSE, in synthetic demand units.
- Mean signed error, positive for overforecasting.
- sMAPE as a fraction in [0, 2].
- WAPE = sum absolute error / sum absolute actuals.
- `bias_fraction` = sum signed error / sum actuals; multiply by 100 for percent.
- `relative_metrics_status`: `defined` or `undefined_near_zero_actual_total`.
- MASE/RMSSE from origin/series-specific seasonal training differences, with
  pair counts and explicit undefined-scale status.
- Bias percentage and tracking signal.

See [time-series metric definitions](time-series-metrics.md); MASE is the
default **selection** metric in the separate comparison workflow below.

The inherited WAPE/bias helpers use a `1e-9` denominator threshold. Undefined
relative metrics are `NaN`, paired with explicit status, not reported as zero.
Zero-demand forecasts still have meaningful MAE/RMSE and sMAPE.

Overlapping horizons count a target once for each origin/horizon forecast.
Portfolio metrics weight forecast rows, not unique target dates. A raw backtest
does not itself select a model or define a separate final holdout. Use the
comparison workflow for that separation.

## Registry-driven statistical comparison and final holdout

```bash
uv run revenue-prediction demand-models
uv run revenue-prediction compare-demand --metric mase --out outputs/demand-comparison
uv run revenue-prediction compare-demand --frequency W-MON --window 156 --out outputs/demand-weekly-comparison
```

The default synthetic comparison evaluates the original seven time-series algorithms.
Use `--models` to select additional registry IDs, and `--tuning-grid` for
[validation-only parameter tuning](tuning.md).
It uses a trailing validation block whose target periods all finish before
holdout, and an evaluation cutoff at the first holdout origin. Default three
origins and three horizons apply to each block. Training history is expanding
unless `--window` sets a calendar sliding window.

Choose `--metric mase|rmsse|wape`, `--minimum-improvement`,
`--max-abs-bias` (fraction) and `--max-horizon-degradation` according to business
requirements. MASE is default; there is no automatic switch when a scale is
undefined. Unknown/regression metrics such as R2 are rejected.

The persisted experiment declares all models/parameters/gates before evaluation.
Selection ranks validation portfolio metrics only, requiring strict improvement
over seasonal naive for alternatives. Holdout then evaluates just the selected
algorithm and mandatory baseline anchors, without reselection.

Outputs are `actuals.parquet`, `validation_predictions.parquet`,
`validation_metrics.csv`, `validation_ranking.csv`, `holdout_predictions.parquet`,
`holdout_metrics.csv`, `data_config.json`, `experiment_config.json`,
`model_capabilities.json` and the frozen `selection.json`. They are local and
unapproved. Rerunning overwrites these named generated files.

Also persisted: `validation_series_metrics.csv`, `validation_segment_metrics.csv`
and the equivalent holdout files, plus first-validation-origin
`training_diagnostics.csv` and `model_eligibility.csv`. CSV timestamps are
ISO-formatted strings; parse them explicitly when loading diagnostics.
Predictions are canonically ordered by origin/model/series/horizon so direct
and tuned configurations reproduce the same rows and summaries.

Python entry point: `run_demand_selection(actuals, DemandSelectionConfig(...))`
with explicit `validation` and `holdout` `DemandBacktestConfig` objects. Declare
the same model/parameter/frequency/horizon/window/scope policy in both. Targets
must not overlap and validation actuals must be known by the holdout start.
The validation cohort is carried into holdout, excluding later launches.

Later holdout origins refit the selected algorithm using newly arrived actuals;
identity/parameters stay fixed. This is prequential reforecasting, not
holdout-driven model tuning. Keep a separate experiment if you revise the
policy after looking at these final results.

## Tests and limits

```bash
uv run pytest -q tests/unit/test_demand_forecasting.py \
  tests/contract/test_demand_forecasts.py tests/smoke/test_demand_backtest_cli.py
make check
```

Exact-value tests cover calendar behavior, expanding/sliding windows, two
series, delayed arrivals, long horizons, future-value invariance, fixed cohort,
missing truth, zero-demand metrics, strict output rejection and artifact
roundtrips.

For coverage, use the filesystem scope `--cov=src/revenue_prediction` with
the same focused pytest command, then restrict the report to the relevant files
with `coverage report --include`. Module-named coverage discovery encountered a
NumPy import failure in the authoring environment; filesystem-scoped coverage
ran successfully without changing dependencies.

Additional exact algorithm/scaling/selection tests are in
`tests/unit/test_demand_time_series.py`; persisted comparison tests are in
`tests/smoke/test_demand_comparison_cli.py`.

Still pending: broader statistical variants and ML/deep adapters, revision
vintages, planned/external covariates, segment/cost gates, intervals,
reconciliation, model cards and governance.
See [leakage contracts](../data/leakage-prevention.md) and
[ADR 0006](../architecture/adr/0006-demand-origins-and-baselines.md).
