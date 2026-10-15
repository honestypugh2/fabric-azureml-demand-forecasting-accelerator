# ADR 0006: Availability-aware demand origins and offline baselines

- Status: Accepted
- Date: 2026-10-13
- Scope: Monthly/weekly historical-actuals point forecasts and rolling evaluation

## Context

Demand actuals carry explicit arrival timestamps.
Revenue snapshot features and month-based regression splitting cannot simply
be relabeled as future-demand inference. Demand baselines must establish a
leakage-safe benchmark before learned models, ensembles or cloud submission.

## Decision

1. Keep the implementation additive under the inherited namespace; leave
   revenue training, API/UI and defaults unchanged.
2. Use `DemandForecastRequest` for date-only origins, `MS`/`W-MON` frequency,
   positive horizon and optional explicit series keys. An aligned origin
   forecasts that complete period; a mid-period origin forecasts the next
   complete period. Partial-period nowcasting is not implemented.
3. Treat dates as timezone-naive UTC-convention calendar dates. At origin `t`,
   only actuals with `actual_available_at <= t` and `period_end < t` are usable.
   No row count, period-start comparison or future actual value substitutes
   for this availability test.
4. A sliding window is a number of calendar periods before the first forecast,
   not the most recent N arrived observations. Missing arrivals remain gaps.
   Expanding windows use all origin-available history.
5. Build calendar lag 1, a complete-window rolling mean, most recent available
   value and seasonal references. Include source dates/arrival timestamps.
   Exclude realized prices/promotions and raw target columns from features.
   No preprocessing is fitted; there is no learned imputation or future-label
   feedback.
6. Naive repeats the most recent available value. Seasonal naive repeats the
   previous calendar cycle, frozen at origin, including for horizons longer
   than a season. Defaults are 12 monthly or 52 weekly periods, configurable.
   A 52-week season is not an ISO-year calendar alignment rule.
7. Missing seasonal sources fail explicitly. No automatic substitution,
   forward fill, cross-series pooling or silent model fallback is permitted.
   A caller may explicitly request only naive.
8. Scope may be supplied independently of actuals. Otherwise inference
   discovers only series with origin-available history. Backtests fix this
   cohort at the first origin so later launches cannot silently change the
   comparison population. New-product forecasts require a separate design.
9. Version point outputs as `demand-point-v1`, with `forecast_kind=baseline`
   and `approval_status=unapproved`. Validate keys, complete horizons, dates,
   finite nonnegative predictions and source/training provenance.
   These are local outputs, not governed official forecasts.
10. Rolling backtests forecast before joining truth. Join by exact period and
    series keys; require every requested target to have arrived by the explicit
    `evaluation_as_of`. Never drop unmatched targets or randomly split rows.
11. Report portfolio and per-horizon MAE, RMSE, mean signed error, sMAPE, WAPE
    and relative bias using existing metric helpers. When actual totals are
    below the helper's `1e-9` threshold, report undefined relative metrics with
    an explicit status, not zero-valued success.
12. Provide a synthetic-only CLI demonstration, serialized configs and
    Parquet output. No extra dependencies, cloud execution or production
    approval is introduced.

## Consequences and limitations

Origin filtering prevents future actuals from becoming inputs, but the schema
still represents a single actual vintage. A revised old value with a falsely
backdated availability timestamp cannot be detected. Revision history,
external-signal vintages and planned future covariates need separate contracts.

Input datasets must satisfy the original strict calendar/hierarchy contract;
arrival-induced gaps in the filtered history are supported. Missing raw
periods, changing hierarchy versions, stock-out censoring, launch/retirement
policies and new-product inference remain separate work.

Overlapping forecast horizons produce repeated evaluations of a target at
different origins. Metrics weight forecast rows, not unique actual periods.
Results do not select/approve a champion or calculate ensemble weights. A final
holdout, segment/business-cost gates, scaled metrics, uncertainty and
reconciliation require separate evaluation and governance contracts.

Forecast provenance is validated metadata, not an independent proof of model
lineage or a digital signature. Registry versions, run identifiers and approval
audits belong in the governed production contract.

## Evidence

Tests assert exact monthly/weekly multi-horizon predictions, availability at
the origin boundary, delayed arrival, calendar-window gaps, per-series
isolation, no future-value sensitivity, fixed backtest cohort, full target
coverage, output rejection and persisted CLI reproducibility.

See [backtesting guide](../../modeling/backtesting.md) and
[leakage prevention](../../data/leakage-prevention.md).
