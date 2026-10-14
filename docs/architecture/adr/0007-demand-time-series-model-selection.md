# ADR 0007: Demand is time-series forecasting, not revenue regression

- Status: Accepted
- Date: 2026-10-13
- Scope: Demand model interface, statistical algorithms, metrics and selection

## Context

Demand forecasts future complete periods at explicit origins and horizons from
ordered related series. The inherited revenue accelerator estimates a current
month's scalar from partial-month snapshots. Its regression task, candidate
factory, R2/MAPE reporting and AutoML regression job are not the demand modeling
strategy. ADR 0003 remains valid only for that preserved revenue workflow.

## Decision

1. Treat demand as panel time-series forecasting. Keep frequency, seasonal
   period, origin, horizon, arrival latency and series identity explicit.
2. Add a `Forecaster` protocol with `fit`, `predict` and executable capability
   metadata. Fit independent univariate states per series at every origin.
   Select models through the demand registry, not the revenue regression
   factory. CLI discovery exposes the same registry.
3. Always compare naive and seasonal-naive anchors. Add moving average,
   seasonal moving average, drift, simple exponential smoothing (SES) and
   additive damped Holt. These are original deterministic NumPy implementations,
   not a borrowed donor model or a fitted tabular regressor.
4. Predeclare SES/Holt smoothing coefficients and moving-average windows.
   Fit levels/trends on training history only. This slice does not optimize
   likelihood, auto-select an ETS error/trend/seasonal configuration or tune
   parameters using validation/holdout results.
5. Enforce history eligibility: contiguous available calendar history for
   statistical/window models, enough observations for their configured windows,
   and actual lead distance from the last observation to each forecast.
   Delayed final observations do not magically become one-step leads.
6. Clip negative point projections to zero, reflecting the current
   nonnegative-unit target contract. Evaluate these clipped predictions.
   Returns/net-demand targets require a different explicit contract.
7. Use **MASE by default for model selection**, with RMSSE and WAPE as
   configurable alternatives. Scales use only origin-available seasonal
   differences within each series/window. Never scale by validation/test
   actuals, pool unrelated series, or compress arrival gaps into row shifts.
8. Retain horizon-specific WAPE, MAE, RMSE, sMAPE and bias; add MASE, RMSSE,
   bias percentage and tracking signal. R2 and MAPE are not demand-selection
   objectives. Undefined scales/relative metrics have explicit statuses.
9. Select using rolling validation only, with a fixed cohort and identical
   parameters across validation/holdout. Enforce nonoverlapping target periods
   and validation actuals available by the first holdout origin.
10. Candidates must strictly outperform seasonal naive on the configured
    validation metric and meet optional relative-improvement, absolute-bias and
    horizon-degradation gates. Ties retain seasonal naive; other ties are
    deterministic by model ID. If no model passes, fail explicitly.
11. Freeze algorithm/parameters before evaluating final holdout. Holdout
    reforecasting may refit that fixed algorithm on newly arrived observations
    at later origins. Holdout errors never trigger reselection.
12. Output local unapproved predictions/decisions, not a production champion.
    The point contract retains its fields/version and adds registry-backed
    model IDs and `statistical` kind; existing baseline outputs remain valid.

## Consequences

The demand workflow does not inherit revenue's regression-first algorithm
choice or threshold assumptions. A lag-based ML forecaster can be considered
later, but its task remains forecasting: origin-safe lag construction, temporal
training/validation, multi-step strategy and horizon evaluation are mandatory.
Using a regression estimator internally would not make it the revenue task.

There is no universal winning algorithm or universal accuracy threshold.
Seasonality/trend, intermittency, length, structural changes, covariate
availability, horizon and cost determine candidate eligibility and gates.

ETS/Holt-Winters, ARIMA/SARIMA/SARIMAX, intermittent-demand algorithms,
probabilistic output, learned global models, LSTM and foundation models remain
future adapters. LSTM is not the default. Registry metadata truthfully marks
current models as local, univariate, point-only and without covariates or
reconciliation; no live Azure execution is claimed.

Azure ML AutoML demand must use the forecasting task, not the inherited
regression job. No cloud task is implemented or submitted in this slice.

## Evidence and references

Tests verify exact recurrence/seasonal values, lead distances, calendar scales,
undefined metrics, eligibility, persisted configs and selection invariance
under holdout mutations. See the [catalog](../../modeling/model-catalog.md),
[time-series metrics](../../modeling/time-series-metrics.md) and
[comparison protocol](../../modeling/backtesting.md).

General mathematical reference: [Forecasting: Principles and Practice](https://otexts.com/fpp3/).
No source code or prose from that reference is copied.
