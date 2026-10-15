# ADR 0010: Separate temporal residual calibration for demand intervals

- Status: Accepted for the bounded offline slice-6 increment
- Date: 2026-10-14

## Context

Demand is time-series forecasting. Point accuracy and validation-only selection
do not measure uncertainty. The existing point schema, serialization, model
cards and local approvals must retain their meaning while uncertainty starts
as a separately evaluated capability.

## Decision

Use a predeclared validation -> calibration -> final-holdout design, with
nonoverlapping closed target blocks and origin-available evaluation cutoffs.
Reuse the existing selection and rolling-backtest implementations. Select
only on validation; calibrate only the winner and mandatory anchors; evaluate
fixed radii on final holdout without reselection.

Estimate each model/series/horizon's symmetric radius with the absolute-residual
order statistic at `ceil((n+1)*coverage)`. Fail for insufficient calibration,
missing actuals or changed cohorts rather than pooling cells, shrinking the
evaluation set, interpolating a lower rank or fabricating finite output.
Clip lower bounds at zero because demand is nonnegative. Retain original
point predictions and provenance unchanged.

Add explicit `demand-interval-v1` and numeric `demand-calibration-v1` contracts.
Report coverage, width, interval score and nominal-tail pinball losses by
horizon and series. Persist experiment, selection, calibration and scored
predictions for reproduction through a dedicated offline command.

## Consequences

No new statistical dependency, optimizer, regression assumption or stochastic
sampling is introduced. The wrapper works with existing monthly/weekly point
algorithms, but does not turn them into learned quantile models. Pinball losses
score the bounds at nominal tail levels; the point is not declared a median.

Time dependence and changing forecast fits invalidate an unconditional claim
of exchangeable conformal coverage. Coverage must be assessed empirically, not
guaranteed. Small synthetic calibration windows are illustrative only.

Calibration records are experiment evidence, not approved model bundles.
Interval-aware artifact hashing, model cards, production registration and
approval gates require a future explicit schema/lifecycle change. Slice-5
point artifacts and approvals are unchanged and do not authorize intervals.
The existing fingerprint scans all demand core modules: this additive source
change intentionally requires repackaging and independent review of previously
saved point bundles. No hash bypass or automatic migration is introduced.
Multiple coverage levels, adaptive/asymmetric calibration, reconciliation,
ensembles and cloud integration remain out of scope.
