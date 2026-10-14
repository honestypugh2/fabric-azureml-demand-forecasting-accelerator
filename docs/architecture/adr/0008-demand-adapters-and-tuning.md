# ADR 0008: Optional seasonal adapters and temporal parameter selection

- Status: Accepted for local synthetic execution
- Date: 2026-10-13
- Extends: [ADR 0007](0007-demand-time-series-model-selection.md)

## Context

Demand forecasts future ordered periods, rather than a revenue snapshot scalar.
Fixed smoothing/baselines do not cover fitted seasonality/autocorrelation or
intermittent arrival dynamics. More algorithms need explicit licensing,
eligibility and temporal tuning, not inherited regression/random-row AutoML.
The prior weekly validation winner underperformed seasonal naive on holdout;
this is evidence to preserve holdout separation, not to choose retrospectively.

## Decision

Add original Croston/SBA/TSB recurrences to the core and ETS/Holt-Winters,
ARIMA/SARIMA through guarded statsmodels APIs. The optional `forecasting` extra
pins the supported statsmodels range; the lockfile records the resolved version.
No third-party source is copied. Keep the original seven-model CLI default.
Explicit model IDs are required for the new adapters.

Use local univariate point forecasts with the existing MS/W-MON contract,
origin-arrival filtering, calendar lead, complete cohort and nonnegative
clipping policy. Require contiguous arrived history and documented minimums.
Reject nonconvergence/numeric fit failures rather than substituting forecasts.
Statsmodels estimates coefficients on each training window only; fixed
original smoothing/intermittent coefficients remain predeclared.

A frozen grid configuration reuses rolling validation and existing business
gates. Model/parameter trials must cover every series/origin/horizon; failed
trials remain explicit in persisted evidence. Missing dependencies abort with
setup instructions. Choose a global algorithm/parameter configuration from
validation only, freeze it, then evaluate the separate final holdout without
replacement or reselection. Persist both the original search and the frozen
reproducible experiment. No automatic metric fallback or production approval.

Add original sparse/long-zero/obsolescence synthetic scenarios and training-only
pattern/structural diagnostics. Persist per-series and segment/horizon metrics
using the same scaled-error formulas. These are neither reconciliation nor
segment/service-cost acceptance gates.

## Alternatives and consequences

Statsmodels supplies mature optimized local estimators without reimplementing
ARIMA optimization. Guarded imports keep core execution independent of it.
Full automatic ETS/ARIMA search, multiplicative components, external predictors,
per-series tuning and probabilistic outputs are deliberately excluded.

Grid selection shares validation for model and parameter search; validation
scores are optimistic search evidence, while the untouched holdout is the final
assessment. Statistical fits can be expensive and can fail. Record that
honestly; never reduce the comparison cohort or conceal a weaker holdout.

Repeated-origin refitting remains allowed after selection; only algorithm and
parameters are frozen. Canonical prediction/metric ordering permits direct
reproduction. Model serialization, production approvals and cloud execution
remain future work.

## License and validation

Statsmodels 0.14.6 installed license metadata was inspected: BSD-3-Clause.
Its Patsy dependency identifies BSD-2-Clause. Package license texts remain in
their distributions; see [third-party notices](../../../THIRD_PARTY_NOTICES.md).
Wrappers and synthetic/intermittent implementations are original MIT work.

Acceptance covers exact intermittent formulas, provider-equivalent statistical
forecasts and delayed calendar leads, finite-output/convergence enforcement,
optional-import guards, sparse-data determinism, holdout-mutation invariance,
failed-cohort trials, grid/gate validation and persisted roundtrips.
See [runnable workflow](../../modeling/tuning.md).
