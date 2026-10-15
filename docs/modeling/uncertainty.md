# Demand prediction intervals and governed local artifacts

This is an **offline time-series** uncertainty workflow, not a regression
confidence interval or a production forecast approval. It wraps the frozen
validation winner and mandatory naive/seasonal-naive anchors without changing
their point predictions. No new dependencies or third-party source are used.

## Run and reproduce

```bash
uv run revenue-prediction demand-uncertainty \
  --config configs/examples/demand-uncertainty.json \
  --out outputs/demand-uncertainty

uv run revenue-prediction demand-uncertainty \
  --config outputs/demand-uncertainty/uncertainty_config.json \
  --actuals outputs/demand-uncertainty/actuals.parquet \
  --out outputs/demand-uncertainty-replay
```

Directories must be new; failures are explicit and never overwrite an existing
experiment. Default actuals come only from the synthetic generator. An explicit
`--actuals` input must satisfy the historical demand contract, including parsed
datetime columns; use the generated Parquet for a lossless replay.

The example predeclares 80% coverage, five calibration origins, two horizons,
and MASE selection over naive, seasonal naive, drift, SES and damped Holt.
Configuration supports `MS` and `W-MON`, expanding/sliding windows, registered
point algorithms and the existing validation selection gates. All three blocks
must share frequency, horizon, candidate models, parameters, window, season
length and optional explicit cohort. Optional statistical models still require
the `forecasting` extra. This increment does not perform parameter-grid tuning.

## Three chronological blocks

1. **Validation:** select model identity/parameters using the existing
   time-series metrics and gates. No calibration or holdout errors rank models.
2. **Calibration:** forecast only the frozen winner and mandatory anchors.
   For each model, series and horizon independently, collect absolute forecast
   errors from rolling origins. All targets must have closed and arrived by the
   declared calibration cutoff.
3. **Final holdout:** attach the frozen calibration radii and measure interval
   quality. Neither model identity/parameters, coverage nor radii are changed
   using holdout errors. Fixed algorithms can refit at later origins using
   newly arrived history, matching the existing reforecasting policy.

Validation targets must close and be evaluated before/by the first calibration
origin; calibration targets must close and be evaluated before/by the first
holdout origin. Target blocks cannot overlap. The cohort discovered at the first
validation origin remains fixed. Missing truth or sparse calibration fails the
whole experiment; no series/horizon pooling, row dropping or zero-width fallback
is used.

## Algorithm and output contract

For nominal coverage `c` and `n` residuals in a model/series/horizon cell:

```text
k = ceil((n + 1) * c)
radius = kth smallest absolute residual
lower = max(0, point_prediction - radius)
upper = point_prediction + radius
```

`k > n` fails rather than returning a misleading finite interval. Thus 80%
coverage needs at least four observations per cell; 90% needs at least nine.
The example's five observations are illustrative and coarse, not a
production sample-size recommendation. Nonnegative clipping preserves containment
of the point and cannot exclude a valid nonnegative target that was inside the
unclipped interval.

`demand-interval-v1` adds coverage, bounds, method, radius, cell sample count
and calibration cutoff to the unchanged `demand-point-v1` columns. Point
`schema_version` and provenance remain intact; `interval_schema_version`
identifies the additive interval contract. All outputs remain **unapproved**.
Nonfinite values, invalid provenance, crossed/incorrect bounds, unsupported
coverage counts and future calibration timestamps are rejected.

`demand-calibration-v1` is strict, data-only JSON with frequency, coverage,
cutoff, final calibration target date and cell radii/counts. It can reproduce
intervals on the persisted point forecasts without refitting or recalibration.
The application helper requires the same complete model/series/horizon scope
at each later origin and no overlap with calibration targets. Calibration JSON
alone is **not** a source-hashed, approved fitted-model artifact; use the
persisted experiment configuration and point predictions together. Do not
attach it to independently changed model parameters or training policies.

## Metrics: uncertainty differs from point model selection

Both portfolio and per-series reports include each horizon and `horizon=0`
for all horizons, with equal forecast-row weighting:

- **Empirical coverage:** fraction with `lower <= actual <= upper`, including
  endpoints.
- **Coverage gap:** empirical minus nominal coverage; negative is undercoverage.
- **Mean interval width:** sharpness in demand units, not meaningful alone.
- **Interval score:** for `alpha = 1 - c`,
  `width + 2/alpha * (max(lower-actual, 0) + max(actual-upper, 0))`.
  Lower is better; penalizes narrow intervals that miss actuals.
- **Pinball losses:** evaluated separately at nominal lower/upper levels
  `alpha/2` and `1-alpha/2`, using
  `max(tau*(actual-bound), (tau-1)*(actual-bound))`. Lower is better.

The endpoints are symmetric absolute-residual interval bounds evaluated at
these nominal tail levels, **not separately fitted conditional quantiles**.
The point forecast is not relabeled a median. Existing MASE/RMSSE/WAPE, bias
and tracking-signal reports remain separate from interval quality.

Rolling time-series residuals are dependent, may share target periods across
horizons, and may drift. This conformal-style finite-sample order statistic
does **not** establish exchangeability or guarantee time-series coverage.
Measure out-of-sample coverage and width by series/horizon; do not claim
calibration quality from nominal coverage or synthetic examples alone.

## Persisted evidence and boundaries

The command writes actuals; validation/calibration/final-holdout point rows;
final-holdout intervals; validation ranking; point and interval metrics;
per-series interval metrics; and frozen experiment, selection and calibration
JSON. Holdout truth is used only to score the already constructed intervals.

## Governed uncertainty lifecycle

The follow-up lifecycle packages a `demand-bundle-v2` containing the fitted
winner, complete uncertainty experiment, calibration, per-series/horizon
holdout interval metrics and content lineage. Packaging **recomputes** the
three blocks from actuals/configuration rather than trusting standalone
calibration JSON, predictions, CSV scores or a selection file.

See the [interval lifecycle demonstration](../governance/demand-model-lifecycle.md#interval-artifacts-and-local-approval)
for packaging, explicit interval acceptance policy, independent review,
frozen interval inference and revocation. The unregistered helper remains a
research API, not authorization. A point-only approval never enables intervals.

Still not included:

- Learned asymmetric quantiles, multiple coverage levels or adaptive calibration.
- Calibration tuning/search or holdout-based interval/model reselection.
- Joint multi-horizon/portfolio coverage or hierarchical reconciliation.
- Demand UI/API, cloud jobs, Fabric publication or official forecast approval.

Point-only v1 serialization and approval semantics remain available; v2 explicitly
adds interval capability and review. The existing
strict source fingerprint includes **all** demand core modules. Adding these
interval modules therefore intentionally invalidates bundles built from the
slice-5 source tree. Repackage and independently review a new artifact under
the current source version, or use the matching slice-5 checkpoint for an old
bundle. Do not edit saved hashes, bypass verification or reuse an old approval
for a newly packaged artifact.

See [ADR 0010](../architecture/adr/0010-demand-temporal-interval-calibration.md),
[ADR 0011](../architecture/adr/0011-demand-interval-artifacts-and-review.md)
and the [remaining gaps](../assessment/gap-analysis.md).
