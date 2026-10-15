# Out-of-sample weighted time-series ensembles

`weighted_ensemble` blends registered **time-series** forecasts, not revenue
regression estimators. It is opt-in; the default seven-model comparison is
unchanged. There are no new dependencies or copied third-party algorithms.

## Weight learning and honest evaluation

Predeclare `ensemble_members` as at least two unique IDs also present in
`validation.models`. Each component uses identical parameters, complete
series/horizon grids and origin-available expanding/sliding histories. Optional
ETS/ARIMA/SARIMA members still require the `forecasting` extra.

For component j, let e_j be its portfolio, combined-horizon validation **MASE**
(or the experiment's explicit RMSSE/WAPE). For positive errors:

```text
q_j = min(e) / e_j
w_j = q_j / sum(q)
forecast = sum(w_j * component_forecast_j)
```

This is normalized inverse error, evaluated in a stable ratio form. Exact-zero
members share all weight equally and other members receive zero. Undefined,
negative or nonfinite errors fail; there is no default/equal-weight fallback
for undefined scores. Weights are finite, nonnegative, normalized, deterministic,
and shared across series and horizons. Extreme floating-point ratios can
underflow to zero; no component is removed.

Component validation forecasts are out of sample relative to fitting, but
validation **learns weights and selects the winner**. Do not present that blend
score as unbiased final accuracy. Weights freeze before the separate holdout:
its targets close after validation, and validation truth must be available by
the first holdout origin. Holdout reports the blend, every member, selected
algorithm, naive and seasonal naive without weight changes or reselection.
Component histories may refit at successive holdout origins using newly arrived
labels; model identities, parameters and weights remain frozen.

The blend must compete under the existing strict seasonal-baseline improvement,
bias and horizon gates. It is not automatically selected or promoted. MASE,
RMSSE, WAPE, MAE/RMSE, bias/tracking and per-series/horizon diagnostics retain
their existing definitions; this is not a regression R2/MAPE workflow.

Every declared component must fit and forecast, even a zero-weight component.
Missing optional dependencies, insufficient/gapped history, missing truth,
incomplete grids and nonfinite output fail the whole run. There is no silent
member removal or inference renormalization.

## Run point comparison or composed hierarchy experiment

```bash
uv run revenue-prediction compare-demand \
  --models naive,seasonal_naive,drift \
  --ensemble-members naive,seasonal_naive,drift \
  --skus 3 --channels 1 --geographies 1 \
  --out outputs/demand-ensemble-comparison

uv run revenue-prediction reconcile-demand \
  --config configs/examples/demand-ensemble.json \
  --out outputs/demand-ensemble-monthly

uv run revenue-prediction reconcile-demand \
  --config configs/examples/demand-ensemble-weekly.json \
  --out outputs/demand-ensemble-weekly
```

Use fresh directories. The hierarchy command never overwrites; the inherited
comparison command overwrites its named directory. The monthly/weekly configs
predeclare identical components; change frequency, temporal blocks and season
length deliberately rather than applying a monthly calendar to weekly data.

The comparison persists experiment/decision JSON (including errors and weights),
validation ranking and row/portfolio/series/segment evidence. The hierarchy
workflow writes its existing 16 files, now also binding ensemble decisions,
all components on holdout, and per-level/node metrics. Replay with the same
config, actuals and hierarchy to reproduce all semantic records.

## Safe artifacts and local review

```bash
uv run revenue-prediction demand-lifecycle package \
  --comparison-dir outputs/demand-ensemble-comparison \
  --origin 2025-01-01 --actor ACTOR-001 \
  --out outputs/demand-ensemble-artifact

uv run revenue-prediction demand-lifecycle package \
  --comparison-dir outputs/demand-ensemble-monthly --include-hierarchy \
  --origin 2025-01-01 --actor ACTOR-001 \
  --out outputs/demand-hierarchy-ensemble-artifact
```

Packaging recomputes learning/selection/holdout from actuals and config, not
editable scores or weights in `selection.json`. `demand-bundle-v3` and
`demand-model-card-v3` include method, scope, errors, weights, all component
capabilities, failure rules and holdout evidence. When selected, the ensemble
stores each member's numeric state in `demand-state-v2`, matching full shared
training observations and parameters. Restore uses saved coefficients/recurrences,
not optimization. Exact selected provider/runtime and implementation hashes
must match. Old artifacts need their matching environment or new review.

Ensemble point rows use `demand-point-v2`, retaining the same 17 columns and
`forecast_kind=ensemble`; other model rows retain `demand-point-v1`. Both are
strictly checked against the model ID. Bottom-up outputs retain
`demand-hierarchy-point-v1` and the original ensemble model ID.

Use [ensemble-only policy](../../configs/examples/demand-ensemble-approval.json)
or [composed policy](../../configs/examples/demand-hierarchy-ensemble-approval.json)
with the existing [local review workflow](../governance/demand-model-lifecycle.md).
V3 requires explicit maximum component weight and minimum validation origins.
Composed models also require every hierarchy node/level/horizon gate. These
are illustrative policies, **not universal business thresholds**; failure must
not trigger loosening after seeing holdout. Old v1/v2 policies cannot approve
ensemble evidence. The same separation of duties, revision checks, rejection
and revocation apply.

Approved inference serves only the reviewed winner (which can be a baseline),
with optional bound hierarchy and a v3 sidecar hashing the weight evidence and
approval revision. Forecast rows are still **unapproved**, not official outputs.
Uncertainty and tuning-grid combination are deliberately unsupported for this
bounded ensemble increment. No interval averaging or coverage claim is made.

See [ADR 0014](../architecture/adr/0014-demand-out-of-sample-ensembles.md).
