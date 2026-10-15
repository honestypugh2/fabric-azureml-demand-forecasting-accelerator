# ADR 0014: Transparent out-of-sample inverse-error demand ensembles

- Status: Accepted

## Decision

Implement the requested transparent inverse-validation-error method, not
regression stacking. Predeclare at least two unique component algorithms in
the selection design. Use their identical complete rolling-origin validation
grids and the experiment's primary time-series metric (MASE by default,
RMSSE/WAPE alternatives) to compute one portfolio/all-horizon weight per
component. Positive errors normalize as `(min(error)/error)/sum(...)`, avoiding
overflow from reciprocals. Exact-zero-error members share all mass; no epsilon,
undefined-metric substitution or learned per-series/horizon weights.

Validation is out of sample for component fitting, but is used for both weight
learning and selection. Its ensemble score is therefore not unbiased final
performance. Freeze weights before the temporally separated final holdout.
Evaluate the frozen blend, every declared member, the selected algorithm and
both mandatory anchors on the complete fixed cohort. Later holdout origins
may refit component states using arrived history, never relearn weights or
reselect models.

Expose `weighted_ensemble` as an opt-in composition capability requiring learned
weights; preserve the existing default seven-model comparison. The ensemble
competes under the existing validation improvement/bias/horizon gates, so it
need not win. No holdout-based replacement is allowed. Every member must fit
and predict, including zero-weight members; any failure aborts the whole run.
There is no component removal, pooling, fallback or inference renormalization.

Version ensemble points as `demand-point-v2` with the same columns and
`forecast_kind=ensemble`; existing models retain point v1. Store complete safe
component states in `demand-state-v2` and learned errors/weights in the decision,
v3 bundle and card. Validate errors against recorded validation metrics, exact
inverse-error weights and shared component training/configuration. Restore
original recurrences or fixed statistical coefficients, never optimize on load.
Pin selected component provider runtimes.

Use the same v3 lifecycle as hierarchy artifacts. Require explicit ensemble
policy for minimum validation origins and maximum component weight, plus
existing selected-model holdout gates. A composed hierarchy/ensemble artifact
requires both policies. Cards and inference sidecars bind weight evidence
even when validation retains a nonensemble winner; inference serves only the
reviewed selected model.

## Boundaries

Only univariate nonnegative point ensembles are implemented. No averaging of
interval bounds, ensemble uncertainty, tuning-grid combination, adaptive
weights, multilevel weights, covariance optimization, new dependency or cloud
execution is claimed. More origins and an additional weight-fitting/selection
block would be needed for less optimistic validation comparison. Synthetic
results do not establish business acceptance. Local roles/hashes do not
constitute production identity or authorization.

See [ensemble guide](../../modeling/ensembles.md).
