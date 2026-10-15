# ADR 0011: Bind temporal calibration to independently reviewed demand artifacts

- Status: Accepted for the offline slice-6 lifecycle follow-up
- Date: 2026-10-15

## Context

The first slice-6 increment persisted residual calibration separately from
point-model artifacts. Pairing arbitrary calibration with model inference
could change model parameters, cohort or training policy without explicit
review. Point-only model approval must not silently authorize uncertainty.

## Decision

Add opt-in `demand-bundle-v2` and `demand-model-card-v2` while retaining point-only
v1 serialization/behavior. Reuse existing model fitting, state restoration,
canonical hashing and the transactional local registry, not a parallel registry.

Recompute the full predeclared validation/calibration/holdout experiment during
packaging. Bind experiment, calibration, complete scope/counts, per-series and
portfolio holdout interval metrics and evidence hashes to the fitted model
artifact. Selection stays validation-only; calibration and holdout cannot
reselect algorithms, change parameters or tune radii to pass review.
Calibration belongs to the fixed algorithm/parameters/training policy;
deployment state is a later closed/arrived snapshot, as in point reforecasting.

Require a separately versioned `demand-local-policy-v2` with explicit nominal
coverage, calibration count and empirical coverage/width/interval-score/pinball
thresholds. Apply all existing point gates, then interval gates for the selected
model at portfolio/every-series/every-horizon scope, including the combined
horizon view. Do not invent business thresholds or allow a point policy to
approve intervals. Registry policy is fixed; create a separate store/version.

Extend existing package/forecast commands with `--include-intervals`. Interval
inference restores only the artifact's selected model/calibration, supports
shorter evaluated horizons and writes a v2 sidecar binding calibration/design
hashes and exact exported rows to the approved audit revision. It never accepts
a new origin, new observations or an external calibration file.

## Consequences and trust limits

Point-only APIs and JSON remain available; model approval and forecast approval
remain separate. Interval rows are always unapproved, even after local model
approval. Rejection/revocation, reviewer separation, acknowledgement, stale
revision checks and audit-chain/head verification apply to the whole artifact.

No new dependencies, third-party source, cloud deployment, authenticated
identity, signed storage, learned quantiles or coverage guarantee are added.
Nominal interval tails are scored with pinball loss but are not independently
learned conditional quantiles. Dependent rolling errors require empirical
time-series validation; small synthetic examples do not establish acceptance.

The source fingerprint deliberately includes all demand modules and registry
code. Older artifacts require the matching source environment or repackaging
and independent review. No hash bypass, schema migration or approval transfer
is introduced. Local actor labels/hashes do not protect against an owner
rewriting both database records and hashes; production authorization remains
separate work.
