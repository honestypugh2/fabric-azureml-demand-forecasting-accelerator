# Demand model artifacts, lineage and local approval

Slice 5 implements an **offline synthetic model lifecycle**. It is separate
from the inherited revenue MLflow/joblib path and from production Azure ML or
Fabric authorization. Nothing in this workflow deploys a model, writes an
official forecast or authenticates a person.

## Safe, fitted model serialization

All thirteen registry models export `demand-state-v1` numeric JSON state.
`demand-bundle-v1` packages the selected model per series, its exact parameters,
closed/arrived training observations, frozen forecast origin, evaluated maximum
horizon and runtime/source compatibility metadata.

- Original NumPy models restore deterministic state by replaying their fixed
  recurrences over the saved training values; there is no coefficient search.
- ETS/Holt-Winters saves optimized smoothing/damping coefficients and initial
  level/trend/seasonal states. Restoration uses known initialization and
  `optimized=False`, not another fit optimization.
- ARIMA/SARIMA saves the estimated coefficient vector and applies the same
  estimator's filter to saved observations; it never runs the optimizer.
- No pickle, joblib, arbitrary class/module import, `eval`, or executable
  deserialization is used. The artifact accepts only registered model IDs,
  versioned schemas, finite numeric state and validated scope/provenance.
- Statistical restores require the optional `forecasting` extra. Loading
  original models does not import statsmodels.
- Exact Python/library versions and demand-source hashes must match. Changed
  code/environments fail closed; restore the recorded environment or rebuild
  and review a new artifact. A wheel built from the same sources is compatible.

The artifact stores synthetic training values; treat it as data, not merely
code. `data_classification=synthetic` is the caller's declared contract, not a
detector that can prove arbitrary uploaded data is synthetic. Default CLI inputs
come from the existing synthetic comparison path. No real data should be used.

Inference is a **frozen-origin snapshot**, not automatic reforecasting. It can
request fewer horizons, never more than the evaluated maximum, and cannot
change origin, series, parameters or training data. A new origin/window/data
revision requires a new artifact/version and independent review. Baselines
retain their individual seasonal/last-value source provenance.

## Recomputed evidence and lineage

Packaging reads synthetic actuals and the frozen experiment configuration,
then recomputes selection and holdout evidence. It does not trust editable CSV
scores or a standalone `selection.json`. When a comparison includes a tuning
configuration, packaging replays the predeclared grid and requires that its
winner/configuration reproduce the frozen experiment.

Fit the deployment snapshot only at an origin on/after the complete holdout
evaluation cutoff. Its cohort is fixed by validation; later launches are not
silently added. Only actuals closed and available at that origin, within any
calendar training window, enter the fitted state.

The content-addressed artifact ID is SHA-256 over canonical validated payload
JSON, including creation metadata. Identical training/evidence with a different
creation time therefore produces a different artifact version. Lineage binds:

- Source actuals, exact training/state records and frozen experiment.
- Validation and holdout predictions separately.
- Original tuning design/trial ranking when applicable.
- Executable demand sources (including uncommitted changes) and runtime versions.

Future source actuals can change the source-version hash but cannot change
earlier fitted state, decision or evaluation where they were unavailable.
Hashes are reproducibility/integrity identifiers, **not signatures or proof of
data authenticity**.

## Model cards

Packaging writes immutable `model_card.json` and human-readable `model_card.md`.
Cards include algorithm/provider/license, target/grain scope, hyperparameters,
training boundaries/counts, frequency/origin/horizon, validation/holdout
portfolio and per-horizon MASE/RMSSE/WAPE/bias, evaluation/gate configuration,
lineage, versions and limitations. Undefined metrics use JSON `null`, never
nonstandard NaN/Infinity or a misleading zero.

The full series/segment diagnostics remain in the linked comparison artifacts;
the card does not invent segment acceptance. Selection remains validation-only.
Holdout may prevent approval but never causes replacement/reselection.

Cards do **not** own mutable approval status. The registry's current verified
audit chain is the source of truth. Updating a Markdown/JSON card or setting
a field in a forecast file cannot approve a model.

## Local registry and state machine

The SQLite registry owns a fixed policy, immutable-by-API artifact records,
append-only-by-API events and an independently stored audit head:

```text
registration -> draft -> in_review -> approved -> revoked
                              |
                              +----> rejected
```

Rejected/revoked versions are terminal. Rework requires a newly packaged
artifact, not an overwritten version or reset approval. Registration never
auto-promotes or replaces an existing artifact.

Policies explicitly declare neutral actor IDs, local roles and thresholds:

- Builder registers only the artifact naming that creator.
- Submitter requests independent review.
- Reviewer approves/rejects/revokes, but cannot be the creator or submitter.
  Holding multiple roles cannot bypass separation of duties.
- Approval requires an audit reason and explicit acknowledgement of card
  limitations. Reasons must also stay synthetic/customer-neutral.
- At portfolio **and every evaluated horizon**, the selected model must have
  defined primary metric/bias, satisfy the explicit absolute metric/bias
  thresholds, and (by default) not degrade seasonal naive.
- Limits apply to the experiment's selected metric (`mase`, `rmsse` or `wape`).
  There is no universal business threshold or automatic fallback.

The [example policy](../../configs/examples/demand-local-approval.json) uses
illustrative MASE <= 1.0 and absolute bias fraction <= 0.2. These are **not**
recommended production thresholds. Its noninferiority check additionally
requires performance no worse than seasonal naive at every horizon. Declare
the policy before evaluation/review for an honest acceptance exercise; do not
loosen it after seeing holdout just to obtain approval.

Writers use SQLite transactions and an exact expected revision. Stale/racing
requests fail without appending an event. Every read verifies artifact/card/
policy digests, revision ordering, UTC timestamp shape, roles, transitions, hash links,
the audit head and all approval conditions. Changed/removed events (including
the last event) fail closed.
Revisions and hash links order events; UTC wall-clock timestamps are recorded
as observed and are not required to increase when a system clock is corrected.

**Trust boundary:** actor IDs are caller assertions, not authenticated Entra
identities. Anyone controlling this process/files can impersonate configured
actors or rewrite both the database and its hashes. This is not a multi-user
authorization service, immutable ledger or adversarial tamper-proof storage.
Restrict local file permissions and do not expose these commands as a service.
Production needs authenticated principals, independently protected policy/
artifact storage, signed or externally anchored audit evidence, retention and
deployment authorization.

## Runnable monthly demonstration

Choose fresh output/registry locations; packaging and inference never overwrite.
The original comparison command still overwrites its named experiment outputs.

```bash
uv run revenue-prediction compare-demand \
  --models naive,seasonal_naive --skus 1 --channels 1 --geographies 1 \
  --out outputs/demand-lifecycle-comparison

uv run revenue-prediction demand-lifecycle package \
  --comparison-dir outputs/demand-lifecycle-comparison \
  --origin 2025-01-01 --actor ACTOR-001 \
  --out outputs/demand-lifecycle-artifact

uv run revenue-prediction demand-lifecycle init-registry \
  --registry outputs/demand-lifecycle.sqlite \
  --policy configs/examples/demand-local-approval.json

uv run revenue-prediction demand-lifecycle register \
  --registry outputs/demand-lifecycle.sqlite \
  --bundle outputs/demand-lifecycle-artifact/model.json \
  --actor ACTOR-001 --reason "Synthetic model card ready"
```

Substitute the actual 64-character artifact ID printed by `package` for
`ARTIFACT_SHA256` below:

```bash
uv run revenue-prediction demand-lifecycle transition \
  --registry outputs/demand-lifecycle.sqlite --artifact-id ARTIFACT_SHA256 \
  --actor ACTOR-001 --action submit --expected-revision 1 \
  --reason "Request independent synthetic review"

uv run revenue-prediction demand-lifecycle transition \
  --registry outputs/demand-lifecycle.sqlite --artifact-id ARTIFACT_SHA256 \
  --actor ACTOR-002 --action approve --expected-revision 2 \
  --acknowledge-limitations --reason "Reviewed evidence and limitations"

uv run revenue-prediction demand-lifecycle forecast \
  --registry outputs/demand-lifecycle.sqlite --artifact-id ARTIFACT_SHA256 \
  --out outputs/demand-lifecycle-inference

uv run revenue-prediction demand-lifecycle transition \
  --registry outputs/demand-lifecycle.sqlite --artifact-id ARTIFACT_SHA256 \
  --actor ACTOR-002 --action revoke --expected-revision 3 \
  --reason "End local demonstration"

uv run revenue-prediction demand-lifecycle status \
  --registry outputs/demand-lifecycle.sqlite --artifact-id ARTIFACT_SHA256
```

Approval can fail if the declared gates are not met. That is an intended
outcome; do not select again or change thresholds using this holdout.
Forecasting before approval or after revocation must fail without new outputs.

For weekly inputs use `--frequency W-MON` in the comparison and an aligned
snapshot origin on/after its actual availability cutoff (the default generator
supports 2025-01-06). Tuned statistical comparisons follow the same commands
after `uv sync --extra api --extra forecasting`.

## Inference output and approval separation

The approved-model command writes `predictions.parquet` using the unchanged
`demand-point-v1` contract plus `lineage.json` binding the exact row digest,
artifact, approval-event hash and registry revision. Approval is a
transactionally consistent **snapshot**, not a promise that it remains current
after export; downstream consumers must recheck current status before later use.
Revocation blocks subsequent calls but cannot retract files already exported.

Point rows **always retain `approval_status=unapproved`**. Local approval of a
model is not approval of its forecasts, official planning consensus, cloud
deployment or publication. Standalone `predict_demand_bundle` is a research
API for unapproved outputs and does not grant governance permission.

## Verification and remaining work

Tests cover all thirteen models in monthly/weekly JSON roundtrips, optional ETS
components, optimizer-free restore, fresh-process/core-only reload, future-value
invariance, windows/provenance, card/evidence completeness, corrupt/schema/
runtime/source rejection, roles/self-review, acknowledgement, undefined/failed
gates, stale revisions, rollback, hash-chain/tail checks and persistent CLI
registration/approval/inference/revocation.

No cloud registration, authenticated identity, cryptographic signing, schema
migration, probabilistic output, automated schedules, retention or official
forecast approval is claimed. See [ADR 0009](../architecture/adr/0009-demand-artifacts-and-local-governance.md).
