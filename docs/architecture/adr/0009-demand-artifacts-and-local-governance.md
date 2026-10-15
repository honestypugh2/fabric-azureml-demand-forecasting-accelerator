# ADR 0009: Safe demand snapshots and separate local model approval

- Status: Accepted for offline synthetic execution
- Date: 2026-10-14
- Extends: [ADR 0007](0007-demand-time-series-model-selection.md) and
  [ADR 0008](0008-demand-adapters-and-tuning.md)

## Context

The forecasting registry now supports temporal selection/tuning, but persisted
predictions/configurations alone are not a fitted model artifact. Demand needs
reproducible snapshots, model cards, training/evaluation lineage and deliberate
approval controls without treating inherited revenue bundles as demand models
or turning model selection into approval.

## Decision

Use strict, versioned data-only JSON snapshots. Preserve origin-available
observations, parameters and optimized statistical coefficients/initialization.
Restore original recurrences deterministically, ETS with known initialization/
fixed coefficients, and ARIMA/SARIMA through fixed-parameter filtering. No
pickle/executable deserialization, optimization on load or arbitrary model
imports. Exact demand-source/runtime compatibility is required.

Package only after recomputing the declared comparison and optional tuning
evidence. Bind source/training/configuration/prediction/search/code hashes and
runtime versions into the content-addressed artifact. Preserve validation-only
selection, fixed cohort, later-origin training availability and maximum
evaluated horizon. A new origin requires a new artifact/review.

Generate machine-readable and Markdown model cards with scope, hyperparameters,
training/evaluation evidence, baseline comparisons, lineage and limitations.
Cards are immutable evidence, never the mutable approval authority.

Implement local SQLite policy/artifact/audit persistence with transactional
expected-revision writes and verified hash chains plus a stored audit head.
Require explicit local roles, independent reviewer, reasons, limitations
acknowledgement and predeclared holdout metric/bias/noninferiority gates.
Registration is draft, submission is in-review, and only independent review
may approve or reject. Approved models can be revoked; rejected/revoked versions
are terminal. Never reselect a replacement on holdout to satisfy approval.

Approved-model inference captures the artifact/approval revision and exact
forecast digest. The unchanged point-output contract remains unapproved:
model approval and official forecast approval are separate decisions.

## Trust boundaries and alternatives

JSON is more verbose than pickle, but no executable object loader is needed.
Replaying deterministic recurrences/filtering requires saved history and exact
libraries; it avoids reoptimization and preserves delayed calendar leads.
Broad version portability and automatic schema upgrades are deliberately
excluded until migration is separately tested.

Local actor IDs are asserted, not authenticated. Hashes detect inconsistent
records, not an owner rewriting all records and hashes. The registry is not
production RBAC, a signed artifact repository or immutable ledger. Production
Azure ML/Entra/Fabric integrations require authenticated authorization,
protected policy/storage, independent audit anchoring, retention and separate
forecast/publication approval. No cloud resource or new dependency is required.

Holdout can block promotion, not alter the selected model. Thresholds are
explicit local policy, not universal business acceptance. Inference status is
an audit snapshot; already exported files cannot be recalled by revocation.

## Validation

Require all thirteen models' monthly/weekly state roundtrips, optimizer-free
statistical restoration, fresh-process/core-only loading, exact provenance/
window checks, evidence integrity, schema/runtime compatibility, independent
review, undefined/failed gates, stale/racing updates, rollback, audit/tail
consistency and end-to-end local lifecycle CLI tests.

See [runnable lifecycle and limitations](../../governance/demand-model-lifecycle.md).
