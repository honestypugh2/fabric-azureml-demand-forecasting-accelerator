# ADR 0013: Hierarchy-bound local model review and snapshot inference

- Status: Accepted

## Decision

Extend the existing artifact/card/registry rather than create a second approval
store. `demand-bundle-v3` binds a complete origin-available static hierarchy,
bottom-up method, recomputed validation/holdout level and node metrics, complete
coherence counts/residuals and semantic output hashes. Packaging reruns the
experiment from synthetic actuals/configuration, never trusts editable reports,
and requires the hierarchy known by the first validation origin.

An explicit `demand-local-policy-v3` covers all four levels, every node/scope
and every horizon, including the combined-horizon view. Undefined selected
metric/bias blocks approval; seasonal noninferiority and absolute limits are
declared before review. Bound coherence must satisfy the explicit residual
limit as well as the reconciliation contract's numerical tolerances.

Approved-model hierarchy inference restores the frozen selected states, then
reuses the exact bottom-up helper and bound membership. Shorter evaluated
horizons are allowed, but not new origins, data, parents or cohorts. A v3
sidecar binds the hierarchy/evidence, output hash and approval audit revision.
Rejection, stale revisions, independent review and revocation use the same
transactional state machine. All forecast rows remain unapproved.

Move immutable JSON/hash primitives to a shared demand data module to avoid
a hierarchy/artifact import cycle; preserve existing artifact helper imports
as explicit reexports. They remain covered by the strict implementation hash.

## Boundaries

Existing point v1 and interval v2 serialization/policies remain unchanged.
Their policies cannot approve hierarchy capability. This increment is point
only: it rejects combined hierarchy/interval requests. It adds no aggregate
coverage claim, effective-dated hierarchy, MinT, cloud deployment, dependency,
namespace migration or authenticated authorization. Existing source-pinned
artifacts require matching source or new packaging/review, not transferred
approval. Hashes, local actor IDs and hierarchy availability remain local
assertions, not signatures or a protected ledger.

See [model lifecycle](../../governance/demand-model-lifecycle.md).
