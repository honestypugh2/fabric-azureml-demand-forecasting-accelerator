# ADR 0012: Frozen-cohort bottom-up demand point reconciliation

- Status: Accepted
- Date: 2026-10-15

## Context

Demand is a multi-series time-series task. Segment error summaries do not
produce coherent parent forecasts or evaluate aggregated errors. The original
demand data validates static hierarchy labels, but has no versioned hierarchy,
reconciliation output contract or aggregate seasonal scales.

## Decision

Add a strict content-addressed hierarchy snapshot, known by the first rolling
validation origin, with a complete explicit SKU/channel/geography cohort.
Discover labels only from arrived training history, or require a supplied
snapshot matching that origin-available scope. Freeze the version throughout
validation and holdout; do not silently add future launches or shrink totals.
The version is an as-of static classification, not an effective-dated hierarchy.

Reconcile point forecasts bottom-up at SKU, product, family and brand levels,
keeping channel/geography separate. Preserve original bottom point predictions,
model identity and source provenance. Validate exact node/leaf/count metadata
against source points and test immediate parent-child coherence with
`rtol=1e-12`, `atol=1e-9` for finite floating-point sums.

Reuse existing validation-only point-model selection and mandatory holdout
anchors. Report MAE/WAPE/bias and existing point diagnostics at each level and
node. For MASE/RMSSE, aggregate only complete arrived historical node periods
inside each origin's training window, then compute exact seasonal calendar
pairs. Never sum leaf scales or treat missing descendants as zero. Retain
undefined scales/counts; missing holdout truth fails rather than dropping rows.

Provide a dedicated offline command with frozen hierarchy/config/decision,
bottom/node evidence, metrics, coherence and content/runtime/source lineage.
Use only the existing synthetic generator and dependencies.

## Consequences and boundaries

Totals cover the explicit versioned cohort and compatible additive units,
not an implicitly growing enterprise population. Upper-level accuracy may
improve or deteriorate through aggregation; it never reselects the winner on
holdout. Different levels are not pooled into one metric.

Summing marginal interval bounds does not preserve coverage. The bottom-up
contract rejects interval inputs and makes no aggregate uncertainty claim.
MinT, grouped/temporal hierarchies, planned reparenting, adaptive uncertainty
and cloud/application publication are outside this contract. Ensemble weights
are governed separately in [ADR 0014](0014-demand-out-of-sample-ensembles.md).

Hierarchy outputs remain unapproved and do not inherit local model approval.
Hashes and declared hierarchy availability are not authenticated metadata,
signatures or adversarial tamper protection. Existing source fingerprints stay
strict: older model bundles require matching source or new packaging/review,
not an automatic migration or transferred approval.
