# ADR 0005: Incremental demand accelerator through repository inheritance

- Status: Accepted
- Date: 2026-10-13
- Scope: Repository strategy and first implementation slice only

## Context

This demand repository already inherits the separately reusable revenue
accelerator. It contains working offline workflows, tests, an educational app,
SDK v2 integration patterns and cloud assets. Revenue snapshot regression is
not equivalent to future-period demand forecasting.

The [reuse assessment](../../assessment/reuse-decision.md) compares inheritance,
directly changing the revenue project, starting from the LSTM donor, and
greenfield development. Inheritance has the strongest fit without disrupting
the original project's purpose.

## Decision

1. Work only in this repository; preserve upstream and reference repositories.
2. Use inherit -> generalize -> extend -> validate -> document.
3. Begin with assessments, migration maps and an additive synthetic historical
   demand contract. Do not rename/remove the revenue package in the first slice.
4. Keep generation in `revenue_prediction.core.data.synthetic` during this
   slice. Place demand configuration/schema separately from the revenue schema.
5. Propose `demand_forecasting`, `demand-forecasting` and a neutral application
   identity for a later coordinated migration.
6. Keep existing regression behavior and ADR 0003; an LSTM is not the default.
7. Treat the MIT LSTM repository as a selective donor, not a merge source.
   Its model/preprocessing/dependency files were inspected; no new code is
   accepted from it now. Future adaptation needs per-component review,
   compatibility/leakage tests, optional dependencies and full attribution.
8. All defaults remain original synthetic, neutral data. No real customer
   names/data, personal information or cloud identifiers enter reusable assets.
9. Cloud integrations/deployments remain opt-in. Record local evidence separately
   from live validation.

## Consequences

The demand foundation temporarily lives under the legacy import namespace.
This is deliberate, reversible and requires no compatibility aliases. Existing
CLI/UI/training continue to consume the unchanged revenue schema.

The first demand dataset contains closed-period actuals, not future inference
features or a working forecasting pipeline. Actual availability metadata does
not replace origin-filtered feature joins.

Hybrid architecture, OneLake access, hosting, model interfaces, hierarchy,
ensembles, scenario/consensus persistence, identity, approval and IaC selection
need separate decisions before implementation. This ADR does not approve a
cloud deployment or production resource rename.

## Validation

Baseline: `make check` (71 passed, 2 live tests skipped), frontend
TypeScript/Vite build, lock consistency, and `make data && make train` passed.
The additive contract/generator receive focused tests and the full existing
quality gate before completion. Existing frontend dependency advisories remain
explicitly tracked in the assessment.
