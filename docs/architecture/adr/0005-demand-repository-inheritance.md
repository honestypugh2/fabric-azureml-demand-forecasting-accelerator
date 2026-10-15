# ADR 0005: Incremental demand accelerator through repository inheritance

- Status: Accepted
- Date: 2026-10-13
- Scope: Repository inheritance and compatibility strategy

## Context

This demand repository already inherits the separately reusable revenue
accelerator. It contains working offline workflows, tests, an educational app,
SDK v2 integration patterns and cloud assets. Revenue snapshot regression is
not equivalent to future-period demand forecasting.

Inheriting this foundation preserves its reusable workflows without changing
the original revenue project. Starting from a single LSTM implementation would
privilege one model family before establishing forecasting baselines and
governance; a greenfield rewrite would discard the existing tested foundation.

## Decision

1. Work only in this repository; preserve upstream and reference repositories.
2. Use inherit -> generalize -> extend -> validate -> document.
3. Keep demand contracts and workflows additive to the supported revenue
   package; do not remove or relabel existing revenue interfaces.
4. Keep generation in `revenue_prediction.core.data.synthetic`.
   Place demand configuration/schema separately from the revenue schema.
5. Propose `demand_forecasting`, `demand-forecasting` and a neutral application
   identity for a later coordinated migration.
6. Keep existing regression behavior and ADR 0003; an LSTM is not the default.
7. Treat the MIT LSTM repository as a selective donor, not a merge source.
   Adaptation needs per-component review,
   compatibility/leakage tests, optional dependencies and full attribution.
8. All defaults remain original synthetic, neutral data. No real customer
   names/data, personal information or cloud identifiers enter reusable assets.
9. Cloud integrations/deployments remain opt-in. Record local evidence separately
   from live validation.

## Consequences

The demand foundation temporarily lives under the legacy import namespace.
This is deliberate, reversible and requires no compatibility aliases. Existing
CLI/UI/training continue to consume the unchanged revenue schema.

The historical demand dataset contains closed-period actuals, not future
inference features. Actual availability metadata does not replace
origin-filtered feature joins.

Hybrid architecture, OneLake access, hosting, model interfaces, hierarchy,
ensembles, scenario/consensus persistence, identity, approval and IaC selection
require their own architectural decisions. This ADR does not approve a
cloud deployment or production resource rename.
