# Demand-forecasting gap analysis

Status after slice 5 and the bounded slice-6 interval increment: offline forecasting,
safe point-model artifacts, local governance and separately evaluated residual intervals. "Inherited" means implementation is
present, not that the cloud path has been validated.

| Requirement area | Current evidence | Gap / acceptance evidence | Phase |
| --- | --- | --- | --- |
| Repository identity | Demand repository name; revenue package/UI | Coordinated namespace, entry-point, config and asset migration, wheel/import tests | 3 |
| Synthetic history | Monthly/weekly generator, five-year defaults, intermittent/long-zero/obsolescence scenarios | Weather, sentiment, events, stock-outs, broader lifecycle, missingness and raw/medallion tables | 4 |
| Contracts | Strict actuals, monthly/weekly origin requests and versioned unapproved point outputs | Configurable target/grain; governed production lineage contract | 4 |
| Availability/leakage | Origin-filtered calendar lags/rolling features, arrival provenance and future-value invariance tests | Vintage-aware external signals, revised values and planned covariates | 4 |
| Reforecasting | Monthly/weekly local baseline inference with configurable origin/horizon | Automated schedules and application/cloud integration | 4-5 |
| Baselines/backtesting | Five baselines, rolling origins, predeclared validation-only parameter grids, series/segment metrics and separated final holdout | Strategic segment/cost gates, broader search and business acceptance | 4-5 |
| Model breadth | Thirteen point forecasters with numeric JSON state and frozen-origin bundles; revenue regression is separate | Additional statistical variants, ML/deep adapters, probabilistic interfaces and tested schema migrations | 5 |
| AutoML | SDK v2 regression job builder | SDK v2 forecasting configuration, explicit temporal validation and cloud opt-in tests | 5 |
| Pipelines | Local validation/backtesting, recomputed artifact evidence, model cards and transactional model-review lifecycle | Cloud component integration, reconciliation, publication and authenticated approval | 5, 8 |
| Uncertainty | Separate versioned residual intervals/calibration JSON; temporal validation/calibration/holdout; coverage, width, interval score and nominal-tail pinball losses by series/horizon | Learned asymmetric/multilevel quantiles, adaptive calibration, interval-aware artifacts/cards/approval; no time-series coverage guarantee | 5 |
| Evaluation | MASE/RMSSE, WAPE, bias, tracking signal, series/segment views, training-only diagnostics/trial failures, validation gates and separate interval holdout metrics | Costs/segment gates, residual acceptance, planner comparison and interval acceptance policies | 5 |
| Hierarchy | Static synthetic SKU-to-product/family/brand relationships validated | Reconciliation, grouped/temporal hierarchies, coherence tests and hierarchy versions | 5 |
| Ensembles | Candidate comparison | Out-of-sample weights, version/approval metadata and explicit failed-model handling | 5 |
| Fabric | Parquet/local I/O and examples | Medallion tables, Delta transactions, approved forecast records, semantic model and tested permissions | 6 |
| Scenarios | Not implemented for demand | Versioned conditional assumptions, isolated results, provenance and approval | 7 |
| Consensus | Not implemented | Immutable override history, approved official forecast, actuals comparison; no label contamination | 7 |
| Learn/Build | Revenue educational app | Demand modules/workflow, accessible components and frontend tests | 7 |
| API/identity | Educational API | Versioning, Entra authentication, authorization, asynchronous jobs, idempotency and audit events | 7-8 |
| Conversational access | Not implemented | Provider interfaces and authorized deterministic tools; no LLM official-forecast calculation | 7-8 |
| Infrastructure | Multiple inherited profiles | Evidence-led primary IaC choice, app hosting, identity/network profiles and validation | 8 |
| CI/CD | Manual workflows | Trusted push/PR offline gates, frontend tests, scans, links, containers; cloud remains manual | 8 |
| Operations/governance | Demand model cards, source/training/evaluation lineage, local roles/reviewer separation, audit chain/head, holdout gates and revocation | Authenticated identity, protected/signed storage/audit, production registry, monitoring, retraining and retention | 8 |
| Cost/support/effort | Inherited cost/operations docs | Parameterized profiles, priced assumptions with source/date, ownership and effort uncertainty | 8 |
| Demonstration | Synthetic comparisons plus local packaging/cards, registration, independent review, inference and revocation | Full application/governed official publication; live steps separately evidenced | 9 |

## First-slice blockers versus follow-up risks

No credentials are needed for the implemented slice. Two existing npm
high-severity dependency advisories block the inherited CI audit gate.
Resolve them in a targeted frontend dependency slice with lockfile and build
validation; do not silently declare the security gate green.

Other follow-ups: remove cloud SDK duplication from core deliberately; establish
automatic CI; choose IaC only after comparing validated coverage; distinguish
Parquet file outputs from Fabric Gold Delta tables. None was altered here.

## Data-discovery questions

- Is the target shipments, orders, depletions, observed sales, or estimated
  unconstrained demand? Which unit and returns policy applies?
- Which grain, weekly boundary, month convention, timezone and hierarchy
  versions does the planning process use?
- When do actuals close, arrive, and get revised? Are vintages retained?
- How are stock-outs, backorders, launches and retirements represented?
- Which future prices/promotions are committed plans rather than actuals?
- Which external sources are licensed, timestamped and usable at each origin?
- What horizons, reforecast schedules, business costs and segment gates matter?
- Who may change scenarios, override forecasts, approve models and approve
  official forecasts? What retention and tenant boundaries apply?

Use discovery answers as configuration and documented decisions, not embedded
customer assumptions.
