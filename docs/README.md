# Documentation index

One entry point for the accelerator's documentation. All examples use
**synthetic data** and **placeholder identifiers** (`FAC-001`,
`WORKSPACE_PLACEHOLDER`). Review every output with your finance, data, security,
privacy, compliance, and operational stakeholders before any production use.

## Start here

### Demand: time-series forecasting, not revenue regression

The React application and original revenue commands still use the inherited workflow.
Demand's additive path implements a forecasting capability registry, fitted
statistical time-series models, training-scaled MASE/RMSSE, rolling validation
selection, validation-only parameter grids, sparse-demand methods and separate
final holdout. Optional seasonal statistical adapters use statsmodels. The application and inherited commands
remain revenue-specific; demand has dedicated local CLI commands.
Covariate/ML/deep forecasters, scenarios, consensus and cloud publication remain
future work.

| Topic | Read |
| --- | --- |
| Inventory, lineage and baseline results | [assessment/current-state.md](assessment/current-state.md) |
| Gaps and data-discovery questions | [assessment/gap-analysis.md](assessment/gap-analysis.md) |
| Scored inheritance/donor comparison | [assessment/reuse-decision.md](assessment/reuse-decision.md) |
| Incremental phases and architecture boundaries | [assessment/migration-map.md](assessment/migration-map.md) |
| Requirements and implementation status | [assessment/customer-requirements-traceability.md](assessment/customer-requirements-traceability.md) |
| Coordinated rename specification | [assessment/domain-rename-map.md](assessment/domain-rename-map.md) |
| Repository inheritance decision | [ADR 0005](architecture/adr/0005-demand-repository-inheritance.md) |
| Proposed package structure | [architecture/demand-target-structure.md](architecture/demand-target-structure.md) |
| Runnable demand example, dictionary and availability | [data/synthetic-generator.md](data/synthetic-generator.md) |
| Forecast origins and inference/output contracts | [data/leakage-prevention.md](data/leakage-prevention.md) |
| Runnable monthly/weekly baseline backtests | [modeling/backtesting.md](modeling/backtesting.md) |
| Origin and baseline design decision | [ADR 0006](architecture/adr/0006-demand-origins-and-baselines.md) |
| Time-series algorithms and eligibility | [modeling/model-catalog.md](modeling/model-catalog.md) |
| MASE/RMSSE and validation/holdout selection | [modeling/time-series-metrics.md](modeling/time-series-metrics.md) |
| Demand time-series modeling decision | [ADR 0007](architecture/adr/0007-demand-time-series-model-selection.md) |
| Parameter tuning, sparse-demand demos and diagnostics | [modeling/tuning.md](modeling/tuning.md) |
| Optional statistical adapter and tuning decision | [ADR 0008](architecture/adr/0008-demand-adapters-and-tuning.md) |
| Model serialization, cards, lineage and local approval | [governance/demand-model-lifecycle.md](governance/demand-model-lifecycle.md) |
| Artifact and approval trust boundaries | [ADR 0009](architecture/adr/0009-demand-artifacts-and-local-governance.md) |

### Inherited revenue documentation

| I want to… | Read |
| --- | --- |
| Understand the whole system | [architecture/overview.md](architecture/overview.md) |
| See how it maps to MLOps v2 | [architecture/mlops-v2.md](architecture/mlops-v2.md) |
| Understand the package layout | [architecture/repository-structure.md](architecture/repository-structure.md) |
| Run a full end-to-end **demo** | [demo/end-to-end-demo-script.md](demo/end-to-end-demo-script.md) |
| Walk it end to end in a **notebook** | [`notebooks/end_to_end/00_end_to_end_walkthrough.ipynb`](../notebooks/end_to_end/00_end_to_end_walkthrough.ipynb) |
| Run it **offline** on my laptop | [development/getting-started.md](development/getting-started.md) |

## Inherited revenue Azure ML authoring patterns

These paths currently implement revenue regression, not demand AutoML
forecasting. Do not reuse their task/target or metrics unchanged for demand.

Start at the hub: [patterns/end-to-end-patterns.md](patterns/end-to-end-patterns.md).
For **selecting, comparing, and evaluating** models across all four, see
[patterns/model-selection-and-evaluation.md](patterns/model-selection-and-evaluation.md).

| # | Pattern | Guide |
| --- | --- | --- |
| 1 | Automated ML via the Studio **UI** | [patterns/automl-ui.md](patterns/automl-ui.md) |
| 2 | Automated ML via the **SDK** v2 | [patterns/automl-sdk.md](patterns/automl-sdk.md) |
| 3 | Code-first via the **SDK** v2 (champion path) | [patterns/aml-sdk.md](patterns/aml-sdk.md) |
| 4 | Code-first via the Studio **UI** / Designer | [patterns/aml-ui.md](patterns/aml-ui.md) |

## By topic

- **Architecture & decisions:** [architecture/](architecture/) (incl. [ADRs](architecture/adr/))
- **Modeling:** [modeling/strategy.md](modeling/strategy.md), [modeling/automl.md](modeling/automl.md),
  [modeling/models-and-metrics.md](modeling/models-and-metrics.md),
  [modeling/success-metrics-and-kpis.md](modeling/success-metrics-and-kpis.md)
- **Fabric & OneLake:** [fabric/integration.md](fabric/integration.md)
- **Security & networking:** [security/networking.md](security/networking.md)
- **Deployment:** [deployment/guide.md](deployment/guide.md),
  [deployment/checklist.md](deployment/checklist.md),
  [deployment/python-dependencies-and-vnet.md](deployment/python-dependencies-and-vnet.md)
- **Operations:** [operations/runbooks/](operations/runbooks/README.md) (task runbooks index),
  [operations/workspace-tour.md](operations/workspace-tour.md) (Azure ML Studio UI tour),
  [operations/inference-in-production.md](operations/inference-in-production.md),
  [operations/monitoring.md](operations/monitoring.md),
  [operations/retraining.md](operations/retraining.md),
  [operations/onnx-optimization.md](operations/onnx-optimization.md),
  [operations/cost-and-cleanup.md](operations/cost-and-cleanup.md)
- **Governance & responsible AI:** [governance/model-governance.md](governance/model-governance.md),
  [governance/responsible-ai.md](governance/responsible-ai.md)
- **Education & workshops:** [education/inspiration.md](education/inspiration.md),
  [workshops/facilitator-guide.md](workshops/facilitator-guide.md),
  [workshops/exercises.md](workshops/exercises.md),
  [workshops/knowledge-checks.md](workshops/knowledge-checks.md)
- **Development:** [development/getting-started.md](development/getting-started.md),
  [development/dependencies.md](development/dependencies.md)

## Inherited revenue conventions

Demand's grain, time-series metrics and validation policy are described above;
the following conventions apply only to the preserved revenue path.

- **Grain:** `facility_id × accounting_month × snapshot_date` (`snapshot_day`).
- **Target:** `actual_month_end_net_revenue` (known only after accounting close).
- **Primary metric:** WAPE (dollar-weighted); bias reported alongside.
- **Leakage safety:** a feature for a snapshot never uses information from after
  its `snapshot_date`; splits are time-aware, never random rows.
</content>
</invoke>
