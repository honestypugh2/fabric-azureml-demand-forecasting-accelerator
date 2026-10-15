# Demand package architecture and proposed namespace

Demand and revenue workflows coexist in the installed `revenue_prediction`
namespace. See [the repository layout](repository-structure.md) and
[namespace compatibility guide](domain-rename-map.md) before changing public
imports, commands or deployment assets.

## Implemented demand module boundaries

Paths below are relative to `src/revenue_prediction/`.

| Responsibility | Modules |
| --- | --- |
| Historical data and synthetic generation | `core/data/demand_schema.py`, `core/data/synthetic.py` |
| Origin/output and interval contracts | `core/data/demand_forecast.py`, `core/data/demand_intervals.py` |
| Availability-aware historical features | `core/features/demand.py` |
| Model capabilities and safe component states | `core/models/demand_forecasters.py` |
| Rolling backtests, selection and tuning | `core/training/demand_backtesting.py`, `core/evaluation/demand_selection.py`, `core/evaluation/demand_tuning.py` |
| Temporal residual calibration | `core/evaluation/demand_uncertainty.py` |
| Frozen hierarchy, bottom-up sums and aggregate evaluation | `core/data/demand_hierarchy.py`, `core/models/demand_reconciliation.py`, `core/evaluation/demand_reconciliation.py` |
| Immutable records and content hashing | `core/data/demand_records.py` |
| Fitted bundles, model cards and snapshot inference | `core/inference/demand_artifacts.py` |
| Local model review and approval audit | `ops/demand_registry.py` |
| CLI adapters | `interfaces/cli/main.py`, `interfaces/cli/demand_lifecycle.py` |

Point-only v1, residual-interval v2 and hierarchy/ensemble v3 artifacts share the
same registry and inference boundary. Ensemble weights are learned from rolling
validation only, then frozen before final holdout. Optional statistical imports
remain guarded. Local model approval is not official forecast approval or
authenticated production authorization.

See [model lifecycle](../governance/demand-model-lifecycle.md),
[reconciliation](../modeling/reconciliation.md) and
[ensembles](../modeling/ensembles.md) for contracts and limitations.

## Proposed namespace

The following is a design for a coordinated migration, not an installed
`demand_forecasting` package or a request to scaffold empty modules.

```text
src/demand_forecasting/             # proposed, not the installed namespace
  config/                          # layered settings, proposed DF_ prefix
  core/
    data/                          # actuals, inference/output contracts, synthetic
    features/                      # origin-aware joins/lags and provenance
    models/                        # protocol, capability registry, adapters
    training/                      # temporal splits, backtests and training
    evaluation/                    # point/probabilistic/business-cost gates
    inference/                     # versioned bundles and batch forecasts
    hierarchy/                     # reconciliation and coherence
    ensembles/                     # out-of-sample approved weights
    scenarios/                     # conditional deterministic projections
    consensus/                     # overrides, approvals and official records
  integrations/
    azureml/ automl/ fabric/        # optional SDK v2 and storage boundaries
    conversation/                  # replaceable providers and authorized tools
  interfaces/
    cli/ api/ ui/                  # thin adapters, versioned API
  education/                       # Learn content and runnable labs
  ops/                             # monitoring, retraining and lineage
  security/                        # authorization/audit/redaction helpers
  pipelines/                       # local orchestration and cloud adapters
tests/                             # unit, contract, API/UI, smoke, opt-in live
configs/ data/ docs/ fabric/        # existing top-level responsibilities
frontend/ infra/ mlops/ notebooks/
scripts/ docker/
```

Create a module only when implementing its behavior; this proposal is not a
request to scaffold empty abstractions. Reuse existing helpers where semantics
match. Optional libraries stay behind extras and guarded imports.

## Deployment and source-of-truth boundaries

For production adaptation, keep model, forecast, scenario, override and official
record ownership explicit:

- Azure ML model registry can own approved production model versions, with
  MLflow recording experiment lineage.
- Fabric Gold can own approved business forecast, scenario, consensus, override
  and assumption records.
- Batch components execute deterministic forecasting, reconciliation and
  ensemble calculations. An LLM must not invent official forecast values.
- APIs must authorize model/record access; experimental workflows require
  explicit roles. Human overrides remain separate from training labels.
- React/FastAPI is the existing application stack. Container Apps and App
  Service are hosting options for a containerized API/UI; confirm identity,
  networking and operating requirements before selecting a deployment profile.
- Hosting, identity, networking and publication paths require deployment-specific
  design and validation; these recommendations are not deployed resources.

### OneLake access qualification

[OneLake/Azure ML integration guidance](https://learn.microsoft.com/fabric/onelake/onelake-azure-machine-learning)
distinguishes datastore file access, UI-linked table copies, shortcut-backed
batch paths and direct API/ABFSS access. It also documents OneLake-backed MLTable
streaming limitations.

Do not infer zero-copy AutoML MLTable support from a successful direct file
read. Validate the chosen datastore/shortcut or curated-copy path for the
relevant SDK/runtime, permissions, execution identity and network access.
A Parquet upload is not transactional Delta table publication.

## Migration boundaries

- Revenue snapshot data/bundles are never silently accepted as demand inputs.
- Train preprocessing only on training windows; split by forecast origin, not
  random rows, and account for label availability.
- Separate model, ensemble, scenario, constraint, override, consensus and
  approved official outputs.
- Feature attributions describe associations, not causal effects.
- Preserve `tests/` organization and offline checks throughout the migration.

See the [namespace compatibility guide](domain-rename-map.md) for affected
consumers and validation requirements.
