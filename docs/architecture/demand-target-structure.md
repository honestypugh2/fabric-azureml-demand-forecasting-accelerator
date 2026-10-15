# Proposed demand package and repository structure

Status: proposal, not the currently installed namespace. The first slice
retains [the inherited layout](repository-structure.md) and adds only a demand
contract/configuration module and a generator function under its data package.

```text
src/demand_forecasting/             # phase 3, not yet created
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
match. Optional libraries stay behind extras and guarded imports. The new
first-slice code adds no dependencies and leaves manifests/locks unchanged.

Slice 5 remains in the inherited namespace: demand forecaster state is in
`core/models/demand_forecasters.py`, safe artifacts/cards and snapshot inference
in `core/inference/demand_artifacts.py`, local governance in
`ops/demand_registry.py`, and commands in `interfaces/cli/demand_lifecycle.py`.
These are implemented surfaces, not empty target-namespace scaffolding.
See [model lifecycle](../governance/demand-model-lifecycle.md) for scope and trust
boundaries. Slice 4's optional statistical extra remains the only added model
dependency; slice 5 adds no dependency.

## Migration boundaries

- Revenue snapshot data/bundles are never silently accepted as demand inputs.
- Train preprocessing only on training windows; split by forecast origin, not
  random rows, and account for label availability.
- Separate model, ensemble, scenario, constraint, override, consensus and
  approved official outputs.
- Feature attributions describe associations, not causal effects.
- Preserve `tests/` organization and offline checks throughout the migration.

See the [rename map](../assessment/domain-rename-map.md) and
[phase backlog](../assessment/migration-map.md) for consumers and exit criteria.
