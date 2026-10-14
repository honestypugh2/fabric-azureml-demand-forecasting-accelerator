# Incremental migration map

Operating model: **inherit -> generalize -> extend -> validate -> document**.
The revenue accelerator remains independently reusable; changes occur only in
this repository. Never repurpose revenue labels as future-demand targets.

| Phase | Deliverable | Entry / exit evidence |
| --- | --- | --- |
| 0: safety/discovery | Clean-state check, lineage, inventories, baseline runs | Completed locally; cloud and IaC validation explicitly excluded |
| 1: assessment | Six assessment documents, scored reuse comparison and rename map | This slice; requirements linked to evidence or backlog |
| 2: architecture | Inheritance ADR and target structure now; hybrid/source-of-truth, identity, plugins, IaC and governance ADRs later | Before material implementation, resolve ownership and exact supported APIs |
| 3: generalization | `demand_forecasting`, `demand-forecasting`, proposed `DF_` prefix, migrated consumers | Import/wheel/CLI/config tests, explicit compatibility policy, coordinated assets and locks |
| 4: data/baselines | Configurable demand contracts, origin-aware features, temporal splits, baselines | Raw actuals, monthly/weekly requests/output contracts, origin-frozen lags, naive/seasonal-naive and expanding/sliding backtests implemented locally; broader signals/baselines pending |
| 5: breadth | Plugins, statistical/ML/optional deep-learning models, SDK v2 forecasting/pipelines, uncertainty, hierarchy, ensembles | Compare out-of-sample to baselines; no test-derived weights |
| 6: Fabric | Medallion examples, permission-tested access, governed Gold publication and semantic model | Explicit local vs mocked vs live status; validate Delta rather than file-write proxy |
| 7: application | Demand Learn/Build, scenarios, overrides, consensus, explanations, API/RBAC, agent interfaces | Frontend/API tests, approval boundaries and scenario isolation |
| 8: MLOps/governance | Primary IaC, CI/CD, approvals, monitoring, security/RAI, cost/support/effort | Offline gates and opt-in cloud evidence; no automatic deployment |
| 9: demonstration | Reproducible demand demonstration and model card | Deterministic local run; live steps conditional and separately reported |

## Proposed architecture boundaries

- Fabric Gold is the recommended business-facing source of approved forecast,
  scenario, consensus, override and assumption records.
- Azure ML model registry is the recommended source of approved production
  model artifacts/versions; MLflow records experiments and lineage.
- Batch-first training/scoring executes deterministic code. Reconciliation,
  explanations and out-of-sample ensemble weights belong in versioned pipeline
  components, not in LLM narratives.
- The API authorizes access to approved records/models; experimental workflows
  require an explicit role. Human override records remain separate from labels.
- React/FastAPI remains the application stack. Container Apps is a provisional
  hosting preference for containerized API/UI; App Service is the alternative.
  Confirm identity/network/operating requirements before a hosting ADR or IaC.
- Conversational providers remain replaceable. Tools retrieve and explain
  governed outputs; an LLM never invents official forecast values.

These recommendations need dedicated ADRs and live configuration validation.
No updated hybrid diagram is supplied until those decisions are finalized.

## OneLake qualification

Official [OneLake/Azure ML integration guidance](https://learn.microsoft.com/fabric/onelake/onelake-azure-machine-learning)
was reviewed on 2026-10-13. It distinguishes datastore file access, UI-linked
table copies, shortcut-backed batch paths and direct API/ABFSS access. It also
documents OneLake-backed MLTable streaming limitations.

Do not promise zero-copy AutoML MLTable execution merely because direct file
reading works. Choose an explicitly tested datastore/shortcut or curated-copy
path for the relevant SDK/runtime. A Parquet file upload is not a Delta table
publication. Fabric permissions, execution identity and network access must be
validated for each path.

## First-slice compatibility and rollback

The new demand API is additive under the existing data package. The revenue CLI,
settings, schemas, models, tests, cloud assets and UI retain their contracts.
No compatibility alias is necessary because no public name was removed.

Future migration must distinguish syntactic compatibility from semantic
compatibility. An import alias cannot turn a revenue bundle into a demand model.
Version schemas/bundles and reject incompatible artifacts explicitly.

If rolling back this slice, remove only its named additions and additive
generator function in a reviewed change; do not reset the worktree or original
repository. Generated offline artifacts are not migration inputs.

## Second-slice progress

[ADR 0006](../architecture/adr/0006-demand-origins-and-baselines.md) establishes
calendar origins, arrival-time filtering, fixed-cohort rolling evaluation,
explicit missing-source/target errors and unapproved point output contracts.
The additive `backtest-demand` command under the retained CLI runs synthetic
monthly/weekly demonstrations without Azure/Fabric access.

Next: establish the model capability interface and a learned lag-feature
candidate against these baselines, with origin-specific training and a separate
holdout. Add actual revision vintages and external/planned covariate contracts
before expanding signal coverage. Package migration remains a coordinated
future boundary, not a prerequisite to testing these semantics.

## Third-slice progress: demand is time-series forecasting

[ADR 0007](../architecture/adr/0007-demand-time-series-model-selection.md)
separates demand's time-series algorithms/metrics from revenue regression.
The implemented registry exposes five baselines and two fitted smoothing models.
MASE/RMSSE use origin-only seasonal scales; validation-only ranking and optional
bias/horizon gates precede a nonoverlapping final holdout. Dedicated synthetic
CLI commands persist configurations, capabilities and an unapproved decision.

The second-slice recommendation above is historical; a tabular regression
candidate was **not** made the new default. Future global lag-based ML remains
an optional forecasting adapter requiring its own time-series protocol.
Next: add well-licensed seasonal/ARIMA or intermittent-demand adapters and
probabilistic evaluation, preserving this holdout boundary. Revision vintages,
future covariates and serialization remain prerequisites for broader production
integration.
