# Domain rename map

This is a migration specification, not a global replacement list. First-slice
changes are additive: all existing revenue names continue to work.

| Class / surface | Current reference | Proposed target / action | Compatibility and validation |
| --- | --- | --- | --- |
| Repository | `fabric-azureml-demand-forecasting-accelerator` | Already correct; preserve remotes | Original revenue repository unchanged |
| Distribution metadata | Demand project name with revenue description/keywords/authors | Neutral demand description and maintainers in phase 3 | Regenerate uv lock only with manifest changes |
| Public imports | `src/revenue_prediction/`, absolute imports in source/tests/scripts | `src/demand_forecasting/` | Inventory callers; wheel/import tests; aliases only for verified callers |
| Public CLI | `revenue-prediction`, Typer revenue commands | `demand-forecasting` with demand-specific inputs | Preserve legacy during transition only if needed; entry-point tests |
| Environment prefix | `RPA_`, `RPA_RUN_LIVE` | Proposed `DF_`, `DF_RUN_LIVE` | Document precedence/deprecation; reject contradictory settings |
| Settings/config paths | Revenue `DataConfig`, `configs/base` -> environment | Separate demand schema/settings, same layering | No automatic translation of facility counts/snapshots |
| Grain | Facility x accounting month x snapshot date | Period x configured series dimensions | Semantic redesign, not identifier substitution |
| Identifiers | `facility_id`, `FAC-001` | Synthetic `sku_id`/channel/geography initially | Preserve revenue contract; later configurable grain |
| Dates | `accounting_month`, `snapshot_date`, `snapshot_day` | Period start/end, forecast origin, actual availability | Snapshot date is not interchangeable with forecast origin |
| Target | `actual_month_end_net_revenue` | Initial historical `demand_units`; final target configurable | Clarify sales vs unconstrained demand, units, returns and censoring |
| Features | Partial-month charges/payment and prior revenue | Demand lags/calendar/plans/external forecasts | Reimplement availability semantics; fit preprocessing on training only |
| Training | Revenue regression factories and temporal month splits | Forecaster registry and frequency-aware backtests | Preserve working regression; do not default to LSTM |
| Bundle/output | Revenue target and prediction fields | Versioned demand model/forecast/interval records | Reject incompatible legacy bundles; no relabeling |
| API | `/api/*`, revenue Pydantic objects | Versioned `/api/v1/*` demand requests/jobs/records | Contract/auth tests; legacy routes only if callers need them |
| UI | Revenue app identity, facility charts, snapshot workflow | Demand Learn/Build; series/horizons/scenarios/consensus | Update API client and education together; component/E2E tests |
| Frontend metadata | `revenue-prediction-frontend` in npm manifest/lock | `demand-forecasting-frontend` | Coordinated package/lock and TypeScript/Vite build |
| Pipeline assets | Revenue component/module/data/model names | Neutral versioned demand components and outputs | SDK v2 construction tests, opt-in live submission |
| Infrastructure | Revenue names in Bicep/Terraform/default parameters | Demand prefixes with environment/suffix configuration | Never rename live resources implicitly; validate chosen IaC |
| OneLake paths | `Files/revenue/input`, `Files/revenue/predictions` | Demand Files staging and governed Gold Delta tables | Migration manifest, transactional publication and schema tests |
| Containers | Installed package/CLI in Docker assets | Demand wheel/CLI once migrated | Build/import/entry-point tests |
| Workflows/tooling | Coverage, Ruff path ignores, neutrality allowlists, Make targets | Update specific consumers for new namespace/prefix | Preserve live opt-in and untrusted-PR boundaries |
| Notebooks/Fabric | Revenue notebook filenames and cells | Demand labs with tested imports/contracts | Inspect cells/output metadata; clear generated sensitive content |
| Documentation | Healthcare/revenue prose, historical ADRs | Current demand guides; retain labeled historical ADRs | Link validation; never rewrite architectural history blindly |
| Generated artifacts | Data samples, outputs, MLflow, caches, frontend build | Regenerate only relevant synthetic/build outputs | Not source migration inputs; no hand-renaming models/data |
| Tests | Revenue fixtures and behavior assertions | Keep legacy coverage; add demand tests | Preserve original behavior until explicit retirement decision |

## Inventory evidence and sequencing

The initial scoped search found 510 matches in 125 files across source, tests,
configuration, docs, cloud assets, frontend source and notebooks. This is a
search result, not an exhaustive semantic reference count. Packaging,
environment, Makefile, Docker and workflow surfaces were separately inspected.

Before phase 3, refresh a full rename manifest from tracked files and symbol
references. Migrate internal imports and packaging together, then configuration
and deployment consumers, then API/UI and notebooks/docs. Run import, CLI,
wheel, config, offline training and build tests at each boundary.

Do not mechanically replace `snapshot_date` with `date`, `revenue` with `demand`,
or dollar metrics with unit metrics. Those require semantic design decisions.

## Retirement candidates (not approved for removal)

Legacy revenue namespace/CLI aliases, revenue-specific labs/assets and duplicate
IaC paths may eventually be retired in this demand repository once replacements
and consumer migration are verified. Historical ADRs and attribution remain.
Nothing in this first slice is moved or removed.
