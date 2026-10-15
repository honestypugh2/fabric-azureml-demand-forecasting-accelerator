# Demand-planning requirements traceability

Customer-neutral interpretation of the supplied proof-of-value brief. No
customer names, identifiers, real business data or confidential schemas are
embedded. Status is explicit: implemented means locally tested; proposed and
backlog do not mean executable cloud capability.

| ID | Requirement | Evidence / planned surface | Status / phase |
| --- | --- | --- | --- |
| R01 | Monthly reforecasting | Demand `MS` requests, point baselines and rolling backtests | Local baseline path implemented; scheduler/integration pending |
| R02 | Weekly reforecasting | Demand `W-MON` requests, point baselines and rolling backtests | Local baseline path implemented; scheduler/integration pending |
| R03 | Configurable horizons | Frequency-aware requests, versioned point/interval outputs and bound interval artifacts/review | Implemented locally; production contract pending |
| R04 | What-if planning | Versioned scenario store, conditional deterministic tools | Backlog; 7 |
| R05 | Governed consensus | Separate baseline/ensemble/scenario/override/consensus/official records | Backlog; 7 |
| R06 | Assumptions/external-signal explanations | Origin-filtered historical feature/source provenance | Historical source metadata implemented; external signals/explanations remaining 4-7 |
| R07 | Multiple model families | Thirteen registry forecasters including optional ETS/ARIMA/SARIMA and original Croston/SBA/TSB; revenue regression is separate | Implemented locally; additional variants and ML/deep adapters pending |
| R08 | Transparent comparison | MASE/RMSSE, validation-only grids, series/segment metrics and separate holdout; recomputed evidence and explicit local approval gates/cards | Implemented locally; segment/cost and authenticated production approval pending |
| R09 | Transparent ensemble weights | Out-of-sample weighting with version/scope/approval records | Backlog; 5 |
| R10 | Replaceable models | Forecaster fit/predict/export-state protocol, safe JSON restore, versioned bundles, runtime/source checks and local review registry | All thirteen serialized locally; production model registry/schema migrations pending |
| R11 | Replaceable LLMs, no official calculation | Provider interface and authorized deterministic tools | Proposed boundary; 7 |
| R12 | Weather/sentiment/events/price/promotion/constraints | Synthetic signals and known-future/observed/scenario distinctions | Realized price/promotion only; 4-7 |
| R13 | Existing planning application outputs | Versioned approved forecast export schema | Backlog; 6 |
| R14 | Power BI-ready outputs | Gold Delta forecast records and semantic model | Backlog; inherited Parquet is not sufficient; 6 |
| R15 | Authorized Copilot/Teams access | Entra/RBAC, allowlisted forecast/scenario/explanation tools | Backlog; 7-8 |
| R16 | Indicative architecture | [Migration map](migration-map.md), [target structure](../architecture/demand-target-structure.md) | Proposed, not deployed; 2 |
| R17 | Parameterized cost framework | Region/currency/date/usage profiles and official price references | Backlog; 8 |
| R18 | Implementation-effort framework | Role-based phases with data-discovery assumptions | Phase backlog recorded; estimates pending; 8 |
| R19 | Support/operating model | Ownership, incidents, monitoring, retention and recovery | Inherited revenue guidance; demand backlog; 8 |
| R20 | Approximately five years for proof of value | 60 monthly or 261 weekly synthetic periods by default | Implemented; no real history supplied |
| R21 | Reproducible demos/measurable evaluation | Monthly/weekly/sparse CLI comparisons with persisted grids, trials, diagnostics, frozen configs and holdout | Local tuned forecasting demos implemented; complete governed demand demo backlog; 9 |

## Cross-cutting acceptance requirements

| ID | Requirement | Evidence after baseline slice / next acceptance test |
| --- | --- | --- |
| C01 | Synthetic and neutral defaults | Neutral identifiers, original seeded generator, no donor dataset reuse |
| C02 | Leakage prevention | Origin-filtered historical lags and forecasts tested against future-value mutations/delayed arrivals; revisions/external vintages pending |
| C03 | No random-row primary split | Demand expanding/sliding calendar rolling origins implemented; existing revenue temporal tests preserved |
| C04 | SDK v2 only | Existing SDK v2 paths preserved; scoped v1 source search found no matches |
| C05 | Offline core and optional cloud | Tests need no credentials; core cloud dependency duplication remains a recorded gap |
| C06 | Separate approval sources of truth | Local model registry owns approval status; point rows always remain unapproved; official/Gold approval pending |
| C07 | Probabilistic/hierarchy evaluation | Temporal residual calibration, series/horizon holdout metrics and explicit local interval review gates; learned quantiles and reconciliation pending |
| C08 | Forecast/scenario/override audit lineage | Model/source/evaluation hashes and inference sidecars bind local approval revision; scenario/override/official lineage pending |
| C09 | Local/cloud execution honesty | Live tests skipped; no cloud resources created or validated |
| C10 | License/attribution | MIT references/inherited notices retained; optional statsmodels BSD-3-Clause and Patsy BSD-2-Clause metadata verified; no donor/library source copied |
| C11 | Incremental migration | No namespace replacement, deletions or file moves in first slice |
| C12 | Security/quality gates | Baseline quality/build passed; two inherited npm advisories recorded, not repaired |

Business acceptance thresholds must be agreed per horizon and segment after
data discovery. Synthetic data tests do not establish customer forecast accuracy
or production readiness. Open data questions are in the [gap analysis](gap-analysis.md).
