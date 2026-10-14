# Current-state assessment

Assessed 2026-10-13 on branch `feat/demand-forecasting-foundation`. This is a
first-slice assessment, not a production-readiness certification.

## Safety and lineage

- Repository: `honestypugh2/fabric-azureml-demand-forecasting-accelerator`.
- `origin`: `https://github.com/honestypugh2/fabric-azureml-demand-forecasting-accelerator.git`.
- `upstream`: `https://github.com/honestypugh2/aml-revenue-prediction-accelerator.git`.
- Initial `git status --short` was empty; 242 tracked files were inventoried.
- References were read using GitHub APIs, without cloning or changing them.
- Revenue reference `main` observed at `2313f64137703d4223f0d0bcb9415eb85e19b4c5`.
- LSTM reference `main` observed at `f3f69c491a17fef87be7e0f37040f221d5a15022`.
- Neither remotes nor Git history were modified. No deployment was performed.

## Inventory and disposition

| Surface | Evidence | Preserve / generalize |
| --- | --- | --- |
| Python packaging | [pyproject.toml](../../pyproject.toml), [uv.lock](../../uv.lock) | `src/` layout, Python 3.11/3.12, Pydantic v2, uv; rename import/CLI separately |
| Configuration | [models.py](../../src/revenue_prediction/config/models.py), [base config](../../configs/base/config.yaml), [.env.example](../../.env.example) | Layering and environment-only secrets; `RPA_`, snapshot, facility and revenue settings need migration |
| Data | [synthetic.py](../../src/revenue_prediction/core/data/synthetic.py), [schema.py](../../src/revenue_prediction/core/data/schema.py), [contracts.py](../../src/revenue_prediction/core/data/contracts.py) | Deterministic synthetic data and leakage contracts; not a future-demand contract |
| Training | [splitting.py](../../src/revenue_prediction/core/training/splitting.py), [train.py](../../src/revenue_prediction/core/training/train.py) | Blocked, sliding and expanding month splits; demand origins need frequency-aware implementation |
| Models | [baselines.py](../../src/revenue_prediction/core/models/baselines.py), [factory.py](../../src/revenue_prediction/core/models/factory.py) | Regression baselines, linear/tree candidates; no demand plugin registry |
| Evaluation | [metrics.py](../../src/revenue_prediction/core/evaluation/metrics.py), [selection.py](../../src/revenue_prediction/core/evaluation/selection.py) | WAPE/bias and candidate comparison; probabilistic and hierarchy evaluation pending |
| Inference | [inference/](../../src/revenue_prediction/core/inference/) | Serialized bundles, batch scoring, MLflow wrapper, optional ONNX |
| Azure ML | [integrations/azureml/](../../src/revenue_prediction/integrations/azureml/), [AutoML regression](../../src/revenue_prediction/integrations/automl/regression.py) | SDK v2, deferred imports; AutoML regression is not AutoML forecasting |
| MLOps | [mlops/](../../mlops/), [azureml_pipeline.py](../../mlops/pipelines/azureml_pipeline.py) | Versionable components, training/scoring/evaluation and endpoint assets; cloud behavior unverified in this slice |
| Fabric | [onelake.py](../../src/revenue_prediction/integrations/fabric/onelake.py), [fabric/](../../fabric/) | Local/remote Parquet I/O, scoring notebook and pipeline guidance; governed Delta publication pending |
| Backend | [app.py](../../src/revenue_prediction/interfaces/api/app.py), [schemas.py](../../src/revenue_prediction/interfaces/api/schemas.py) | Educational API, not yet the required versioned, authorized planning service |
| Frontend | [package.json](../../frontend/package.json), [package-lock.json](../../frontend/package-lock.json), [src/](../../frontend/src/) | React 19, TypeScript, Vite and educational experience; no frontend test script |
| Infrastructure | [infra/README.md](../../infra/README.md), [Bicep](../../infra/bicep/), [Terraform](../../infra/terraform/) | Several inherited profiles; select one maintained primary path after validation |
| CI | [ci.yml](../../.github/workflows/ci.yml), [terraform-validate.yml](../../.github/workflows/terraform-validate.yml) | Quality, frontend and IaC jobs exist; workflows are manual-only |
| Containers/scripts | [docker/](../../docker/), [scripts/](../../scripts/), [Makefile](../../Makefile) | Packaging, Bash/PowerShell setup, UI lifecycle and environment checking |
| Notebooks | [notebooks/](../../notebooks/), [Fabric notebook](../../fabric/notebooks/revenue_scoring_fabric.ipynb) | Five learning notebooks plus Fabric scoring; revenue-specific, not executed here |
| Tests | [tests/](../../tests/) | Unit, contract, API, UI-logic, smoke and opt-in live tests |
| Documentation | [index](../README.md), [ADRs](../architecture/adr/), [notices](../../THIRD_PARTY_NOTICES.md) | Preserve historical decisions; label target demand architecture separately |

## Baseline execution

| Command | Observed result before code changes |
| --- | --- |
| `make check` | Ruff passed; Pyright 0 errors; pytest 71 passed, 2 live tests skipped |
| `uv lock --check` | Passed, 258 packages resolved |
| `npm --prefix frontend run build` | Initially failed: TypeScript executable absent |
| `npm --prefix frontend ci` then `npm --prefix frontend run build` | Restored dependencies; TypeScript/Vite build passed, 93 modules transformed |
| `npm --prefix frontend audit --audit-level=high` | Failed: two inherited high-severity advisories in `nanoid` and `source-map-js`; not fixed in this slice |
| `make data && make train` | Passed offline; six candidates compared, gradient boosting selected, champion bundle written |
| Scoped search for `from azureml.`, `import azureml`, SDK v1 dependency names | No matches in searched Python, TOML, YAML and Markdown files; not a transitive-dependency audit |

MLflow emitted nine Pydantic deprecation warnings during baseline tests.
No live tests, notebooks, IaC compilation, container build, cloud job, or
deployment were executed. A successful local path is not cloud validation.

## Post-change quality gates

`make check` passed after the additive demand implementation: Ruff passed,
Pyright reported zero errors, and pytest reported **113 passed, 2 live tests
skipped**, with the same nine inherited MLflow/Pydantic warnings. The focused
demand plus inherited data tests reported **55 passed**. This adds 42 demand
test cases without removing inherited tests.

The editor test runner did not discover the requested files, so the focused
tests were executed successfully using the repository's `uv run pytest` path.

## Important qualifications

The manifest includes `azure-ai-ml` and `azure-identity` in both core and extras,
contrary to the intended cloud-optional dependency policy. Deferred imports help,
but do not make dependency installation cloud-free.

The inherited OneLake client writes Parquet files. A file upload is not a
governed Delta transaction, semantic model, or automatically Direct Lake-ready
table. Preserve its offline test utility without carrying that assumption into
the demand architecture.

The first slice adds a separate historical demand schema and generator inside
the inherited data package. Existing revenue training and UI retain their
snapshot contracts. The CLI now also has a separate synthetic demand backtest
command. See [gap analysis](gap-analysis.md) for remaining work.

## Second slice: offline origins, features, baselines and backtesting

Added monthly/weekly forecast requests and versioned unapproved point outputs,
origin-available historical lags with source metadata, naive and seasonal-naive
baselines, expanding/sliding rolling origins, complete evaluation joins and
portfolio/per-horizon point metrics. Added the synthetic-only `backtest-demand`
command, saved configs and persisted Parquet/CSV outputs.

| Validation | Executed result |
| --- | --- |
| Focused demand forecasting/contract/CLI tests | 78 passed |
| `make check` | Ruff passed; Pyright zero errors; 191 passed, 2 live tests skipped, 9 inherited warnings |
| `make data && make train` | Revenue generation/training passed, gradient boosting champion unchanged |
| Monthly CLI demonstration | 720 actual rows, 216 evaluated forecasts; naive portfolio WAPE 0.1468, seasonal-naive 0.1063 |
| Weekly sliding CLI demonstration (`--window 104`) | 3,132 actual rows, 216 evaluated forecasts; naive portfolio WAPE 0.1046, seasonal-naive 0.0957 |
| `uv build --wheel` | Wheel built successfully with the additive package modules |
| Lock consistency, new-test typing, formatting and whitespace | Passed |
| Pylance syntax, documented examples, local links and neutrality | Passed |
| Filesystem-scoped focused coverage | 78 passed; all four new demand modules have 100% statement and branch coverage (234 statements, 68 branches) |

Module-named coverage discovery initially failed before the tests with the
NumPy import error `cannot load module more than once per process`, including
with the direct coverage runner. Using `--cov=src/revenue_prediction` instead
succeeded without dependency/environment changes; the report was restricted
to the four new demand modules. This does not claim 100% coverage of the whole
package or CLI.

The CLI tests caught a timestamp-resolution mismatch on Parquet roundtrip.
The producer now explicitly emits `evaluation_as_of` as `datetime64[ns]`;
persisted predictions/configs reproduce the original results without dtype
relaxation.

These synthetic accuracy figures illustrate the comparison, not customer
forecast performance or acceptance thresholds. No new model dependencies,
namespace rename, donor code reuse, frontend edits, cloud resources or
production approvals were introduced. The frontend build was not rerun for
this backend-only slice; its prior baseline result remains above.

## Third slice: time-series registry, scaled metrics and frozen selection

Demand forecasting is explicitly separate from inherited snapshot regression.
The executable local registry includes naive, seasonal naive, moving average,
seasonal moving average, drift, SES and damped Holt. Models fit independently
per series and origin; smoothing coefficients are predeclared, not optimized
on holdout. No inherited regression or LSTM model became the demand default.

MASE is the default selection metric, with explicit RMSSE/WAPE alternatives.
Seasonal denominators use only origin-available calendar pairs within the
training window. Summaries also expose bias percent and tracking signal.
Undefined scales cannot silently drop difficult series; numerical scale
overflow fails explicitly. Optional improvement, bias and horizon-degradation
gates apply to validation only. A separate final holdout evaluates the frozen
algorithm plus the mandatory naive/seasonal-naive anchors without reselection.

| Validation | Executed result |
| --- | --- |
| Combined forecasting/contract/CLI tests | 127 passed |
| `make check` | Ruff passed; Pyright zero errors; 240 passed, 2 live tests skipped, 9 inherited warnings |
| Focused third-slice coverage | 49 passed; metrics 100%, selection 98%, forecasters 94%; combined statement/branch coverage 96% |
| Monthly comparison | 756 validation and 324 holdout forecast rows; selected seasonal moving average |
| Weekly comparison (`--window 156`) | 756 validation and 324 holdout forecast rows; selected SES |
| Persistence reproducibility | Both demonstrations reproduced decisions, rankings, Parquet predictions and CSV metrics exactly |
| `make data && make train` | Inherited revenue generation/training passed; gradient boosting remains champion |
| Wheel build | Succeeded; all three new forecasting modules included |
| New-test Pyright, changed-file formatting, lock consistency, whitespace | Passed |
| Local documentation links and neutrality patterns | Passed for 39 modified/added files, including untracked additions |
| Pylance syntax checks after final Python edits | Passed for updated metrics and time-series tests |

| Synthetic demonstration | Selected validation MASE | Selected holdout MASE | Seasonal-naive holdout MASE |
| --- | --- | --- | --- |
| Monthly | 1.0058 | 0.8851 | 1.0032 |
| Weekly | 0.6744 | 1.1800 | 1.0736 |

The weekly validation winner performs worse than seasonal naive on holdout.
This is preserved and reported, not concealed by selecting again on the test
block. These demonstrations are not customer performance claims or production
acceptance. All decisions remain unapproved.

A repository-wide formatting check also identified seven unchanged inherited
files requiring formatting; they were deliberately not rewritten. Formatting
of all third-slice source/test files passed. No new dependencies, donor code,
namespace migration, frontend changes, commits, deployments or production
approvals were introduced. Broader statistical/intermittent adapters,
probabilistic calibration, revision vintages, model artifacts and live
Fabric/Azure ML validation remain future slices. See the
[model catalog](../modeling/model-catalog.md) and
[time-series metrics](../modeling/time-series-metrics.md).

## Fourth slice: seasonal/intermittent adapters and temporal tuning

The registry now exposes thirteen local point forecasters. Added guarded
statsmodels ETS/Holt-Winters, ARIMA and SARIMA adapters plus original
Croston/SBA/TSB recurrences. The optional `forecasting` extra and lockfile were
updated; installed statsmodels 0.14.6/Patsy 1.0.3 license metadata was verified.
The original seven-model comparison remains the default.

Added seeded intermittent, long-zero-run and obsolescence synthetic scenarios,
training-only pattern/structural diagnostics, and series/segment/horizon metrics.
The predeclared parameter-grid tuner reuses validation gates and requires the
complete fixed cohort for every candidate. Failed model trials remain explicit;
missing dependencies abort with setup instructions. Winner/parameters are frozen
before holdout, with no automatic metric fallback, replacement or approval.

| Validation | Executed result |
| --- | --- |
| Final `make check` | Ruff passed; Pyright zero errors; 299 passed, 2 live tests skipped, 9 inherited warnings |
| New-test typing, lock consistency, changed-file formatting and whitespace | Passed |
| Focused adapter/tuning coverage run | 99 passed; tuner 100% statement/branch coverage; five forecasting modules combined 96% |
| Guarded core execution | Separate process blocked statsmodels and still imported the CLI/registry and forecast Croston/SBA/TSB |
| Statistical adapters | Monthly/weekly provider-equivalent forecasts, correct delayed leads, convergence/nonfinite failures and future-value invariance passed |
| Final artifact reproduction | All three demos: JSON/Parquet exact; CSV trial/portfolio/series/segment metrics and parsed diagnostics within 1e-12 |
| `make data && make train` | Inherited revenue path passed; gradient boosting champion unchanged |
| Wheel build/content | Passed; adapter/tuner/diagnostic modules and optional dependency metadata included |
| Local links and neutrality patterns | Passed for 48 modified/added files, including untracked additions |
| Pylance syntax | Passed for forecasters, tuner and diagnostics |

The focused coverage run preceded the final calendar-gap diagnostic test;
the final full suite includes that additional test. New CLI roundtrip tests
caught a producer ordering difference between per-model tuned trials and direct
backtests. Both now use canonical origin/model/series/horizon ordering and the
same summary helper; no dtype/order relaxation was used to hide that difference.
CSV diagnostic dates are parsed explicitly because CSV does not preserve
datetime dtypes.

| Synthetic demonstration | Validation / holdout rows | Selected model | Selected holdout MASE | Seasonal-naive holdout MASE |
| --- | --- | --- | --- | --- |
| Monthly, one series, statistical/intermittent candidates | 90 / 18 | Seasonal naive | 0.4201 | 0.4201 |
| Weekly, one series, window 156 and seasonal/smoothing candidates | 45 / 27 | Damped Holt | 1.0202 | 0.5608 |
| Intermittent monthly, twelve series | 540 / 216 | Seasonal naive | 0.9200 | 0.9200 |

All three chose predeclared grid trial 0. Monthly/sparse runs retained the
seasonal baseline because alternatives did not improve validation MASE. The
weekly winner underperformed both anchors on holdout; this is reported without
reselection. WAPE can exceed one on sparse demand, and zero-volume windows can
make it undefined. These are synthetic tests, not customer acceptance.

No namespace migration, donor/library source copying, UI changes, commits,
cloud deployment or production approval occurred. Original statistical wrappers,
intermittent methods and diagnostics remain offline and customer-neutral.
Next work is model serialization, model cards, lineage and approval controls;
probabilistic forecasts, inventory costs, residual/segment acceptance,
reconciliation and live integrations remain future work.
See [tuning workflow](../modeling/tuning.md) and
[ADR 0008](../architecture/adr/0008-demand-adapters-and-tuning.md).
