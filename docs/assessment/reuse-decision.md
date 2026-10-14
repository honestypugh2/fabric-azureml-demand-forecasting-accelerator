# Reuse decision

Inheritance is already selected. The comparison below records why, rather than
reopening the repository choice. Scores are ordinal architecture-fit judgments:
1 = substantial new work, 3 = useful foundation, 5 = strong fit. They are not
benchmark results or cloud/security certifications.

Evidence: [current-state inventory](current-state.md), the reference repositories'
read-only file trees and MIT licenses, and selected donor model/preprocessor and
dependency files. Uninspected donor surfaces are conservatively scored; their
implementation quality remains unverified.

| Criterion | Inherit this repository | Revenue accelerator directly | Start from LSTM donor | Greenfield |
| --- | --- | --- | --- | --- |
| Fabric readiness | 3 | 3 | 1 | 1 |
| OneLake readiness | 3 | 3 | 1 | 1 |
| AutoML forecasting readiness | 2 | 2 | 1 | 1 |
| Azure ML SDK v2 readiness | 4 | 4 | 3 | 1 |
| Forecasting breadth | 2 | 2 | 2 | 1 |
| Educational UI maturity | 4 | 4 | 1 | 1 |
| MLOps maturity | 3 | 3 | 3 | 1 |
| Security foundation | 2 | 2 | 2 | 1 |
| Testing | 4 | 4 | 2 | 1 |
| Documentation | 4 | 4 | 2 | 1 |
| Migration risk (higher = lower risk) | 4 | 1 | 2 | 2 |
| Proof-of-value readiness | 3 | 2 | 2 | 1 |
| Long-term maintainability | 4 | 2 | 2 | 3 |
| Total, equally weighted / 65 | 42 | 36 | 24 | 16 |

The inherited and revenue columns share capabilities. Their differences are
ownership and migration risk: transforming the original would compromise the
requirement to preserve a separately reusable revenue accelerator. Greenfield
offers naming freedom but discards working educational and test foundations.
Starting from the LSTM donor overweights one model family before establishing
baselines and governance.

## Selected disposition

- **Preserve:** layering, uv, contracts, temporal splitting patterns,
  educational UI, SDK v2 import boundaries and offline test organization.
- **Adapt later:** names, schemas, temporal origins, inference outputs,
  orchestration, deployment assets and business-facing documentation.
- **Reimplement:** demand feature availability, plugins, reconciliation,
  out-of-sample ensembles, scenarios and consensus where no suitable inherited
  implementation exists.
- **Reject:** wholesale donor merge, confidential datasets, default LSTM,
  random-row evaluation, and a competing production approval path in Fabric.

## Donor decision log for this slice

Reference: `honestypugh2/aml-v2-lstm-ts-forecasting-demo`,
commit `f3f69c491a17fef87be7e0f37040f221d5a15022`, MIT license verified.

| Inspected component | Observation | Decision |
| --- | --- | --- |
| `pyproject.toml` | SDK v2, MLflow and CPU-index PyTorch dependency; broad modeling dependencies | Do not copy dependencies; evaluate an optional deep-learning extra later |
| `src/models/lstm_model.py` | Batch-first LSTM with dense output head and configurable dimensions | Candidate for later adaptation behind a tested forecaster interface; no code accepted now |
| `src/data_processing/preprocessor.py` | Scaling, sequence windows, non-shuffled train/validation/test split | Reimplement origin-aware split-first preprocessing; caller must fit scaler on training only and separate multi-step label windows |
| File tree / infrastructure layout | Training, inference, sequence-to-sequence, SDK job, notebooks and tests present | Inventory only; inspect each implementation before component-specific acceptance |
| `LICENSE` | MIT permits licensed reuse with notice retention | Preserve existing attribution; retain full upstream notice for any later copied component |

No new donor implementation was copied or executed. `shuffle=False` alone does
not prove leakage safety: scaling scope and multi-step label overlap require
tests. Existing infrastructure inspiration remains covered by the inherited
notices and ADR 0002.

Decision: [ADR 0005](../architecture/adr/0005-demand-repository-inheritance.md).
