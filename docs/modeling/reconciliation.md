# Bottom-up demand point reconciliation and hierarchy-bound lifecycle

This offline synthetic workflow creates coherent SKU -> product -> family ->
brand point forecasts, preserving **channel and geography separately**.
It is additive to the time-series model/interval lifecycle, not a namespace
migration or a production publication path. There are no new dependencies,
copied third-party algorithms or cloud calls.

## Run monthly, weekly and replay

```bash
uv run revenue-prediction reconcile-demand \
  --config configs/examples/demand-reconciliation.json \
  --out outputs/demand-reconciled-monthly

uv run revenue-prediction reconcile-demand \
  --config configs/examples/demand-reconciliation-weekly.json \
  --out outputs/demand-reconciled-weekly

uv run revenue-prediction reconcile-demand \
  --config outputs/demand-reconciled-monthly/experiment_config.json \
  --actuals outputs/demand-reconciled-monthly/actuals.parquet \
  --hierarchy outputs/demand-reconciled-monthly/hierarchy.json \
  --out outputs/demand-reconciled-monthly-replay
```

Output directories must be new and are never overwritten. The defaults call
the existing synthetic generator with nine SKUs, one channel and one geography,
so the demonstration includes multiple products, families and brands. Use
`--skus`, `--channels`, `--geographies` and `--seed` to configure generated
synthetic inputs. `--actuals` supplies an existing synthetic historical contract
instead; use generated Parquet to preserve datetime types and precision.

Both example designs predeclare naive, seasonal naive and drift, rolling
validation/holdout blocks, MASE selection and three forecast horizons. Monthly
season length defaults to 12; weekly to 52 (a cycle, not ISO-year alignment).
All standalone point models can be declared in the existing selection config;
optional statistical adapters still require the `forecasting` extra.
Opt-in [ensemble members](ensembles.md) use this same config;
`weighted_ensemble` itself is not an input model ID for selection.
This workflow does not add tuning grids.

## Frozen hierarchy version and availability

`demand-hierarchy-v1` is strict numeric/string JSON with:

- A declared `known_at` date and synthetic data classification.
- Complete, unique, sorted bottom memberships:
  `sku_id x channel_id x geography_id`, product, family and brand.
- Stable parents for each SKU, product and family across scopes.
- A SHA-256 `hierarchy_id` over the canonical payload, including `known_at`.

When no `--hierarchy` is supplied, membership and parent labels are discovered
**only from closed, arrived training rows at the first validation origin**,
respecting the explicit cohort and any sliding window. Future launches and
future-only labels cannot add nodes. An explicit snapshot must have been known
by that origin and match the complete origin-available cohort/parents. A
changed membership/availability declaration creates a different version.

Every forecast origin must be on/after `known_at`. The mapping is frozen
throughout validation and holdout; later launches are excluded from **all**
levels. Totals mean **totals over this version's cohort**, not an enterprise
total that silently incorporates future members. The helper rejects missing
members rather than publishing partial parent sums. There is no assumption
that every SKU has every channel/geography combination.

The snapshot is an as-of classification applied to all forecast horizons.
Effective-dated reparenting, historical hierarchy vintages and future planned
classification changes are not implemented. `known_at` is a local declaration,
not an authenticated metadata timestamp; externally supplied snapshots need
appropriate governance. The historical actuals contract still requires stable
hierarchy labels. Package is a SKU attribute, not an extra hierarchy level.

## Reconciliation contract and coherence

Bottom-up forecasts are deterministic sums of unchanged bottom-level
versioned demand point predictions (`demand-point-v1`, or v2 for weighted
ensembles). Each model and origin is reconciled separately.
Channels/geographies are never pooled, no new optimizer is run, and no
independently fitted upper-level forecast is substituted.

`demand-hierarchy-point-v1` records:

- Origin, target date, horizon, original model ID, frequency.
- Hierarchy level/node, channel/geography and descendant-series count.
- Nonnegative finite predicted units, hierarchy ID and availability date.
- Latest used source date/arrival, earliest training start and latest training
  end across the contributing series.
- `reconciliation_method=bottom_up`, `approval_status=unapproved`.

Parent training boundaries describe the **union** of used bottom histories;
they do not claim every descendant has data throughout that span. The scale
pair counts below separately describe complete aggregate evaluation history.
All quantities being summed must share compatible additive demand units.

The contract validator recomputes complete canonical nodes from original point
rows, checking exact leaf values, node scope/counts, version and provenance.
A separate immediate-child coherence report covers product/family/brand
nodes and rejects incomplete/duplicate/nonfinite node sets rather than hiding
missing levels through an inner join. Floating sums use documented tolerances
`rtol=1e-12`, `atol=1e-9`.
Canonical input ordering makes reconciliation independent of point row order.

**Intervals are not reconciled.** The helper requires exactly the point
contract and rejects interval/scored frames. Summing bounds does not preserve
nominal aggregate coverage or cross-series dependence. There are no aggregate
quantiles, intervals or coverage claims in these outputs.

## Time-series evaluation at each level

Selection remains at the bottom level using validation only. The final holdout
reports the frozen winner and mandatory anchors; hierarchical metrics never
reselect the model. Each level is evaluated independently, including each
horizon and `horizon=0` for the combined-horizon view.

MAE, WAPE and signed bias (prediction minus actual), along with RMSE, tracking
signal and other existing point diagnostics, are computed **after aggregating
both predictions and actuals to the node**. Pooling leaf errors under a
segment label is not equivalent: positive/negative errors can cancel in a
parent forecast. Reports include level-wide and individual node/channel/
geography views. Do not pool levels into one headline metric that counts the
same demand repeatedly.

MASE/RMSSE use a separate training scale for each origin and aggregate node:

1. Filter closed/arrived history within the configured training window.
2. Sum a historical period only if **every fixed descendant** is present.
3. Compute differences at exact seasonal calendar pairs using the shared
   time-series scale helper. Late arrivals leave gaps; no compressed row-offset
   matching, zero padding, imputation or summing/averaging of leaf scales.
4. Preserve undefined/zero scales and pair counts explicitly. If any row in a
   metric group lacks a valid scale, the group's MASE/RMSSE is undefined.

Missing evaluation truth fails the entire backtest rather than dropping a
descendant or shrinking a parent total. Zero/near-zero actual denominators
retain the existing explicit undefined WAPE/bias status. Numerical overflow
fails with rescaling instructions, never a successful infinite metric.

## Persisted evidence and trust boundaries

Each run writes 16 evidence files: synthetic actuals; frozen selection config;
hierarchy and decision JSON; validation ranking; bottom scored rows; node scored
rows; level/node metrics and coherence reports for each block; and
`demand-reconciliation-lineage-v1`.

The sidecar hashes source actuals, hierarchy version, experiment, decision,
bottom point/evaluation rows, node rows, metrics and coherence evidence, plus
implementation/runtime versions. Hashes are canonical semantic record hashes,
not signatures or hashes of CSV file bytes. Parquet preserves exact records;
CSV metrics are human-readable exports. JSON and CSV bytes and Parquet record
values reproduce on a matching-source/runtime replay.

These standalone outputs are **research reconciliation evidence**, not approved
model artifacts, official forecasts, Fabric Gold tables or authenticated
publication. Arbitrarily reconciling approved-model points does not transfer
approval. Hierarchy capability requires an explicit v3 packaging/review path:
`demand-lifecycle package --include-hierarchy` recomputes and binds this
snapshot/evidence, then requires hierarchy-specific level/node/horizon policy.
`demand-lifecycle forecast --include-hierarchy` uses only that reviewed mapping
and winner. All rows still remain unapproved.
See [hierarchy lifecycle](../governance/demand-model-lifecycle.md) and
[ADR 0013](../architecture/adr/0013-demand-hierarchy-lifecycle.md).

The existing strict implementation fingerprint includes new demand modules.
Saved point/interval bundles from an older source tree require their matching
environment or new packaging and independent review; no hash bypass or
automatic migration is introduced.

MinT/top-down/middle-out methods, aggregate
uncertainty, grouped/temporal reconciliation, authenticated production approval,
effective-dated metadata, UI/API and cloud integration remain future work.
See [ADR 0012](../architecture/adr/0012-demand-bottom-up-reconciliation.md).
