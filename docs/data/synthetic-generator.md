# Synthetic demand foundation

This generator provides additive, offline historical-actuals examples.
It does not replace revenue generation, train a demand model, or connect to
Fabric/Azure. All values are original, seeded synthetic examples.

## Run locally

After the existing `uv sync` setup, run:

```bash
uv run python - <<'PY'
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset

config = DemandDataConfig(seed=42)
actuals = generate_synthetic_demand_dataset(config)
print(actuals.shape)  # (720, 14): 60 months x 3 SKUs x 2 channels x 2 regions
print(actuals.head())
PY
```

PowerShell uses the same Python code in a here-string:

```powershell
@'
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
actuals = generate_synthetic_demand_dataset()
print(actuals.shape)
'@ | uv run python -
```

The function returns a validated dataframe; it never writes files implicitly.
Persist explicitly with the existing dataset I/O helper if needed. Do not pass
this frame to `train-local`, which still requires revenue snapshots.

## Configuration

`DemandDataConfig` is an immutable Pydantic v2 model and rejects unknown keys,
nonpositive counts, invalid frequencies, negative/nonfinite noise, negative
seeds, and unaligned start dates.

| Setting | Default | Meaning |
| --- | --- | --- |
| `seed` | 42 | Independent local NumPy random generator per call |
| `frequency` | `MS` | Month-start periods; `W-MON` supports Monday-start weeks |
| `start_date` | unset | Resolves to 2020-01-01 monthly, 2020-01-06 weekly |
| `n_periods` | unset | Resolves to 60 monthly or 261 weekly periods (at least five years of period coverage) |
| `n_skus` | 3 | Neutral leaf series dimensions |
| `n_channels` | 2 | Neutral channels |
| `n_geographies` | 2 | Neutral regions |
| `noise_scale` | 0.05 | Standard deviation of synthetic multiplicative noise |
| `close_delay_days` | 1 | Actual availability delay from inclusive period-end date; minimum one day |
| `demand_pattern` | `regular` | `regular`, `intermittent`, `long_zero_runs` or `obsolescence` |
| `occurrence_probability` | 0.2 | Positive-arrival probability for sparse patterns, in (0,1] |
| `zero_run_periods` | 6 | Positive integer; long-zero scenario alternates blocked/eligible blocks of this length |
| `obsolete_after_fraction` | 0.7 | In (0,1); obsolete scenario stops arrivals at `int(period_count * fraction)` |

Row count = periods x SKUs x channels x geographies. Use small explicit counts
in CI. Dates are timezone-naive calendar dates under a UTC convention, not
real local-system event timestamps. Align an explicit start to day 1 or Monday.

Generation contains annual seasonality, gradual trend, neutral series effects,
noise, promotion uplift and synthetic price discounts. Weekly and monthly
examples are generated independently; they are **not temporally reconciled**.
`demand_units` represents an illustrative quantity in one consistent unit, not
a calibrated customer measurement or inferred unconstrained demand.

`regular` preserves the original generation sequence and values. Other patterns
gate the same synthetic positive sizes with seeded Bernoulli occurrences.
`long_zero_runs` forces alternating zero blocks before applying occurrences;
`obsolescence` forces the tail to zero. Zeros are observed demand, not missing
actuals or simulated stock-outs. No occurrence count is artificially guaranteed;
an all-zero sample is legitimate and must expose undefined scale metrics.
Scenarios test algorithm behavior, not calibrated inventory or lifecycle models.
See [sparse-demand comparison and tuning](../modeling/tuning.md).

## Data dictionary and availability

Every column is required, non-null, and classified in
`DEMAND_COLUMN_AVAILABILITY`. No unreviewed extra columns are accepted.

| Column(s) | Type / expectation | Availability |
| --- | --- | --- |
| `date` | `datetime64[ns]`, midnight period start | Known future calendar |
| `sku_id`, `channel_id`, `geography_id` | Neutral `SKU-001`, `CHANNEL-001`, `REGION-001` patterns | Known at origin in this static synthetic hierarchy |
| `brand_id`, `product_family_id`, `product_id`, `package_id` | Neutral hierarchy identifiers; stable parents | Known at origin in this static example |
| `frequency` | `MS` or `W-MON`, one per dataset | Known future calendar |
| `period_end` | Inclusive last calendar date: month end or Sunday | Known future calendar |
| `actual_available_at` | `datetime64[ns]`, no earlier than the day after period end | Post-close record arrival |
| `unit_price` | Finite positive float, arbitrary synthetic currency/unit | Realized historical actual, post-close |
| `promotion` | Boolean, realized historical promotion | Post-close, not a known-future promotion plan |
| `demand_units` | Finite nonnegative float, target | Post-close actual, never an inference feature |

Grain: `date x sku_id x channel_id x geography_id`. The schema rejects duplicate
keys, wrong dtypes, missing/extra columns, nonfinite values and invalid IDs.
Semantic validation rejects mixed frequencies, intraday/unaligned period starts,
wrong period ends, internal calendar gaps, reversed series order, in-period
availability and inconsistent SKU/product/family parents.

Series can have different start/end boundaries; validation checks internal
continuity, not a universal launch/retirement calendar. Missing leading/trailing
periods cannot be inferred without an external expected calendar.

## Leakage boundary

The raw frame intentionally contains future labels across its history.
At forecast origin `t`, the demand feature builder first restricts actuals to
`actual_available_at <= t`, then build per-series lags/rolling features. A
period-start filter alone is unsafe. Keep targets separate from inputs.

See [the implemented origin/feature contracts](leakage-prevention.md) and
[baseline backtests](../modeling/backtesting.md) for the additive demand path.

Price and promotion here are realized values. Future plans require separate
versioned records available at the origin; weather actuals, forecasts and
scenario assumptions must also be distinct. Historical arrival filtering is now
implemented; revision vintages and future-covariate feature building are not.

Tests:
[generator](../../tests/unit/test_synthetic_demand.py) and
[contract](../../tests/contract/test_demand_contracts.py).
Run `uv run pytest -q tests/unit/test_synthetic_demand.py tests/contract/test_demand_contracts.py`,
then `make check`.

## Scope and input limitations

Synthetic histories do not establish production accuracy or business acceptance.
Weather/sentiment/event inputs, stock-out censoring, inventory/production
constraints, revision vintages and future-covariate feature contracts require
separate domain designs. Parquet examples are not governed Fabric Delta tables.
For forecasting outputs and supported algorithms, use the
[model catalog](../modeling/model-catalog.md),
[uncertainty guide](../modeling/uncertainty.md) and
[hierarchy guide](../modeling/reconciliation.md).

## Data-discovery questions

Before adapting the accelerator to a real planning process, establish:

- Whether the target is shipments, orders, depletions, observed sales or
  estimated unconstrained demand, including units and returns policy.
- Series grain, weekly boundary, month convention, timezone and hierarchy
  versions.
- Actual close, arrival and revision timing, and whether vintages are retained.
- Representation of stock-outs, backorders, launches and retirements.
- Which future prices/promotions are committed plans rather than actuals.
- External-source licensing, timestamps and availability at each origin.
- Required horizons, reforecast schedules, business costs and segment gates.
- Model/scenario/override/official-forecast approval roles, retention and
  tenant boundaries.

Use discovery answers as configuration and documented decisions, not embedded
customer assumptions.
