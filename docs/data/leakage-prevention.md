# Demand origins, inference features and leakage prevention

The offline demand path is additive under `revenue_prediction`. It is separate
from the preserved revenue snapshot transformer and does not use its regression
input schema.

## Inference request

`DemandForecastRequest` accepts:

| Field | Meaning |
| --- | --- |
| `forecast_origin` | Date-only UTC-convention origin; arrival at exactly midnight of this date is available |
| `frequency` | `MS` or `W-MON` |
| `horizon` | Positive integer number of complete forecast periods |
| `series` | Optional nonempty tuple of unique `DemandSeriesKey` records (`sku_id`, `channel_id`, `geography_id`) |

An origin on 2021-01-01 forecasts January onward. An origin on 2021-01-15
forecasts February onward. A Wednesday origin forecasts the next Monday-start
week. This slice does not forecast the partly completed period.

Explicit scope is independent of future actuals. If omitted, discover only
series with actuals available at origin. Reject requested series without usable
history rather than inventing a cold-start value.

## Filtering and feature definitions

First validate the full raw dataset. Then restrict it to:

```text
actual_available_at <= forecast_origin
period_end < forecast_origin
```

If configured, restrict dates to the calendar training window before the first
forecast period. Delayed rows remain absent; do not shift or compress time.

`build_demand_features` returns one row per known series and requested horizon:

| Feature | Definition / missing behavior |
| --- | --- |
| `last_value` | Most recent available actual, with `last_source_date` and arrival timestamp |
| `lag_1` | Actual for the calendar period before the first forecast; absent if not received |
| `lag_1_source_date`, `lag_1_source_available_at` | Provenance for lag 1; missing together when unavailable |
| `rolling_mean` | Mean over `rolling_periods` calendar periods before the first forecast; requires all values |
| `rolling_start`, `rolling_end`, `rolling_source_available_at` | Window dates and latest source arrival; missing when incomplete |
| `seasonal_value` | Previous seasonal cycle's corresponding calendar position, frozen at origin |
| `seasonal_source_date`, `seasonal_source_available_at` | Exact selected seasonal source; missing when unavailable |
| `month` | Forecast-period calendar month, known future |
| `training_start`, `training_end`, `n_training_periods` | Per-series usable history boundaries and actual row count |

Lag 1 and rolling mean are **origin-relative**, not recalculated with actuals
from later horizons. The seasonal feature cycles through the pre-origin season
for horizons longer than the configured seasonal length. Derived missing values
remain `NaN`/`NaT`; there is no imputation. Baselines fail if a required source
is absent, but nullable unused features need not block naive inference.

The defaults are a three-period rolling mean and 12 monthly/52 weekly seasonal
periods. `DemandFeatureConfig` can change these and set a calendar training
window. A short window may make seasonal naive ineligible.

Raw `demand_units`, realized `unit_price` and realized `promotion` are never
included as inference columns. The target is consumed only as historical
source data after arrival. Future plans and external forecast vintages are not
supported by this slice.

## Versioned point forecast output

`forecast_demand_baselines` and the registry-driven `forecast_demand_models`
return strict `demand-point-v1` dataframes. Existing baseline fields remain
unchanged; the allowed model IDs/kinds now include the implemented demand
time-series registry:

| Columns | Contract |
| --- | --- |
| `date`, series identifiers | Complete future period x series grain |
| `frequency`, `forecast_origin`, `horizon` | Match request; horizons 1..H in order for every model/series |
| `model_id` | Implemented registry ID; see [catalog](../modeling/model-catalog.md) |
| `predicted_demand_units` | Finite nonnegative float |
| `source_date`, `source_available_at` | Baseline copy source, or latest fitted input period and latest arrival across statistical training inputs; no later than origin |
| `training_start`, `training_end`, `n_training_periods` | Valid calendar history; source inside bounds, last period closed before origin |
| `schema_version` | `demand-point-v1` |
| `forecast_kind`, `approval_status` | Registry-matched `baseline` or `statistical`, always `unapproved` |

Outputs reject target columns, extra fields, duplicate keys, wrong dtypes,
missing horizons, inconsistent model scope, wrong request scope and unavailable
source metadata. Output validation alone cannot prove that a caller has not
forged provenance; use the provided builder and forecasting function.

## Leakage evidence and remaining boundaries

Tests mutate future demand, prices and promotions and require identical
features/predictions at the earlier origin. They verify exact availability
boundaries, absent delayed calendar lags, no cross-series pooling, and seasonal
sources that never advance into the future.

Evaluation actuals are joined only after forecasting and may legitimately arrive
after origin, but must arrive by `evaluation_as_of`. Their values never feed
that origin's baseline features.

This does not solve revised-history leakage: the raw contract stores one value
per key, not multiple vintages. Correctly model revision arrival and validate
temporal joins before accepting revised or external data. See
[ADR 0006](../architecture/adr/0006-demand-origins-and-baselines.md).
