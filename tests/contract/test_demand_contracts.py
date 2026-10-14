"""Reject malformed historical actuals and unsafe availability metadata."""

from __future__ import annotations

import pandas as pd
import pytest

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_schema import DemandDataConfig, validate_demand_actuals
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset

pytestmark = pytest.mark.contract


@pytest.fixture
def demand() -> pd.DataFrame:
    return generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=4, n_skus=2, n_channels=1, n_geographies=1)
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("sku_id", "invalid"),
        ("frequency", "D"),
        ("demand_units", -1.0),
        ("demand_units", float("nan")),
        ("demand_units", float("inf")),
        ("unit_price", 0.0),
        ("unit_price", float("inf")),
        ("date", pd.NaT),
        ("period_end", pd.Timestamp("2020-01-30")),
        ("actual_available_at", pd.Timestamp("2020-01-15")),
        ("product_id", "PRODUCT-999"),
        ("brand_id", "BRAND-999"),
    ],
)
def test_invalid_rows_rejected(demand: pd.DataFrame, column: str, value: object) -> None:
    demand.loc[0, column] = value
    with pytest.raises(ContractViolation):
        validate_demand_actuals(demand)


def test_schema_rejects_missing_extra_and_wrong_type_columns(demand: pd.DataFrame) -> None:
    for invalid in [
        demand.drop(columns=["promotion"]),
        demand.assign(unreviewed_future_signal=1.0),
        demand.assign(date=demand["date"].astype(str)),
        demand.assign(demand_units=demand["demand_units"].astype(int)),
        demand.assign(promotion="yes"),
    ]:
        with pytest.raises(ContractViolation, match="schema"):
            validate_demand_actuals(invalid)


def test_empty_and_duplicate_grain_rejected(demand: pd.DataFrame) -> None:
    with pytest.raises(ContractViolation, match="empty"):
        validate_demand_actuals(demand.iloc[:0])
    with pytest.raises(ContractViolation, match="schema"):
        validate_demand_actuals(pd.concat([demand, demand.iloc[:1]], ignore_index=True))


def test_missing_periods_and_time_order_rejected(demand: pd.DataFrame) -> None:
    with pytest.raises(ContractViolation, match="Missing periods"):
        validate_demand_actuals(demand.drop(index=2))
    with pytest.raises(ContractViolation, match="chronologically"):
        validate_demand_actuals(demand.iloc[::-1])


def test_mixed_frequencies_rejected(demand: pd.DataFrame) -> None:
    demand.loc[0, "frequency"] = "W-MON"
    with pytest.raises(ContractViolation, match="one frequency"):
        validate_demand_actuals(demand)


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_misaligned_dates_rejected(frequency: str) -> None:
    demand = generate_synthetic_demand_dataset(
        DemandDataConfig.model_validate({"frequency": frequency, "n_periods": 2})
    )
    demand["date"] += pd.Timedelta(days=1)
    with pytest.raises(ContractViolation, match="align"):
        validate_demand_actuals(demand)


def test_intraday_period_start_rejected(demand: pd.DataFrame) -> None:
    demand["date"] += pd.Timedelta(hours=1)
    with pytest.raises(ContractViolation, match="midnight"):
        validate_demand_actuals(demand)


def test_parent_consistency_across_skus(demand: pd.DataFrame) -> None:
    demand.loc[demand["sku_id"] == "SKU-002", "product_family_id"] = "FAMILY-002"
    with pytest.raises(ContractViolation, match="product_id"):
        validate_demand_actuals(demand)


def test_validation_preserves_input(demand: pd.DataFrame) -> None:
    before = demand.copy(deep=True)
    pd.testing.assert_frame_equal(validate_demand_actuals(demand), before)
    pd.testing.assert_frame_equal(demand, before)
