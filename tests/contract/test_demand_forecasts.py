"""Forecast-output rejection tests, independent of historical actuals schema."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    DemandForecastRequest,
    DemandSeriesKey,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.models.demand_baselines import forecast_demand_baselines

pytestmark = pytest.mark.contract


@pytest.fixture
def forecast_request() -> DemandForecastRequest:
    return DemandForecastRequest(forecast_origin=date(2021, 1, 1), horizon=2)


@pytest.fixture
def forecast(forecast_request: DemandForecastRequest) -> pd.DataFrame:
    actuals = generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=24, n_skus=2, n_channels=1, n_geographies=1)
    )
    return forecast_demand_baselines(actuals, forecast_request)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("predicted_demand_units", -1.0, "schema"),
        ("predicted_demand_units", float("inf"), "schema"),
        ("predicted_demand_units", float("nan"), "schema"),
        ("horizon", 0, "schema"),
        ("frequency", "W-MON", "schema"),
        ("model_id", "unknown", "schema"),
        ("sku_id", "bad", "schema"),
        ("n_training_periods", 0, "schema"),
        ("schema_version", "unknown", "schema"),
        ("forecast_kind", "official", "schema"),
        ("approval_status", "approved", "schema"),
        ("date", pd.Timestamp("2021-02-01"), "calendar/horizon"),
        ("forecast_origin", pd.Timestamp("2021-01-02"), "origins"),
        ("source_available_at", pd.Timestamp("2021-01-02"), "provenance"),
        ("training_start", pd.Timestamp("2021-01-01"), "provenance"),
        ("training_end", pd.Timestamp("2021-01-01"), "provenance"),
        ("source_date", pd.Timestamp("2020-11-02"), "aligned"),
        ("source_available_at", pd.Timestamp("2020-12-31"), "closes"),
    ],
)
def test_invalid_forecast_row(
    forecast: pd.DataFrame,
    forecast_request: DemandForecastRequest,
    column: str,
    value: object,
    message: str,
) -> None:
    forecast.loc[0, column] = value
    with pytest.raises(ContractViolation, match=message):
        validate_demand_forecasts(forecast, forecast_request)


def test_schema_empty_missing_extra_and_duplicate_rejection(
    forecast: pd.DataFrame, forecast_request: DemandForecastRequest
) -> None:
    with pytest.raises(ContractViolation, match="empty"):
        validate_demand_forecasts(forecast.iloc[:0], forecast_request)
    for invalid in [
        forecast.drop(columns=["source_available_at"]),
        forecast.assign(demand_units=1.0),
        pd.concat([forecast, forecast.iloc[:1]], ignore_index=True),
        forecast.assign(date=forecast.date.astype(str)),
    ]:
        with pytest.raises(ContractViolation, match="schema"):
            validate_demand_forecasts(invalid, forecast_request)


def test_missing_or_reversed_horizons_rejected(
    forecast: pd.DataFrame, forecast_request: DemandForecastRequest
) -> None:
    for invalid in [forecast.drop(index=0), forecast.iloc[::-1]]:
        with pytest.raises(ContractViolation, match="all horizons in order"):
            validate_demand_forecasts(invalid, forecast_request)


def test_inconsistent_model_scope_rejected(
    forecast: pd.DataFrame, forecast_request: DemandForecastRequest
) -> None:
    invalid = forecast.loc[~((forecast.model_id == "naive") & (forecast.sku_id == "SKU-002"))]
    with pytest.raises(ContractViolation, match="same series scope"):
        validate_demand_forecasts(invalid, forecast_request)


def test_explicit_scope_rejected(
    forecast: pd.DataFrame, forecast_request: DemandForecastRequest
) -> None:
    key = DemandSeriesKey(sku_id="SKU-001", channel_id="CHANNEL-001", geography_id="REGION-001")
    scoped = forecast_request.model_copy(update={"series": (key,)})
    with pytest.raises(ContractViolation, match="explicit request scope"):
        validate_demand_forecasts(forecast, scoped)


def test_midperiod_origin_rejects_unclosed_training_period() -> None:
    actuals = generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=24, n_skus=1, n_channels=1, n_geographies=1)
    )
    configured = DemandForecastRequest(forecast_origin=date(2021, 1, 15))
    forecast = forecast_demand_baselines(actuals, configured)
    forecast["training_end"] = pd.Timestamp("2021-01-01")
    with pytest.raises(ContractViolation, match="not closed"):
        validate_demand_forecasts(forecast, configured)
