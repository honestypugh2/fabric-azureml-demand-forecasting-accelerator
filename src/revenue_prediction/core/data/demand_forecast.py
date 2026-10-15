"""Calendar-based demand forecast requests and point-output contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ContractViolation
from .demand_schema import DEMAND_SERIES_COLUMNS

DemandFrequency = Literal["MS", "W-MON"]
DemandBaseline = Literal["naive", "seasonal_naive"]
FORECAST_SCHEMA_VERSION = "demand-point-v1"
ENSEMBLE_FORECAST_SCHEMA_VERSION = "demand-point-v2"
FORECAST_COLUMNS = [
    "date",
    *DEMAND_SERIES_COLUMNS,
    "frequency",
    "forecast_origin",
    "horizon",
    "model_id",
    "predicted_demand_units",
    "source_date",
    "source_available_at",
    "training_start",
    "training_end",
    "n_training_periods",
    "schema_version",
    "forecast_kind",
    "approval_status",
]


def demand_period_offset(frequency: DemandFrequency) -> pd.DateOffset:
    """Return one monthly or Monday-start weekly period."""
    return pd.offsets.MonthBegin() if frequency == "MS" else pd.offsets.Week(weekday=0)


class DemandSeriesKey(BaseModel):
    """Explicit neutral series scope, independent of future actuals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sku_id: str = Field(pattern=r"^SKU-\d{3,}$")
    channel_id: str = Field(pattern=r"^CHANNEL-\d{3,}$")
    geography_id: str = Field(pattern=r"^REGION-\d{3,}$")

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the canonical series key."""
        return self.sku_id, self.channel_id, self.geography_id


class DemandForecastRequest(BaseModel):
    """Forecast complete future periods from a date-only UTC origin.

    A mid-period origin forecasts the next complete period, not the period
    already underway. Optional scope otherwise discovers only series with
    actuals available at the origin.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    forecast_origin: date
    frequency: DemandFrequency = "MS"
    horizon: int = Field(default=1, ge=1, strict=True)
    series: tuple[DemandSeriesKey, ...] | None = None

    @model_validator(mode="after")
    def _unique_scope(self) -> DemandForecastRequest:
        if self.series is not None:
            keys = [series.key for series in self.series]
            if not keys or len(keys) != len(set(keys)):
                raise ValueError("series must be nonempty and unique; supply distinct series keys.")
        return self

    @property
    def origin(self) -> pd.Timestamp:
        """Return the UTC-convention, timezone-naive midnight origin."""
        return pd.Timestamp(self.forecast_origin)

    @property
    def forecast_dates(self) -> pd.DatetimeIndex:
        """Return exactly horizon aligned complete-period start dates."""
        first = demand_period_offset(self.frequency).rollforward(self.origin)
        return pd.date_range(start=first, periods=self.horizon, freq=self.frequency)


def validate_demand_forecasts(frame: pd.DataFrame, request: DemandForecastRequest) -> pd.DataFrame:
    """Validate point output shape, horizon/calendar and historical provenance."""
    from pandera import errors as pa_errors

    from revenue_prediction.core.models.demand_forecasters import demand_model_capabilities

    capabilities = {item.model_id: item for item in demand_model_capabilities()}

    if frame.empty:
        raise ContractViolation(
            "Forecast output is empty; supply forecasts for the requested scope."
        )
    columns: dict[str, Column] = {
        column: Column("datetime64[ns]", nullable=False)
        for column in [
            "date",
            "forecast_origin",
            "source_date",
            "source_available_at",
            "training_start",
            "training_end",
        ]
    }
    for column, prefix in zip(DEMAND_SERIES_COLUMNS, ["SKU", "CHANNEL", "REGION"], strict=True):
        columns[column] = Column(str, pa.Check.str_matches(rf"^{prefix}-\d{{3,}}$"))
    columns.update(
        {
            "frequency": Column(str, pa.Check.eq(request.frequency)),
            "model_id": Column(str, pa.Check.isin(list(capabilities))),
            "schema_version": Column(
                str, pa.Check.isin([FORECAST_SCHEMA_VERSION, ENSEMBLE_FORECAST_SCHEMA_VERSION])
            ),
            "forecast_kind": Column(str, pa.Check.isin(["baseline", "statistical", "ensemble"])),
            "approval_status": Column(str, pa.Check.eq("unapproved")),
            "horizon": Column(int, pa.Check.in_range(1, request.horizon)),
            "n_training_periods": Column(int, pa.Check.ge(1)),
            "predicted_demand_units": Column(
                float, [pa.Check.ge(0), pa.Check(lambda s: np.isfinite(s.to_numpy()).all())]
            ),
        }
    )
    schema = DataFrameSchema(
        columns,
        strict=True,
        coerce=False,
        unique=["forecast_origin", "model_id", *DEMAND_SERIES_COLUMNS, "horizon"],
    )
    try:
        validated = schema.validate(frame, lazy=True)
    except (pa_errors.SchemaError, pa_errors.SchemaErrors) as exc:
        raise ContractViolation(f"Demand forecast schema failed: {exc}") from exc

    dates = request.forecast_dates
    expected_kinds = validated["model_id"].map(
        {key: item.family for key, item in capabilities.items()}
    )
    expected_versions = validated["model_id"].map(
        lambda name: ENSEMBLE_FORECAST_SCHEMA_VERSION
        if name == "weighted_ensemble"
        else FORECAST_SCHEMA_VERSION
    )
    if not np.all(validated["schema_version"].eq(expected_versions).to_numpy()):
        raise ContractViolation("Point schema version must match the model capability.")
    if not np.all(validated["forecast_kind"].eq(expected_kinds).to_numpy()):
        raise ContractViolation(
            "Demand forecast schema failed: forecast_kind must match model capability."
        )
    expected_dates = validated["horizon"].map(dict(enumerate(dates, start=1)))
    if not np.all(validated["date"].eq(expected_dates).to_numpy()):
        raise ContractViolation("Forecast dates do not match the requested calendar/horizon.")
    if not np.all(validated["forecast_origin"].eq(request.origin).to_numpy()):
        raise ContractViolation("Forecast origins must equal the request origin.")
    if not np.all(
        (
            (validated["source_available_at"] <= request.origin)
            & (validated["training_start"] <= validated["source_date"])
            & (validated["source_date"] <= validated["training_end"])
            & (validated["training_end"] < dates[0])
        ).to_numpy()
    ):
        raise ContractViolation("Forecast provenance uses unavailable or out-of-training data.")
    for column in ["source_date", "training_start", "training_end"]:
        values = validated[column]
        aligned = values.dt.day.eq(1) if request.frequency == "MS" else values.dt.dayofweek.eq(0)
        if not np.all((aligned & values.eq(values.dt.normalize())).to_numpy()):
            raise ContractViolation(f"{column} must be an aligned period-start date.")
    source_end = (
        validated["source_date"] + pd.offsets.MonthEnd(0)
        if request.frequency == "MS"
        else validated["source_date"] + pd.Timedelta(days=6)
    )
    if not np.all((validated["source_available_at"] > source_end).to_numpy()):
        raise ContractViolation("Forecast source must be available only after its period closes.")
    training_end_close = (
        validated["training_end"] + pd.offsets.MonthEnd(0)
        if request.frequency == "MS"
        else validated["training_end"] + pd.Timedelta(days=6)
    )
    if not np.all((training_end_close < request.origin).to_numpy()):
        raise ContractViolation("Training includes a period that has not closed at the origin.")
    scope = {item.key for item in request.series} if request.series is not None else None
    model_scopes: list[set[tuple[str, str, str]]] = []
    for _, model in validated.groupby("model_id", sort=False):
        keys = set(model[DEMAND_SERIES_COLUMNS].itertuples(index=False, name=None))
        model_scopes.append(keys)
        if scope is not None and keys != scope:
            raise ContractViolation("Forecast series do not match the explicit request scope.")
        for _, series in model.groupby(DEMAND_SERIES_COLUMNS, sort=False):
            if series["horizon"].tolist() != list(range(1, request.horizon + 1)):
                raise ContractViolation("Each model/series must have all horizons in order.")
    if any(keys != model_scopes[0] for keys in model_scopes):
        raise ContractViolation("All baseline models must forecast the same series scope.")
    return validated
