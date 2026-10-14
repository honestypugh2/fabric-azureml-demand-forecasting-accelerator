"""Additive historical demand contract; independent of revenue snapshot inputs."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ContractViolation

DEMAND_TARGET = "demand_units"
DEMAND_SERIES_COLUMNS = ["sku_id", "channel_id", "geography_id"]
DEMAND_KEY_COLUMNS = ["date", *DEMAND_SERIES_COLUMNS]
DEMAND_HIERARCHY_COLUMNS = ["brand_id", "product_family_id", "product_id", "package_id"]
DEMAND_COLUMNS = [
    *DEMAND_KEY_COLUMNS,
    *DEMAND_HIERARCHY_COLUMNS,
    "frequency",
    "period_end",
    "actual_available_at",
    "unit_price",
    "promotion",
    DEMAND_TARGET,
]

FeatureAvailability = Literal[
    "known_at_forecast_origin", "known_future", "available_after_period_close"
]
DEMAND_COLUMN_AVAILABILITY: dict[str, FeatureAvailability] = {
    **{column: "known_at_forecast_origin" for column in DEMAND_SERIES_COLUMNS},
    **{column: "known_at_forecast_origin" for column in DEMAND_HIERARCHY_COLUMNS},
    "date": "known_future",
    "frequency": "known_future",
    "period_end": "known_future",
    "actual_available_at": "available_after_period_close",
    "unit_price": "available_after_period_close",
    "promotion": "available_after_period_close",
    DEMAND_TARGET: "available_after_period_close",
}


class DemandDataConfig(BaseModel):
    """Configure neutral synthetic series with five years of history by default."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    seed: int = Field(default=42, ge=0, strict=True)
    frequency: Literal["MS", "W-MON"] = "MS"
    start_date: date | None = None
    n_periods: int | None = Field(default=None, ge=1, strict=True)
    n_skus: int = Field(default=3, ge=1, strict=True)
    n_channels: int = Field(default=2, ge=1, strict=True)
    n_geographies: int = Field(default=2, ge=1, strict=True)
    noise_scale: float = Field(default=0.05, ge=0)
    close_delay_days: int = Field(default=1, ge=1, strict=True)
    demand_pattern: Literal["regular", "intermittent", "long_zero_runs", "obsolescence"] = "regular"
    occurrence_probability: float = Field(default=0.2, gt=0, le=1)
    zero_run_periods: int = Field(default=6, ge=1, strict=True)
    obsolete_after_fraction: float = Field(default=0.7, gt=0, lt=1)

    @property
    def first_period(self) -> date:
        """Return an aligned default date when no explicit start is supplied."""
        if self.start_date is not None:
            return self.start_date
        return date(2020, 1, 1) if self.frequency == "MS" else date(2020, 1, 6)

    @property
    def period_count(self) -> int:
        """Return 60 monthly or 261 weekly periods unless explicitly configured."""
        if self.n_periods is not None:
            return self.n_periods
        return 60 if self.frequency == "MS" else 261

    @model_validator(mode="after")
    def _aligned_start(self) -> DemandDataConfig:
        if self.frequency == "MS" and self.first_period.day != 1:
            raise ValueError("For MS frequency, set start_date to the first day of a month.")
        if self.frequency == "W-MON" and self.first_period.weekday() != 0:
            raise ValueError("For W-MON frequency, set start_date to a Monday.")
        return self


def demand_actuals_schema() -> DataFrameSchema:
    """Return the strict, non-coercing schema for closed-period actuals."""
    columns: dict[str, Column] = {
        "date": Column("datetime64[ns]", nullable=False),
        "period_end": Column("datetime64[ns]", nullable=False),
        "actual_available_at": Column("datetime64[ns]", nullable=False),
        "frequency": Column(str, pa.Check.isin(["MS", "W-MON"]), nullable=False),
        "promotion": Column(bool, nullable=False),
    }
    prefixes = {
        "sku_id": "SKU",
        "channel_id": "CHANNEL",
        "geography_id": "REGION",
        "brand_id": "BRAND",
        "product_family_id": "FAMILY",
        "product_id": "PRODUCT",
        "package_id": "PACKAGE",
    }
    for column, prefix in prefixes.items():
        columns[column] = Column(str, pa.Check.str_matches(rf"^{prefix}-\d{{3,}}$"), nullable=False)
    for column in ["unit_price", DEMAND_TARGET]:
        columns[column] = Column(
            float,
            [
                pa.Check(lambda series: np.isfinite(series.to_numpy()).all()),
                pa.Check.gt(0) if column == "unit_price" else pa.Check.ge(0),
            ],
            nullable=False,
        )
    return DataFrameSchema(columns, strict=True, coerce=False, unique=DEMAND_KEY_COLUMNS)


def validate_demand_actuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate grain, dates, frequency, hierarchy and post-close availability.

    This validates historical actuals, not inference features. Values must be
    filtered by ``actual_available_at <= forecast_origin`` before any future
    feature builder consumes them.
    """
    from pandera import errors as pa_errors

    if frame.empty:
        raise ContractViolation("Demand actuals must not be empty; supply closed-period rows.")
    try:
        validated = demand_actuals_schema().validate(frame, lazy=True)
    except (pa_errors.SchemaError, pa_errors.SchemaErrors) as exc:
        raise ContractViolation(f"Demand actuals schema failed: {exc}") from exc

    errors: list[str] = []
    frequencies = validated["frequency"].unique()
    if len(frequencies) != 1:
        errors.append("Use one frequency per dataset; aggregate weekly and monthly separately.")
    else:
        frequency = str(frequencies[0])
        dates = validated["date"]
        if not np.all(dates.eq(dates.dt.normalize()).to_numpy()):
            errors.append("date must be a midnight period-start date.")
        if frequency == "MS":
            aligned = dates.dt.day.eq(1).all()
            expected_end = dates + pd.offsets.MonthEnd(0)
        else:
            aligned = dates.dt.dayofweek.eq(0).all()
            expected_end = dates + pd.Timedelta(days=6)
        if not aligned:
            errors.append("date must align to MS (month start) or W-MON (Monday).")
        if not np.all(validated["period_end"].eq(expected_end).to_numpy()):
            errors.append("period_end must be the inclusive last date of its period.")
        for _, series in validated.groupby(DEMAND_SERIES_COLUMNS, sort=False):
            series_dates = series["date"]
            if not series_dates.is_monotonic_increasing:
                errors.append("Order each series chronologically before validation.")
                break
            expected = pd.date_range(
                start=series_dates.iloc[0], periods=len(series_dates), freq=frequency
            )
            if not np.array_equal(series_dates.to_numpy(), expected.to_numpy()):
                errors.append("Missing periods detected; complete each series calendar explicitly.")
                break

    if not (
        validated["actual_available_at"] >= validated["period_end"] + pd.Timedelta(days=1)
    ).all():
        errors.append("actual_available_at must be after period close, never inside its period.")
    relationships = {
        "sku_id": DEMAND_HIERARCHY_COLUMNS,
        "product_id": ["product_family_id", "brand_id"],
        "product_family_id": ["brand_id"],
    }
    for child, parents in relationships.items():
        if (validated.groupby(child)[parents].nunique() > 1).any().any():
            errors.append(f"Hierarchy inconsistent: each {child} must have stable parents.")
    if errors:
        raise ContractViolation(" ".join(errors))
    return validated
