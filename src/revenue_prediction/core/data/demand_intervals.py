"""Versioned, unapproved central prediction intervals alongside point forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    DemandForecastRequest,
    validate_demand_forecasts,
)

INTERVAL_SCHEMA_VERSION = "demand-interval-v1"
INTERVAL_METHOD = "absolute_residual_order_statistic"
INTERVAL_COLUMNS = [
    *FORECAST_COLUMNS,
    "interval_schema_version",
    "interval_method",
    "nominal_coverage",
    "lower_demand_units",
    "upper_demand_units",
    "calibration_radius",
    "n_calibration_predictions",
    "calibration_as_of",
]


def validate_demand_intervals(frame: pd.DataFrame, request: DemandForecastRequest) -> pd.DataFrame:
    """Require complete point provenance, finite ordered bounds and past calibration."""
    if set(frame.columns) != set(INTERVAL_COLUMNS) or frame.columns.duplicated().any():
        raise ContractViolation("Demand interval columns must match demand-interval-v1 exactly.")
    validate_demand_forecasts(frame[FORECAST_COLUMNS], request)
    for column in [
        "nominal_coverage",
        "lower_demand_units",
        "upper_demand_units",
        "calibration_radius",
    ]:
        if frame[column].dtype.kind != "f" or not np.isfinite(frame[column].to_numpy()).all():
            raise ContractViolation(f"Interval {column} must contain finite floats.")
    if (
        frame["n_calibration_predictions"].dtype.kind not in "iu"
        or not (frame["n_calibration_predictions"] >= 1).all()
    ):
        raise ContractViolation("Intervals require positive integer calibration counts.")
    counts = frame["n_calibration_predictions"].to_numpy(dtype=float)
    if not np.all(frame["nominal_coverage"].to_numpy() <= counts / (counts + 1)):
        raise ContractViolation("Too few calibration predictions for finite interval coverage.")
    if (
        frame["calibration_as_of"].dtype != np.dtype("datetime64[ns]")
        or not (
            frame["calibration_as_of"].notna() & (frame["calibration_as_of"] <= request.origin)
        ).all()
    ):
        raise ContractViolation("Calibration must be timestamped and available at the origin.")
    point = frame["predicted_demand_units"].to_numpy(dtype=float)
    radius = frame["calibration_radius"].to_numpy(dtype=float)
    with np.errstate(over="ignore"):
        expected_upper = point + radius
    if not (
        np.all(frame["interval_schema_version"].eq(INTERVAL_SCHEMA_VERSION).to_numpy())
        and np.all(frame["interval_method"].eq(INTERVAL_METHOD).to_numpy())
        and np.all(frame["nominal_coverage"].between(0, 1, inclusive="neither").to_numpy())
        and (radius >= 0).all()
        and np.array_equal(frame["lower_demand_units"].to_numpy(), np.maximum(0.0, point - radius))
        and np.array_equal(frame["upper_demand_units"].to_numpy(), expected_upper)
    ):
        raise ContractViolation("Invalid interval version, method, coverage or symmetric bounds.")
    return frame
