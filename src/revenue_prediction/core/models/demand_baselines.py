"""Explicit point baselines for origin-filtered monthly and weekly demand."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    FORECAST_SCHEMA_VERSION,
    DemandBaseline,
    DemandForecastRequest,
    validate_demand_forecasts,
)
from revenue_prediction.core.features.demand import DemandFeatureConfig, build_demand_features


def forecast_demand_baselines(
    actuals: pd.DataFrame,
    request: DemandForecastRequest,
    config: DemandFeatureConfig | None = None,
    models: tuple[DemandBaseline, ...] = ("naive", "seasonal_naive"),
) -> pd.DataFrame:
    """Forecast all requested horizons from history known at the origin.

    Naive repeats the last available value. Seasonal naive repeats the prior
    calendar cycle, never substituting another model if a source is missing.
    This function does not train a regression model or approve an official forecast.
    """
    if not models or len(set(models)) != len(models) or set(models) - {"naive", "seasonal_naive"}:
        raise ValueError("models must be unique selections of naive and/or seasonal_naive.")
    features = build_demand_features(actuals, request, config)
    forecasts: list[pd.DataFrame] = []
    for model_id in models:
        prefix = "last" if model_id == "naive" else "seasonal"
        feature = "last_value" if model_id == "naive" else "seasonal_value"
        if np.any(features[feature].isna().to_numpy()):
            raise ContractViolation(
                f"{model_id} lacks an origin-available source for a requested series/horizon; "
                "supply sufficient seasonal history, enlarge the window, or request naive explicitly."
            )
        forecast = features.rename(
            columns={
                feature: "predicted_demand_units",
                f"{prefix}_source_date": "source_date",
                f"{prefix}_source_available_at": "source_available_at",
            }
        ).copy()
        forecast["model_id"] = model_id
        forecast["schema_version"] = FORECAST_SCHEMA_VERSION
        forecast["forecast_kind"] = "baseline"
        forecast["approval_status"] = "unapproved"
        forecasts.append(forecast[FORECAST_COLUMNS])
    return validate_demand_forecasts(pd.concat(forecasts, ignore_index=True), request)
