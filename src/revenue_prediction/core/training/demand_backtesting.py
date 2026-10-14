"""Frequency-aware rolling-origin point evaluation without random row splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    DemandForecastRequest,
    DemandFrequency,
    DemandSeriesKey,
    demand_period_offset,
)
from revenue_prediction.core.data.demand_schema import (
    DEMAND_HIERARCHY_COLUMNS,
    DEMAND_KEY_COLUMNS,
    DEMAND_SERIES_COLUMNS,
    DEMAND_TARGET,
    validate_demand_actuals,
)
from revenue_prediction.core.evaluation.demand_metrics import (
    scaled_error_summary,
    seasonal_training_scales,
)
from revenue_prediction.core.evaluation.metrics import bias, smape, wape
from revenue_prediction.core.features.demand import DemandFeatureConfig, available_demand_history
from revenue_prediction.core.models.demand_forecasters import (
    TimeSeriesModelConfig,
    demand_model_capability,
    forecast_demand_models,
)


class DemandBacktestConfig(BaseModel):
    """Explicit rolling origins, evaluation cutoff and expanding/sliding history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first_origin: date
    evaluation_as_of: date
    frequency: DemandFrequency = "MS"
    horizon: int = Field(default=1, ge=1, strict=True)
    n_origins: int = Field(default=3, ge=1, strict=True)
    step_periods: int = Field(default=1, ge=1, strict=True)
    strategy: Literal["expanding", "sliding"] = "expanding"
    training_window_periods: int | None = Field(default=None, ge=1, strict=True)
    seasonal_periods: int | None = Field(default=None, ge=1, strict=True)
    models: tuple[str, ...] = ("naive", "seasonal_naive")
    model_parameters: TimeSeriesModelConfig = Field(default_factory=TimeSeriesModelConfig)
    series: tuple[DemandSeriesKey, ...] | None = None

    @model_validator(mode="after")
    def _valid_windows(self) -> DemandBacktestConfig:
        request = DemandForecastRequest(
            forecast_origin=self.first_origin, frequency=self.frequency, series=self.series
        )
        if request.forecast_dates[0] != request.origin:
            raise ValueError("first_origin must be an aligned month start or Monday.")
        if (self.strategy == "sliding") != (self.training_window_periods is not None):
            raise ValueError("Supply training_window_periods only for the sliding strategy.")
        if self.evaluation_as_of <= self.first_origin:
            raise ValueError("evaluation_as_of must be after first_origin.")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be nonempty and unique.")
        for model_id in self.models:
            demand_model_capability(model_id)
        return self


@dataclass(frozen=True)
class DemandBacktestResult:
    """Unapproved row-level predictions and portfolio/horizon point metrics."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    series_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    segment_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_demand_backtest(
    actuals: pd.DataFrame, config: DemandBacktestConfig
) -> DemandBacktestResult:
    """Evaluate a fixed series cohort at successive forecast origins.

    Each origin re-filters training labels by arrival time. Truth is joined
    only after forecasting, using an explicit evaluation cutoff. Missing or
    unavailable truth fails rather than silently shrinking the evaluation set.
    """
    validated = validate_demand_actuals(actuals)
    features = DemandFeatureConfig(
        seasonal_periods=config.seasonal_periods,
        training_window_periods=config.training_window_periods,
    )
    first_request = DemandForecastRequest(
        forecast_origin=config.first_origin,
        frequency=config.frequency,
        horizon=config.horizon,
        series=config.series,
    )
    first_history = available_demand_history(validated, first_request, features)
    scope = config.series or tuple(
        DemandSeriesKey(**dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)))
        for key in first_history[DEMAND_SERIES_COLUMNS]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    truth = validated.loc[
        validated["actual_available_at"] <= pd.Timestamp(config.evaluation_as_of),
        [*DEMAND_KEY_COLUMNS, *DEMAND_HIERARCHY_COLUMNS, DEMAND_TARGET, "actual_available_at"],
    ].rename(columns={"actual_available_at": "actual_evaluated_available_at"})
    predictions: list[pd.DataFrame] = []
    for origin_index in range(config.n_origins):
        origin = first_request.origin + (
            origin_index * config.step_periods * demand_period_offset(config.frequency)
        )
        request = DemandForecastRequest(
            forecast_origin=origin.date(),
            frequency=config.frequency,
            horizon=config.horizon,
            series=scope,
        )
        forecast = forecast_demand_models(
            validated, request, features, config.models, config.model_parameters
        )
        history = available_demand_history(validated, request, features)
        scales = seasonal_training_scales(
            history,
            config.frequency,
            config.seasonal_periods or (12 if config.frequency == "MS" else 52),
        )
        scored = forecast.merge(truth, on=DEMAND_KEY_COLUMNS, how="left", validate="many_to_one")
        if np.any(scored[DEMAND_TARGET].isna().to_numpy()):
            raise ContractViolation(
                f"Backtest at {origin.date()} lacks complete evaluation actuals by evaluation_as_of; "
                "supply all targets, advance the evaluation cutoff, or shorten the backtest."
            )
        scored["evaluation_as_of"] = pd.Series(
            pd.Timestamp(config.evaluation_as_of), index=scored.index, dtype="datetime64[ns]"
        )
        scored["error"] = scored["predicted_demand_units"] - scored[DEMAND_TARGET]
        scored = scored.merge(scales, on=DEMAND_SERIES_COLUMNS, how="left", validate="many_to_one")
        predictions.append(scored)
    return summarise_demand_backtest(pd.concat(predictions, ignore_index=True))


def summarise_demand_backtest(predictions: pd.DataFrame) -> DemandBacktestResult:
    """Canonical row ordering and identical metrics for direct and tuned runs."""
    combined = predictions.sort_values(
        ["forecast_origin", "model_id", *DEMAND_SERIES_COLUMNS, "horizon"], kind="stable"
    ).reset_index(drop=True)
    metrics = summarise_demand_predictions(combined)
    series_metrics = summarise_demand_predictions(combined, DEMAND_SERIES_COLUMNS)
    segment_metrics = pd.concat(
        [
            summarise_demand_predictions(combined, [column])
            .rename(columns={column: "segment_id"})
            .assign(segment_type=column)
            for column in [*DEMAND_HIERARCHY_COLUMNS, "channel_id", "geography_id"]
        ],
        ignore_index=True,
    )
    return DemandBacktestResult(combined, metrics, series_metrics, segment_metrics)


def summarise_demand_predictions(
    predictions: pd.DataFrame,
    dimensions: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the same metrics to portfolio, series and segment/horizon views."""
    dimensions = dimensions or []
    summaries: list[dict[str, object]] = []
    columns = ["model_id", *dimensions]
    for key, model in predictions.groupby(columns, sort=True):
        groups = [(0, model), *list(model.groupby("horizon", sort=True))]
        for horizon, group in groups:
            actual = group[DEMAND_TARGET].to_numpy(dtype=float)
            predicted = group["predicted_demand_units"].to_numpy(dtype=float)
            error = predicted - actual
            total = float(np.sum(actual))
            summaries.append(
                {
                    **dict(zip(columns, key, strict=True)),
                    "horizon": horizon,
                    "n_predictions": len(group),
                    "n_origins": group["forecast_origin"].nunique(),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mean_error": float(np.mean(error)),
                    "bias_percent": 100 * bias(actual, predicted),
                    "tracking_signal": float(np.sum(error) / np.mean(np.abs(error)))
                    if np.any(error != 0)
                    else 0.0,
                    "smape": smape(actual, predicted),
                    "wape": wape(actual, predicted),
                    "bias_fraction": bias(actual, predicted),
                    "relative_metrics_status": "defined"
                    if total >= 1e-9
                    else "undefined_near_zero_actual_total",
                    **scaled_error_summary(group),
                }
            )
    return pd.DataFrame(summaries)
