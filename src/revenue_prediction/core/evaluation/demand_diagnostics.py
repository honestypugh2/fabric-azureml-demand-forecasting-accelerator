"""Training-only intermittent-demand and structural-eligibility diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandForecastRequest, demand_period_offset
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DEMAND_TARGET
from revenue_prediction.core.evaluation.demand_metrics import seasonal_training_scales
from revenue_prediction.core.features.demand import DemandFeatureConfig, available_demand_history
from revenue_prediction.core.models.demand_forecasters import (
    TimeSeriesModelConfig,
    validate_demand_model_history,
)


def demand_training_diagnostics(
    actuals: pd.DataFrame,
    request: DemandForecastRequest,
    features: DemandFeatureConfig | None = None,
    models: tuple[str, ...] = ("naive", "seasonal_naive"),
    parameters: TimeSeriesModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe only arrived training observations, not validation/holdout patterns.

    Eligibility is structural, not a claim that fitting will converge or that
    a model is suitable. Baseline calendar-source coverage is checked during
    actual forecasting, where the requested horizons are known.
    """
    features = features or DemandFeatureConfig()
    parameters = parameters or TimeSeriesModelConfig()
    history = available_demand_history(actuals, request, features)
    season = features.seasonal_periods or (12 if request.frequency == "MS" else 52)
    records: list[dict[str, object]] = []
    eligibility: list[dict[str, object]] = []
    for key, series in history.groupby(DEMAND_SERIES_COLUMNS, sort=True):
        ids = dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True))
        values = series[DEMAND_TARGET].to_numpy(dtype=float)
        positive = values[values > 0]
        longest = trailing = 0
        dates = pd.DatetimeIndex(series["date"])
        contiguous = dates.equals(
            pd.date_range(start=dates[0], periods=len(dates), freq=request.frequency)
        )
        for index, value in enumerate(values):
            if index and dates[index] != dates[index - 1] + demand_period_offset(request.frequency):
                trailing = 0
            trailing = trailing + 1 if value == 0 else 0
            longest = max(longest, trailing)
        records.append(
            {
                **ids,
                "forecast_origin": request.origin,
                "training_start": series["date"].iloc[0],
                "training_end": series["date"].iloc[-1],
                "n_training_periods": len(values),
                "calendar_is_contiguous": contiguous,
                "n_positive_periods": len(positive),
                "zero_fraction": float(np.mean(values == 0)),
                "average_demand_interval": float(len(values) / len(positive))
                if len(positive)
                else np.nan,
                "positive_size_cv_squared": float((np.std(positive) / np.mean(positive)) ** 2)
                if len(positive)
                else np.nan,
                "longest_zero_run": longest,
                "trailing_zero_run": trailing,
                "intermittency_status": "defined" if len(positive) else "all_zero_training",
            }
        )
        for model_id in models:
            status, reason = "structurally_eligible", "requirements_satisfied"
            try:
                validate_demand_model_history(
                    model_id, series, request.frequency, season, parameters
                )
            except ContractViolation as exc:
                status, reason = "ineligible", str(exc)
            eligibility.append(
                {
                    **ids,
                    "forecast_origin": request.origin,
                    "model_id": model_id,
                    "eligibility_status": status,
                    "reason": reason,
                }
            )
    scales = seasonal_training_scales(history, request.frequency, season)
    return pd.DataFrame(records).merge(
        scales, on=DEMAND_SERIES_COLUMNS, validate="one_to_one"
    ), pd.DataFrame(eligibility)
