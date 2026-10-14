"""Origin-filtered demand lags, calendar features and source provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    DemandForecastRequest,
    demand_period_offset,
)
from revenue_prediction.core.data.demand_schema import (
    DEMAND_SERIES_COLUMNS,
    DEMAND_TARGET,
    validate_demand_actuals,
)


class DemandFeatureConfig(BaseModel):
    """Calendar-relative feature windows; no learned imputation or fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seasonal_periods: int | None = Field(default=None, ge=1, strict=True)
    rolling_periods: int = Field(default=3, ge=1, strict=True)
    training_window_periods: int | None = Field(default=None, ge=1, strict=True)


def available_demand_history(
    actuals: pd.DataFrame,
    request: DemandForecastRequest,
    config: DemandFeatureConfig | None = None,
) -> pd.DataFrame:
    """Return only closed actuals received by origin, within an optional window.

    Calendar gaps caused by delayed arrival remain gaps; rows are never shifted
    or imputed to pretend they are different historical periods.
    """
    config = config or DemandFeatureConfig()
    validated = validate_demand_actuals(actuals)
    if str(validated["frequency"].iloc[0]) != request.frequency:
        raise ContractViolation("Request frequency differs from actuals; aggregate explicitly.")
    history = validated.loc[
        (validated["actual_available_at"] <= request.origin)
        & (validated["period_end"] < request.origin)
    ].copy()
    if config.training_window_periods is not None:
        cutoff = request.forecast_dates[0] - (
            config.training_window_periods * demand_period_offset(request.frequency)
        )
        history = history.loc[history["date"] >= cutoff].copy()
    if request.series is not None:
        scope = {item.key for item in request.series}
        keys = history[DEMAND_SERIES_COLUMNS].itertuples(index=False, name=None)
        history = history.loc[[key in scope for key in keys]].copy()
        present = set(history[DEMAND_SERIES_COLUMNS].itertuples(index=False, name=None))
        if present != scope:
            raise ContractViolation(
                "Requested series lack origin-available history in the training window; "
                "supply history or adjust the scope/window."
            )
    if history.empty:
        raise ContractViolation(
            "No actuals available at origin; move the origin or supply history."
        )
    return history.reset_index(drop=True)


def build_demand_features(
    actuals: pd.DataFrame,
    request: DemandForecastRequest,
    config: DemandFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build one origin-frozen feature row per series and forecast horizon.

    Realized prices, promotions and targets are never included in the output.
    ``lag_1`` is the calendar period before the first forecast; ``last_value``
    is the most recent available observation, potentially older. Seasonal
    sources repeat the prior calendar cycle for horizons longer than a season.
    """
    config = config or DemandFeatureConfig()
    history = available_demand_history(actuals, request, config)
    dates = request.forecast_dates
    offset = demand_period_offset(request.frequency)
    season = config.seasonal_periods or (12 if request.frequency == "MS" else 52)
    lag_date = dates[0] - offset
    rolling_dates = pd.date_range(
        end=lag_date, periods=config.rolling_periods, freq=request.frequency
    )
    records: list[dict[str, object]] = []
    for key, series in history.groupby(DEMAND_SERIES_COLUMNS, sort=True):
        by_date = series.set_index("date")
        last = series.iloc[-1]
        rolling = by_date.reindex(rolling_dates)
        complete_rolling = bool(rolling[DEMAND_TARGET].notna().all())
        for horizon, target_date in enumerate(dates, start=1):
            # Freeze the reference cycle at origin, even beyond the first season.
            cycles = (horizon - 1) // season + 1
            seasonal_date = target_date - cycles * season * offset
            seasonal = by_date.loc[seasonal_date] if seasonal_date in by_date.index else None
            records.append(
                {
                    **dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)),
                    "date": target_date,
                    "forecast_origin": request.origin,
                    "frequency": request.frequency,
                    "horizon": horizon,
                    "month": target_date.month,
                    "last_value": float(last[DEMAND_TARGET]),
                    "last_source_date": last["date"],
                    "last_source_available_at": last["actual_available_at"],
                    "lag_1": float(by_date.loc[lag_date, DEMAND_TARGET])
                    if lag_date in by_date.index
                    else np.nan,
                    "lag_1_source_date": lag_date if lag_date in by_date.index else pd.NaT,
                    "lag_1_source_available_at": by_date.loc[lag_date, "actual_available_at"]
                    if lag_date in by_date.index
                    else pd.NaT,
                    "rolling_mean": float(rolling[DEMAND_TARGET].mean())
                    if complete_rolling
                    else np.nan,
                    "rolling_source_available_at": rolling["actual_available_at"].max()
                    if complete_rolling
                    else pd.NaT,
                    "rolling_start": rolling_dates[0] if complete_rolling else pd.NaT,
                    "rolling_end": rolling_dates[-1] if complete_rolling else pd.NaT,
                    "seasonal_value": float(seasonal[DEMAND_TARGET])
                    if seasonal is not None
                    else np.nan,
                    "seasonal_source_date": seasonal_date if seasonal is not None else pd.NaT,
                    "seasonal_source_available_at": seasonal["actual_available_at"]
                    if seasonal is not None
                    else pd.NaT,
                    "training_start": series["date"].iloc[0],
                    "training_end": last["date"],
                    "n_training_periods": len(series),
                }
            )
    return pd.DataFrame(records)
