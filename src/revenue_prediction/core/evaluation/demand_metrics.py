"""Seasonal training-only scales and multi-series time-series accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandFrequency, demand_period_offset
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DEMAND_TARGET


def seasonal_training_scales(
    history: pd.DataFrame, frequency: DemandFrequency, seasonal_periods: int
) -> pd.DataFrame:
    """Compute exact calendar seasonal-difference scales from available history.

    Arrival-induced gaps are never compressed into row-offset differences.
    Constant seasonal histories and insufficient pairs have undefined scales.
    """
    if seasonal_periods < 1:
        raise ValueError("seasonal_periods must be positive.")
    records: list[dict[str, object]] = []
    for key, series in history.groupby(DEMAND_SERIES_COLUMNS, sort=True):
        values = series.set_index("date")[DEMAND_TARGET]
        prior_dates = pd.DatetimeIndex(values.index) - seasonal_periods * demand_period_offset(
            frequency
        )
        prior = values.reindex(prior_dates).to_numpy(dtype=float)
        current = values.to_numpy(dtype=float)
        valid = np.isfinite(prior)
        try:
            with np.errstate(over="raise", invalid="raise"):
                differences = current[valid] - prior[valid]
                mae_scale = float(np.mean(np.abs(differences))) if len(differences) else np.nan
                mse_scale = float(np.mean(differences**2)) if len(differences) else np.nan
        except FloatingPointError as exc:
            raise ContractViolation(
                "Seasonal scales overflowed; rescale demand units before evaluation."
            ) from exc
        status = "defined"
        if not len(differences):
            status = "insufficient_seasonal_pairs"
        elif mae_scale == 0 or mse_scale == 0:
            status = "zero_seasonal_scale"
        records.append(
            {
                **dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)),
                "mae_scale": mae_scale,
                "mse_scale": mse_scale,
                "n_scale_pairs": len(differences),
                "scaled_metrics_status": status,
            }
        )
    return pd.DataFrame(records)


def scaled_error_summary(group: pd.DataFrame) -> dict[str, object]:
    """Summarize MASE/RMSSE only when every forecast row has a valid scale."""
    valid = group["scaled_metrics_status"].eq("defined").to_numpy()
    n_valid = int(np.sum(valid))
    if n_valid != len(group):
        return {
            "mase": np.nan,
            "rmsse": np.nan,
            "n_scaled_predictions": n_valid,
            "scaled_metrics_status": "undefined_training_scale",
        }
    error = group["error"].to_numpy(dtype=float)
    mae_scale = group["mae_scale"].to_numpy(dtype=float)
    mse_scale = group["mse_scale"].to_numpy(dtype=float)
    return {
        "mase": float(np.mean(np.abs(error) / mae_scale)),
        "rmsse": float(np.sqrt(np.mean(error**2 / mse_scale))),
        "n_scaled_predictions": n_valid,
        "scaled_metrics_status": "defined",
    }
