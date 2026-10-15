"""Temporal residual calibration and untouched-holdout interval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    DemandForecastRequest,
    DemandFrequency,
    DemandSeriesKey,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_intervals import (
    INTERVAL_COLUMNS,
    INTERVAL_METHOD,
    INTERVAL_SCHEMA_VERSION,
    validate_demand_intervals,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DEMAND_TARGET
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    DemandSelectionDecision,
    evaluate_demand_holdout,
    run_demand_selection,
)
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    DemandBacktestResult,
)

CALIBRATION_KEYS = ["model_id", *DEMAND_SERIES_COLUMNS, "horizon"]


def _order_rank(n: int, coverage: float) -> int:
    # Decimal avoids promoting exact integer ranks such as 10 * 0.9 to the next rank.
    return int((Decimal(n + 1) * Decimal(str(coverage))).to_integral_value(rounding=ROUND_CEILING))


class DemandUncertaintyConfig(BaseModel):
    """Predeclare selection, a later calibration block, and a final holdout."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    selection: DemandSelectionConfig
    calibration: DemandBacktestConfig
    nominal_coverage: float = Field(default=0.8, gt=0, lt=1)

    @model_validator(mode="after")
    def _three_temporal_blocks(self) -> DemandUncertaintyConfig:
        if self.selection.ensemble_members is not None:
            raise ValueError(
                "Ensemble interval calibration is not supported by this bounded point-only increment."
            )
        for validation, holdout in [
            (self.selection.validation, self.calibration),
            (self.calibration, self.selection.holdout),
        ]:
            DemandSelectionConfig(validation=validation, holdout=holdout)
        if _order_rank(self.calibration.n_origins, self.nominal_coverage) > (
            self.calibration.n_origins
        ):
            raise ValueError(
                "Too few calibration origins for finite intervals at this coverage; "
                "add origins or predeclare lower coverage."
            )
        return self


class CalibrationCell(BaseModel):
    """One fixed model/series/horizon's absolute residual order statistic."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_id: str
    series: DemandSeriesKey
    horizon: int = Field(ge=1, strict=True)
    n_predictions: int = Field(ge=1, strict=True)
    radius: float = Field(ge=0)

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return self.model_id, *self.series.key, self.horizon


class DemandIntervalCalibration(BaseModel):
    """Numeric JSON calibration state; not a fitted/approved model artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["demand-calibration-v1"] = "demand-calibration-v1"
    method: Literal["absolute_residual_order_statistic"] = INTERVAL_METHOD
    frequency: DemandFrequency
    nominal_coverage: float = Field(gt=0, lt=1)
    calibration_as_of: date
    last_target_date: date
    cells: tuple[CalibrationCell, ...]

    @model_validator(mode="after")
    def _valid_cells(self) -> DemandIntervalCalibration:
        if not self.cells or len({cell.key for cell in self.cells}) != len(self.cells):
            raise ValueError("Calibration cells must be nonempty and unique.")
        last = pd.Timestamp(self.last_target_date)
        close = (
            last + pd.offsets.MonthEnd(0) if self.frequency == "MS" else last + pd.Timedelta(days=6)
        )
        aligned = last.day == 1 if self.frequency == "MS" else last.dayofweek == 0
        if not aligned or close >= pd.Timestamp(self.calibration_as_of):
            raise ValueError(
                "Calibration targets must be aligned and closed before calibration_as_of."
            )
        for cell in self.cells:
            if _order_rank(cell.n_predictions, self.nominal_coverage) > cell.n_predictions:
                raise ValueError("Insufficient calibration predictions for finite coverage.")
        return self


def _validate_scored(predictions: pd.DataFrame, frequency: DemandFrequency) -> None:
    required = {
        *FORECAST_COLUMNS,
        DEMAND_TARGET,
        "actual_evaluated_available_at",
        "evaluation_as_of",
    }
    if (
        not required.issubset(predictions.columns)
        or predictions.columns.duplicated().any()
        or predictions.empty
    ):
        raise ContractViolation("Supply nonempty, complete scored demand backtest predictions.")
    if predictions["forecast_origin"].dtype != np.dtype("datetime64[ns]") or np.any(
        predictions["forecast_origin"].isna().to_numpy()
    ):
        raise ContractViolation("Scored forecasts require nonmissing datetime64[ns] origins.")
    for _, group in predictions.groupby("forecast_origin", sort=True):
        request = DemandForecastRequest(
            forecast_origin=group["forecast_origin"].iloc[0].date(),
            frequency=frequency,
            horizon=int(group["horizon"].max()),
        )
        validate_demand_forecasts(group[FORECAST_COLUMNS], request)
    actual = predictions[DEMAND_TARGET].to_numpy(dtype=float)
    arrival = predictions["actual_evaluated_available_at"]
    cutoff = predictions["evaluation_as_of"]
    close = (
        predictions["date"] + pd.offsets.MonthEnd(0)
        if frequency == "MS"
        else predictions["date"] + pd.Timedelta(days=6)
    )
    if (
        not np.isfinite(actual).all()
        or (actual < 0).any()
        or arrival.dtype != np.dtype("datetime64[ns]")
        or cutoff.dtype != np.dtype("datetime64[ns]")
        or not (arrival.notna() & cutoff.notna() & (arrival > close) & (arrival <= cutoff)).all()
    ):
        raise ContractViolation(
            "Scored demand requires finite actuals available by evaluation cutoff."
        )


def fit_demand_intervals(
    calibration: pd.DataFrame,
    frequency: DemandFrequency,
    calibration_as_of: date,
    nominal_coverage: float = 0.8,
) -> DemandIntervalCalibration:
    """Use only closed, arrived calibration targets, never validation or holdout errors."""
    if not np.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1:
        raise ContractViolation(
            "nominal_coverage must be finite and strictly between zero and one."
        )
    _validate_scored(calibration, frequency)
    cutoff = pd.Timestamp(calibration_as_of)
    if not np.all(calibration["evaluation_as_of"].eq(cutoff).to_numpy()):
        raise ContractViolation("Calibration evaluation cutoff must equal calibration_as_of.")
    scopes = [
        set(group[CALIBRATION_KEYS].itertuples(index=False, name=None))
        for _, group in calibration.groupby("forecast_origin", sort=True)
    ]
    if any(scope != scopes[0] for scope in scopes):
        raise ContractViolation("Every calibration origin must use the same complete cohort.")
    cells: list[CalibrationCell] = []
    for key, group in calibration.groupby(CALIBRATION_KEYS, sort=True):
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = np.abs(
                group[DEMAND_TARGET].to_numpy(dtype=float)
                - group["predicted_demand_units"].to_numpy(dtype=float)
            )
        n = len(residuals)
        rank = _order_rank(n, nominal_coverage)
        if rank > n:
            raise ContractViolation(
                f"Insufficient calibration residuals for {key}: need rank {rank}, have {n}; "
                "add origins or predeclare lower coverage. No pooling or finite fallback."
            )
        if not np.isfinite(residuals).all():
            raise ContractViolation("Calibration residuals overflowed; rescale demand units.")
        cells.append(
            CalibrationCell(
                model_id=key[0],
                series=DemandSeriesKey(**dict(zip(DEMAND_SERIES_COLUMNS, key[1:4], strict=True))),
                horizon=int(key[4]),
                n_predictions=n,
                radius=float(np.partition(residuals, rank - 1)[rank - 1]),
            )
        )
    return DemandIntervalCalibration(
        frequency=frequency,
        nominal_coverage=nominal_coverage,
        calibration_as_of=calibration_as_of,
        last_target_date=calibration["date"].max().date(),
        cells=tuple(cells),
    )


def apply_demand_intervals(
    points: pd.DataFrame, calibration: DemandIntervalCalibration
) -> pd.DataFrame:
    """Attach frozen radii to later point forecasts with identical model/series/horizon scope."""
    if points.empty or not set(FORECAST_COLUMNS).issubset(points.columns):
        raise ContractViolation("Supply complete nonempty point forecasts for interval inference.")
    cells = {cell.key: cell for cell in calibration.cells}
    for _, group in points.groupby("forecast_origin", sort=True):
        request = DemandForecastRequest(
            forecast_origin=group["forecast_origin"].iloc[0].date(),
            frequency=calibration.frequency,
            horizon=int(group["horizon"].max()),
        )
        validate_demand_forecasts(group[FORECAST_COLUMNS], request)
        keys = set(group[CALIBRATION_KEYS].itertuples(index=False, name=None))
        if keys != set(cells):
            raise ContractViolation(
                "Point model/series/horizon scope must exactly match calibration."
            )
    if not (
        (points["forecast_origin"] >= pd.Timestamp(calibration.calibration_as_of)).all()
        and (points["date"] > pd.Timestamp(calibration.last_target_date)).all()
    ):
        raise ContractViolation("Calibration must precede forecast origins and all target periods.")
    result = points[FORECAST_COLUMNS].copy()
    keys = list(points[CALIBRATION_KEYS].itertuples(index=False, name=None))
    radius = np.array([cells[key].radius for key in keys], dtype=float)
    point = points["predicted_demand_units"].to_numpy(dtype=float)
    with np.errstate(over="ignore"):
        upper = point + radius
    if not np.isfinite(upper).all():
        raise ContractViolation("Interval bounds overflowed; rescale demand units.")
    result["interval_schema_version"] = INTERVAL_SCHEMA_VERSION
    result["interval_method"] = calibration.method
    result["nominal_coverage"] = float(calibration.nominal_coverage)
    result["lower_demand_units"] = np.maximum(0.0, point - radius)
    result["upper_demand_units"] = upper
    result["calibration_radius"] = radius
    result["n_calibration_predictions"] = np.array(
        [cells[key].n_predictions for key in keys], dtype=int
    )
    result["calibration_as_of"] = pd.Series(
        pd.Timestamp(calibration.calibration_as_of), index=result.index, dtype="datetime64[ns]"
    )
    return result[INTERVAL_COLUMNS]


def summarise_demand_intervals(
    predictions: pd.DataFrame, dimensions: list[str] | None = None
) -> pd.DataFrame:
    """Coverage, width, interval score and nominal-tail pinball losses, including horizon zero."""
    if predictions.empty or "frequency" not in predictions:
        raise ContractViolation("Supply nonempty scored intervals.")
    frequency = predictions["frequency"].iloc[0]
    _validate_scored(predictions, frequency)
    for _, group in predictions.groupby("forecast_origin", sort=True):
        validate_demand_intervals(
            group[INTERVAL_COLUMNS],
            DemandForecastRequest(
                forecast_origin=group["forecast_origin"].iloc[0].date(),
                frequency=frequency,
                horizon=int(group["horizon"].max()),
            ),
        )
    summaries: list[dict[str, object]] = []
    columns = ["model_id", "nominal_coverage", *(dimensions or [])]
    for key, model in predictions.groupby(columns, sort=True):
        for horizon, group in [(0, model), *list(model.groupby("horizon", sort=True))]:
            actual = group[DEMAND_TARGET].to_numpy(dtype=float)
            lower = group["lower_demand_units"].to_numpy(dtype=float)
            upper = group["upper_demand_units"].to_numpy(dtype=float)
            coverage = float(group["nominal_coverage"].iloc[0])
            alpha = 1 - coverage
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    lower_error, upper_error = actual - lower, actual - upper
                    lower_loss = np.maximum(alpha / 2 * lower_error, (alpha / 2 - 1) * lower_error)
                    upper_loss = np.maximum((1 - alpha / 2) * upper_error, -alpha / 2 * upper_error)
                    width = upper - lower
                    score = width + 2 / alpha * (
                        np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0)
                    )
                    metrics = {
                        "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                        "mean_interval_width": float(np.mean(width)),
                        "mean_interval_score": float(np.mean(score)),
                        "lower_pinball_loss": float(np.mean(lower_loss)),
                        "upper_pinball_loss": float(np.mean(upper_loss)),
                    }
            except FloatingPointError as exc:
                raise ContractViolation(
                    "Interval metrics overflowed; rescale demand units."
                ) from exc
            summaries.append(
                {
                    **dict(zip(columns, key, strict=True)),
                    "horizon": horizon,
                    "n_predictions": len(group),
                    **metrics,
                    "coverage_gap": metrics["empirical_coverage"] - coverage,
                    "lower_quantile_level": alpha / 2,
                    "upper_quantile_level": 1 - alpha / 2,
                }
            )
    return pd.DataFrame(summaries)


@dataclass(frozen=True)
class DemandUncertaintyResult:
    decision: DemandSelectionDecision
    ranking: pd.DataFrame
    validation: DemandBacktestResult
    calibration_backtest: DemandBacktestResult
    calibration: DemandIntervalCalibration
    holdout: DemandBacktestResult
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    series_metrics: pd.DataFrame


def run_demand_uncertainty(
    actuals: pd.DataFrame, config: DemandUncertaintyConfig
) -> DemandUncertaintyResult:
    """Select on validation, calibrate frozen winner/anchors, then measure final holdout."""
    calibration_experiment = config.selection.model_copy(update={"holdout": config.calibration})
    selected = run_demand_selection(actuals, calibration_experiment)
    calibration = fit_demand_intervals(
        selected.holdout.predictions,
        config.calibration.frequency,
        config.calibration.evaluation_as_of,
        config.nominal_coverage,
    )
    holdout = evaluate_demand_holdout(
        actuals, config.selection, selected.validation, selected.decision
    )
    intervals = apply_demand_intervals(holdout.predictions, calibration)
    for column in [DEMAND_TARGET, "actual_evaluated_available_at", "evaluation_as_of"]:
        intervals[column] = holdout.predictions[column].to_numpy()
    return DemandUncertaintyResult(
        selected.decision,
        selected.ranking,
        selected.validation,
        selected.holdout,
        calibration,
        holdout,
        intervals,
        summarise_demand_intervals(intervals),
        summarise_demand_intervals(intervals, DEMAND_SERIES_COLUMNS),
    )
