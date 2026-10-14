"""Validation-only time-series selection with a temporally separated holdout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    DemandForecastRequest,
    DemandSeriesKey,
    demand_period_offset,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS
from revenue_prediction.core.models.demand_forecasters import TimeSeriesModelConfig
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    DemandBacktestResult,
    run_demand_backtest,
)


class DemandSelectionConfig(BaseModel):
    """Fix validation, holdout and business gates before observing holdout errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    validation: DemandBacktestConfig
    holdout: DemandBacktestConfig
    primary_metric: Literal["mase", "rmsse", "wape"] = "mase"
    minimum_relative_improvement: float = Field(default=0.0, ge=0, lt=1)
    max_abs_bias_fraction: float | None = Field(default=None, ge=0)
    max_horizon_relative_degradation: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _separated_evaluation(self) -> DemandSelectionConfig:
        validation, holdout = self.validation, self.holdout
        if not {"naive", "seasonal_naive"}.issubset(validation.models):
            raise ValueError("Selection must compare both naive and seasonal_naive baselines.")
        for field in [
            "frequency",
            "horizon",
            "models",
            "model_parameters",
            "seasonal_periods",
            "strategy",
            "training_window_periods",
            "series",
        ]:
            if getattr(validation, field) != getattr(holdout, field):
                raise ValueError(
                    f"Validation/holdout must use identical {field}; predeclare one comparison."
                )
        last_origin = pd.Timestamp(validation.first_origin) + (
            (validation.n_origins - 1)
            * validation.step_periods
            * demand_period_offset(validation.frequency)
        )
        final_validation_date = DemandForecastRequest(
            forecast_origin=last_origin.date(),
            frequency=validation.frequency,
            horizon=validation.horizon,
        ).forecast_dates[-1]
        final_validation_close = (
            final_validation_date + pd.offsets.MonthEnd(0)
            if validation.frequency == "MS"
            else final_validation_date + pd.Timedelta(days=6)
        )
        selection_origin = pd.Timestamp(holdout.first_origin)
        if (
            final_validation_close >= selection_origin
            or validation.evaluation_as_of > holdout.first_origin
        ):
            raise ValueError(
                "Validation targets must close and be evaluated before/by the first holdout origin; "
                "use nonoverlapping target periods and an origin-available validation cutoff."
            )
        return self


class DemandSelectionDecision(BaseModel):
    """Frozen local selection metadata, not production approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_model: str
    primary_metric: str
    validation_metric: float
    seasonal_naive_metric: float
    selected_at: date
    reason: Literal["validation_winner", "seasonal_baseline_retained"]
    approval_status: Literal["unapproved"] = "unapproved"
    selected_parameters: TimeSeriesModelConfig = Field(default_factory=TimeSeriesModelConfig)


@dataclass(frozen=True)
class DemandSelectionResult:
    """Separate validation ranking, frozen decision and final holdout evaluation."""

    decision: DemandSelectionDecision
    ranking: pd.DataFrame
    validation: DemandBacktestResult
    holdout: DemandBacktestResult


def rank_demand_validation(
    validation: DemandBacktestResult, config: DemandSelectionConfig
) -> pd.DataFrame:
    """Apply predeclared gates without selecting or seeing any holdout results."""
    metric = config.primary_metric
    ranking = validation.metrics.loc[validation.metrics["horizon"] == 0].copy()
    ranking["eligible"] = True
    ranking["selection_reason"] = "eligible"
    baseline = ranking.loc[ranking["model_id"] == "seasonal_naive"].iloc[0]
    reference = float(baseline[metric])
    if not np.isfinite(reference):
        raise ContractViolation(
            f"Seasonal-naive validation {metric} is undefined; supply sufficient nonconstant "
            "training history or explicitly predeclare an alternative selection metric."
        )
    for index, row in ranking.iterrows():
        reason = "eligible"
        value = float(row[metric])
        if not np.isfinite(value):
            reason = "undefined_primary_metric"
        elif config.max_abs_bias_fraction is not None and (
            not np.isfinite(row["bias_fraction"])
            or abs(row["bias_fraction"]) > config.max_abs_bias_fraction
        ):
            reason = "bias_gate"
        elif row["model_id"] != "seasonal_naive" and not (
            value < reference and value <= reference * (1 - config.minimum_relative_improvement)
        ):
            reason = "does_not_improve_seasonal_naive"
        elif config.max_horizon_relative_degradation is not None:
            candidate = validation.metrics.loc[
                (validation.metrics["model_id"] == row["model_id"])
                & (validation.metrics["horizon"] > 0)
            ].set_index("horizon")[metric]
            seasonal = validation.metrics.loc[
                (validation.metrics["model_id"] == "seasonal_naive")
                & (validation.metrics["horizon"] > 0)
            ].set_index("horizon")[metric]
            if not np.all(np.isfinite(candidate.to_numpy())) or not np.all(
                candidate.to_numpy()
                <= seasonal.to_numpy() * (1 + config.max_horizon_relative_degradation)
            ):
                reason = "horizon_degradation_gate"
        ranking.loc[index, "eligible"] = reason == "eligible"
        ranking.loc[index, "selection_reason"] = reason
    return ranking.sort_values([metric, "model_id"], kind="stable").reset_index(drop=True)


def _select_on_validation(
    validation: DemandBacktestResult, config: DemandSelectionConfig
) -> tuple[DemandSelectionDecision, pd.DataFrame]:
    metric = config.primary_metric
    ranking = rank_demand_validation(validation, config)
    reference = float(ranking.loc[ranking["model_id"] == "seasonal_naive", metric].iloc[0])
    eligible = ranking.loc[ranking["eligible"]].sort_values([metric, "model_id"], kind="stable")
    if eligible.empty:
        raise ContractViolation(
            "No model passes validation selection gates; revise the design, not the holdout."
        )
    selected = eligible.iloc[0]
    decision = DemandSelectionDecision(
        selected_model=str(selected["model_id"]),
        primary_metric=metric,
        validation_metric=float(selected[metric]),
        seasonal_naive_metric=reference,
        selected_at=config.holdout.first_origin,
        selected_parameters=config.validation.model_parameters,
        reason="seasonal_baseline_retained"
        if selected["model_id"] == "seasonal_naive"
        else "validation_winner",
    )
    return decision, ranking.sort_values([metric, "model_id"], kind="stable").reset_index(drop=True)


def run_demand_selection(
    actuals: pd.DataFrame, config: DemandSelectionConfig
) -> DemandSelectionResult:
    """Select from rolling validation only, then evaluate fixed algorithms on holdout.

    At later holdout origins the selected algorithm is refit using newly arrived
    history, as in reforecasting. Model identity and parameters are never
    reselected using holdout errors.
    """
    validation = run_demand_backtest(actuals, config.validation)
    decision, ranking = _select_on_validation(validation, config)
    holdout = evaluate_demand_holdout(actuals, config, validation, decision)
    return DemandSelectionResult(
        decision=decision, ranking=ranking, validation=validation, holdout=holdout
    )


def evaluate_demand_holdout(
    actuals: pd.DataFrame,
    config: DemandSelectionConfig,
    validation: DemandBacktestResult,
    decision: DemandSelectionDecision,
) -> DemandBacktestResult:
    """Refit only the frozen winner and anchors on the validation's fixed cohort."""
    scope = tuple(
        DemandSeriesKey(**dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)))
        for key in validation.predictions[DEMAND_SERIES_COLUMNS]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    fixed_models = tuple(dict.fromkeys(("naive", "seasonal_naive", decision.selected_model)))
    holdout_config = config.holdout.model_copy(update={"models": fixed_models, "series": scope})
    holdout = run_demand_backtest(actuals, holdout_config)
    return holdout
