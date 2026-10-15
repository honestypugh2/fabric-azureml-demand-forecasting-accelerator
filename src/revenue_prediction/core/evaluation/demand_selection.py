"""Validation-only time-series selection with a temporally separated holdout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    ENSEMBLE_FORECAST_SCHEMA_VERSION,
    DemandForecastRequest,
    DemandSeriesKey,
    demand_period_offset,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS
from revenue_prediction.core.models.demand_forecasters import (
    EnsembleWeight,
    TimeSeriesModelConfig,
    inverse_error_weights,
)
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    DemandBacktestResult,
    run_demand_backtest,
    summarise_demand_backtest,
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
    ensemble_members: tuple[str, ...] | None = Field(default=None, min_length=2)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result = handler(self)
        if self.ensemble_members is None:
            result.pop("ensemble_members", None)
        return result

    @property
    def validation_models(self) -> set[str]:
        return set(self.validation.models) | (
            {"weighted_ensemble"} if self.ensemble_members else set()
        )

    def holdout_models(self, selected_model: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    "naive",
                    "seasonal_naive",
                    selected_model,
                    *(self.ensemble_members or ()),
                    *(("weighted_ensemble",) if self.ensemble_members else ()),
                )
            )
        )

    @model_validator(mode="after")
    def _separated_evaluation(self) -> DemandSelectionConfig:
        validation, holdout = self.validation, self.holdout
        if "weighted_ensemble" in validation.models or validation.model_parameters.ensemble_weights:
            raise ValueError(
                "Predeclare ensemble_members, not learned weights or weighted_ensemble in selection models."
            )
        if self.ensemble_members is not None and (
            len(set(self.ensemble_members)) != len(self.ensemble_members)
            or not set(self.ensemble_members).issubset(validation.models)
        ):
            raise ValueError("Ensemble members must be unique, predeclared validation algorithms.")
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

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    selected_model: str
    primary_metric: str
    validation_metric: float
    seasonal_naive_metric: float
    selected_at: date
    reason: Literal["validation_winner", "seasonal_baseline_retained"]
    approval_status: Literal["unapproved"] = "unapproved"
    selected_parameters: TimeSeriesModelConfig = Field(default_factory=TimeSeriesModelConfig)
    ensemble_weights: tuple[EnsembleWeight, ...] = ()

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result = handler(self)
        if not self.ensemble_weights:
            result.pop("ensemble_weights", None)
        return result


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
    validation: DemandBacktestResult,
    config: DemandSelectionConfig,
    ensemble_weights: tuple[EnsembleWeight, ...] = (),
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
        selected_parameters=TimeSeriesModelConfig.model_validate(
            {
                **config.validation.model_parameters.model_dump(),
                "ensemble_weights": ensemble_weights,
            }
        )
        if selected["model_id"] == "weighted_ensemble"
        else config.validation.model_parameters,
        ensemble_weights=ensemble_weights,
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
    weights: tuple[EnsembleWeight, ...] = ()
    if config.ensemble_members is not None:
        weights = learn_demand_ensemble_weights(validation, config)
        validation = add_demand_ensemble_validation(validation, weights)
    decision, ranking = _select_on_validation(validation, config, weights)
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
    fixed_models = config.holdout_models(decision.selected_model)
    parameters = TimeSeriesModelConfig.model_validate(
        {
            **config.holdout.model_parameters.model_dump(),
            "ensemble_weights": decision.ensemble_weights,
        }
    )
    holdout_config = config.holdout.model_copy(
        update={"models": fixed_models, "series": scope, "model_parameters": parameters}
    )
    holdout = run_demand_backtest(actuals, holdout_config)
    return holdout


def learn_demand_ensemble_weights(
    validation: DemandBacktestResult, config: DemandSelectionConfig
) -> tuple[EnsembleWeight, ...]:
    """Learn one portfolio/horizon-shared blend from rolling out-of-sample errors."""
    if config.ensemble_members is None:
        raise ContractViolation("Ensemble members must be predeclared.")
    names = sorted(config.ensemble_members)
    metrics = validation.metrics.loc[validation.metrics.horizon == 0].set_index("model_id")
    errors = tuple(float(metrics.loc[name, config.primary_metric]) for name in names)
    weights = inverse_error_weights(errors)
    return tuple(
        EnsembleWeight(model_id=name, validation_error=error, weight=weight)
        for name, error, weight in zip(names, errors, weights, strict=True)
    )


def add_demand_ensemble_validation(
    validation: DemandBacktestResult, weights: tuple[EnsembleWeight, ...]
) -> DemandBacktestResult:
    """Blend identical complete out-of-sample grids; never drop a failed member."""
    TimeSeriesModelConfig(ensemble_weights=weights)
    keys = ["forecast_origin", *DEMAND_SERIES_COLUMNS, "horizon"]
    members = [
        validation.predictions.loc[validation.predictions.model_id == member.model_id]
        .set_index(keys)
        .sort_index()
        for member in weights
    ]
    if not members or any(
        frame.empty or not frame.index.is_unique or not frame.index.equals(members[0].index)
        for frame in members
    ):
        raise ContractViolation(
            "Ensemble validation requires every component on the identical complete grid."
        )
    common = [
        "date",
        "demand_units",
        "mae_scale",
        "mse_scale",
        "training_start",
        "training_end",
        "n_training_periods",
    ]
    if any(not frame[common].equals(members[0][common]) for frame in members[1:]):
        raise ContractViolation("Ensemble component truth/scales/training scope differ.")
    ensemble = members[0].copy()
    values = np.stack([frame.predicted_demand_units.to_numpy(dtype=float) for frame in members])
    if not np.isfinite(values).all():
        raise ContractViolation("Ensemble validation component predictions must be finite.")
    ensemble["predicted_demand_units"] = np.asarray([item.weight for item in weights]) @ values
    if not np.isfinite(ensemble.predicted_demand_units.to_numpy()).all():
        raise ContractViolation("Ensemble validation predictions overflowed.")
    ensemble["model_id"] = "weighted_ensemble"
    ensemble["forecast_kind"] = "ensemble"
    ensemble["schema_version"] = ENSEMBLE_FORECAST_SCHEMA_VERSION
    ensemble["source_date"] = ensemble["training_end"]
    ensemble["source_available_at"] = pd.concat(
        [frame.source_available_at for frame in members], axis=1
    ).max(axis=1)
    ensemble["error"] = ensemble.predicted_demand_units - ensemble.demand_units
    return summarise_demand_backtest(
        pd.concat([validation.predictions, ensemble.reset_index()], ignore_index=True)
    )
