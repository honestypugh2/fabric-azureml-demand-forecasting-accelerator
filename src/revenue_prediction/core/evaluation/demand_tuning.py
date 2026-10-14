"""Predeclared parameter grids evaluated only at temporal validation origins."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandSeriesKey
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    DemandSelectionResult,
    _select_on_validation,
    evaluate_demand_holdout,
    rank_demand_validation,
)
from revenue_prediction.core.models.demand_forecasters import TimeSeriesModelConfig
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestResult,
    run_demand_backtest,
    summarise_demand_backtest,
)


class DemandTuningConfig(BaseModel):
    """Fix the complete grid and evaluation design before examining holdout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: DemandSelectionConfig
    parameter_grid: tuple[TimeSeriesModelConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_trials(self) -> DemandTuningConfig:
        if len(set(self.parameter_grid)) != len(self.parameter_grid):
            raise ValueError("Parameter-grid trials must be unique.")
        return self


@dataclass(frozen=True)
class DemandTuningResult:
    """Complete trial/gate evidence, selected configuration and untouched holdout."""

    selection: DemandSelectionResult
    frozen_config: DemandSelectionConfig
    trial_ranking: pd.DataFrame
    trial_metrics: pd.DataFrame
    selected_trial: int


def run_demand_tuning(actuals: pd.DataFrame, config: DemandTuningConfig) -> DemandTuningResult:
    """Tune global parameters/model identity; do not tune independently per series.

    A candidate must forecast every validation origin/series or the whole
    candidate trial fails. Failures remain in the trial ranking; they never
    reduce the shared cohort. Missing optional dependencies abort with setup
    instructions rather than being treated as evidence against an algorithm.
    """
    experiment = config.experiment
    anchors = run_demand_backtest(
        actuals, experiment.validation.model_copy(update={"models": ("naive", "seasonal_naive")})
    )
    scope = tuple(
        DemandSeriesKey(**dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)))
        for key in anchors.predictions[DEMAND_SERIES_COLUMNS]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    rankings: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    validations: list[DemandBacktestResult] = []
    experiments: list[DemandSelectionConfig] = []
    for trial, parameters in enumerate(config.parameter_grid):
        results = [anchors]
        failures: list[dict[str, object]] = []
        successful = ["naive", "seasonal_naive"]
        for model_id in experiment.validation.models:
            if model_id in successful:
                continue
            candidate = experiment.validation.model_copy(
                update={"models": (model_id,), "model_parameters": parameters, "series": scope}
            )
            try:
                evaluation = run_demand_backtest(actuals, candidate)
            except ContractViolation as exc:
                failures.append(
                    {
                        "model_id": model_id,
                        "eligible": False,
                        "selection_reason": "fit_or_evaluation_failed",
                        "failure": str(exc),
                    }
                )
                continue
            results.append(evaluation)
            successful.append(model_id)
        validation = summarise_demand_backtest(
            pd.concat([item.predictions for item in results], ignore_index=True)
        )
        frozen = experiment.model_copy(
            update={
                "validation": experiment.validation.model_copy(
                    update={"models": tuple(successful), "model_parameters": parameters}
                ),
                "holdout": experiment.holdout.model_copy(
                    update={"models": tuple(successful), "model_parameters": parameters}
                ),
            }
        )
        ranking = rank_demand_validation(validation, frozen)
        ranking["failure"] = "not_applicable"
        if failures:
            ranking = pd.concat([ranking, pd.DataFrame(failures)], ignore_index=True)
        ranking["trial"] = trial
        ranking["parameters_json"] = parameters.model_dump_json()
        rankings.append(ranking)
        metrics.append(validation.metrics.assign(trial=trial))
        validations.append(validation)
        experiments.append(frozen)
    trial_ranking = pd.concat(rankings, ignore_index=True)
    eligible = trial_ranking.loc[trial_ranking["eligible"]].sort_values(
        [experiment.primary_metric, "model_id", "trial"], kind="stable"
    )
    if eligible.empty:
        raise ContractViolation(
            "No parameter/model trial passes validation gates; revise the grid, not the holdout."
        )
    selected_trial = int(eligible.iloc[0]["trial"])
    frozen = experiments[selected_trial]
    validation = validations[selected_trial]
    decision, ranking = _select_on_validation(validation, frozen)
    holdout = evaluate_demand_holdout(actuals, frozen, validation, decision)
    return DemandTuningResult(
        selection=DemandSelectionResult(decision, ranking, validation, holdout),
        frozen_config=frozen,
        trial_ranking=trial_ranking,
        trial_metrics=pd.concat(metrics, ignore_index=True),
        selected_trial=selected_trial,
    )
