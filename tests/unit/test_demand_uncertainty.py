"""Origin-safe interval calibration, temporal separation and uncertainty metrics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import FORECAST_COLUMNS, DemandForecastRequest
from revenue_prediction.core.data.demand_intervals import (
    INTERVAL_COLUMNS,
    validate_demand_intervals,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.evaluation.demand_uncertainty import (
    CALIBRATION_KEYS,
    DemandIntervalCalibration,
    DemandUncertaintyConfig,
    apply_demand_intervals,
    fit_demand_intervals,
    run_demand_uncertainty,
    summarise_demand_intervals,
)
from revenue_prediction.core.training.demand_backtesting import DemandBacktestConfig

pytestmark = pytest.mark.unit


def history_and_config(frequency: str = "MS") -> tuple[pd.DataFrame, DemandUncertaintyConfig]:
    actuals = generate_synthetic_demand_dataset(
        DemandDataConfig.model_validate(
            {
                "frequency": frequency,
                "n_periods": 60,
                "n_skus": 2,
                "n_channels": 1,
                "n_geographies": 1,
            }
        )
    )
    dates = actuals["date"].drop_duplicates().sort_values()
    common = {
        "frequency": frequency,
        "horizon": 2,
        "seasonal_periods": 4,
        "models": ("naive", "seasonal_naive", "drift"),
    }

    def block(start: int, cutoff: int, origins: int) -> DemandBacktestConfig:
        return DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[start].date(),
                "evaluation_as_of": dates.iloc[cutoff].date(),
                "n_origins": origins,
            }
        )

    return actuals, DemandUncertaintyConfig(
        selection=DemandSelectionConfig(validation=block(25, 29, 2), holdout=block(40, 45, 2)),
        calibration=block(30, 37, 5),
    )


@pytest.fixture(scope="module")
def experiment():
    actuals, config = history_and_config()
    return actuals, config, run_demand_uncertainty(actuals, config)


def known_residuals(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["predicted_demand_units"] = 10.0
    for _, group in result.groupby(CALIBRATION_KEYS, sort=True):
        result.loc[group.index, "demand_units"] = group[
            "predicted_demand_units"
        ].to_numpy() + np.arange(1, len(group) + 1)
    return result


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_temporal_workflow_preserves_points_and_reports_all_horizons(frequency: str) -> None:
    actuals, config = history_and_config(frequency)
    result = run_demand_uncertainty(actuals, config)
    pd.testing.assert_frame_equal(
        result.predictions[FORECAST_COLUMNS], result.holdout.predictions[FORECAST_COLUMNS]
    )
    assert result.decision.selected_at == config.calibration.first_origin
    assert result.calibration.calibration_as_of == config.calibration.evaluation_as_of
    assert set(result.predictions.model_id) == {
        "naive",
        "seasonal_naive",
        result.decision.selected_model,
    }
    assert result.predictions.approval_status.eq("unapproved").all()
    assert set(result.metrics.horizon) == {0, 1, 2}
    assert len(result.series_metrics) == len(result.metrics) * 2
    assert result.metrics.empirical_coverage.between(0, 1).all()
    assert (result.metrics.mean_interval_width >= 0).all()
    for origin, group in result.predictions.groupby("forecast_origin"):
        validate_demand_intervals(
            group[INTERVAL_COLUMNS],
            DemandForecastRequest(
                forecast_origin=origin.date(), frequency=config.calibration.frequency, horizon=2
            ),
        )


def test_order_statistic_json_roundtrip_and_no_pooling(experiment) -> None:
    _, config, result = experiment
    predictions = known_residuals(result.calibration_backtest.predictions)
    calibration = fit_demand_intervals(predictions, "MS", config.calibration.evaluation_as_of, 0.8)
    assert all(cell.radius == 5 and cell.n_predictions == 5 for cell in calibration.cells)
    restored = DemandIntervalCalibration.model_validate_json(calibration.model_dump_json())
    pd.testing.assert_frame_equal(
        apply_demand_intervals(result.holdout.predictions, calibration),
        apply_demand_intervals(result.holdout.predictions, restored),
    )
    first_origin = predictions.forecast_origin.min()
    four = predictions.loc[predictions.forecast_origin != first_origin]
    assert all(
        cell.n_predictions == 4
        for cell in fit_demand_intervals(four, "MS", config.calibration.evaluation_as_of).cells
    )
    with pytest.raises(ContractViolation, match="Insufficient"):
        fit_demand_intervals(four, "MS", config.calibration.evaluation_as_of, 0.9)
    incomplete = predictions.drop(predictions.index[0])
    with pytest.raises(ContractViolation):
        fit_demand_intervals(incomplete, "MS", config.calibration.evaluation_as_of)


def test_numeric_metrics_clipping_and_inclusive_coverage(experiment) -> None:
    _, config, result = experiment
    calibration = fit_demand_intervals(
        known_residuals(result.calibration_backtest.predictions),
        "MS",
        config.calibration.evaluation_as_of,
    )
    points = result.holdout.predictions.copy()
    points["predicted_demand_units"] = 2.0
    intervals = apply_demand_intervals(points, calibration)
    assert intervals.lower_demand_units.eq(0).all()
    assert intervals.upper_demand_units.eq(7).all()
    for column in ["actual_evaluated_available_at", "evaluation_as_of"]:
        intervals[column] = points[column]
    intervals["demand_units"] = 10.0
    metrics = summarise_demand_intervals(intervals)
    for column, expected in {
        "empirical_coverage": 0.0,
        "mean_interval_width": 7.0,
        "mean_interval_score": 37.0,
        "lower_pinball_loss": 1.0,
        "upper_pinball_loss": 2.7,
        "coverage_gap": -0.8,
    }.items():
        np.testing.assert_allclose(metrics[column], expected)
    intervals["demand_units"] = 7.0
    assert summarise_demand_intervals(intervals).empirical_coverage.eq(1).all()
    intervals["demand_units"] = 0.0
    assert summarise_demand_intervals(intervals).empirical_coverage.eq(1).all()


def test_holdout_and_future_changes_do_not_change_selection_or_calibration(experiment) -> None:
    actuals, config, original = experiment
    modified = actuals.copy()
    holdout_rows = modified.date >= pd.Timestamp(config.selection.holdout.first_origin)
    modified.loc[holdout_rows, "demand_units"] *= 10
    changed = run_demand_uncertainty(modified, config)
    assert changed.decision == original.decision
    assert changed.calibration == original.calibration
    pd.testing.assert_frame_equal(changed.ranking, original.ranking)
    first = pd.Timestamp(config.selection.holdout.first_origin)
    pd.testing.assert_frame_equal(
        original.predictions.loc[original.predictions.forecast_origin == first],
        changed.predictions.loc[changed.predictions.forecast_origin == first].assign(
            demand_units=original.predictions.loc[
                original.predictions.forecast_origin == first, "demand_units"
            ].to_numpy()
        ),
    )
    pd.testing.assert_frame_equal(
        original.predictions[["calibration_radius", "n_calibration_predictions"]],
        changed.predictions[["calibration_radius", "n_calibration_predictions"]],
    )
    assert not changed.metrics.equals(original.metrics)


def test_calibration_changes_do_not_reselect_model(experiment) -> None:
    actuals, config, original = experiment
    changed = actuals.copy()
    mask = (changed.date >= pd.Timestamp(config.calibration.first_origin)) & (
        changed.date <= pd.Timestamp(original.calibration.last_target_date)
    )
    changed.loc[mask, "demand_units"] *= 2
    result = run_demand_uncertainty(changed, config)
    assert result.decision == original.decision
    assert result.calibration != original.calibration


@pytest.mark.parametrize("field", ["first_origin", "evaluation_as_of", "models", "horizon"])
def test_reject_overlap_or_incompatible_calibration(experiment, field: str) -> None:
    _, config, _ = experiment
    payload = config.model_dump(mode="json")
    payload["calibration"][field] = {
        "first_origin": config.selection.validation.first_origin.isoformat(),
        "evaluation_as_of": config.selection.holdout.evaluation_as_of.isoformat(),
        "models": ["naive"],
        "horizon": 3,
    }[field]
    with pytest.raises(ValidationError):
        DemandUncertaintyConfig.model_validate(payload)


def test_insufficient_calibration_and_invalid_state(experiment) -> None:
    _, config, result = experiment
    payload = config.model_dump(mode="json")
    payload["nominal_coverage"] = 0.99
    with pytest.raises(ValidationError, match="Too few"):
        DemandUncertaintyConfig.model_validate(payload)
    for update in [
        {"cells": []},
        {"cells": [result.calibration.cells[0].model_dump()] * 2},
        {"nominal_coverage": float("nan")},
        {"calibration_as_of": result.calibration.last_target_date},
        {"last_target_date": date(2022, 1, 2)},
        {"nominal_coverage": 0.999},
    ]:
        with pytest.raises(ValidationError):
            DemandIntervalCalibration.model_validate({**result.calibration.model_dump(), **update})


@pytest.mark.parametrize(
    "column,value",
    [
        ("actual_evaluated_available_at", pd.Timestamp("2099-01-01")),
        ("actual_evaluated_available_at", pd.NaT),
        ("demand_units", float("inf")),
        ("demand_units", -1.0),
    ],
)
def test_reject_unavailable_or_invalid_calibration_truth(experiment, column: str, value) -> None:
    _, config, result = experiment
    predictions = result.calibration_backtest.predictions.copy()
    predictions.loc[predictions.index[0], column] = value
    with pytest.raises(ContractViolation):
        fit_demand_intervals(predictions, "MS", config.calibration.evaluation_as_of)


def test_reject_future_calibration_and_changed_cohort(experiment) -> None:
    _, config, result = experiment
    with pytest.raises(ContractViolation, match="precede"):
        apply_demand_intervals(result.calibration_backtest.predictions, result.calibration)
    other_scope = result.holdout.predictions.loc[
        result.holdout.predictions.sku_id == result.holdout.predictions.sku_id.iloc[0]
    ]
    with pytest.raises(ContractViolation, match="scope"):
        apply_demand_intervals(other_scope, result.calibration)
    with pytest.raises(ContractViolation, match="cutoff"):
        fit_demand_intervals(result.calibration_backtest.predictions, "MS", date(2030, 1, 1))
    for coverage in [0, 1, float("nan")]:
        with pytest.raises(ContractViolation):
            fit_demand_intervals(
                result.calibration_backtest.predictions,
                "MS",
                config.calibration.evaluation_as_of,
                coverage,
            )
    empty = result.holdout.predictions.iloc[:0]
    for operation in [
        lambda: apply_demand_intervals(empty, result.calibration),
        lambda: summarise_demand_intervals(empty),
        lambda: fit_demand_intervals(empty, "MS", config.calibration.evaluation_as_of),
    ]:
        with pytest.raises(ContractViolation):
            operation()


@pytest.mark.parametrize(
    "column,value",
    [
        ("upper_demand_units", -1.0),
        ("lower_demand_units", float("nan")),
        ("nominal_coverage", 1.0),
        ("calibration_radius", -1.0),
        ("n_calibration_predictions", 0),
        ("interval_schema_version", "future"),
        ("calibration_as_of", pd.Timestamp("2099-01-01")),
    ],
)
def test_interval_contract_rejects_invalid_rows(experiment, column: str, value) -> None:
    _, _, result = experiment
    origin = result.predictions.forecast_origin.min()
    rows = result.predictions.loc[
        result.predictions.forecast_origin == origin, INTERVAL_COLUMNS
    ].copy()
    rows.loc[rows.index[0], column] = value
    request = DemandForecastRequest(forecast_origin=origin.date(), horizon=2)
    with pytest.raises(ContractViolation):
        validate_demand_intervals(rows, request)
    with pytest.raises(ContractViolation, match="columns"):
        validate_demand_intervals(rows.assign(extra=1), request)


def test_no_pooled_calibration_across_series_or_horizons(experiment) -> None:
    _, config, result = experiment
    predictions = known_residuals(result.calibration_backtest.predictions)
    mask = predictions.sku_id.eq(predictions.sku_id.iloc[0]) & predictions.horizon.eq(2)
    predictions.loc[mask, "demand_units"] += 100
    calibration = fit_demand_intervals(predictions, "MS", config.calibration.evaluation_as_of)
    for cell in calibration.cells:
        expected = (
            105 if cell.series.sku_id == predictions.sku_id.iloc[0] and cell.horizon == 2 else 5
        )
        assert cell.radius == expected
    metrics = summarise_demand_intervals(result.predictions, DEMAND_SERIES_COLUMNS)
    assert set(metrics.sku_id) == set(predictions.sku_id)


def test_exact_ninety_percent_rank_and_zero_residuals(experiment) -> None:
    _, config, result = experiment
    predictions = known_residuals(result.calibration_backtest.predictions)
    extra = predictions.loc[predictions.forecast_origin < predictions.forecast_origin.max()].copy()
    extra["forecast_origin"] -= pd.offsets.MonthBegin(10)
    extra["date"] -= pd.offsets.MonthBegin(10)
    for column in ["source_date", "training_start", "training_end"]:
        extra[column] -= pd.offsets.MonthBegin(10)
    extra["source_available_at"] -= pd.offsets.MonthBegin(10)
    extra["actual_evaluated_available_at"] -= pd.offsets.MonthBegin(10)
    nine = pd.concat([extra, predictions], ignore_index=True).sort_values(
        ["forecast_origin", *CALIBRATION_KEYS], kind="stable"
    )
    calibration = fit_demand_intervals(nine, "MS", config.calibration.evaluation_as_of, 0.9)
    assert all(cell.n_predictions == 9 for cell in calibration.cells)
    assert all(cell.radius == 5 for cell in calibration.cells)
    zero = predictions.assign(predicted_demand_units=0.0, demand_units=0.0)
    fitted = fit_demand_intervals(zero, "MS", config.calibration.evaluation_as_of)
    assert all(cell.radius == 0 for cell in fitted.cells)
    points = result.holdout.predictions.assign(predicted_demand_units=0.0, demand_units=0.0)
    intervals = apply_demand_intervals(points, fitted)
    for column in ["demand_units", "evaluation_as_of", "actual_evaluated_available_at"]:
        intervals[column] = points[column]
    metrics = summarise_demand_intervals(intervals)
    assert metrics.empirical_coverage.eq(1).all()
    assert metrics.mean_interval_width.eq(0).all()
    assert metrics.mean_interval_score.eq(0).all()


def test_missing_whole_series_or_origin_provenance_is_rejected(experiment) -> None:
    _, config, result = experiment
    predictions = result.calibration_backtest.predictions
    missing_series = predictions.loc[
        ~(
            predictions.sku_id.eq(predictions.sku_id.iloc[0])
            & predictions.forecast_origin.eq(predictions.forecast_origin.min())
        )
    ]
    with pytest.raises(ContractViolation, match="cohort"):
        fit_demand_intervals(missing_series, "MS", config.calibration.evaluation_as_of)
    missing_origin = predictions.copy()
    missing_origin.loc[missing_origin.index[0], "forecast_origin"] = pd.NaT
    with pytest.raises(ContractViolation, match="origins"):
        fit_demand_intervals(missing_origin, "MS", config.calibration.evaluation_as_of)


def test_overflow_is_explicit_not_infinite_success(experiment) -> None:
    _, _, result = experiment
    huge = result.calibration.model_copy(
        update={
            "cells": tuple(
                cell.model_copy(update={"radius": 1e308}) for cell in result.calibration.cells
            )
        }
    )
    with pytest.raises(ContractViolation, match="bounds overflowed"):
        apply_demand_intervals(
            result.holdout.predictions.assign(predicted_demand_units=1e308), huge
        )
    with pytest.raises(ContractViolation, match="metrics overflowed"):
        summarise_demand_intervals(result.predictions.assign(demand_units=1e308))


def test_delayed_calibration_truth_fails_without_cohort_shrinkage(experiment) -> None:
    actuals, config, _ = experiment
    delayed = actuals.copy()
    first = delayed.date.eq(pd.Timestamp(config.calibration.first_origin))
    delayed.loc[first, "actual_available_at"] = pd.Timestamp(
        config.selection.holdout.evaluation_as_of
    )
    with pytest.raises(ContractViolation, match="complete evaluation actuals"):
        run_demand_uncertainty(delayed, config)
