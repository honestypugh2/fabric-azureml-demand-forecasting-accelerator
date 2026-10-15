"""Exact statistical forecasts, seasonal scaling and holdout isolation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    DemandForecastRequest,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_metrics import (
    scaled_error_summary,
    seasonal_training_scales,
)
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    _select_on_validation,
    run_demand_selection,
)
from revenue_prediction.core.features.demand import DemandFeatureConfig
from revenue_prediction.core.models.demand_forecasters import (
    DEFAULT_DEMAND_MODELS,
    TimeSeriesModelConfig,
    create_demand_forecaster,
    demand_model_capabilities,
    forecast_demand_models,
)
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    run_demand_backtest,
)

pytestmark = pytest.mark.unit


def history(values: list[float], frequency: str = "MS") -> pd.DataFrame:
    frame = generate_synthetic_demand_dataset(
        DemandDataConfig.model_validate(
            {
                "frequency": frequency,
                "n_periods": len(values),
                "n_skus": 1,
                "n_channels": 1,
                "n_geographies": 1,
            }
        )
    )
    frame["demand_units"] = np.asarray(values, dtype=float)
    return frame


def selection_config(**gates: object) -> DemandSelectionConfig:
    models = DEFAULT_DEMAND_MODELS
    validation = DemandBacktestConfig(
        first_origin=date(2022, 1, 1),
        evaluation_as_of=date(2022, 4, 1),
        models=models,
        horizon=2,
        n_origins=2,
    )
    holdout = DemandBacktestConfig(
        first_origin=date(2022, 4, 1),
        evaluation_as_of=date(2025, 1, 1),
        models=models,
        horizon=2,
        n_origins=2,
    )
    return DemandSelectionConfig.model_validate(
        {
            "validation": validation,
            "holdout": holdout,
            **gates,
        }
    )


def test_capability_registry_is_time_series_not_revenue_regression() -> None:
    capabilities = demand_model_capabilities()
    assert {item.model_id for item in capabilities} == {
        "naive",
        "seasonal_naive",
        "moving_average",
        "seasonal_moving_average",
        "drift",
        "ses",
        "holt_damped",
        "croston",
        "sba",
        "tsb",
        "ets",
        "arima",
        "sarima",
        "weighted_ensemble",
    }
    assert all(item.frequencies == ("MS", "W-MON") for item in capabilities)
    assert all(
        not item.supported_covariates and not item.supports_probabilistic_output
        for item in capabilities
    )
    assert all(item.local_execution and not item.azure_execution_validated for item in capabilities)
    with pytest.raises(ValueError, match="Unknown demand forecaster"):
        create_demand_forecaster("elastic_net")


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("naive", [4, 4, 4, 4]),
        ("seasonal_naive", [3, 4, 3, 4]),
        ("moving_average", [3, 3, 3, 3]),
        ("seasonal_moving_average", [2, 3, 2, 3]),
        ("drift", [5, 6, 7, 8]),
        ("ses", [3.125] * 4),
        ("holt_damped", [5, 6, 7, 8]),
    ],
)
def test_exact_fitted_time_series_algorithms(model_id: str, expected: list[float]) -> None:
    forecaster = create_demand_forecaster(
        model_id, TimeSeriesModelConfig(alpha=0.5, beta=0.5, damping=1)
    )
    forecaster.fit(history([1, 2, 3, 4]), "MS", 2)
    np.testing.assert_allclose(
        forecaster.predict(pd.date_range("2020-05-01", periods=4, freq="MS")), expected
    )


def test_damped_holt_matches_level_trend_recurrences() -> None:
    values = [10.0, 12.0, 11.0, 15.0]
    alpha, beta, phi = 0.4, 0.3, 0.8
    level, trend = values[0], values[1] - values[0]
    for value in values[1:]:
        previous = level
        level = alpha * value + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - previous) + (1 - beta) * phi * trend
    expected = [level + trend * sum(phi**step for step in range(1, h + 1)) for h in (1, 2, 3)]
    forecaster = create_demand_forecaster(
        "holt_damped", TimeSeriesModelConfig(alpha=alpha, beta=beta, damping=phi)
    )
    forecaster.fit(history(values), "MS", 12)
    np.testing.assert_allclose(
        forecaster.predict(pd.date_range("2020-05-01", periods=3, freq="MS")), expected
    )


def test_actual_lead_accounts_for_delayed_final_observation() -> None:
    forecaster = create_demand_forecaster("drift")
    forecaster.fit(history([1, 2, 3]), "MS", 12)
    np.testing.assert_allclose(
        forecaster.predict(pd.date_range("2020-05-01", periods=2, freq="MS")), [5, 6]
    )


def test_fitted_state_is_not_mutated_by_caller_and_refit_is_independent() -> None:
    first = history([1, 2, 3, 4])
    forecaster = create_demand_forecaster("drift")
    forecaster.fit(first, "MS", 2)
    first["demand_units"] = 999.0
    np.testing.assert_allclose(
        forecaster.predict(pd.date_range("2020-05-01", periods=2, freq="MS")), [5, 6]
    )
    forecaster.fit(history([4, 3, 2, 1]), "MS", 2)
    np.testing.assert_allclose(
        forecaster.predict(pd.date_range("2020-05-01", periods=2, freq="MS")), [0, 0]
    )
    with pytest.raises(ContractViolation, match="at least"):
        forecaster.fit(history([1]), "MS", 2)
    with pytest.raises(RuntimeError, match="Fit the forecaster"):
        forecaster.predict(pd.date_range("2020-05-01", periods=1, freq="MS"))


@pytest.mark.parametrize(
    "model_id", ["drift", "ses", "holt_damped", "moving_average", "seasonal_moving_average"]
)
def test_statistical_models_reject_arrival_gaps(model_id: str) -> None:
    frame = history([1, 2, 3, 4, 5, 6])
    frame.loc[2, "actual_available_at"] = pd.Timestamp("2021-01-01")
    with pytest.raises(ContractViolation, match="contiguous"):
        forecast_demand_models(
            frame,
            DemandForecastRequest(forecast_origin=date(2020, 7, 1)),
            DemandFeatureConfig(seasonal_periods=2),
            models=(model_id,),
        )


def test_missing_history_and_bad_prediction_dates_are_explicit_errors() -> None:
    forecaster = create_demand_forecaster("ses")
    with pytest.raises(RuntimeError, match="Fit"):
        forecaster.predict(pd.date_range("2020-05-01", periods=1, freq="MS"))
    forecaster.fit(history([1, 2, 3, 4]), "MS", 2)
    for dates in [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2020-04-01"]),
        pd.DatetimeIndex(["2020-05-02"]),
        pd.DatetimeIndex(["2020-05-01", "2020-07-01"]),
    ]:
        with pytest.raises(ValueError):
            forecaster.predict(dates)


@pytest.mark.parametrize(
    "parameters",
    [
        {"alpha": 0},
        {"alpha": 1.1},
        {"beta": -1},
        {"damping": 0},
        {"alpha": float("nan")},
        {"average_periods": 0},
        {"average_seasons": True},
    ],
)
def test_invalid_smoothing_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TimeSeriesModelConfig.model_validate(parameters)


def test_scale_values_and_scaled_errors_are_exact() -> None:
    scales = seasonal_training_scales(history([1, 2, 4, 8]), "MS", 2)
    assert scales.mae_scale.iloc[0] == 4.5
    assert scales.mse_scale.iloc[0] == 22.5
    assert scales.n_scale_pairs.iloc[0] == 2
    group = pd.concat([scales, scales], ignore_index=True)
    group["error"] = [4.5, -9.0]
    summary = scaled_error_summary(group)
    assert summary["mase"] == 1.5
    assert summary["rmsse"] == pytest.approx(np.sqrt((4.5**2 + 9.0**2) / 45))


def test_scales_preserve_calendar_pairs_across_arrival_gaps() -> None:
    frame = history([1, 2, 4, 8, 16])
    scales = seasonal_training_scales(frame.drop(index=2), "MS", 2)
    assert scales.n_scale_pairs.iloc[0] == 1  # April minus February, not compressed rows
    assert scales.mae_scale.iloc[0] == 6.0


@pytest.mark.parametrize(
    ("values", "status"),
    [
        ([1, 2], "insufficient_seasonal_pairs"),
        ([5, 5, 5, 5], "zero_seasonal_scale"),
        ([0, 0, 0, 0], "zero_seasonal_scale"),
    ],
)
def test_undefined_scaled_metrics_not_zero_success(values: list[float], status: str) -> None:
    scales = seasonal_training_scales(history(values), "MS", 2)
    assert scales.scaled_metrics_status.iloc[0] == status
    scales["error"] = 0.0
    summary = scaled_error_summary(scales)
    assert np.isnan(summary["mase"])
    assert np.isnan(summary["rmsse"])
    assert summary["n_scaled_predictions"] == 0


def test_selection_is_time_series_validation_only_and_has_separate_holdout() -> None:
    result = run_demand_selection(history(list(np.arange(1.0, 61))), selection_config())
    assert result.decision.selected_model in {"drift", "holt_damped"}
    assert result.decision.primary_metric == "mase"
    assert result.decision.validation_metric == 0.0
    assert result.decision.seasonal_naive_metric == 1.0
    assert result.decision.approval_status == "unapproved"
    assert result.validation.predictions.date.max() < result.holdout.predictions.date.min()
    assert result.validation.predictions.actual_evaluated_available_at.max() <= pd.Timestamp(
        result.decision.selected_at
    )
    assert set(result.holdout.predictions.model_id) == {
        "naive",
        "seasonal_naive",
        result.decision.selected_model,
    }


def test_holdout_mutations_never_change_selection_or_validation_scales() -> None:
    frame = history(list(np.arange(1.0, 61)))
    config = selection_config()
    before = run_demand_selection(frame, config)
    frame.loc[frame.date >= pd.Timestamp(config.holdout.first_origin), "demand_units"] += 500.0
    after = run_demand_selection(frame, config)
    assert before.decision == after.decision
    pd.testing.assert_frame_equal(before.ranking, after.ranking)
    pd.testing.assert_frame_equal(before.validation.predictions, after.validation.predictions)
    assert not before.holdout.metrics.equals(after.holdout.metrics)


def test_undefined_mase_requires_explicit_metric_choice() -> None:
    frame = history([10.0] * 60)
    with pytest.raises(ContractViolation, match="validation mase is undefined"):
        run_demand_selection(frame, selection_config())
    result = run_demand_selection(frame, selection_config(primary_metric="wape"))
    assert result.decision.selected_model == "seasonal_naive"
    assert result.decision.reason == "seasonal_baseline_retained"


def test_configurable_gates_and_baseline_retention() -> None:
    frame = history(list(np.arange(1.0, 61)))
    result = run_demand_selection(
        frame,
        selection_config(
            primary_metric="rmsse",
            minimum_relative_improvement=0.99,
            max_abs_bias_fraction=0.0,
            max_horizon_relative_degradation=0.0,
        ),
    )
    assert result.decision.validation_metric == 0
    assert result.ranking.selection_reason.str.contains("bias_gate").any()
    noisy = history([10.0 + index % 12 for index in range(60)])
    result = run_demand_selection(noisy, selection_config(primary_metric="wape"))
    assert result.decision.selected_model == "seasonal_naive"


def test_invalid_temporal_selection_configs_rejected() -> None:
    config = selection_config()
    for update in [
        {"primary_metric": "r2"},
        {"minimum_relative_improvement": 1.0},
        {"holdout": config.holdout.model_copy(update={"first_origin": date(2022, 3, 1)})},
        {"validation": config.validation.model_copy(update={"evaluation_as_of": date(2023, 1, 1)})},
        {
            "holdout": config.holdout.model_copy(
                update={"model_parameters": TimeSeriesModelConfig(alpha=0.8)}
            )
        },
        {"validation": config.validation.model_copy(update={"models": ("ses",)})},
    ]:
        with pytest.raises(ValidationError):
            DemandSelectionConfig.model_validate({**config.model_dump(), **update})


def test_point_contract_accepts_registry_models_but_not_wrong_kind() -> None:
    frame = history(list(np.arange(1.0, 61)))
    request = DemandForecastRequest(forecast_origin=date(2022, 1, 1))
    forecast = forecast_demand_models(frame, request, models=("ses",))
    assert forecast.forecast_kind.eq("statistical").all()
    forecast["forecast_kind"] = "baseline"
    with pytest.raises(ContractViolation, match="forecast_kind"):
        validate_demand_forecasts(forecast, request)


def test_weekly_scaled_metrics_and_selection() -> None:
    frame = history(list(np.arange(1.0, 262)), "W-MON")
    first = frame.date.iloc[156].date()
    holdout_first = frame.date.iloc[160].date()
    models = ("naive", "seasonal_naive", "drift", "ses", "holt_damped", "seasonal_moving_average")
    config = DemandSelectionConfig(
        validation=DemandBacktestConfig(
            first_origin=first,
            evaluation_as_of=holdout_first,
            frequency="W-MON",
            models=models,
            horizon=2,
            n_origins=2,
        ),
        holdout=DemandBacktestConfig(
            first_origin=holdout_first,
            evaluation_as_of=frame.actual_available_at.max().date(),
            frequency="W-MON",
            models=models,
            horizon=2,
            n_origins=2,
        ),
    )
    result = run_demand_selection(frame, config)
    assert result.decision.validation_metric == 0.0
    assert result.decision.seasonal_naive_metric == 1.0
    assert result.validation.predictions.mae_scale.eq(52.0).all()


def test_partial_scale_coverage_cannot_rank_only_easy_series() -> None:
    valid = seasonal_training_scales(history([1, 2, 3, 4]), "MS", 2)
    invalid = seasonal_training_scales(history([5, 5, 5, 5]), "MS", 2)
    group = pd.concat([valid, invalid], ignore_index=True)
    group["error"] = [1.0, 1.0]
    result = scaled_error_summary(group)
    assert np.isnan(result["mase"])
    assert result["n_scaled_predictions"] == 1


def test_horizon_gate_rejects_aggregate_winner_that_degrades_one_horizon() -> None:
    config = selection_config(max_horizon_relative_degradation=0)
    validation = run_demand_backtest(history(list(np.arange(1.0, 61))), config.validation)
    mask = (validation.metrics.model_id == "drift") & (validation.metrics.horizon == 2)
    validation.metrics.loc[mask, "mase"] = 2.0
    decision, ranking = _select_on_validation(validation, config)
    assert (
        ranking.loc[ranking.model_id == "drift", "selection_reason"].iloc[0]
        == "horizon_degradation_gate"
    )
    assert decision.selected_model != "drift"


def test_failed_business_gates_never_silently_approve_a_baseline() -> None:
    frame = generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=60, n_skus=1, n_channels=1, n_geographies=1)
    )
    with pytest.raises(ContractViolation, match="No model passes"):
        run_demand_selection(frame, selection_config(max_abs_bias_fraction=0))


def test_extreme_training_scale_fails_explicitly_and_season_must_be_positive() -> None:
    with pytest.raises(ContractViolation, match="scales overflowed"):
        seasonal_training_scales(history([1.0, 2.0, 1e200, 2e200]), "MS", 2)
    with pytest.raises(ValueError, match="positive"):
        seasonal_training_scales(history([1, 2, 3, 4]), "MS", 0)


def test_tracking_signal_and_bias_percent_have_forecasting_sign_convention() -> None:
    config = selection_config()
    result = run_demand_backtest(history(list(np.arange(1.0, 61))), config.validation)
    for _, row in result.metrics.iterrows():
        assert row.bias_percent == pytest.approx(100 * row.bias_fraction)
        if row.mae == 0:
            assert row.tracking_signal == 0
        else:
            assert row.tracking_signal == pytest.approx(
                row.mean_error * row.n_predictions / row.mae
            )
