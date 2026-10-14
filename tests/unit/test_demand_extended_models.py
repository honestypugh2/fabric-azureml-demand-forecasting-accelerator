"""Forecasting recurrences, statistical adapters, sparse data and temporal tuning."""

from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandForecastRequest
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_diagnostics import demand_training_diagnostics
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    run_demand_selection,
)
from revenue_prediction.core.evaluation.demand_tuning import DemandTuningConfig, run_demand_tuning
from revenue_prediction.core.models.demand_forecasters import (
    TimeSeriesModelConfig,
    create_demand_forecaster,
    forecast_demand_models,
)
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    run_demand_backtest,
)

pytestmark = pytest.mark.unit


def data(frequency: str = "MS", **kwargs: object) -> pd.DataFrame:
    return generate_synthetic_demand_dataset(
        DemandDataConfig.model_validate(
            {
                "frequency": frequency,
                "n_skus": 1,
                "n_channels": 1,
                "n_geographies": 1,
                **kwargs,
            }
        )
    )


def experiment(frame: pd.DataFrame, models: tuple[str, ...]) -> DemandSelectionConfig:
    dates = frame.date.drop_duplicates().sort_values()
    common = {"models": models, "frequency": frame.frequency.iloc[0], "horizon": 2, "n_origins": 2}
    return DemandSelectionConfig(
        validation=DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[-10].date(),
                "evaluation_as_of": dates.iloc[-5].date(),
            }
        ),
        holdout=DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[-5].date(),
                "evaluation_as_of": frame.actual_available_at.max().date(),
            }
        ),
    )


@pytest.mark.parametrize(
    ("model_id", "expected"), [("croston", 6.0), ("sba", 4.5), ("tsb", 8.4375)]
)
def test_intermittent_exact_recurrences(model_id: str, expected: float) -> None:
    frame = data(n_periods=5)
    frame["demand_units"] = [0.0, 10.0, 0.0, 0.0, 20.0]
    model = create_demand_forecaster(model_id, TimeSeriesModelConfig(alpha=0.5, beta=0.5))
    model.fit(frame, "MS", 2)
    np.testing.assert_allclose(
        model.predict(pd.date_range("2020-06-01", periods=4, freq="MS")), expected
    )


@pytest.mark.parametrize("model_id", ["croston", "sba", "tsb"])
def test_all_zero_is_explicit_valid_intermittent_state(model_id: str) -> None:
    frame = data(n_periods=8)
    frame["demand_units"] = 0.0
    model = create_demand_forecaster(model_id)
    model.fit(frame, "MS", 2)
    np.testing.assert_array_equal(
        model.predict(pd.date_range("2020-09-01", periods=3, freq="MS")), [0, 0, 0]
    )


def test_tsb_decays_during_obsolescence_but_croston_does_not() -> None:
    frame = data(n_periods=8)
    frame["demand_units"] = [10.0, 10.0, *([0.0] * 6)]
    forecasts = {}
    for model_id in ("croston", "sba", "tsb"):
        model = create_demand_forecaster(model_id, TimeSeriesModelConfig(alpha=0.2, beta=0.5))
        model.fit(frame, "MS", 2)
        forecasts[model_id] = model.predict(pd.DatetimeIndex(["2020-09-01"]))[0]
    assert forecasts == pytest.approx({"croston": 10, "sba": 9, "tsb": 10 * 0.5**6})


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
@pytest.mark.parametrize("pattern", ["regular", "intermittent", "long_zero_runs", "obsolescence"])
def test_pattern_determinism_and_zero_run_contract(frequency: str, pattern: str) -> None:
    kwargs = {"demand_pattern": pattern, "zero_run_periods": 8, "occurrence_probability": 0.4}
    frame = data(frequency, **kwargs)
    pd.testing.assert_frame_equal(frame, data(frequency, **kwargs))
    if pattern == "regular":
        assert frame.demand_units.gt(0).all()
    else:
        assert frame.demand_units.eq(0).any()
        assert frame.demand_units.gt(0).any()
    if pattern == "long_zero_runs":
        assert frame.demand_units.iloc[:8].eq(0).all()
    if pattern == "obsolescence":
        assert frame.demand_units.iloc[int(len(frame) * 0.7) :].eq(0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"occurrence_probability": 0},
        {"occurrence_probability": 1.1},
        {"zero_run_periods": 0},
        {"obsolete_after_fraction": 1},
        {"demand_pattern": "invalid"},
        {"arima_order": (1.5, 0, 0)},
        {"arima_order": (True, 0, 0)},
        {"sarima_order": (6, 0, 0)},
        {"ets_trend": None},
        {"statistical_maxiter": 0},
    ],
)
def test_invalid_new_configuration_rejected(kwargs: dict[str, object]) -> None:
    model_keys = {"arima_order", "sarima_order", "ets_trend", "statistical_maxiter"}
    config = TimeSeriesModelConfig if model_keys.intersection(kwargs) else DemandDataConfig
    with pytest.raises(ValidationError):
        config.model_validate(kwargs)


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
@pytest.mark.parametrize("model_id", ["ets", "arima", "sarima"])
def test_statistical_adapter_matches_provider_and_calendar_lead(
    frequency: str, model_id: str
) -> None:
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    frame = data(frequency, n_periods=48)
    values = (
        20
        + np.arange(40) * 0.1
        + np.sin(np.arange(40) * np.pi / 2)
        + np.random.default_rng(4).normal(0, 0.1, 40)
    )
    training = frame.iloc[:40].copy()
    training["demand_units"] = values
    model = create_demand_forecaster(model_id)
    model.fit(training, frequency, 4)
    dates = pd.DatetimeIndex(frame.date.iloc[42:45])
    if model_id == "ets":
        fitted = ExponentialSmoothing(
            values,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=4,
            initialization_method="estimated",
        ).fit(optimized=True, minimize_kwargs={"options": {"maxiter": 200}})
    else:
        fitted = SARIMAX(
            values,
            order=(1, 1, 0),
            seasonal_order=(1, 0, 0, 4) if model_id == "sarima" else (0, 0, 0, 0),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False, maxiter=200)
    expected = np.maximum(np.asarray(fitted.forecast(5))[2:5], 0)
    np.testing.assert_allclose(model.predict(dates), expected, rtol=1e-8, atol=1e-8)
    assert model.predict(dates).shape == (3,)


@pytest.mark.parametrize("model_id", ["ets", "arima", "sarima", "croston", "sba", "tsb"])
def test_extended_models_do_not_observe_future_actuals(model_id: str) -> None:
    if model_id in {"ets", "arima", "sarima"}:
        pytest.importorskip("statsmodels")
    frame = data()
    request = DemandForecastRequest(forecast_origin=frame.date.iloc[48].date(), horizon=3)
    before = forecast_demand_models(frame, request, models=(model_id,))
    frame.loc[frame.actual_available_at > request.origin, "demand_units"] += 10000
    pd.testing.assert_frame_equal(
        before, forecast_demand_models(frame, request, models=(model_id,))
    )
    assert before.forecast_kind.eq("statistical").all()


def test_missing_statistical_dependency_is_actionable_and_invalidates_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("statsmodels"):
            raise ImportError("blocked optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    model = create_demand_forecaster("ets")
    with pytest.raises(ImportError, match="uv sync --extra forecasting"):
        model.fit(data().iloc[:36], "MS", 12)
    with pytest.raises(RuntimeError, match="Fit"):
        model.predict(pd.DatetimeIndex(["2023-01-01"]))


@pytest.mark.parametrize("model_id", ["ets", "sarima"])
def test_seasonal_eligibility_rejects_short_history_and_invalid_season(model_id: str) -> None:
    model = create_demand_forecaster(model_id)
    with pytest.raises(ContractViolation, match="at least"):
        model.fit(data().iloc[:10], "MS", 12)
    with pytest.raises(ContractViolation, match="at least two"):
        model.fit(data(), "MS", 1)


def test_statistical_nonconvergence_rejects_and_invalidates_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("statsmodels")
    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = create_demand_forecaster("ets")
    frame = data().iloc[:36]
    model.fit(frame, "MS", 12)

    def failed_fit(*args: object, **kwargs: object) -> None:
        warnings.warn("Did not converge", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(ExponentialSmoothing, "fit", failed_fit)
    with pytest.raises(ContractViolation, match="fit failed.*Did not converge"):
        model.fit(frame, "MS", 12)
    with pytest.raises(RuntimeError, match="Fit"):
        model.predict(pd.DatetimeIndex(["2023-01-01"]))


def test_statistical_nonfinite_output_is_not_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    class InvalidFit:
        def forecast(self, steps: int) -> np.ndarray:
            return np.full(steps, np.inf)

    monkeypatch.setattr(ExponentialSmoothing, "fit", lambda *args, **kwargs: InvalidFit())
    model = create_demand_forecaster("ets")
    model.fit(data().iloc[:36], "MS", 12)
    with pytest.raises(ContractViolation, match="nonfinite"):
        model.predict(pd.DatetimeIndex(["2023-01-01"]))


def test_training_diagnostics_do_not_use_future_and_describe_zero_runs() -> None:
    frame = data(demand_pattern="long_zero_runs")
    request = DemandForecastRequest(forecast_origin=frame.date.iloc[40].date())
    before = demand_training_diagnostics(frame, request, models=("tsb", "ets"))
    frame.loc[frame.actual_available_at > request.origin, "demand_units"] = 99999
    after = demand_training_diagnostics(frame, request, models=("tsb", "ets"))
    for first, second in zip(before, after, strict=True):
        pd.testing.assert_frame_equal(first, second)
    row = before[0].iloc[0]
    assert row.zero_fraction > 0.5
    assert row.longest_zero_run >= 6
    assert row.training_end < request.origin
    assert row.n_positive_periods < row.n_training_periods


def test_diagnostics_never_bridge_unavailable_calendar_periods() -> None:
    frame = data(n_periods=5)
    frame["demand_units"] = 0.0
    frame.loc[2, "actual_available_at"] = pd.Timestamp("2021-01-01")
    diagnostic, eligibility = demand_training_diagnostics(
        frame,
        DemandForecastRequest(forecast_origin=pd.Timestamp("2020-06-01").date()),
        models=("naive", "croston"),
    )
    row = diagnostic.iloc[0]
    assert not row.calendar_is_contiguous
    assert row.longest_zero_run == row.trailing_zero_run == 2
    assert row.n_training_periods == 4
    assert row.intermittency_status == "all_zero_training"
    assert np.isnan(row.average_demand_interval)
    assert np.isnan(row.positive_size_cv_squared)
    assert (
        eligibility.loc[eligibility.model_id == "croston", "eligibility_status"].iloc[0]
        == "ineligible"
    )


def test_tuning_selects_parameters_without_holdout_leakage_and_reproduces() -> None:
    frame = data()
    frame["demand_units"] = 20 + np.arange(60) + np.sin(np.arange(60) * np.pi / 6)
    frame.loc[frame.index >= 45, "demand_units"] = 200.0
    frame.loc[49, "demand_units"] = 230.0
    config = DemandTuningConfig(
        experiment=experiment(frame, ("naive", "seasonal_naive", "ses")),
        parameter_grid=(TimeSeriesModelConfig(alpha=0.1), TimeSeriesModelConfig(alpha=0.8)),
    )
    before = run_demand_tuning(frame, config)
    assert before.selected_trial == 1
    assert before.selection.decision.selected_model == "ses"
    assert before.selection.decision.selected_parameters.alpha == 0.8
    reproduced = run_demand_selection(frame, before.frozen_config)
    assert reproduced.decision == before.selection.decision
    pd.testing.assert_frame_equal(
        reproduced.holdout.predictions, before.selection.holdout.predictions
    )
    frame.loc[
        frame.date >= pd.Timestamp(config.experiment.holdout.first_origin), "demand_units"
    ] *= 10
    after = run_demand_tuning(frame, config)
    assert before.selection.decision == after.selection.decision
    assert before.frozen_config == after.frozen_config
    pd.testing.assert_frame_equal(before.trial_ranking, after.trial_ranking)
    pd.testing.assert_frame_equal(before.trial_metrics, after.trial_metrics)
    assert not before.selection.holdout.metrics.equals(after.selection.holdout.metrics)


def test_ineligible_series_fails_entire_trial_without_shrinking_cohort() -> None:
    frame = data(n_skus=2)
    dates = frame.date.drop_duplicates().sort_values()
    frame = frame.loc[~((frame.sku_id == "SKU-002") & (frame.date < dates.iloc[23]))].copy()
    config = DemandTuningConfig(
        experiment=experiment(frame, ("naive", "seasonal_naive", "moving_average", "ses")),
        parameter_grid=(TimeSeriesModelConfig(average_periods=40),),
    )
    result = run_demand_tuning(frame, config)
    failed = result.trial_ranking.loc[result.trial_ranking.model_id == "moving_average"].iloc[0]
    assert failed.selection_reason == "fit_or_evaluation_failed"
    assert not failed.eligible
    assert "40 available periods" in failed.failure
    assert result.selection.validation.predictions.sku_id.nunique() == 2
    assert "moving_average" not in result.frozen_config.validation.models


def test_empty_or_duplicate_grid_rejected() -> None:
    config = experiment(data(), ("naive", "seasonal_naive", "ses"))
    for grid in ((), (TimeSeriesModelConfig(), TimeSeriesModelConfig())):
        with pytest.raises(ValidationError):
            DemandTuningConfig(experiment=config, parameter_grid=grid)


def test_tuning_does_not_override_failed_business_gates() -> None:
    config = experiment(data(), ("naive", "seasonal_naive", "ses")).model_copy(
        update={"max_abs_bias_fraction": 0}
    )
    with pytest.raises(ContractViolation, match="No parameter/model trial passes"):
        run_demand_tuning(
            data(),
            DemandTuningConfig(
                experiment=config,
                parameter_grid=(TimeSeriesModelConfig(alpha=0.1), TimeSeriesModelConfig(alpha=0.8)),
            ),
        )


def test_segment_and_series_metrics_preserve_scales_and_row_counts() -> None:
    frame = data(n_skus=2, n_channels=2, demand_pattern="intermittent")
    result = run_demand_backtest(
        frame, experiment(frame, ("naive", "seasonal_naive", "tsb")).validation
    )
    portfolio = result.metrics.loc[result.metrics.horizon == 0]
    series = result.series_metrics.loc[result.series_metrics.horizon == 0]
    pd.testing.assert_series_equal(
        portfolio.set_index("model_id").n_predictions.sort_index(),
        series.groupby("model_id").n_predictions.sum().sort_index(),
    )
    assert set(result.segment_metrics.segment_type) == {
        "brand_id",
        "product_family_id",
        "product_id",
        "package_id",
        "channel_id",
        "geography_id",
    }
    assert {"mase", "rmsse", "bias_fraction", "scaled_metrics_status"}.issubset(
        result.series_metrics
    )
