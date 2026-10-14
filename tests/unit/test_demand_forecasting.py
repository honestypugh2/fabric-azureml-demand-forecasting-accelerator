"""Exact calendar baselines, leakage invariance and temporal evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    DemandForecastRequest,
    DemandSeriesKey,
)
from revenue_prediction.core.data.demand_schema import (
    DEMAND_SERIES_COLUMNS,
    DemandDataConfig,
)
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.features.demand import (
    DemandFeatureConfig,
    available_demand_history,
    build_demand_features,
)
from revenue_prediction.core.models.demand_baselines import forecast_demand_baselines
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    run_demand_backtest,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def actuals() -> pd.DataFrame:
    frame = generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=24, n_skus=2, n_channels=1, n_geographies=1)
    )
    for index, (_, series) in enumerate(frame.groupby(DEMAND_SERIES_COLUMNS)):
        frame.loc[series.index, "demand_units"] = np.arange(1, 25, dtype=float) + index * 100
    return frame


def request(**values: object) -> DemandForecastRequest:
    return DemandForecastRequest.model_validate({"forecast_origin": "2021-01-01", **values})


def backtest_config(**values: object) -> DemandBacktestConfig:
    return DemandBacktestConfig.model_validate(
        {
            "first_origin": "2021-01-01",
            "evaluation_as_of": "2022-01-01",
            "n_origins": 3,
            "horizon": 2,
            **values,
        }
    )


@pytest.mark.parametrize(
    ("values", "first"),
    [
        ({}, "2021-01-01"),
        ({"forecast_origin": "2021-01-15"}, "2021-02-01"),
        ({"forecast_origin": "2020-02-29"}, "2020-03-01"),
        ({"forecast_origin": "2021-01-04", "frequency": "W-MON"}, "2021-01-04"),
        ({"forecast_origin": "2021-01-06", "frequency": "W-MON"}, "2021-01-11"),
    ],
)
def test_forecast_origin_calendar(values: dict[str, object], first: str) -> None:
    configured = request(horizon=4, **values)
    assert len(configured.forecast_dates) == 4
    assert configured.forecast_dates[0] == pd.Timestamp(first)
    assert configured.forecast_dates[0] >= configured.origin


@pytest.mark.parametrize(
    "values",
    [
        {"horizon": 0},
        {"horizon": True},
        {"horizon": 1.5},
        {"frequency": "D"},
        {"forecast_origin": "2021-01-01T01:00:00"},
        {"series": []},
        {"unknown_feature": 1},
        {"series": [{"sku_id": "bad", "channel_id": "CHANNEL-001", "geography_id": "REGION-001"}]},
        {
            "series": [
                DemandSeriesKey(
                    sku_id="SKU-001", channel_id="CHANNEL-001", geography_id="REGION-001"
                )
            ]
            * 2
        },
    ],
)
def test_invalid_forecast_requests_fail(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        request(**values)


def test_exact_multiseries_naive_and_seasonal_forecasts(actuals: pd.DataFrame) -> None:
    forecast = forecast_demand_baselines(actuals, request(horizon=14))
    assert list(forecast.columns) == FORECAST_COLUMNS
    assert len(forecast) == 2 * 2 * 14
    naive = forecast.loc[forecast.model_id == "naive"]
    seasonal = forecast.loc[forecast.model_id == "seasonal_naive"]
    assert naive.loc[naive.sku_id == "SKU-001", "predicted_demand_units"].tolist() == [12.0] * 14
    assert naive.loc[naive.sku_id == "SKU-002", "predicted_demand_units"].tolist() == [112.0] * 14
    assert seasonal.loc[seasonal.sku_id == "SKU-001", "predicted_demand_units"].tolist() == [
        *np.arange(1.0, 13.0),
        1.0,
        2.0,
    ]
    assert seasonal.loc[seasonal.sku_id == "SKU-002", "predicted_demand_units"].tolist() == [
        *np.arange(101.0, 113.0),
        101.0,
        102.0,
    ]
    assert (forecast.source_available_at <= forecast.forecast_origin).all()
    assert (forecast.source_date < forecast.date).all()
    assert forecast.n_training_periods.eq(12).all()


def test_origin_frozen_lags_and_rolling_mean(actuals: pd.DataFrame) -> None:
    frame = build_demand_features(actuals, request(horizon=3))
    one = frame.loc[frame.sku_id == "SKU-001"]
    assert one.lag_1.tolist() == [12.0] * 3
    assert one.rolling_mean.tolist() == [11.0] * 3
    assert one.seasonal_value.tolist() == [1.0, 2.0, 3.0]
    assert one.month.tolist() == [1, 2, 3]
    assert {"unit_price", "promotion", "demand_units", "actual_available_at"}.isdisjoint(
        frame.columns
    )


def test_future_actual_and_realized_covariate_mutations_do_not_change_features(
    actuals: pd.DataFrame,
) -> None:
    before = actuals.copy(deep=True)
    features = build_demand_features(actuals, request(horizon=3))
    forecast = forecast_demand_baselines(actuals, request(horizon=3))
    changed = actuals.copy()
    future = changed.actual_available_at > request().origin
    changed.loc[future, "demand_units"] += 1e6
    changed.loc[future, "unit_price"] += 1000.0
    changed.loc[future, "promotion"] = ~changed.loc[future, "promotion"]
    pd.testing.assert_frame_equal(features, build_demand_features(changed, request(horizon=3)))
    pd.testing.assert_frame_equal(forecast, forecast_demand_baselines(changed, request(horizon=3)))
    pd.testing.assert_frame_equal(actuals, before)


def test_delayed_arrival_does_not_shift_calendar_lags(actuals: pd.DataFrame) -> None:
    actuals.loc[actuals.date == pd.Timestamp("2020-12-01"), "actual_available_at"] = pd.Timestamp(
        "2021-01-02"
    )
    features = build_demand_features(actuals, request())
    assert features.last_value.tolist() == [11.0, 111.0]
    assert features.lag_1.isna().all()
    assert features.rolling_mean.isna().all()
    forecast = forecast_demand_baselines(actuals, request(), models=("naive",))
    assert forecast.predicted_demand_units.tolist() == [11.0, 111.0]
    actuals.loc[actuals.date == pd.Timestamp("2020-01-01"), "actual_available_at"] = pd.Timestamp(
        "2021-01-02"
    )
    with pytest.raises(ContractViolation, match="seasonal_naive lacks"):
        forecast_demand_baselines(actuals, request())


def test_history_window_is_calendar_based_not_row_count(actuals: pd.DataFrame) -> None:
    config = DemandFeatureConfig(training_window_periods=3)
    actuals.loc[actuals.date == pd.Timestamp("2020-11-01"), "actual_available_at"] = pd.Timestamp(
        "2021-02-01"
    )
    history = available_demand_history(actuals, request(), config)
    assert set(history.date) == {pd.Timestamp("2020-10-01"), pd.Timestamp("2020-12-01")}
    assert len(history) == 4
    features = build_demand_features(actuals, request(), config)
    assert features.n_training_periods.eq(2).all()
    assert features.rolling_mean.isna().all()
    with pytest.raises(ContractViolation, match="seasonal_naive lacks"):
        forecast_demand_baselines(actuals, request(), config)


def test_explicit_scope_and_unavailable_series_fail(actuals: pd.DataFrame) -> None:
    series = DemandSeriesKey(sku_id="SKU-002", channel_id="CHANNEL-001", geography_id="REGION-001")
    forecast = forecast_demand_baselines(actuals, request(series=(series,)))
    assert forecast.sku_id.tolist() == ["SKU-002"] * 2
    missing = series.model_copy(update={"sku_id": "SKU-999"})
    with pytest.raises(ContractViolation, match="Requested series lack"):
        forecast_demand_baselines(actuals, request(series=(missing,)))
    with pytest.raises(ContractViolation, match="No actuals available"):
        forecast_demand_baselines(actuals, request(forecast_origin="2020-01-01"))
    with pytest.raises(ContractViolation, match="frequency differs"):
        forecast_demand_baselines(actuals, request(frequency="W-MON"))


def test_future_launch_does_not_change_origin_scope(actuals: pd.DataFrame) -> None:
    extra = generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=24, n_skus=3, n_channels=1, n_geographies=1)
    )
    extra = extra.loc[(extra.sku_id == "SKU-003") & (extra.date >= pd.Timestamp("2021-01-01"))]
    enlarged = pd.concat([actuals, extra], ignore_index=True)
    pd.testing.assert_frame_equal(
        forecast_demand_baselines(actuals, request()),
        forecast_demand_baselines(enlarged, request()),
    )


def test_weekly_season_and_origin_boundary() -> None:
    frame = generate_synthetic_demand_dataset(
        DemandDataConfig(frequency="W-MON", n_periods=60, n_skus=1, n_channels=1, n_geographies=1)
    )
    frame["demand_units"] = np.arange(1.0, 61.0)
    origin = frame.date.iloc[52].date()
    forecast = forecast_demand_baselines(
        frame, DemandForecastRequest(frequency="W-MON", forecast_origin=origin, horizon=54)
    )
    assert (
        forecast.loc[forecast.model_id == "naive", "predicted_demand_units"].tolist() == [52.0] * 54
    )
    assert forecast.loc[
        forecast.model_id == "seasonal_naive", "predicted_demand_units"
    ].tolist() == [
        *np.arange(1.0, 53.0),
        1.0,
        2.0,
    ]


@pytest.mark.parametrize("models", [(), ("naive", "naive"), ("bad",)])
def test_invalid_model_selection_rejected(actuals: pd.DataFrame, models: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="models must"):
        forecast_demand_baselines(actuals, request(), models=models)


@pytest.mark.parametrize(
    "values",
    [
        {"strategy": "random"},
        {"n_origins": 0},
        {"step_periods": 0},
        {"horizon": -1},
        {"first_origin": "2021-01-15"},
        {"evaluation_as_of": "2020-01-01"},
        {"strategy": "sliding"},
        {"training_window_periods": 12},
        {"models": []},
        {"models": ["naive", "naive"]},
        {"models": ["unknown"]},
        {"seasonal_periods": 0},
    ],
)
def test_invalid_backtest_config(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        backtest_config(**values)


@pytest.mark.parametrize("strategy", ["expanding", "sliding"])
def test_backtest_windows_row_counts_and_exact_metrics(
    actuals: pd.DataFrame, strategy: str
) -> None:
    values: dict[str, object] = {"strategy": strategy}
    if strategy == "sliding":
        values["training_window_periods"] = 12
    result = run_demand_backtest(actuals, backtest_config(**values))
    assert len(result.predictions) == 3 * 2 * 2 * 2
    assert len(result.metrics) == 2 * 3
    naive = result.predictions.loc[result.predictions.model_id == "naive"]
    seasonal = result.predictions.loc[result.predictions.model_id == "seasonal_naive"]
    assert naive.error.tolist() == [-1.0, -2.0] * 6
    assert seasonal.error.eq(-12.0).all()
    metric = result.metrics.loc[
        (result.metrics.model_id == "naive") & (result.metrics.horizon == 0)
    ].iloc[0]
    assert metric.mae == 1.5
    assert metric.rmse == pytest.approx(np.sqrt(2.5))
    assert metric.mean_error == -1.5
    assert metric.wape == pytest.approx(18.0 / 774.0)
    assert metric.bias_fraction == pytest.approx(-18.0 / 774.0)
    counts = naive.groupby("forecast_origin").n_training_periods.first().tolist()
    assert counts == ([12, 13, 14] if strategy == "expanding" else [12, 12, 12])
    assert (result.predictions.source_available_at <= result.predictions.forecast_origin).all()
    assert (
        result.predictions.actual_evaluated_available_at <= result.predictions.evaluation_as_of
    ).all()


def test_backtest_origin_spacing_and_repeatability(actuals: pd.DataFrame) -> None:
    config = backtest_config(step_periods=2, horizon=1)
    first = run_demand_backtest(actuals, config)
    second = run_demand_backtest(actuals, config)
    assert list(first.predictions.forecast_origin.unique()) == list(
        pd.to_datetime(
            [
                "2021-01-01",
                "2021-03-01",
                "2021-05-01",
            ]
        )
    )
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)


def test_backtest_predictions_at_first_origin_ignore_later_truth_changes(
    actuals: pd.DataFrame,
) -> None:
    config = backtest_config(n_origins=1)
    baseline = run_demand_backtest(actuals, config)
    actuals.loc[actuals.date >= pd.Timestamp("2021-01-01"), "demand_units"] += 500.0
    changed = run_demand_backtest(actuals, config)
    pd.testing.assert_frame_equal(
        baseline.predictions[FORECAST_COLUMNS], changed.predictions[FORECAST_COLUMNS]
    )
    assert not baseline.metrics.equals(changed.metrics)


def test_backtest_fails_on_missing_or_unavailable_truth(actuals: pd.DataFrame) -> None:
    with pytest.raises(ContractViolation, match="complete evaluation actuals"):
        run_demand_backtest(actuals, backtest_config(n_origins=12, horizon=2))
    with pytest.raises(ContractViolation, match="complete evaluation actuals"):
        run_demand_backtest(actuals, backtest_config(evaluation_as_of="2021-02-01"))
    actuals.loc[actuals.date == pd.Timestamp("2021-02-01"), "actual_available_at"] = pd.Timestamp(
        "2022-02-01"
    )
    with pytest.raises(ContractViolation, match="complete evaluation actuals"):
        run_demand_backtest(actuals, backtest_config())


def test_zero_actuals_have_explicit_undefined_relative_metric_status(actuals: pd.DataFrame) -> None:
    actuals["demand_units"] = 0.0
    result = run_demand_backtest(actuals, backtest_config())
    assert result.metrics.mae.eq(0.0).all()
    assert result.metrics.smape.eq(0.0).all()
    assert result.metrics.wape.isna().all()
    assert result.metrics.bias_fraction.isna().all()
    assert result.metrics.relative_metrics_status.eq("undefined_near_zero_actual_total").all()


def test_backtest_keeps_first_origin_cohort(actuals: pd.DataFrame) -> None:
    extra = generate_synthetic_demand_dataset(
        DemandDataConfig(n_periods=24, n_skus=3, n_channels=1, n_geographies=1)
    )
    extra = extra.loc[(extra.sku_id == "SKU-003") & (extra.date >= pd.Timestamp("2021-01-01"))]
    result = run_demand_backtest(pd.concat([actuals, extra], ignore_index=True), backtest_config())
    assert set(result.predictions.sku_id) == {"SKU-001", "SKU-002"}


def test_weekly_rolling_backtest() -> None:
    actuals = generate_synthetic_demand_dataset(
        DemandDataConfig(frequency="W-MON", n_periods=60, n_skus=1, n_channels=1, n_geographies=1)
    )
    actuals["demand_units"] = np.arange(1.0, 61.0)
    config = DemandBacktestConfig(
        first_origin=actuals.date.iloc[52].date(),
        evaluation_as_of=actuals.actual_available_at.max().date(),
        frequency="W-MON",
        n_origins=3,
        horizon=2,
    )
    result = run_demand_backtest(actuals, config)
    assert len(result.predictions) == 12
    assert (
        result.predictions.loc[result.predictions.model_id == "seasonal_naive", "error"]
        .eq(-52.0)
        .all()
    )


@pytest.mark.parametrize(
    "values", [{"seasonal_periods": 0}, {"rolling_periods": 0}, {"training_window_periods": True}]
)
def test_invalid_feature_config(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DemandFeatureConfig.model_validate(values)
