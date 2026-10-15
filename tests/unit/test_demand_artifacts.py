"""Safe fitted-state persistence, origin lineage and model cards."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandForecastRequest
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.evaluation.demand_tuning import DemandTuningConfig
from revenue_prediction.core.inference.demand_artifacts import (
    DemandModelBundle,
    build_demand_bundle,
    content_digest,
    demand_model_card,
    load_demand_bundle,
    predict_demand_bundle,
    render_demand_model_card,
    save_demand_bundle,
    verify_demand_bundle,
)
from revenue_prediction.core.models.demand_forecasters import (
    DemandForecasterState,
    TimeSeriesModelConfig,
    create_demand_forecaster,
    demand_model_capabilities,
    forecast_demand_models,
    restore_demand_forecaster,
)
from revenue_prediction.core.training.demand_backtesting import DemandBacktestConfig

pytestmark = pytest.mark.unit
MODEL_IDS = tuple(item.model_id for item in demand_model_capabilities())


def synthetic_history(frequency: str = "MS", n_skus: int = 1, n_periods: int = 60) -> pd.DataFrame:
    return generate_synthetic_demand_dataset(
        DemandDataConfig.model_validate(
            {
                "frequency": frequency,
                "n_skus": n_skus,
                "n_channels": 1,
                "n_geographies": 1,
                "n_periods": n_periods,
            }
        )
    )


def artifact_experiment(
    frame: pd.DataFrame, models: tuple[str, ...] = ("naive", "seasonal_naive", "drift")
) -> DemandSelectionConfig:
    dates = frame.date.drop_duplicates().sort_values()
    common = {
        "models": models,
        "horizon": 2,
        "n_origins": 2,
        "frequency": frame.frequency.iloc[0],
        "seasonal_periods": 4,
    }
    return DemandSelectionConfig(
        validation=DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[40].date(),
                "evaluation_as_of": dates.iloc[45].date(),
            }
        ),
        holdout=DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[45].date(),
                "evaluation_as_of": dates.iloc[50].date(),
            }
        ),
    )


@pytest.fixture(scope="module")
def drift_bundle() -> DemandModelBundle:
    frame = synthetic_history(n_skus=2)
    for _, series in frame.groupby("sku_id"):
        frame.loc[series.index, "demand_units"] = np.arange(len(series), dtype=float) + 10
    return build_demand_bundle(frame, artifact_experiment(frame), date(2025, 1, 1), "ACTOR-001")


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_all_models_json_roundtrip_with_long_horizons_and_delayed_leads(
    model_id: str, frequency: str
) -> None:
    if model_id in {"ets", "arima", "sarima"}:
        pytest.importorskip("statsmodels")
    frame = synthetic_history(frequency)
    model = create_demand_forecaster(model_id)
    model.fit(frame.iloc[:40], frequency, 4)
    first = 40 if model_id == "seasonal_naive" else 42
    dates = pd.DatetimeIndex(frame.date.iloc[first : first + 15])
    state_json = model.export_state().model_dump_json()
    assert "NaN" not in state_json and "Infinity" not in state_json
    state = DemandForecasterState.model_validate_json(state_json)
    restored = restore_demand_forecaster(state)
    np.testing.assert_allclose(
        model.predict(dates), restored.predict(dates), rtol=1e-12, atol=1e-12
    )
    assert restored.export_state() == state


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ets_trend": None, "ets_damped": False, "ets_seasonal": None},
        {"ets_trend": None, "ets_damped": False},
        {"ets_seasonal": None},
        {"ets_damped": False},
    ],
)
def test_optional_ets_components_restore_without_optimizer(
    kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    frame = synthetic_history()
    model = create_demand_forecaster("ets", TimeSeriesModelConfig.model_validate(kwargs))
    model.fit(frame.iloc[:40], "MS", 4)
    state = model.export_state()
    original = ExponentialSmoothing.fit

    def fixed_fit(self: object, *args: object, **options: object) -> object:
        assert options["optimized"] is False
        return original(self, *args, **options)

    monkeypatch.setattr(ExponentialSmoothing, "fit", fixed_fit)
    restored = restore_demand_forecaster(state)
    dates = pd.DatetimeIndex(frame.date.iloc[40:50])
    np.testing.assert_allclose(
        model.predict(dates), restored.predict(dates), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("model_id", ["arima", "sarima"])
def test_state_space_restoration_does_not_fit(
    model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    frame = synthetic_history()
    model = create_demand_forecaster(model_id)
    model.fit(frame.iloc[:40], "MS", 4)
    state = model.export_state()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Restoration must not optimize on load.")

    monkeypatch.setattr(SARIMAX, "fit", forbidden)
    restored = restore_demand_forecaster(state)
    dates = pd.DatetimeIndex(frame.date.iloc[40:50])
    np.testing.assert_allclose(
        model.predict(dates), restored.predict(dates), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_export_requires_successful_fit(model_id: str) -> None:
    with pytest.raises(RuntimeError, match="Fit"):
        create_demand_forecaster(model_id).export_state()


def test_invalid_json_state_rejected_without_executable_deserialization() -> None:
    model = create_demand_forecaster("ses")
    model.fit(synthetic_history().iloc[:30], "MS", 4)
    state = model.export_state().model_dump(mode="json")
    for updates in [
        {"schema_version": "future"},
        {"model_id": "arbitrary.module"},
        {"values": [float("nan")] * 30},
        {"values": [1]},
        {"dates": ["2020-01-02T00:00:00"] * 30},
        {"dates": ["2020-01-01T00:00:00+00:00"] * 30},
        {"arima_parameters": [1]},
        {"python_object": "os.system"},
    ]:
        with pytest.raises((ValidationError, ContractViolation)):
            DemandForecasterState.model_validate({**state, **updates})


def test_bundle_roundtrip_and_point_contract(
    drift_bundle: DemandModelBundle, tmp_path: Path
) -> None:
    path = tmp_path / "model.json"
    save_demand_bundle(drift_bundle, path)
    loaded = load_demand_bundle(path)
    assert loaded == drift_bundle
    forecast = predict_demand_bundle(loaded)
    np.testing.assert_array_equal(forecast.predicted_demand_units, [70, 71, 70, 71])
    assert forecast.approval_status.eq("unapproved").all()
    assert len(predict_demand_bundle(loaded, 1)) == 2
    with pytest.raises(FileExistsError):
        save_demand_bundle(drift_bundle, path)
    for horizon in (0, 3):
        with pytest.raises(ContractViolation, match="evaluated horizon"):
            predict_demand_bundle(loaded, horizon)


@pytest.mark.parametrize("model_id", ["naive", "seasonal_naive"])
def test_frozen_baseline_provenance_matches_existing_contract(model_id: str) -> None:
    frame = synthetic_history()
    if model_id == "naive":
        frame["demand_units"] = [
            *(10.0 + np.arange(40) * 0.1 + np.arange(40) % 4),
            *([20.0] * 20),
        ]
    else:
        frame["demand_units"] = 10.0 + np.arange(60) * 0.01 + np.tile([0, 5, 0, -5], 15)
    config = artifact_experiment(frame, ("naive", "seasonal_naive"))
    bundle = build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001")
    assert bundle.payload.decision.selected_model == model_id
    expected = (
        forecast_demand_models(
            frame,
            DemandForecastRequest(forecast_origin=date(2025, 1, 1), horizon=2),
            models=(model_id,),
            features=None,
        )
        if model_id == "naive"
        else None
    )
    if expected is None:
        from revenue_prediction.core.features.demand import DemandFeatureConfig

        expected = forecast_demand_models(
            frame,
            DemandForecastRequest(forecast_origin=date(2025, 1, 1), horizon=2),
            models=(model_id,),
            features=DemandFeatureConfig(seasonal_periods=4),
        )
    pd.testing.assert_frame_equal(predict_demand_bundle(bundle), expected)


def test_bundle_integrity_schema_and_compatibility_fail_closed(
    drift_bundle: DemandModelBundle, tmp_path: Path
) -> None:
    raw = drift_bundle.model_dump(mode="json")
    raw["payload"]["origin"] = "2025-02-01"
    with pytest.raises(ValidationError, match="digest mismatch"):
        DemandModelBundle.model_validate(raw)
    raw = drift_bundle.model_dump(mode="json")
    raw["payload"]["schema_version"] = "revenue-bundle-v1"
    with pytest.raises(ValidationError):
        DemandModelBundle.model_validate(raw)
    for field, replacement in [
        ("implementation_sha256", "0" * 64),
        ("runtime", {"python": "0.0"}),
    ]:
        raw = drift_bundle.model_dump(mode="json")
        raw["payload"]["lineage"][field] = replacement
        raw["artifact_id"] = content_digest(raw["payload"])
        altered = DemandModelBundle.model_validate(raw)
        with pytest.raises(ContractViolation, match="differs"):
            verify_demand_bundle(altered)
    path = tmp_path / "untrusted.json"
    path.write_bytes(b"\x80\x04pickle-is-not-json")
    with pytest.raises((ValueError, UnicodeDecodeError)):
        load_demand_bundle(path)


def test_lineage_rejects_changed_history_and_future_arrivals(
    drift_bundle: DemandModelBundle,
) -> None:
    raw = drift_bundle.model_dump(mode="json")
    raw["payload"]["series"][0]["state"]["values"][0] += 1
    with pytest.raises(ValidationError, match="training observations"):
        DemandModelBundle.model_validate(raw)
    raw = drift_bundle.model_dump(mode="json")
    series = raw["payload"]["series"]
    series[0]["observations"][-1]["actual_available_at"] = "2026-01-01T00:00:00"
    raw["payload"]["lineage"]["training_sha256"] = content_digest(series)
    raw["artifact_id"] = content_digest(raw["payload"])
    with pytest.raises(ValidationError, match="unclosed/unavailable"):
        DemandModelBundle.model_validate(raw)


def test_model_card_contains_scope_evidence_and_no_approval_claim(
    drift_bundle: DemandModelBundle,
) -> None:
    card = demand_model_card(drift_bundle)
    assert card["target"] == "demand_units"
    assert card["artifact_id"] == drift_bundle.artifact_id
    assert len(card["series_training"]) == 2
    assert card["lineage"]["holdout_predictions_sha256"]
    assert card["validation_metrics"] and card["holdout_metrics"]
    assert "never grants approval" in card["approval_authority"]
    text = render_demand_model_card(drift_bundle)
    assert json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0]) == json.loads(
        json.dumps(card)
    )


def test_incomplete_or_inconsistent_evaluation_card_is_rejected(
    drift_bundle: DemandModelBundle,
) -> None:
    for change in ("coverage", "count", "score", "target", "scope"):
        raw = drift_bundle.model_dump(mode="json")
        if change == "coverage":
            raw["payload"]["holdout_metrics"].pop()
        elif change == "count":
            raw["payload"]["holdout_metrics"][0]["n_predictions"] = 1
        elif change == "score":
            raw["payload"]["decision"]["validation_metric"] = 99.0
        elif change == "target":
            raw["payload"]["target"] = "actual_month_end_net_revenue"
        else:
            raw["payload"]["series"].pop()
            raw["payload"]["lineage"]["training_sha256"] = content_digest(raw["payload"]["series"])
        raw["artifact_id"] = content_digest(raw["payload"])
        with pytest.raises(ValidationError):
            DemandModelBundle.model_validate(raw)


def test_missing_seasonal_calendar_source_stays_an_explicit_failure() -> None:
    model = create_demand_forecaster("seasonal_naive")
    frame = synthetic_history()
    model.fit(frame.iloc[:40], "MS", 4)
    restored = restore_demand_forecaster(model.export_state())
    with pytest.raises(ContractViolation, match="calendar sources"):
        restored.predict(pd.DatetimeIndex(frame.date.iloc[42:48]))


def test_sliding_bundle_keeps_only_origin_window() -> None:
    frame = synthetic_history()
    frame["demand_units"] = np.arange(60, dtype=float) + 1
    config = artifact_experiment(frame)
    config = DemandSelectionConfig(
        validation=config.validation.model_copy(
            update={"strategy": "sliding", "training_window_periods": 16}
        ),
        holdout=config.holdout.model_copy(
            update={"strategy": "sliding", "training_window_periods": 16}
        ),
    )
    bundle = build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001")
    assert len(bundle.payload.series[0].observations) == 16
    assert (
        bundle.payload.series[0].observations[0].date == pd.Timestamp("2023-09-01").to_pydatetime()
    )


def test_state_space_parameter_shape_is_checked_before_filtering() -> None:
    pytest.importorskip("statsmodels")
    model = create_demand_forecaster("arima")
    model.fit(synthetic_history().iloc[:40], "MS", 4)
    raw = model.export_state().model_dump()
    raw["arima_parameters"] = (0.1,)
    state = DemandForecasterState.model_validate(raw)
    with pytest.raises(ContractViolation, match="parameter count"):
        restore_demand_forecaster(state)


def test_statistical_winner_bundle_restores_with_provider_lineage(tmp_path: Path) -> None:
    pytest.importorskip("statsmodels")
    frame = synthetic_history()
    index = np.arange(60)
    frame["demand_units"] = (
        20
        + index * 0.5
        + 4 * np.sin(index * np.pi / 2)
        + np.random.default_rng(7).normal(0, 0.1, 60)
    )
    config = artifact_experiment(frame, ("naive", "seasonal_naive", "ets"))
    bundle = build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001")
    assert bundle.payload.decision.selected_model == "ets"
    assert "statsmodels" in bundle.payload.lineage.runtime
    path = tmp_path / "statistical-model.json"
    save_demand_bundle(bundle, path)
    restored = load_demand_bundle(path)
    from revenue_prediction.core.features.demand import DemandFeatureConfig

    expected = forecast_demand_models(
        frame,
        DemandForecastRequest(forecast_origin=date(2025, 1, 1), horizon=2),
        features=DemandFeatureConfig(seasonal_periods=4),
        models=("ets",),
    )
    pd.testing.assert_frame_equal(predict_demand_bundle(restored), expected, rtol=1e-12, atol=1e-12)


def test_packaging_cannot_discard_mismatching_tuning_evidence() -> None:
    frame = synthetic_history()
    config = artifact_experiment(frame, ("naive", "seasonal_naive", "ses"))
    tuning = DemandTuningConfig(
        experiment=config, parameter_grid=(TimeSeriesModelConfig(alpha=0.8),)
    )
    with pytest.raises(ContractViolation, match="does not reproduce"):
        build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001", tuning=tuning)


def test_origin_and_future_value_invariance_of_fitted_artifact() -> None:
    frame = synthetic_history(n_periods=72)
    config = artifact_experiment(frame)
    with pytest.raises(ContractViolation, match="on/after"):
        build_demand_bundle(frame, config, date(2020, 1, 1), "ACTOR-001")
    before = build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001")
    frame.loc[frame.date >= pd.Timestamp("2025-01-01"), "demand_units"] += 10000
    after = build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001")
    assert before.payload.series == after.payload.series
    assert before.payload.decision == after.payload.decision
    assert before.payload.lineage.training_sha256 == after.payload.lineage.training_sha256
    assert (
        before.payload.lineage.source_actuals_sha256 != after.payload.lineage.source_actuals_sha256
    )
    pd.testing.assert_frame_equal(predict_demand_bundle(before), predict_demand_bundle(after))
