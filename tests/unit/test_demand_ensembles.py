"""Out-of-sample weighting, isolated holdout, safe state and composed governance."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    DemandForecastRequest,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_hierarchy import build_demand_hierarchy
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    add_demand_ensemble_validation,
    run_demand_selection,
)
from revenue_prediction.core.evaluation.demand_tuning import DemandTuningConfig
from revenue_prediction.core.inference.demand_artifacts import (
    DemandModelBundle,
    build_demand_bundle,
    content_digest,
    demand_model_card,
    load_demand_bundle,
    predict_demand_bundle,
    predict_demand_hierarchy_bundle,
    runtime_versions,
    save_demand_bundle,
)
from revenue_prediction.core.models.demand_forecasters import (
    DemandForecasterState,
    TimeSeriesModelConfig,
    create_demand_forecaster,
    inverse_error_weights,
    restore_demand_forecaster,
)
from revenue_prediction.core.models.demand_reconciliation import demand_coherence_report
from revenue_prediction.core.training.demand_backtesting import DemandBacktestResult
from revenue_prediction.interfaces.cli.main import app
from revenue_prediction.ops.demand_registry import (
    LocalApprovalPolicy,
    demand_registry_status,
    initialize_demand_registry,
    predict_registered_demand,
    register_demand_model,
    transition_demand_model,
)

pytestmark = pytest.mark.unit


def experiment(frequency="MS"):
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
    for _, group in actuals.groupby("sku_id"):
        t = np.arange(len(group))
        actuals.loc[group.index, "demand_units"] = 100 + 0.5 * t + 2 * np.sin(t * np.pi / 2)
    dates = actuals.date.drop_duplicates().sort_values()
    common = {
        "frequency": frequency,
        "models": ["naive", "seasonal_naive", "drift"],
        "horizon": 2,
        "n_origins": 3,
        "seasonal_periods": 4,
    }
    config = DemandSelectionConfig.model_validate(
        {
            "validation": {
                **common,
                "first_origin": dates.iloc[30].date(),
                "evaluation_as_of": dates.iloc[35].date(),
            },
            "holdout": {
                **common,
                "first_origin": dates.iloc[40].date(),
                "evaluation_as_of": dates.iloc[45].date(),
            },
            "ensemble_members": ["naive", "seasonal_naive", "drift"],
        }
    )
    return actuals, config, dates.iloc[50].date()


def policy(hierarchy=False, **ensemble_updates):
    value = {
        "schema_version": "demand-local-policy-v3",
        "principals": [
            {"actor_id": "ACTOR-001", "roles": ["builder", "submitter"]},
            {"actor_id": "ACTOR-002", "roles": ["reviewer"]},
        ],
        "max_holdout_primary_metric": 100,
        "max_abs_holdout_bias_fraction": 100,
        "require_seasonal_noninferiority": False,
        "ensemble": {
            "maximum_component_weight": 1,
            "minimum_validation_origins": 3,
            **ensemble_updates,
        },
    }
    if hierarchy:
        value["reconciliation"] = {
            "maximum_absolute_coherence_residual": 1e-9,
            "levels": [
                {
                    "hierarchy_level": level,
                    "maximum_primary_metric": 100,
                    "maximum_abs_bias_fraction": 100,
                    "require_seasonal_noninferiority": False,
                }
                for level in ("sku", "product", "family", "brand")
            ],
        }
    return LocalApprovalPolicy.model_validate(value)


def review(path, bundle, limits):
    initialize_demand_registry(path, limits)
    register_demand_model(path, bundle, "ACTOR-001", "Synthetic ensemble")
    transition_demand_model(path, bundle.artifact_id, "ACTOR-001", "submit", "Review", 1)


def approve(path, bundle):
    return transition_demand_model(
        path,
        bundle.artifact_id,
        "ACTOR-002",
        "approve",
        "Review frozen weights and out-of-sample holdout",
        2,
        True,
    )


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        ((2.0, 4.0), (2 / 3, 1 / 3)),
        ((0.0, 4.0), (1.0, 0.0)),
        ((0.0, 0.0, 4.0), (0.5, 0.5, 0.0)),
        ((0.0, 0.0), (0.5, 0.5)),
        ((1e-300, 1e300), (1.0, 0.0)),
    ],
)
def test_exact_inverse_error_and_zero_rules(errors, expected):
    assert inverse_error_weights(errors) == expected
    assert sum(expected) == pytest.approx(1)


@pytest.mark.parametrize("errors", [(1,), (-1, 2), (np.nan, 2), (np.inf, 2)])
def test_invalid_or_undefined_errors_fail(errors):
    with pytest.raises(ContractViolation, match="finite nonnegative"):
        inverse_error_weights(errors)


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_weights_are_validation_only_and_holdout_keeps_all_members(frequency):
    actuals, config, _ = experiment(frequency)
    before = run_demand_selection(actuals, config)
    assert before.decision.selected_model == "weighted_ensemble"
    weights = before.decision.ensemble_weights
    metrics = before.validation.metrics.loc[before.validation.metrics.horizon == 0].set_index(
        "model_id"
    )
    for member in weights:
        assert member.validation_error == metrics.loc[member.model_id, "mase"]
    assert {row.model_id for row in weights} == set(config.ensemble_members)
    assert {row.model_id for row in weights} | {"weighted_ensemble"} == set(
        before.holdout.predictions.model_id
    )
    changed = actuals.copy()
    changed.loc[changed.date >= pd.Timestamp(config.holdout.first_origin), "demand_units"] += 1000
    after = run_demand_selection(changed, config)
    assert before.decision == after.decision
    pd.testing.assert_frame_equal(
        before.validation.predictions, after.validation.predictions, check_exact=True
    )
    assert not before.holdout.metrics.equals(after.holdout.metrics)
    for origin, group in before.holdout.predictions.groupby("forecast_origin"):
        validate_demand_forecasts(
            group[
                [
                    "date",
                    "sku_id",
                    "channel_id",
                    "geography_id",
                    "frequency",
                    "forecast_origin",
                    "horizon",
                    "model_id",
                    "predicted_demand_units",
                    "source_date",
                    "source_available_at",
                    "training_start",
                    "training_end",
                    "n_training_periods",
                    "schema_version",
                    "forecast_kind",
                    "approval_status",
                ]
            ],
            DemandForecastRequest(
                forecast_origin=origin.date(), frequency=config.validation.frequency, horizon=2
            ),
        )
    assert (
        before.holdout.predictions.loc[
            before.holdout.predictions.model_id == "weighted_ensemble", "schema_version"
        ]
        .eq("demand-point-v2")
        .all()
    )


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
@pytest.mark.parametrize("hierarchical", [False, True])
def test_bundle_replay_and_composed_approval(tmp_path, frequency, hierarchical):
    actuals, config, origin = experiment(frequency)
    hierarchy = (
        build_demand_hierarchy(
            actuals,
            DemandForecastRequest(
                forecast_origin=config.validation.first_origin,
                frequency=config.validation.frequency,
            ),
        )
        if hierarchical
        else None
    )
    bundle = build_demand_bundle(actuals, config, origin, "ACTOR-001", hierarchy=hierarchy)
    assert bundle.payload.schema_version == "demand-bundle-v3"
    assert bundle.payload.decision.selected_model == "weighted_ensemble"
    assert demand_model_card(bundle)["ensemble"]["weight_scope"] == "portfolio_all_horizons"
    save_demand_bundle(bundle, tmp_path / "model.json")
    loaded = load_demand_bundle(tmp_path / "model.json")
    points = predict_demand_bundle(loaded)
    pd.testing.assert_frame_equal(points, predict_demand_bundle(bundle), check_exact=True)
    for item in loaded.payload.series:
        restored = restore_demand_forecaster(item.state)
        assert restored.export_state() == item.state
        dates = pd.DatetimeIndex(points.date.drop_duplicates())
        expected = np.asarray(
            [row.weight for row in item.state.parameters.ensemble_weights]
        ) @ np.stack(
            [
                restore_demand_forecaster(component).predict(dates)
                for component in item.state.components
            ]
        )
        np.testing.assert_array_equal(restored.predict(dates), expected)
    registry = tmp_path / "registry.sqlite"
    review(registry, loaded, policy(hierarchical))
    approve(registry, loaded)
    forecast, lineage = predict_registered_demand(
        registry, loaded.artifact_id, include_hierarchy=hierarchical
    )
    assert lineage["schema_version"] == "demand-inference-lineage-v3"
    assert lineage["ensemble_weights_sha256"] == content_digest(
        [item.model_dump(mode="json") for item in loaded.payload.decision.ensemble_weights]
    )
    if hierarchical:
        pd.testing.assert_frame_equal(forecast, predict_demand_hierarchy_bundle(loaded))
        assert demand_coherence_report(forecast, hierarchy).coherent.all()
    else:
        pd.testing.assert_frame_equal(forecast, points)
    assert forecast.approval_status.eq("unapproved").all()
    transition_demand_model(registry, loaded.artifact_id, "ACTOR-002", "revoke", "Suspend", 3)
    with pytest.raises(ContractViolation, match="revoked"):
        predict_registered_demand(registry, loaded.artifact_id, include_hierarchy=hierarchical)


def test_missing_components_and_edited_weight_evidence_fail():
    actuals, config, origin = experiment()
    result = run_demand_selection(actuals, config)
    incomplete = result.validation.predictions.loc[
        ~(
            (result.validation.predictions.model_id == "naive")
            & (result.validation.predictions.horizon == 1)
        )
        & (result.validation.predictions.model_id != "weighted_ensemble")
    ]
    with pytest.raises(ContractViolation, match="identical complete grid"):
        add_demand_ensemble_validation(
            DemandBacktestResult(incomplete, pd.DataFrame()), result.decision.ensemble_weights
        )
    bundle = build_demand_bundle(actuals, config, origin, "ACTOR-001")
    raw = bundle.payload.model_dump(mode="json")
    raw["decision"]["ensemble_weights"][0]["validation_error"] *= 2
    with pytest.raises(ValidationError):
        DemandModelBundle.model_validate({"artifact_id": content_digest(raw), "payload": raw})
    state = bundle.payload.series[0].state.model_dump(mode="json")
    for change in [{"components": state["components"][:-1]}, {"schema_version": "demand-state-v1"}]:
        with pytest.raises(ValidationError):
            DemandForecasterState.model_validate({**state, **change})
    state = copy.deepcopy(state)
    state["components"][0]["values"][0] += 1
    with pytest.raises(ValidationError, match="shared training"):
        DemandForecasterState.model_validate(state)


def test_zero_weight_component_must_still_fit_and_forecast():
    parameters = TimeSeriesModelConfig.model_validate(
        {
            "average_periods": 10,
            "ensemble_weights": [
                {"model_id": "moving_average", "validation_error": 1, "weight": 0},
                {"model_id": "naive", "validation_error": 0, "weight": 1},
            ],
        }
    )
    actuals, _, _ = experiment()
    model = create_demand_forecaster("weighted_ensemble", parameters)
    with pytest.raises(ContractViolation, match="at least 10"):
        model.fit(actuals.loc[actuals.sku_id == "SKU-001"].iloc[:5], "MS", 4)
    with pytest.raises(ContractViolation, match="every fitted component"):
        model.predict(pd.date_range("2025-01-01", periods=2, freq="MS"))
    with pytest.raises(ContractViolation, match="frozen validation"):
        create_demand_forecaster("weighted_ensemble")


@pytest.mark.parametrize(
    "updates",
    [
        {"maximum_component_weight": 0.1},
        {"minimum_validation_origins": 4},
    ],
)
def test_explicit_ensemble_policy_blocks_without_revision_change(tmp_path, updates):
    actuals, config, origin = experiment()
    bundle = build_demand_bundle(actuals, config, origin, "ACTOR-001")
    registry = tmp_path / "registry.sqlite"
    review(registry, bundle, policy(**updates))
    with pytest.raises(ContractViolation, match="ensemble weight/origin"):
        approve(registry, bundle)
    assert demand_registry_status(registry, bundle.artifact_id).revision == 2


def test_undeclared_members_and_manual_weights_cannot_enter_selection():
    _, config, _ = experiment()
    for members in [("naive", "naive"), ("naive", "ets")]:
        with pytest.raises(ValidationError, match="unique, predeclared"):
            DemandSelectionConfig.model_validate(
                {**config.model_dump(), "ensemble_members": members}
            )
    with pytest.raises(ValidationError, match="cannot be combined"):
        DemandTuningConfig(experiment=config, parameter_grid=(TimeSeriesModelConfig(),))


def test_statistical_component_restores_without_optimization(monkeypatch):
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    parameters = TimeSeriesModelConfig.model_validate(
        {
            "ensemble_weights": [
                {"model_id": "ets", "validation_error": 1, "weight": 0.5},
                {"model_id": "naive", "validation_error": 1, "weight": 0.5},
            ],
        }
    )
    actuals, _, _ = experiment()
    model = create_demand_forecaster("weighted_ensemble", parameters)
    model.fit(actuals.loc[actuals.sku_id == "SKU-001"].iloc[:40], "MS", 4)
    state = model.export_state()
    fit = ExponentialSmoothing.fit

    def fixed_fit(self, *args, **kwargs):
        assert kwargs["optimized"] is False
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(ExponentialSmoothing, "fit", fixed_fit)
    restored = restore_demand_forecaster(state)
    dates = pd.DatetimeIndex(actuals.loc[actuals.sku_id == "SKU-001", "date"].iloc[40:45])
    np.testing.assert_array_equal(model.predict(dates), restored.predict(dates))
    assert "statsmodels" in runtime_versions("weighted_ensemble", parameters)


def test_cli_ensemble_comparison_packages_frozen_evidence(tmp_path):
    runner = CliRunner()
    comparison = tmp_path / "comparison"
    compared = runner.invoke(
        app,
        [
            "compare-demand",
            "--models",
            "naive,seasonal_naive,drift",
            "--ensemble-members",
            "naive,seasonal_naive,drift",
            "--skus",
            "1",
            "--channels",
            "1",
            "--geographies",
            "1",
            "--out",
            str(comparison),
        ],
    )
    assert compared.exit_code == 0, compared.output
    output = tmp_path / "artifact"
    packaged = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "package",
            "--comparison-dir",
            str(comparison),
            "--origin",
            "2025-01-01",
            "--actor",
            "ACTOR-001",
            "--out",
            str(output),
        ],
    )
    assert packaged.exit_code == 0, packaged.output
    bundle = load_demand_bundle(output / "model.json")
    decision = json.loads((comparison / "selection.json").read_text())
    assert decision["ensemble_weights"] == [
        row.model_dump(mode="json") for row in bundle.payload.decision.ensemble_weights
    ]
    assert bundle.payload.schema_version == "demand-bundle-v3"


def test_constant_scaled_errors_fail_instead_of_learning_default_weights():
    actuals, config, _ = experiment()
    actuals["demand_units"] = 100.0
    with pytest.raises(ContractViolation, match="finite nonnegative"):
        run_demand_selection(actuals, config)
    wape_config = DemandSelectionConfig.model_validate(
        {**config.model_dump(), "primary_metric": "wape"}
    )
    result = run_demand_selection(actuals, wape_config)
    assert result.decision.selected_model == "seasonal_naive"
    assert [row.weight for row in result.decision.ensemble_weights] == [1 / 3] * 3


def test_zero_weight_component_forecast_failure_is_not_hidden(monkeypatch):
    from revenue_prediction.core.models import demand_forecasters

    parameters = TimeSeriesModelConfig.model_validate(
        {
            "ensemble_weights": [
                {"model_id": "moving_average", "validation_error": 1, "weight": 0},
                {"model_id": "naive", "validation_error": 0, "weight": 1},
            ],
        }
    )
    actuals, _, _ = experiment()
    model = create_demand_forecaster("weighted_ensemble", parameters)
    model.fit(actuals.loc[actuals.sku_id == "SKU-001"].iloc[:40], "MS", 4)
    original = demand_forecasters._LocalForecaster.predict

    def fail_zero_member(self, dates):
        if self.capability.model_id == "moving_average":
            raise ContractViolation("Zero-weight component failed")
        return original(self, dates)

    monkeypatch.setattr(demand_forecasters._LocalForecaster, "predict", fail_zero_member)
    with pytest.raises(ContractViolation, match="Zero-weight component failed"):
        model.predict(pd.date_range("2024-01-01", periods=2, freq="MS"))


def test_old_point_policy_cannot_approve_ensemble(tmp_path):
    actuals, config, origin = experiment()
    bundle = build_demand_bundle(actuals, config, origin, "ACTOR-001")
    raw = policy().model_dump(mode="json")
    raw.pop("ensemble")
    raw["schema_version"] = "demand-local-policy-v1"
    registry = tmp_path / "registry.sqlite"
    review(registry, bundle, LocalApprovalPolicy.model_validate(raw))
    with pytest.raises(ContractViolation, match="explicit ensemble policy"):
        approve(registry, bundle)


@pytest.mark.parametrize("metric", ["rmsse", "wape"])
def test_alternative_time_series_weighting_metrics_are_explicit(metric):
    actuals, config, _ = experiment()
    config = DemandSelectionConfig.model_validate({**config.model_dump(), "primary_metric": metric})
    result = run_demand_selection(actuals, config)
    metrics = result.validation.metrics.loc[result.validation.metrics.horizon == 0].set_index(
        "model_id"
    )
    for member in result.decision.ensemble_weights:
        assert member.validation_error == metrics.loc[member.model_id, metric]


def test_ensemble_point_version_cannot_be_downgraded():
    actuals, config, origin = experiment()
    bundle = build_demand_bundle(actuals, config, origin, "ACTOR-001")
    points = predict_demand_bundle(bundle)
    points["schema_version"] = "demand-point-v1"
    with pytest.raises(ContractViolation, match="schema version"):
        validate_demand_forecasts(
            points,
            DemandForecastRequest(
                forecast_origin=origin,
                frequency="MS",
                horizon=2,
            ),
        )
