"""Bound interval artifacts, explicit review policy and fail-closed inference."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import FORECAST_COLUMNS
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.evaluation.demand_uncertainty import (
    DemandUncertaintyConfig,
    run_demand_uncertainty,
)
from revenue_prediction.core.inference.demand_artifacts import (
    DemandModelBundle,
    build_demand_bundle,
    content_digest,
    dataframe_records,
    demand_model_card,
    load_demand_bundle,
    predict_demand_bundle,
    predict_demand_interval_bundle,
    render_demand_model_card,
    save_demand_bundle,
)
from revenue_prediction.core.training.demand_backtesting import DemandBacktestConfig
from revenue_prediction.ops.demand_registry import (
    LocalApprovalPolicy,
    demand_registry_status,
    initialize_demand_registry,
    predict_registered_demand,
    register_demand_model,
    transition_demand_model,
)

pytestmark = pytest.mark.unit


def experiment(frequency: str = "MS"):
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
    dates = actuals.date.drop_duplicates().sort_values()
    common = {
        "frequency": frequency,
        "models": ["naive", "seasonal_naive", "drift"],
        "seasonal_periods": 4,
        "horizon": 2,
    }

    def block(start: int, cutoff: int, count: int):
        return DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[start].date(),
                "evaluation_as_of": dates.iloc[cutoff].date(),
                "n_origins": count,
            }
        )

    config = DemandUncertaintyConfig(
        selection=DemandSelectionConfig(validation=block(25, 29, 2), holdout=block(40, 45, 2)),
        calibration=block(30, 37, 5),
    )
    bundle = build_demand_bundle(
        actuals, config.selection, dates.iloc[50].date(), "ACTOR-001", uncertainty=config
    )
    return actuals, config, bundle


@pytest.fixture(scope="module")
def packaged():
    return experiment()


def policy(**interval_updates):
    return LocalApprovalPolicy.model_validate(
        {
            "schema_version": "demand-local-policy-v2",
            "principals": [
                {"actor_id": "ACTOR-001", "roles": ["builder", "submitter"]},
                {"actor_id": "ACTOR-002", "roles": ["reviewer"]},
            ],
            "max_holdout_primary_metric": 100,
            "max_abs_holdout_bias_fraction": 100,
            "require_seasonal_noninferiority": False,
            "intervals": {
                "nominal_coverage": 0.8,
                "minimum_empirical_coverage": 0.0,
                "maximum_mean_width": 1e6,
                "maximum_mean_interval_score": 1e6,
                "maximum_pinball_loss": 1e6,
                "minimum_calibration_predictions": 5,
                **interval_updates,
            },
        }
    )


def in_review(path, bundle, approval_policy):
    initialize_demand_registry(path, approval_policy)
    register_demand_model(path, bundle, "ACTOR-001", "Synthetic artifact")
    transition_demand_model(path, bundle.artifact_id, "ACTOR-001", "submit", "Review", 1)


def approve(path, bundle):
    return transition_demand_model(
        path,
        bundle.artifact_id,
        "ACTOR-002",
        "approve",
        "Reviewed intervals and limitations",
        2,
        True,
    )


def rehash(payload):
    return DemandModelBundle.model_validate(
        {"artifact_id": content_digest(payload), "payload": payload}
    )


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_bundle_roundtrip_frozen_intervals_and_complete_lineage(tmp_path, frequency):
    actuals, config, bundle = experiment(frequency)
    evidence = bundle.payload.uncertainty
    assert evidence is not None
    assert bundle.payload.schema_version == "demand-bundle-v2"
    result = run_demand_uncertainty(actuals, config)
    assert evidence.calibration == result.calibration
    assert evidence.calibration_predictions_sha256 == content_digest(
        dataframe_records(result.calibration_backtest.predictions)
    )
    assert evidence.holdout_intervals_sha256 == content_digest(
        dataframe_records(result.predictions)
    )
    path = tmp_path / "model.json"
    save_demand_bundle(bundle, path)
    restored = load_demand_bundle(path)
    intervals = predict_demand_interval_bundle(restored)
    pd.testing.assert_frame_equal(
        intervals, predict_demand_interval_bundle(bundle), check_exact=True
    )
    pd.testing.assert_frame_equal(intervals[FORECAST_COLUMNS], predict_demand_bundle(bundle))
    short = predict_demand_interval_bundle(restored, 1)
    pd.testing.assert_frame_equal(
        short.reset_index(drop=True),
        intervals.loc[intervals.horizon == 1].reset_index(drop=True),
    )
    assert intervals.approval_status.eq("unapproved").all()
    card = demand_model_card(bundle)
    assert card["schema_version"] == "demand-model-card-v2"
    assert card["uncertainty"] == evidence.model_dump(mode="json")
    text = render_demand_model_card(bundle)
    assert "do not guarantee time-series coverage" in text
    assert "no covariates, intervals or reconciliation" not in text
    assert bundle.artifact_id in text


def test_interval_registry_lifecycle_and_lineage(tmp_path, packaged):
    _, _, bundle = packaged
    path = tmp_path / "registry.sqlite"
    in_review(path, bundle, policy())
    with pytest.raises(ContractViolation, match="in_review"):
        predict_registered_demand(path, bundle.artifact_id, include_intervals=True)
    approved = approve(path, bundle)
    intervals, manifest = predict_registered_demand(
        path, bundle.artifact_id, include_intervals=True
    )
    assert manifest["schema_version"] == "demand-inference-lineage-v2"
    assert manifest["forecast_sha256"] == content_digest(dataframe_records(intervals))
    assert manifest["calibration_sha256"] == bundle.payload.uncertainty.calibration_sha256
    assert manifest["approval_event_sha256"] == approved.event_sha256
    assert manifest["forecast_approval_status"] == "unapproved"
    assert approved.events[-1].model_card_sha256 == content_digest(demand_model_card(bundle))
    points, old_manifest = predict_registered_demand(path, bundle.artifact_id)
    pd.testing.assert_frame_equal(intervals[FORECAST_COLUMNS], points)
    assert old_manifest["schema_version"] == "demand-inference-lineage-v1"
    shortened, _ = predict_registered_demand(path, bundle.artifact_id, 1, include_intervals=True)
    assert shortened.horizon.eq(1).all()
    transition_demand_model(path, bundle.artifact_id, "ACTOR-002", "revoke", "Suspend", 3)
    with pytest.raises(ContractViolation, match="revoked"):
        predict_registered_demand(path, bundle.artifact_id, include_intervals=True)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"nominal_coverage": 0.9}, "nominal coverage"),
        ({"minimum_calibration_predictions": 6}, "insufficient calibration"),
        ({"maximum_mean_width": 0}, "width"),
        ({"maximum_mean_interval_score": 0}, "interval score"),
        ({"maximum_pinball_loss": 0}, "pinball loss"),
    ],
)
def test_explicit_interval_gates_rollback(tmp_path, packaged, updates, message):
    _, _, bundle = packaged
    path = tmp_path / "registry.sqlite"
    in_review(path, bundle, policy(**updates))
    with pytest.raises(ContractViolation, match=message):
        approve(path, bundle)
    record = demand_registry_status(path, bundle.artifact_id)
    assert record.revision == 2 and record.status == "in_review"
    with pytest.raises(ContractViolation):
        predict_registered_demand(path, bundle.artifact_id, include_intervals=True)


def test_coverage_gate_checks_each_series_not_only_portfolio(tmp_path, packaged):
    _, _, bundle = packaged
    payload = bundle.payload.model_dump(mode="json")
    row = next(
        row
        for row in payload["uncertainty"]["series_holdout_metrics"]
        if row["model_id"] == bundle.payload.decision.selected_model and row["horizon"] == 1
    )
    row["empirical_coverage"] = 0.0
    row["coverage_gap"] = -row["nominal_coverage"]
    changed = rehash(payload)
    path = tmp_path / "registry.sqlite"
    in_review(path, changed, policy(minimum_empirical_coverage=0.01))
    with pytest.raises(ContractViolation, match="coverage at series"):
        approve(path, changed)
    assert demand_registry_status(path, changed.artifact_id).revision == 2


def test_point_and_interval_approval_scopes_cannot_silently_mix(tmp_path, packaged):
    actuals, config, bundle = packaged
    legacy_policy = LocalApprovalPolicy.model_validate_json(
        Path("configs/examples/demand-local-approval.json").read_text()
    )
    path = tmp_path / "point-policy.sqlite"
    in_review(
        path,
        bundle,
        legacy_policy.model_copy(
            update={
                "max_holdout_primary_metric": 100,
                "max_abs_holdout_bias_fraction": 100,
                "require_seasonal_noninferiority": False,
            }
        ),
    )
    with pytest.raises(ContractViolation, match="explicit interval policy"):
        approve(path, bundle)
    point = build_demand_bundle(actuals, config.selection, bundle.payload.origin, "ACTOR-001")
    assert "uncertainty" not in point.payload.model_dump(mode="json")
    assert point.payload.schema_version == "demand-bundle-v1"
    assert demand_model_card(point)["schema_version"] == "demand-model-card-v1"
    assert "intervals" not in legacy_policy.model_dump(mode="json")
    with pytest.raises(ContractViolation, match="Point-only"):
        predict_demand_interval_bundle(point)
    interval_path = tmp_path / "interval-policy.sqlite"
    in_review(interval_path, point, policy())
    with pytest.raises(ContractViolation, match="requires an interval artifact"):
        approve(interval_path, point)


@pytest.mark.parametrize(
    "mutation",
    [
        "radius",
        "calibration_hash",
        "config_hash",
        "experiment",
        "frequency",
        "cells",
        "count",
        "target",
        "metrics",
        "series_metrics",
        "metric_count",
        "metric_coverage",
        "schema",
        "missing",
        "parameters",
    ],
)
def test_inconsistent_or_tampered_evidence_is_rejected(packaged, mutation):
    _, _, bundle = packaged
    payload = copy.deepcopy(bundle.payload.model_dump(mode="json"))
    evidence = payload["uncertainty"]
    if mutation == "radius":
        evidence["calibration"]["cells"][0]["radius"] += 1
    elif mutation == "calibration_hash":
        evidence["calibration_sha256"] = "0" * 64
    elif mutation == "config_hash":
        evidence["experiment_sha256"] = "0" * 64
    elif mutation == "experiment":
        evidence["experiment"]["selection"]["primary_metric"] = "wape"
    elif mutation == "frequency":
        evidence["calibration"]["frequency"] = "W-MON"
    elif mutation == "cells":
        evidence["calibration"]["cells"].pop()
    elif mutation == "count":
        evidence["calibration"]["cells"][0]["n_predictions"] += 1
    elif mutation == "target":
        evidence["calibration"]["last_target_date"] = "2020-01-01"
    elif mutation == "metrics":
        evidence["holdout_metrics"].pop()
    elif mutation == "series_metrics":
        evidence["series_holdout_metrics"].pop()
    elif mutation == "metric_count":
        evidence["holdout_metrics"][0]["n_predictions"] += 1
    elif mutation == "metric_coverage":
        evidence["holdout_metrics"][0]["coverage_gap"] += 0.1
    elif mutation == "schema":
        payload["schema_version"] = "demand-bundle-v1"
    elif mutation == "missing":
        payload["uncertainty"] = None
    elif mutation == "parameters":
        evidence["experiment"]["calibration"]["model_parameters"]["alpha"] = 0.5
    if mutation in {"cells", "count", "target", "frequency"}:
        evidence["calibration_sha256"] = content_digest(evidence["calibration"])
    with pytest.raises(ValidationError):
        rehash(payload)


def test_raw_tampering_changes_artifact_id(packaged):
    _, _, bundle = packaged
    data = bundle.model_dump(mode="json")
    data["payload"]["uncertainty"]["calibration_predictions_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content digest"):
        DemandModelBundle.model_validate(data)


@pytest.mark.parametrize("horizon", [0, 3, True, 1.5])
def test_invalid_interval_horizons_fail(packaged, horizon):
    with pytest.raises(ContractViolation, match="horizon"):
        predict_demand_interval_bundle(packaged[2], horizon)


def test_mismatched_packaging_experiment_rejected(packaged):
    actuals, config, bundle = packaged
    with pytest.raises(ContractViolation, match="matching selection"):
        build_demand_bundle(
            actuals,
            config.selection.model_copy(update={"primary_metric": "wape"}),
            bundle.payload.origin,
            "ACTOR-001",
            uncertainty=config,
        )
    with pytest.raises(ContractViolation, match="origin"):
        build_demand_bundle(
            actuals, config.selection, date(2020, 1, 1), "ACTOR-001", uncertainty=config
        )


def test_policy_version_requires_interval_thresholds():
    data = policy().model_dump()
    for changes in [
        {"schema_version": "demand-local-policy-v1"},
        {"intervals": None},
        {"intervals": {**data["intervals"], "minimum_calibration_predictions": 0}},
        {"intervals": {**data["intervals"], "maximum_pinball_loss": np.inf}},
    ]:
        with pytest.raises(ValidationError):
            LocalApprovalPolicy.model_validate({**data, **changes})


def test_selected_statistical_interval_artifact_restores_without_optimization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    actuals, config, original = experiment()
    for _, series in actuals.groupby("sku_id"):
        index = np.arange(len(series), dtype=float)
        actuals.loc[series.index, "demand_units"] = (
            20
            + 2 * index
            + 4 * np.sin(index * np.pi / 2)
            + np.random.default_rng(7).normal(0, 0.1, len(index))
        )
    payload = config.model_dump(mode="json")
    for block in [
        payload["selection"]["validation"],
        payload["calibration"],
        payload["selection"]["holdout"],
    ]:
        block["models"] = ["naive", "seasonal_naive", "ets"]
    statistical = DemandUncertaintyConfig.model_validate(payload)
    bundle = build_demand_bundle(
        actuals,
        statistical.selection,
        original.payload.origin,
        "ACTOR-001",
        uncertainty=statistical,
    )
    assert bundle.payload.decision.selected_model == "ets"
    expected = predict_demand_interval_bundle(bundle)
    path = tmp_path / "statistical.json"
    save_demand_bundle(bundle, path)
    fit = ExponentialSmoothing.fit

    def fixed_coefficients_only(self, *args, **kwargs):
        assert kwargs.get("optimized") is False, "Restore must not run an optimizer"
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(ExponentialSmoothing, "fit", fixed_coefficients_only)
    restored = load_demand_bundle(path)
    pd.testing.assert_frame_equal(
        expected, predict_demand_interval_bundle(restored), rtol=1e-12, atol=1e-12
    )


def test_original_interval_bundle_restores_in_fresh_core_only_process(tmp_path: Path, packaged):
    import subprocess
    import sys

    path = tmp_path / "model.json"
    bundle = packaged[2]
    save_demand_bundle(bundle, path)
    expected = content_digest(dataframe_records(predict_demand_interval_bundle(bundle)))
    code = """
import importlib.abc
import sys
from pathlib import Path

class BlockStatistics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "statsmodels" or fullname.startswith("statsmodels."):
            raise ImportError("statsmodels deliberately unavailable")

sys.meta_path.insert(0, BlockStatistics())
from revenue_prediction.core.inference.demand_artifacts import (
    content_digest, dataframe_records, load_demand_bundle, predict_demand_interval_bundle,
)
bundle = load_demand_bundle(Path(sys.argv[1]))
print(content_digest(dataframe_records(predict_demand_interval_bundle(bundle))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == expected
