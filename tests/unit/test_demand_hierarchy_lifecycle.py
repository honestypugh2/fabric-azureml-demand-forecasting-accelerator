"""Hierarchy-bound artifacts and fail-closed local approval/inference."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandForecastRequest
from revenue_prediction.core.data.demand_hierarchy import build_demand_hierarchy
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_reconciliation import run_demand_reconciliation
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.inference.demand_artifacts import (
    DemandModelBundle,
    build_demand_bundle,
    content_digest,
    dataframe_records,
    demand_model_card,
    load_demand_bundle,
    predict_demand_bundle,
    predict_demand_hierarchy_bundle,
    save_demand_bundle,
)
from revenue_prediction.core.models.demand_reconciliation import reconcile_demand_points
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


def experiment(frequency="MS", ensemble=None):
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
        "n_origins": 2,
    }
    design = {
        "validation": dict(
            **common, first_origin=dates.iloc[25].date(), evaluation_as_of=dates.iloc[30].date()
        ),
        "holdout": dict(
            **common, first_origin=dates.iloc[35].date(), evaluation_as_of=dates.iloc[40].date()
        ),
    }
    if ensemble is not None:
        design["ensemble_members"] = ensemble
    config = DemandSelectionConfig.model_validate(design)
    hierarchy = build_demand_hierarchy(
        actuals,
        DemandForecastRequest(
            forecast_origin=config.validation.first_origin, frequency=config.validation.frequency
        ),
    )
    bundle = build_demand_bundle(
        actuals, config, dates.iloc[45].date(), "ACTOR-001", hierarchy=hierarchy
    )
    return actuals, config, bundle


def policy(**updates):
    return LocalApprovalPolicy.model_validate(
        {
            "schema_version": "demand-local-policy-v3",
            "principals": [
                {"actor_id": "ACTOR-001", "roles": ["builder", "submitter"]},
                {"actor_id": "ACTOR-002", "roles": ["reviewer"]},
            ],
            "max_holdout_primary_metric": 100,
            "max_abs_holdout_bias_fraction": 100,
            "require_seasonal_noninferiority": False,
            "reconciliation": {
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
            },
            **updates,
        }
    )


def review(path, bundle, limits):
    initialize_demand_registry(path, limits)
    register_demand_model(path, bundle, "ACTOR-001", "Synthetic hierarchy")
    transition_demand_model(path, bundle.artifact_id, "ACTOR-001", "submit", "Review", 1)


def approve(path, bundle):
    return transition_demand_model(
        path,
        bundle.artifact_id,
        "ACTOR-002",
        "approve",
        "Reviewed complete hierarchy evidence",
        2,
        True,
    )


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_hierarchy_roundtrip_evidence_and_governance(tmp_path, frequency):
    actuals, config, bundle = experiment(frequency)
    evidence = bundle.payload.reconciliation
    assert evidence is not None
    result = run_demand_reconciliation(actuals, config, evidence.hierarchy)
    assert evidence.holdout.predictions_sha256 == content_digest(
        dataframe_records(result.holdout.predictions)
    )
    assert evidence.holdout.coherence_sha256 == content_digest(
        dataframe_records(result.holdout.coherence)
    )
    assert bundle.payload.schema_version == "demand-bundle-v3"
    assert demand_model_card(bundle)["schema_version"] == "demand-model-card-v3"
    save_demand_bundle(bundle, tmp_path / "model.json")
    loaded = load_demand_bundle(tmp_path / "model.json")
    points = predict_demand_bundle(loaded)
    expected = reconcile_demand_points(points, evidence.hierarchy)
    pd.testing.assert_frame_equal(
        predict_demand_hierarchy_bundle(loaded), expected, check_exact=True
    )
    registry = tmp_path / "registry.sqlite"
    review(registry, loaded, policy())
    with pytest.raises(ContractViolation, match="in_review"):
        predict_registered_demand(registry, loaded.artifact_id, include_hierarchy=True)
    approved = approve(registry, loaded)
    nodes, manifest = predict_registered_demand(
        registry, loaded.artifact_id, include_hierarchy=True
    )
    pd.testing.assert_frame_equal(nodes, expected, check_exact=True)
    assert manifest["schema_version"] == "demand-inference-lineage-v3"
    assert manifest["approval_event_sha256"] == approved.event_sha256
    assert manifest["hierarchy_id"] == evidence.hierarchy.hierarchy_id
    assert manifest["forecast_sha256"] == content_digest(dataframe_records(nodes))
    assert nodes.approval_status.eq("unapproved").all()
    short, _ = predict_registered_demand(registry, loaded.artifact_id, 1, include_hierarchy=True)
    pd.testing.assert_frame_equal(short, nodes.loc[nodes.horizon == 1].reset_index(drop=True))
    with pytest.raises(ContractViolation, match="Aggregate intervals"):
        predict_registered_demand(
            registry, loaded.artifact_id, include_hierarchy=True, include_intervals=True
        )
    transition_demand_model(registry, loaded.artifact_id, "ACTOR-002", "revoke", "Suspend", 3)
    with pytest.raises(ContractViolation, match="revoked"):
        predict_registered_demand(registry, loaded.artifact_id, include_hierarchy=True)


@pytest.mark.parametrize("field", ["metrics", "node_metrics", "n_coherence_checks"])
def test_incomplete_evidence_is_rejected(field):
    _, _, bundle = experiment()
    payload = copy.deepcopy(bundle.payload.model_dump(mode="json"))
    block = payload["reconciliation"]["holdout"]
    block[field] = block[field][1:] if isinstance(block[field], list) else 1
    with pytest.raises(ValidationError, match="Hierarchy"):
        DemandModelBundle.model_validate(
            {"artifact_id": content_digest(payload), "payload": payload}
        )


def test_explicit_policy_gates_rollback_and_point_compatibility(tmp_path):
    _, _, bundle = experiment()
    raw = policy().model_dump(mode="json")
    raw["reconciliation"]["levels"][0]["maximum_primary_metric"] = 0
    registry = tmp_path / "blocked.sqlite"
    review(registry, bundle, LocalApprovalPolicy.model_validate(raw))
    with pytest.raises(ContractViolation, match="hierarchy metric"):
        approve(registry, bundle)
    assert demand_registry_status(registry, bundle.artifact_id).revision == 2
    point_policy = LocalApprovalPolicy.model_validate(
        {
            **{key: value for key, value in raw.items() if key != "reconciliation"},
            "schema_version": "demand-local-policy-v1",
        }
    )
    registry = tmp_path / "point.sqlite"
    review(registry, bundle, point_policy)
    with pytest.raises(ContractViolation, match="explicit hierarchy policy"):
        approve(registry, bundle)


def test_cli_recomputes_hierarchy_reports_and_exports(tmp_path):
    actuals, config, bundle = experiment()
    assert bundle.payload.reconciliation is not None
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    actuals.to_parquet(comparison / "actuals.parquet", index=False)
    (comparison / "experiment_config.json").write_text(config.model_dump_json())
    (comparison / "hierarchy.json").write_text(
        bundle.payload.reconciliation.hierarchy.model_dump_json()
    )
    (comparison / "holdout_metrics.csv").write_text("tampered report is not input")
    runner = CliRunner()
    output = tmp_path / "artifact"
    packaged = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "package",
            "--comparison-dir",
            str(comparison),
            "--origin",
            bundle.payload.origin.isoformat(),
            "--actor",
            "ACTOR-001",
            "--include-hierarchy",
            "--out",
            str(output),
        ],
    )
    assert packaged.exit_code == 0, packaged.output
    loaded = load_demand_bundle(output / "model.json")
    assert loaded.payload.reconciliation is not None
    assert loaded.payload.reconciliation == bundle.payload.reconciliation
    registry = tmp_path / "registry.sqlite"
    review(registry, loaded, policy())
    approve(registry, loaded)
    inference = tmp_path / "inference"
    exported = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "forecast",
            "--registry",
            str(registry),
            "--artifact-id",
            loaded.artifact_id,
            "--include-hierarchy",
            "--out",
            str(inference),
        ],
    )
    assert exported.exit_code == 0, exported.output
    assert (
        json.loads((inference / "lineage.json").read_text())["hierarchy_id"]
        == loaded.payload.reconciliation.hierarchy.hierarchy_id
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(inference / "predictions.parquet"), predict_demand_hierarchy_bundle(loaded)
    )


@pytest.mark.parametrize(
    "corruption", ["late_hierarchy", "wrong_cohort", "false_coherence", "wrong_version"]
)
def test_rehashed_hierarchy_scope_and_version_tampering_fails(corruption):
    _, config, bundle = experiment()
    raw = copy.deepcopy(bundle.payload.model_dump(mode="json"))
    evidence = raw["reconciliation"]
    if corruption == "late_hierarchy":
        evidence["hierarchy"]["payload"]["known_at"] = config.holdout.first_origin.isoformat()
    elif corruption == "wrong_cohort":
        evidence["hierarchy"]["payload"]["members"] = evidence["hierarchy"]["payload"]["members"][
            :-1
        ]
    elif corruption == "false_coherence":
        evidence["holdout"]["all_coherent"] = False
    else:
        raw["schema_version"] = "demand-bundle-v1"
    evidence["hierarchy"]["hierarchy_id"] = content_digest(evidence["hierarchy"]["payload"])
    with pytest.raises(ValidationError):
        DemandModelBundle.model_validate({"artifact_id": content_digest(raw), "payload": raw})


def test_every_node_gate_applies_even_when_level_summary_passes(tmp_path):
    _, _, bundle = experiment()
    raw = copy.deepcopy(bundle.payload.model_dump(mode="json"))
    selected = bundle.payload.decision.selected_model
    row = next(
        item
        for item in raw["reconciliation"]["holdout"]["node_metrics"]
        if item["model_id"] == selected
    )
    row["mase"] = 101.0
    edited = DemandModelBundle.model_validate({"artifact_id": content_digest(raw), "payload": raw})
    registry = tmp_path / "registry.sqlite"
    review(registry, edited, policy())
    with pytest.raises(ContractViolation, match="hierarchy metric"):
        approve(registry, edited)
    assert demand_registry_status(registry, edited.artifact_id).revision == 2
