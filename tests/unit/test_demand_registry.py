"""Local roles, separation of duties, fail-closed gates and audit transactions."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.inference.demand_artifacts import (
    DemandModelBundle,
    build_demand_bundle,
    content_digest,
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


@pytest.fixture(scope="module")
def bundle() -> DemandModelBundle:
    frame = generate_synthetic_demand_dataset(
        DemandDataConfig(n_skus=1, n_channels=1, n_geographies=1)
    )
    frame["demand_units"] = np.arange(60, dtype=float) + 1
    common = {"models": ("naive", "seasonal_naive", "drift"), "horizon": 2, "n_origins": 2}
    config = DemandSelectionConfig(
        validation=DemandBacktestConfig(
            first_origin=date(2023, 1, 1), evaluation_as_of=date(2023, 6, 1), **common
        ),
        holdout=DemandBacktestConfig(
            first_origin=date(2023, 6, 1), evaluation_as_of=date(2024, 1, 1), **common
        ),
    )
    return build_demand_bundle(frame, config, date(2025, 1, 1), "ACTOR-001")


def approval_policy(**updates: object) -> LocalApprovalPolicy:
    return LocalApprovalPolicy.model_validate(
        {
            "principals": [
                {"actor_id": "ACTOR-001", "roles": ["builder", "submitter", "reviewer"]},
                {"actor_id": "ACTOR-002", "roles": ["reviewer"]},
                {"actor_id": "ACTOR-003", "roles": ["submitter", "reviewer"]},
            ],
            "max_holdout_primary_metric": 1.0,
            "max_abs_holdout_bias_fraction": 0.1,
            **updates,
        }
    )


@pytest.fixture
def registry(tmp_path: Path, bundle: DemandModelBundle) -> Path:
    path = tmp_path / "models.sqlite"
    initialize_demand_registry(path, approval_policy())
    register_demand_model(path, bundle, "ACTOR-001", "Synthetic evaluation complete")
    return path


def submit(registry: Path, bundle: DemandModelBundle) -> None:
    transition_demand_model(
        registry, bundle.artifact_id, "ACTOR-001", "submit", "Request independent review", 1
    )


def approve(registry: Path, bundle: DemandModelBundle) -> None:
    transition_demand_model(
        registry,
        bundle.artifact_id,
        "ACTOR-002",
        "approve",
        "Reviewed synthetic evidence and limitations",
        2,
        True,
    )


def test_register_submit_approve_forecast_revoke_lifecycle(
    registry: Path, bundle: DemandModelBundle
) -> None:
    assert demand_registry_status(registry, bundle.artifact_id).status == "draft"
    with pytest.raises(ContractViolation, match="draft"):
        predict_registered_demand(registry, bundle.artifact_id)
    submit(registry, bundle)
    with pytest.raises(ContractViolation, match="in_review"):
        predict_registered_demand(registry, bundle.artifact_id)
    approve(registry, bundle)
    record = demand_registry_status(registry, bundle.artifact_id)
    assert record.status == "approved" and record.revision == 3
    assert record.events[1].previous_event_sha256 == content_digest(
        record.events[0].model_dump(mode="json")
    )
    forecast, lineage = predict_registered_demand(registry, bundle.artifact_id)
    assert forecast.approval_status.eq("unapproved").all()
    assert lineage["artifact_id"] == bundle.artifact_id
    assert lineage["registry_revision"] == 3
    assert lineage["approval_event_sha256"] == record.event_sha256
    assert lineage["model_approval_status"] == "approved"
    assert lineage["forecast_approval_status"] == "unapproved"
    revoked = transition_demand_model(
        registry, bundle.artifact_id, "ACTOR-002", "revoke", "Suspend local use", 3
    )
    assert revoked.status == "revoked" and revoked.revision == 4
    with pytest.raises(ContractViolation, match="revoked"):
        predict_registered_demand(registry, bundle.artifact_id)
    with pytest.raises(ContractViolation, match="transition"):
        transition_demand_model(
            registry,
            bundle.artifact_id,
            "ACTOR-002",
            "approve",
            "Cannot revive terminal version",
            4,
            True,
        )


def test_no_auto_promotion_reset_overwrite_or_unknown_artifacts(
    registry: Path, bundle: DemandModelBundle
) -> None:
    with pytest.raises(FileExistsError):
        initialize_demand_registry(registry, approval_policy())
    with pytest.raises(ContractViolation, match="already registered"):
        register_demand_model(registry, bundle, "ACTOR-001", "Cannot reset version")
    with pytest.raises(ContractViolation, match="Unknown"):
        demand_registry_status(registry, "0" * 64)
    with pytest.raises(ContractViolation, match="transition"):
        transition_demand_model(
            registry, bundle.artifact_id, "ACTOR-002", "approve", "Cannot skip submission", 1, True
        )
    assert demand_registry_status(registry, bundle.artifact_id).revision == 1


def test_roles_separation_acknowledgement_and_reasons(
    registry: Path, bundle: DemandModelBundle
) -> None:
    with pytest.raises(ContractViolation, match="submitter"):
        transition_demand_model(
            registry, bundle.artifact_id, "ACTOR-002", "submit", "Not a submitter", 1
        )
    submit(registry, bundle)
    for actor, expected in [("ACTOR-001", "self-review"), ("ACTOR-999", "reviewer")]:
        with pytest.raises(ContractViolation, match=expected):
            transition_demand_model(
                registry, bundle.artifact_id, actor, "approve", "Review", 2, True
            )
    with pytest.raises(ContractViolation, match="acknowledgement"):
        transition_demand_model(registry, bundle.artifact_id, "ACTOR-002", "approve", "Review", 2)
    with pytest.raises(ValidationError, match="reason"):
        transition_demand_model(registry, bundle.artifact_id, "ACTOR-002", "approve", "  ", 2, True)
    assert demand_registry_status(registry, bundle.artifact_id).revision == 2


def test_submitter_cannot_review_even_if_not_creator(
    registry: Path, bundle: DemandModelBundle
) -> None:
    transition_demand_model(
        registry, bundle.artifact_id, "ACTOR-003", "submit", "Separate submitter", 1
    )
    with pytest.raises(ContractViolation, match="self-review"):
        transition_demand_model(
            registry, bundle.artifact_id, "ACTOR-003", "reject", "Cannot review own submission", 2
        )


def test_stale_revision_is_rejected_without_appending(
    registry: Path, bundle: DemandModelBundle
) -> None:
    submit(registry, bundle)
    with pytest.raises(ContractViolation, match="Stale"):
        transition_demand_model(
            registry, bundle.artifact_id, "ACTOR-002", "approve", "Stale client", 1, True
        )
    assert demand_registry_status(registry, bundle.artifact_id).revision == 2
    approve(registry, bundle)
    with pytest.raises(ContractViolation, match="Stale"):
        transition_demand_model(
            registry, bundle.artifact_id, "ACTOR-002", "reject", "Racing stale decision", 2
        )


def test_revision_not_wall_clock_orders_audit_events(
    registry: Path,
    bundle: DemandModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import revenue_prediction.ops.demand_registry as registry_module

    previous = demand_registry_status(registry, bundle.artifact_id).events[-1].recorded_at

    class AdjustedClock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return previous - timedelta(seconds=5)

    monkeypatch.setattr(registry_module, "datetime", AdjustedClock)
    submit(registry, bundle)
    record = demand_registry_status(registry, bundle.artifact_id)
    assert record.revision == 2
    assert record.events[-1].recorded_at < previous


@pytest.mark.parametrize("revision", [0, -1, True, 1.5])
def test_invalid_revision_types_are_rejected(
    registry: Path, bundle: DemandModelBundle, revision: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        transition_demand_model(
            registry, bundle.artifact_id, "ACTOR-001", "submit", "Invalid revision", revision
        )
    assert demand_registry_status(registry, bundle.artifact_id).revision == 1


def test_rejection_is_terminal_but_auditable(registry: Path, bundle: DemandModelBundle) -> None:
    submit(registry, bundle)
    record = transition_demand_model(
        registry, bundle.artifact_id, "ACTOR-002", "reject", "Review requires a new version", 2
    )
    assert record.status == "rejected"
    assert len(record.events) == 3
    with pytest.raises(ContractViolation, match="rejected"):
        predict_registered_demand(registry, bundle.artifact_id)


def test_unauthorized_registration_rolls_back(tmp_path: Path, bundle: DemandModelBundle) -> None:
    registry = tmp_path / "registry.sqlite"
    initialize_demand_registry(registry, approval_policy())
    for actor, match in [("ACTOR-002", "builder"), ("ACTOR-999", "builder")]:
        with pytest.raises(ContractViolation, match=match):
            register_demand_model(registry, bundle, actor, "Role check")
    with pytest.raises(ContractViolation, match="Unknown"):
        demand_registry_status(registry, bundle.artifact_id)
    assert register_demand_model(registry, bundle, "ACTOR-001", "Valid registration").revision == 1


@pytest.mark.parametrize(
    "damage", ["event_hash", "event_delete", "tail_delete", "policy", "artifact"]
)
def test_local_tamper_evidence_fails_closed(
    registry: Path, bundle: DemandModelBundle, damage: str
) -> None:
    submit(registry, bundle)
    approve(registry, bundle)
    with sqlite3.connect(registry) as connection:
        if damage == "event_hash":
            connection.execute("UPDATE events SET event_sha256=? WHERE revision=2", ("0" * 64,))
        elif damage == "event_delete":
            connection.execute("DELETE FROM events WHERE revision=2")
        elif damage == "tail_delete":
            connection.execute("DELETE FROM events WHERE revision=3")
        elif damage == "policy":
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='policy'",
                (approval_policy(max_holdout_primary_metric=2).model_dump_json(),),
            )
        else:
            raw = bundle.model_dump(mode="json")
            raw["payload"]["created_by"] = "ACTOR-003"
            connection.execute("UPDATE artifacts SET bundle_json=?", (json.dumps(raw),))
    with pytest.raises((ContractViolation, ValidationError)):
        demand_registry_status(registry, bundle.artifact_id)
    with pytest.raises((ContractViolation, ValidationError)):
        predict_registered_demand(registry, bundle.artifact_id)


@pytest.mark.parametrize("bad_metric", [None, 2.0])
def test_undefined_or_failed_holdout_gate_cannot_approve(
    tmp_path: Path, bundle: DemandModelBundle, bad_metric: float | None
) -> None:
    raw = bundle.model_dump(mode="json")
    for row in raw["payload"]["holdout_metrics"]:
        if row["model_id"] == "drift":
            row["mase"] = bad_metric
    raw["artifact_id"] = content_digest(raw["payload"])
    failing = DemandModelBundle.model_validate(raw)
    registry = tmp_path / "gates.sqlite"
    initialize_demand_registry(registry, approval_policy())
    register_demand_model(registry, failing, "ACTOR-001", "Evidence requires review")
    submit(registry, failing)
    with pytest.raises(ContractViolation, match="holdout mase"):
        approve(registry, failing)
    assert demand_registry_status(registry, failing.artifact_id).status == "in_review"


@pytest.mark.parametrize("kind", ["bias", "noninferiority"])
def test_bias_and_seasonal_noninferiority_gates(
    tmp_path: Path, bundle: DemandModelBundle, kind: str
) -> None:
    raw = bundle.model_dump(mode="json")
    for row in raw["payload"]["holdout_metrics"]:
        if row["model_id"] == "drift":
            row["bias_fraction"] = 0.2 if kind == "bias" else 0.0
            row["mase"] = 0.5 if kind == "noninferiority" else 0.0
        if row["model_id"] == "seasonal_naive" and kind == "noninferiority":
            row["mase"] = 0.1
    raw["artifact_id"] = content_digest(raw["payload"])
    failing = DemandModelBundle.model_validate(raw)
    registry = tmp_path / "gates.sqlite"
    initialize_demand_registry(registry, approval_policy())
    register_demand_model(registry, failing, "ACTOR-001", "Evidence")
    submit(registry, failing)
    with pytest.raises(ContractViolation, match="bias" if kind == "bias" else "degrades"):
        approve(registry, failing)


def test_invalid_policies_and_missing_stores_are_explicit(tmp_path: Path) -> None:
    for updates in [
        {"max_holdout_primary_metric": -1},
        {"max_abs_holdout_bias_fraction": float("inf")},
        {"principals": [{"actor_id": "ACTOR-001", "roles": ["builder", "submitter", "reviewer"]}]},
        {
            "principals": [
                {"actor_id": "ACTOR-001", "roles": ["builder", "submitter"]},
                {"actor_id": "ACTOR-001", "roles": ["reviewer"]},
            ]
        },
    ]:
        with pytest.raises(ValidationError):
            approval_policy(**updates)
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        demand_registry_status(path, "0" * 64)
    assert not path.exists()
