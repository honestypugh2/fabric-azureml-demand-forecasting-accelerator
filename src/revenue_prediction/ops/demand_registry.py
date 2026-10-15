"""Transactional local model governance, explicitly not an authenticated service."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.inference.demand_artifacts import (
    ActorId,
    ArtifactRecord,
    DemandModelBundle,
    Digest,
    canonical_json,
    content_digest,
    dataframe_records,
    demand_model_card,
    predict_demand_bundle,
    verify_demand_bundle,
)

ModelStatus = Literal["draft", "in_review", "approved", "rejected", "revoked"]
ReviewAction = Literal["register", "submit", "approve", "reject", "revoke"]
Role = Literal["builder", "submitter", "reviewer"]


class LocalPrincipal(ArtifactRecord):
    actor_id: ActorId
    roles: tuple[Role, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_roles(self) -> LocalPrincipal:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("Principal roles must be unique.")
        return self


class LocalApprovalPolicy(ArtifactRecord):
    """Explicit illustrative acceptance limits, fixed when a registry is created."""

    schema_version: Literal["demand-local-policy-v1"] = "demand-local-policy-v1"
    principals: tuple[LocalPrincipal, ...] = Field(min_length=2)
    max_holdout_primary_metric: float = Field(ge=0)
    max_abs_holdout_bias_fraction: float = Field(ge=0)
    require_seasonal_noninferiority: bool = True

    @model_validator(mode="after")
    def _separation_possible(self) -> LocalApprovalPolicy:
        actors = [item.actor_id for item in self.principals]
        if len(set(actors)) != len(actors):
            raise ValueError("Policy actor IDs must be unique.")
        builders = {item.actor_id for item in self.principals if "builder" in item.roles}
        submitters = {item.actor_id for item in self.principals if "submitter" in item.roles}
        reviewers = {item.actor_id for item in self.principals if "reviewer" in item.roles}
        if not any(
            reviewer not in {builder, submitter}
            for builder in builders
            for submitter in submitters
            for reviewer in reviewers
        ):
            raise ValueError("Policy needs builders/submitters and an independent reviewer.")
        return self


class ApprovalEvent(ArtifactRecord):
    schema_version: Literal["demand-approval-event-v1"] = "demand-approval-event-v1"
    artifact_id: Digest
    revision: int = Field(ge=1, strict=True)
    previous_event_sha256: Digest | None
    policy_sha256: Digest
    model_card_sha256: Digest
    action: ReviewAction
    status: ModelStatus
    actor_id: ActorId
    reason: str = Field(min_length=1, max_length=1000)
    acknowledged_limitations: bool
    recorded_at: datetime

    @model_validator(mode="after")
    def _event_fields(self) -> ApprovalEvent:
        if not self.reason.strip():
            raise ValueError("An audit reason is required.")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() != UTC.utcoffset(
            self.recorded_at
        ):
            raise ValueError("Audit timestamps must be UTC-aware.")
        return self


@dataclass(frozen=True)
class RegistryRecord:
    bundle: DemandModelBundle
    status: ModelStatus
    revision: int
    event_sha256: str
    events: tuple[ApprovalEvent, ...]


_TRANSITIONS: dict[tuple[str, str], ModelStatus] = {
    ("absent", "register"): "draft",
    ("draft", "submit"): "in_review",
    ("in_review", "approve"): "approved",
    ("in_review", "reject"): "rejected",
    ("approved", "revoke"): "revoked",
}


def initialize_demand_registry(path: Path, policy: LocalApprovalPolicy) -> None:
    """Create a new local-only registry; never replace an existing policy/store."""
    policy = LocalApprovalPolicy.model_validate(policy.model_dump())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY, bundle_json TEXT NOT NULL,
                head_revision INTEGER NOT NULL DEFAULT 0, head_sha256 TEXT
            );
            CREATE TABLE events (
                artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                revision INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                event_sha256 TEXT NOT NULL,
                PRIMARY KEY (artifact_id, revision)
            );
        """)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("schema_version", "demand-local-registry-v1"),
                ("policy", canonical_json(policy.model_dump(mode="json"))),
            ],
        )


@contextmanager
def _registry(path: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None or row[0] != "demand-local-registry-v1":
                raise ContractViolation("Unsupported demand registry schema.")
            yield connection
    finally:
        connection.close()


def _policy(connection: sqlite3.Connection) -> LocalApprovalPolicy:
    row = connection.execute("SELECT value FROM metadata WHERE key='policy'").fetchone()
    if row is None:
        raise ContractViolation("Demand registry has no approval policy.")
    return LocalApprovalPolicy.model_validate_json(row[0])


def _require_role(policy: LocalApprovalPolicy, actor_id: str, role: Role) -> None:
    if not any(item.actor_id == actor_id and role in item.roles for item in policy.principals):
        raise ContractViolation(f"Local actor lacks the required {role} role.")


def _approval_gates(bundle: DemandModelBundle, policy: LocalApprovalPolicy) -> None:
    payload = bundle.payload
    metric = payload.experiment.primary_metric
    selected = {
        row.horizon: row
        for row in payload.holdout_metrics
        if row.model_id == payload.decision.selected_model
    }
    seasonal = {
        row.horizon: row for row in payload.holdout_metrics if row.model_id == "seasonal_naive"
    }
    for horizon, row in selected.items():
        value = getattr(row, metric)
        reference = getattr(seasonal[horizon], metric)
        if value is None or value > policy.max_holdout_primary_metric:
            raise ContractViolation(
                f"Approval blocked: holdout {metric} at horizon {horizon} is undefined or exceeds policy."
            )
        if (
            row.bias_fraction is None
            or abs(row.bias_fraction) > policy.max_abs_holdout_bias_fraction
        ):
            raise ContractViolation(
                f"Approval blocked: holdout bias at horizon {horizon} is undefined or exceeds policy."
            )
        if policy.require_seasonal_noninferiority and (reference is None or value > reference):
            raise ContractViolation(
                f"Approval blocked: holdout {metric} at horizon {horizon} degrades seasonal naive."
            )


def _validate_event(
    event: ApprovalEvent,
    bundle: DemandModelBundle,
    policy: LocalApprovalPolicy,
    prior: tuple[ApprovalEvent, ...],
    previous_hash: str | None,
) -> None:
    status = prior[-1].status if prior else "absent"
    if (
        event.artifact_id != bundle.artifact_id
        or event.revision != len(prior) + 1
        or event.previous_event_sha256 != previous_hash
    ):
        raise ContractViolation("Broken model approval audit chain.")
    if event.policy_sha256 != content_digest(
        policy.model_dump(mode="json")
    ) or event.model_card_sha256 != content_digest(demand_model_card(bundle)):
        raise ContractViolation("Approval policy/model card differs from audit evidence.")
    if _TRANSITIONS.get((status, event.action)) != event.status:
        raise ContractViolation(f"Invalid model lifecycle transition: {status} -> {event.action}.")
    role: Role = (
        "builder"
        if event.action == "register"
        else "submitter"
        if event.action == "submit"
        else "reviewer"
    )
    _require_role(policy, event.actor_id, role)
    if event.action == "register" and event.actor_id != bundle.payload.created_by:
        raise ContractViolation("Registration actor must match artifact creator.")
    if event.action in {"approve", "reject", "revoke"}:
        submitter = next(item.actor_id for item in prior if item.action == "submit")
        if event.actor_id in {bundle.payload.created_by, submitter}:
            raise ContractViolation(
                "Independent reviewer required; no creator/submitter self-review."
            )
    if event.action == "approve":
        if not event.acknowledged_limitations:
            raise ContractViolation(
                "Approval requires explicit model-card limitations acknowledgement."
            )
        _approval_gates(bundle, policy)


def _read_record(connection: sqlite3.Connection, artifact_id: str) -> RegistryRecord:
    row = connection.execute(
        "SELECT bundle_json, head_revision, head_sha256 FROM artifacts WHERE artifact_id=?",
        (artifact_id,),
    ).fetchone()
    if row is None:
        raise ContractViolation("Unknown model artifact in this local registry.")
    bundle = DemandModelBundle.model_validate_json(row[0])
    if artifact_id != bundle.artifact_id:
        raise ContractViolation("Stored artifact identity mismatch.")
    verify_demand_bundle(bundle)
    policy = _policy(connection)
    events: tuple[ApprovalEvent, ...] = ()
    previous_hash = None
    for stored_revision, event_json, event_hash in connection.execute(
        "SELECT revision, event_json, event_sha256 FROM events WHERE artifact_id=? ORDER BY revision",
        (artifact_id,),
    ):
        event = ApprovalEvent.model_validate_json(event_json)
        if stored_revision != event.revision:
            raise ContractViolation("Stored approval revision differs from audit event.")
        if content_digest(event.model_dump(mode="json")) != event_hash:
            raise ContractViolation("Model approval event digest mismatch.")
        _validate_event(event, bundle, policy, events, previous_hash)
        events += (event,)
        previous_hash = event_hash
    if not events or previous_hash is None:
        raise ContractViolation("Model artifact has no registration audit event.")
    if len(events) != row[1] or previous_hash != row[2]:
        raise ContractViolation("Approval audit head differs; events were removed or replaced.")
    return RegistryRecord(bundle, events[-1].status, events[-1].revision, previous_hash, events)


def _append_event(
    connection: sqlite3.Connection,
    bundle: DemandModelBundle,
    actor_id: str,
    action: ReviewAction,
    reason: str,
    prior: tuple[ApprovalEvent, ...],
    previous_hash: str | None,
    acknowledged: bool,
) -> None:
    policy = _policy(connection)
    current = prior[-1].status if prior else "absent"
    status = _TRANSITIONS.get((current, action))
    if status is None:
        raise ContractViolation(f"Invalid model lifecycle transition: {current} -> {action}.")
    event = ApprovalEvent(
        artifact_id=bundle.artifact_id,
        revision=len(prior) + 1,
        previous_event_sha256=previous_hash,
        policy_sha256=content_digest(policy.model_dump(mode="json")),
        model_card_sha256=content_digest(demand_model_card(bundle)),
        action=action,
        status=status,
        actor_id=actor_id,
        reason=reason,
        acknowledged_limitations=acknowledged,
        recorded_at=datetime.now(UTC),
    )
    _validate_event(event, bundle, policy, prior, previous_hash)
    body = event.model_dump(mode="json")
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?)",
        (
            bundle.artifact_id,
            event.revision,
            canonical_json(body),
            content_digest(body),
        ),
    )
    connection.execute(
        "UPDATE artifacts SET head_revision=?, head_sha256=? WHERE artifact_id=?",
        (event.revision, content_digest(body), bundle.artifact_id),
    )


def register_demand_model(
    path: Path, bundle: DemandModelBundle, actor_id: str, reason: str
) -> RegistryRecord:
    verify_demand_bundle(bundle)
    with _registry(path, write=True) as connection:
        if connection.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id=?", (bundle.artifact_id,)
        ).fetchone():
            raise ContractViolation(
                "Artifact already registered; registration never resets approval state."
            )
        connection.execute(
            "INSERT INTO artifacts (artifact_id, bundle_json) VALUES (?, ?)",
            (bundle.artifact_id, canonical_json(bundle.model_dump(mode="json"))),
        )
        _append_event(connection, bundle, actor_id, "register", reason, (), None, False)
        return _read_record(connection, bundle.artifact_id)


def transition_demand_model(
    path: Path,
    artifact_id: str,
    actor_id: str,
    action: Literal["submit", "approve", "reject", "revoke"],
    reason: str,
    expected_revision: int,
    acknowledge_limitations: bool = False,
) -> RegistryRecord:
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError("expected_revision must be a positive integer.")
    with _registry(path, write=True) as connection:
        record = _read_record(connection, artifact_id)
        if record.revision != expected_revision:
            raise ContractViolation(
                "Stale approval revision; refresh status before changing lifecycle."
            )
        _append_event(
            connection,
            record.bundle,
            actor_id,
            action,
            reason,
            record.events,
            record.event_sha256,
            acknowledge_limitations,
        )
        return _read_record(connection, artifact_id)


def demand_registry_status(path: Path, artifact_id: str) -> RegistryRecord:
    with _registry(path) as connection:
        return _read_record(connection, artifact_id)


def predict_registered_demand(
    path: Path,
    artifact_id: str,
    horizon: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Require approved model at a transactionally consistent audit revision.

    The resulting point forecasts remain unapproved. The sidecar records the
    approval snapshot, not ongoing authorization or an official forecast.
    """
    with _registry(path) as connection:
        record = _read_record(connection, artifact_id)
        if record.status != "approved":
            raise ContractViolation(
                f"Model is {record.status}; approved-model inference is blocked."
            )
        forecast = predict_demand_bundle(record.bundle, horizon)
        manifest: dict[str, object] = {
            "schema_version": "demand-inference-lineage-v1",
            "artifact_id": artifact_id,
            "registry_revision": record.revision,
            "approval_event_sha256": record.event_sha256,
            "model_approval_status": record.status,
            "forecast_approval_status": "unapproved",
            "forecast_sha256": content_digest(dataframe_records(forecast)),
            "forecast_origin": record.bundle.payload.origin.isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        return forecast, manifest
