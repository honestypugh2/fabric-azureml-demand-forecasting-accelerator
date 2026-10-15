"""Offline demand artifact and local-governance commands."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.io import read_dataset, write_dataset
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.evaluation.demand_tuning import DemandTuningConfig
from revenue_prediction.core.evaluation.demand_uncertainty import DemandUncertaintyConfig
from revenue_prediction.core.inference.demand_artifacts import (
    build_demand_bundle,
    canonical_json,
    demand_model_card,
    load_demand_bundle,
    render_demand_model_card,
    save_demand_bundle,
)
from revenue_prediction.ops.demand_registry import (
    LocalApprovalPolicy,
    RegistryRecord,
    demand_registry_status,
    initialize_demand_registry,
    predict_registered_demand,
    register_demand_model,
    transition_demand_model,
)

app = typer.Typer(
    help="Synthetic demand model artifacts and local-only governance (not production authorization).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except (ContractViolation, ValueError, ImportError, OSError, sqlite3.Error) as exc:
        console.print(f"Demand lifecycle failed: {exc}", style="red", markup=False)
        raise typer.Exit(code=1) from exc


def _print_record(record: RegistryRecord) -> None:
    console.print(
        canonical_json(
            {
                "artifact_id": record.bundle.artifact_id,
                "status": record.status,
                "revision": record.revision,
                "event_sha256": record.event_sha256,
                "events": [event.model_dump(mode="json") for event in record.events],
                "scope": "local-model-only; forecasts remain unapproved",
            }
        ),
        markup=False,
        soft_wrap=True,
    )


@app.command("package")
def package(
    comparison_dir: Path = typer.Option(..., "--comparison-dir"),
    origin: str = typer.Option(
        ...,
        "--origin",
        help="Frozen forecast origin YYYY-MM-DD, on/after holdout evaluation cutoff",
    ),
    actor: str = typer.Option(..., "--actor", help="Neutral local actor ID, e.g. ACTOR-001"),
    output_dir: Path = typer.Option(..., "--out", help="New artifact directory; never overwrite"),
    include_intervals: bool = typer.Option(
        False, "--include-intervals", help="Recompute an uncertainty experiment into a v2 bundle"
    ),
) -> None:
    """Recompute synthetic comparison evidence and serialize the fitted winner."""
    with _errors():
        if output_dir.exists():
            raise FileExistsError(
                "Artifact output directory exists; choose a new version directory."
            )
        uncertainty = (
            DemandUncertaintyConfig.model_validate_json(
                (comparison_dir / "uncertainty_config.json").read_text(encoding="utf-8")
            )
            if include_intervals
            else None
        )
        experiment = (
            uncertainty.selection
            if uncertainty is not None
            else (
                DemandSelectionConfig.model_validate_json(
                    (comparison_dir / "experiment_config.json").read_text(encoding="utf-8")
                )
            )
        )
        tuning_path = comparison_dir / "tuning_config.json"
        tuning = (
            DemandTuningConfig.model_validate_json(tuning_path.read_text(encoding="utf-8"))
            if tuning_path.exists()
            else None
        )
        bundle = build_demand_bundle(
            read_dataset(comparison_dir / "actuals.parquet"),
            experiment,
            date.fromisoformat(origin),
            actor,
            tuning=tuning,
            uncertainty=uncertainty,
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        save_demand_bundle(bundle, output_dir / "model.json")
        (output_dir / "model_card.json").write_text(
            canonical_json(demand_model_card(bundle)) + "\n", encoding="utf-8"
        )
        (output_dir / "model_card.md").write_text(
            render_demand_model_card(bundle), encoding="utf-8"
        )
        console.print(
            canonical_json(
                {
                    "artifact_id": bundle.artifact_id,
                    "status": "unapproved",
                    "directory": str(output_dir),
                }
            ),
            markup=False,
            soft_wrap=True,
        )


@app.command("init-registry")
def init_registry(
    registry: Path = typer.Option(..., "--registry"),
    policy: Path = typer.Option(
        ..., "--policy", help="Explicit local roles and acceptance thresholds JSON"
    ),
) -> None:
    """Create a new local registry with an immutable-by-API policy."""
    with _errors():
        initialize_demand_registry(
            registry,
            LocalApprovalPolicy.model_validate_json(policy.read_text(encoding="utf-8")),
        )
        console.print("Created local-only registry. Actor IDs are not authenticated identities.")


@app.command("register")
def register(
    registry: Path = typer.Option(..., "--registry"),
    bundle: Path = typer.Option(..., "--bundle"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Register one content-addressed artifact as draft; never auto-approve."""
    with _errors():
        _print_record(register_demand_model(registry, load_demand_bundle(bundle), actor, reason))


@app.command("transition")
def transition(
    registry: Path = typer.Option(..., "--registry"),
    artifact_id: str = typer.Option(..., "--artifact-id"),
    actor: str = typer.Option(..., "--actor"),
    action: str = typer.Option(..., "--action", help="submit, approve, reject or revoke"),
    reason: str = typer.Option(..., "--reason"),
    expected_revision: int = typer.Option(..., "--expected-revision", min=1),
    acknowledge_limitations: bool = typer.Option(False, "--acknowledge-limitations"),
) -> None:
    """Apply a role/separation/gate-checked transition at an exact audit revision."""
    with _errors():
        chosen: Literal["submit", "approve", "reject", "revoke"]
        if action == "submit":
            chosen = "submit"
        elif action == "approve":
            chosen = "approve"
        elif action == "reject":
            chosen = "reject"
        elif action == "revoke":
            chosen = "revoke"
        else:
            raise ValueError("Use submit, approve, reject or revoke.")
        _print_record(
            transition_demand_model(
                registry,
                artifact_id,
                actor,
                chosen,
                reason,
                expected_revision,
                acknowledge_limitations=acknowledge_limitations,
            )
        )


@app.command("status")
def status(
    registry: Path = typer.Option(..., "--registry"),
    artifact_id: str = typer.Option(..., "--artifact-id"),
) -> None:
    """Verify artifact, policy and complete approval chain before reporting status."""
    with _errors():
        _print_record(demand_registry_status(registry, artifact_id))


@app.command("forecast")
def forecast(
    registry: Path = typer.Option(..., "--registry"),
    artifact_id: str = typer.Option(..., "--artifact-id"),
    horizon: int | None = typer.Option(None, "--horizon", min=1),
    output_dir: Path = typer.Option(..., "--out", help="New inference directory; never overwrite"),
    include_intervals: bool = typer.Option(
        False,
        "--include-intervals",
        help="Require a reviewed v2 artifact and export bound intervals",
    ),
) -> None:
    """Require a locally approved model; output forecasts still need separate approval."""
    with _errors():
        if output_dir.exists():
            raise FileExistsError("Inference output directory exists; choose a new directory.")
        predictions, manifest = predict_registered_demand(
            registry, artifact_id, horizon, include_intervals=include_intervals
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        write_dataset(predictions, output_dir / "predictions.parquet")
        (output_dir / "lineage.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        console.print(
            json.dumps(
                {
                    "model_approval_status": "approved",
                    "forecast_approval_status": "unapproved",
                    "artifact_id": artifact_id,
                    "directory": str(output_dir),
                }
            ),
            markup=False,
            soft_wrap=True,
        )
