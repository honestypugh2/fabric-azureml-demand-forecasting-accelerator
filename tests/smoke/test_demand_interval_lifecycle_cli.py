"""Recomputed packaging, independent approval, interval export and revocation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.inference.demand_artifacts import (
    content_digest,
    dataframe_records,
    load_demand_bundle,
    predict_demand_interval_bundle,
)
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


def invoke(*args: str) -> str:
    response = runner.invoke(app, list(args))
    assert response.exit_code == 0, response.output
    return response.output


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_governed_interval_cli(tmp_path: Path, frequency: str) -> None:
    config = json.loads(Path("configs/examples/demand-uncertainty.json").read_text())
    origin = "2024-01-01"
    if frequency == "W-MON":
        base = pd.Timestamp("2020-01-06")
        for block in [
            config["selection"]["validation"],
            config["calibration"],
            config["selection"]["holdout"],
        ]:
            block["frequency"] = frequency
            block["seasonal_periods"] = 4
            for field in ["first_origin", "evaluation_as_of"]:
                value = pd.Timestamp(block[field])
                index = (value.year - 2020) * 12 + value.month - 1
                block[field] = (base + pd.Timedelta(weeks=index)).date().isoformat()
        origin = (base + pd.Timedelta(weeks=50)).date().isoformat()
    config_path = tmp_path / "uncertainty.json"
    config_path.write_text(json.dumps(config))
    comparison = tmp_path / "comparison"
    invoke("demand-uncertainty", "--config", str(config_path), "--out", str(comparison))
    # Packaging recomputes evidence rather than trusting persisted reports or calibration.
    (comparison / "calibration.json").write_text("{}")
    (comparison / "interval_metrics.csv").write_text("invalid")
    artifact = tmp_path / "artifact"
    package_args = [
        "demand-lifecycle",
        "package",
        "--comparison-dir",
        str(comparison),
        "--include-intervals",
        "--origin",
        origin,
        "--actor",
        "ACTOR-001",
        "--out",
        str(artifact),
    ]
    summary = json.loads(invoke(*package_args))
    artifact_id = summary["artifact_id"]
    bundle = load_demand_bundle(artifact / "model.json")
    assert bundle.payload.uncertainty is not None
    card = json.loads((artifact / "model_card.json").read_text())
    assert card["schema_version"] == "demand-model-card-v2"
    assert "uncertainty" in card
    assert runner.invoke(app, package_args).exit_code == 1
    policy = json.loads(Path("configs/examples/demand-interval-approval.json").read_text())
    policy.update(
        {
            "max_holdout_primary_metric": 100,
            "max_abs_holdout_bias_fraction": 100,
            "require_seasonal_noninferiority": False,
        }
    )
    policy["intervals"].update(
        {
            "minimum_empirical_coverage": 0.0,
            "maximum_mean_width": 1e6,
            "maximum_mean_interval_score": 1e6,
            "maximum_pinball_loss": 1e6,
        }
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    registry = tmp_path / "registry.sqlite"
    invoke(
        "demand-lifecycle",
        "init-registry",
        "--registry",
        str(registry),
        "--policy",
        str(policy_path),
    )
    invoke(
        "demand-lifecycle",
        "register",
        "--registry",
        str(registry),
        "--bundle",
        str(artifact / "model.json"),
        "--actor",
        "ACTOR-001",
        "--reason",
        "Synthetic interval experiment",
    )
    forecast_args = [
        "demand-lifecycle",
        "forecast",
        "--registry",
        str(registry),
        "--artifact-id",
        artifact_id,
        "--include-intervals",
        "--out",
    ]
    blocked = tmp_path / "blocked"
    assert runner.invoke(app, [*forecast_args, str(blocked)]).exit_code == 1
    assert not blocked.exists()
    common = [
        "demand-lifecycle",
        "transition",
        "--registry",
        str(registry),
        "--artifact-id",
        artifact_id,
        "--reason",
        "Independent evidence review",
    ]
    invoke(*common, "--actor", "ACTOR-001", "--action", "submit", "--expected-revision", "1")
    approval = [*common, "--actor", "ACTOR-002", "--action", "approve", "--expected-revision", "2"]
    assert runner.invoke(app, approval).exit_code == 1
    invoke(*approval, "--acknowledge-limitations")
    output = tmp_path / "forecast"
    invoke(*forecast_args, str(output))
    predictions = read_dataset(output / "predictions.parquet")
    pd.testing.assert_frame_equal(
        predictions, predict_demand_interval_bundle(bundle), check_exact=True
    )
    manifest = json.loads((output / "lineage.json").read_text())
    assert manifest["schema_version"] == "demand-inference-lineage-v2"
    assert manifest["calibration_sha256"] == bundle.payload.uncertainty.calibration_sha256
    assert manifest["forecast_sha256"] == content_digest(dataframe_records(predictions))
    assert manifest["registry_revision"] == 3
    assert manifest["forecast_approval_status"] == "unapproved"
    assert runner.invoke(app, [*forecast_args, str(output)]).exit_code == 1
    invoke(*common, "--actor", "ACTOR-002", "--action", "revoke", "--expected-revision", "3")
    assert runner.invoke(app, [*forecast_args, str(blocked)]).exit_code == 1
    assert not blocked.exists()
    assert (
        json.loads(
            invoke(
                "demand-lifecycle",
                "status",
                "--registry",
                str(registry),
                "--artifact-id",
                artifact_id,
            )
        )["status"]
        == "revoked"
    )


def test_interval_package_failure_creates_no_directory(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    response = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "package",
            "--comparison-dir",
            str(tmp_path),
            "--include-intervals",
            "--origin",
            "2025-01-01",
            "--actor",
            "ACTOR-001",
            "--out",
            str(output),
        ],
    )
    assert response.exit_code == 1
    assert "Demand lifecycle failed" in response.output
    assert not output.exists()
