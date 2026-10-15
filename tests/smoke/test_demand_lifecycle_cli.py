"""Offline package/card/registry/inference lifecycle with persistent audit evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.inference.demand_artifacts import (
    content_digest,
    dataframe_records,
    demand_model_card,
    load_demand_bundle,
    predict_demand_bundle,
)
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


def invoke(*arguments: str) -> str:
    result = runner.invoke(app, list(arguments))
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_package_register_approve_forecast_revoke_roundtrip(tmp_path: Path, frequency: str) -> None:
    comparison = tmp_path / "comparison"
    invoke(
        "compare-demand",
        "--frequency",
        frequency,
        "--models",
        "naive,seasonal_naive,tsb",
        "--tuning-grid",
        "configs/examples/demand-tuning.json",
        "--skus",
        "1",
        "--channels",
        "1",
        "--geographies",
        "1",
        "--out",
        str(comparison),
    )
    artifact_dir = tmp_path / "artifact"
    origin = "2025-01-01" if frequency == "MS" else "2025-01-06"
    package = json.loads(
        invoke(
            "demand-lifecycle",
            "package",
            "--comparison-dir",
            str(comparison),
            "--origin",
            origin,
            "--actor",
            "ACTOR-001",
            "--out",
            str(artifact_dir),
        )
    )
    assert package["status"] == "unapproved"
    artifact_id = package["artifact_id"]
    model_path = artifact_dir / "model.json"
    bundle = load_demand_bundle(model_path)
    assert bundle.payload.lineage.tuning_config_sha256 is not None
    assert bundle.payload.lineage.tuning_trials_sha256 is not None
    assert json.loads((artifact_dir / "model_card.json").read_text()) == json.loads(
        json.dumps(demand_model_card(bundle))
    )
    assert bundle.artifact_id in (artifact_dir / "model_card.md").read_text()
    policy_path = tmp_path / "policy.json"
    policy = json.loads(Path("configs/examples/demand-local-approval.json").read_text())
    policy.update(
        max_holdout_primary_metric=10.0,
        max_abs_holdout_bias_fraction=10.0,
        require_seasonal_noninferiority=False,
    )
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
    registered = json.loads(
        invoke(
            "demand-lifecycle",
            "register",
            "--registry",
            str(registry),
            "--bundle",
            str(model_path),
            "--actor",
            "ACTOR-001",
            "--reason",
            "Synthetic artifact registration",
        )
    )
    assert registered["status"] == "draft" and registered["revision"] == 1
    output = tmp_path / "inference"
    blocked = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "forecast",
            "--registry",
            str(registry),
            "--artifact-id",
            artifact_id,
            "--out",
            str(output),
        ],
    )
    assert blocked.exit_code != 0 and not output.exists()
    invoke(
        "demand-lifecycle",
        "transition",
        "--registry",
        str(registry),
        "--artifact-id",
        artifact_id,
        "--actor",
        "ACTOR-001",
        "--action",
        "submit",
        "--reason",
        "Submit synthetic model card",
        "--expected-revision",
        "1",
    )
    approved = json.loads(
        invoke(
            "demand-lifecycle",
            "transition",
            "--registry",
            str(registry),
            "--artifact-id",
            artifact_id,
            "--actor",
            "ACTOR-002",
            "--action",
            "approve",
            "--reason",
            "Reviewed synthetic evidence",
            "--expected-revision",
            "2",
            "--acknowledge-limitations",
        )
    )
    assert approved["status"] == "approved" and approved["revision"] == 3
    invoke(
        "demand-lifecycle",
        "forecast",
        "--registry",
        str(registry),
        "--artifact-id",
        artifact_id,
        "--out",
        str(output),
    )
    forecast = read_dataset(output / "predictions.parquet")
    pd.testing.assert_frame_equal(forecast, predict_demand_bundle(bundle))
    lineage = json.loads((output / "lineage.json").read_text())
    assert lineage["forecast_sha256"] == content_digest(dataframe_records(forecast))
    assert lineage["approval_event_sha256"] == approved["event_sha256"]
    assert lineage["forecast_approval_status"] == "unapproved"
    script = """
import sys
import importlib.abc
class BlockStatsmodels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'statsmodels' or fullname.startswith('statsmodels.'):
            raise ImportError('optional statistical dependency blocked')
sys.meta_path.insert(0, BlockStatsmodels())
from pathlib import Path
from revenue_prediction.core.inference.demand_artifacts import load_demand_bundle,predict_demand_bundle
from revenue_prediction.core.data.io import read_dataset
import pandas as pd
pd.testing.assert_frame_equal(predict_demand_bundle(load_demand_bundle(Path(sys.argv[1]))),read_dataset(Path(sys.argv[2])))
"""
    fresh = subprocess.run(
        [sys.executable, "-c", script, str(model_path), str(output / "predictions.parquet")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fresh.returncode == 0, fresh.stderr
    invoke(
        "demand-lifecycle",
        "transition",
        "--registry",
        str(registry),
        "--artifact-id",
        artifact_id,
        "--actor",
        "ACTOR-002",
        "--action",
        "revoke",
        "--reason",
        "End synthetic demonstration",
        "--expected-revision",
        "3",
    )
    status = json.loads(
        invoke(
            "demand-lifecycle", "status", "--registry", str(registry), "--artifact-id", artifact_id
        )
    )
    assert status["status"] == "revoked" and len(status["events"]) == 4
    revoked_output = tmp_path / "blocked_after_revocation"
    rejected = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "forecast",
            "--registry",
            str(registry),
            "--artifact-id",
            artifact_id,
            "--out",
            str(revoked_output),
        ],
    )
    assert rejected.exit_code != 0 and not revoked_output.exists()
    overwrite = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "package",
            "--comparison-dir",
            str(comparison),
            "--origin",
            origin,
            "--actor",
            "ACTOR-001",
            "--out",
            str(artifact_dir),
        ],
    )
    assert overwrite.exit_code != 0
    assert load_demand_bundle(model_path).artifact_id == artifact_id


@pytest.mark.parametrize("origin", ["not-a-date", "2020-01-01"])
def test_failed_package_writes_no_success_artifacts(tmp_path: Path, origin: str) -> None:
    comparison = tmp_path / "comparison"
    invoke(
        "compare-demand",
        "--models",
        "naive,seasonal_naive",
        "--skus",
        "1",
        "--channels",
        "1",
        "--geographies",
        "1",
        "--out",
        str(comparison),
    )
    out = tmp_path / "failed"
    result = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "package",
            "--comparison-dir",
            str(comparison),
            "--origin",
            origin,
            "--actor",
            "ACTOR-001",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "Demand lifecycle failed" in result.output
    assert not out.exists()


def test_missing_registry_and_invalid_actions_surface_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    result = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "status",
            "--registry",
            str(missing),
            "--artifact-id",
            "0" * 64,
        ],
    )
    assert result.exit_code != 0
    assert "Demand lifecycle failed" in result.output and not missing.exists()
    result = runner.invoke(
        app,
        [
            "demand-lifecycle",
            "transition",
            "--registry",
            str(missing),
            "--artifact-id",
            "0" * 64,
            "--actor",
            "ACTOR-002",
            "--action",
            "invent",
            "--reason",
            "Invalid action",
            "--expected-revision",
            "1",
        ],
    )
    assert result.exit_code != 0 and "Use submit" in result.output
