"""Persisted synthetic uncertainty evidence and explicit command failures."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.demand_forecast import FORECAST_COLUMNS
from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.evaluation.demand_uncertainty import (
    DemandIntervalCalibration,
    apply_demand_intervals,
)
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


def test_uncertainty_cli_persists_reproducible_evidence(tmp_path: Path) -> None:
    output = tmp_path / "uncertainty"
    args = [
        "demand-uncertainty",
        "--config",
        "configs/examples/demand-uncertainty.json",
        "--out",
        str(output),
    ]
    response = runner.invoke(app, args)
    assert response.exit_code == 0, response.output
    summary = json.loads(response.output)
    assert summary["approval_status"] == "unapproved"
    assert summary["nominal_coverage"] == 0.8
    calibration = DemandIntervalCalibration.model_validate_json(
        (output / "calibration.json").read_text()
    )
    points = read_dataset(output / "holdout_point_predictions.parquet")
    persisted = read_dataset(output / "holdout_intervals.parquet")
    reproduced = apply_demand_intervals(points, calibration)
    pd.testing.assert_frame_equal(persisted[reproduced.columns], reproduced)
    pd.testing.assert_frame_equal(persisted[FORECAST_COLUMNS], points[FORECAST_COLUMNS])
    assert {0, 1, 2} == set(read_dataset(output / "interval_metrics.csv").horizon)
    assert (output / "selection.json").exists()
    assert (output / "validation_ranking.csv").exists()
    assert (output / "series_interval_metrics.csv").exists()
    assert runner.invoke(app, args).exit_code == 1
    replay = runner.invoke(
        app, [*args[:-1], str(tmp_path / "replay"), "--actuals", str(output / "actuals.parquet")]
    )
    assert replay.exit_code == 0, replay.output
    assert (output / "calibration.json").read_bytes() == (
        tmp_path / "replay" / "calibration.json"
    ).read_bytes()


@pytest.mark.parametrize("config_content", ["{}", "{not json}"])
def test_invalid_experiment_does_not_create_output(tmp_path: Path, config_content: str) -> None:
    config = tmp_path / "invalid.json"
    config.write_text(config_content)
    output = tmp_path / "missing"
    result = runner.invoke(
        app, ["demand-uncertainty", "--config", str(config), "--out", str(output)]
    )
    assert result.exit_code == 1
    assert "Demand uncertainty failed" in result.output
    assert not output.exists()
