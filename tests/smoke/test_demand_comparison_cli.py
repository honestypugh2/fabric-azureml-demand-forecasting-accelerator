"""Registry-driven comparison, frozen selection and persisted final holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    run_demand_selection,
)
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


def test_cli_registry_is_discoverable_json() -> None:
    response = runner.invoke(app, ["demand-models"])
    assert response.exit_code == 0
    capabilities = json.loads(response.output)
    assert len(capabilities) == 13
    assert all(not item["supports_probabilistic_output"] for item in capabilities)
    assert {"ses", "holt_damped"}.issubset({item["model_id"] for item in capabilities})


@pytest.mark.parametrize(("frequency", "window"), [("MS", 36), ("W-MON", 156)])
def test_comparison_cli_roundtrip(tmp_path: Path, frequency: str, window: int) -> None:
    response = runner.invoke(
        app,
        [
            "compare-demand",
            "--frequency",
            frequency,
            "--window",
            str(window),
            "--out",
            str(tmp_path),
        ],
    )
    assert response.exit_code == 0, response.output
    config = DemandSelectionConfig.model_validate_json(
        (tmp_path / "experiment_config.json").read_text()
    )
    repeated = run_demand_selection(read_dataset(tmp_path / "actuals.parquet"), config)
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection == json.loads(repeated.decision.model_dump_json())
    assert len(repeated.validation.predictions) == 3 * 3 * 12 * 7
    assert repeated.validation.predictions.date.max() < repeated.holdout.predictions.date.min()
    for phase, result in (("validation", repeated.validation), ("holdout", repeated.holdout)):
        pd.testing.assert_frame_equal(
            read_dataset(tmp_path / f"{phase}_predictions.parquet"), result.predictions
        )
        pd.testing.assert_frame_equal(
            read_dataset(tmp_path / f"{phase}_metrics.csv"), result.metrics
        )
    pd.testing.assert_frame_equal(
        read_dataset(tmp_path / "validation_ranking.csv"), repeated.ranking
    )
    assert selection["approval_status"] == "unapproved"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--metric", "r2"],
        ["--frequency", "D"],
        ["--window", "12"],
        ["--minimum-improvement", "1"],
        ["--origins", "999"],
    ],
)
def test_comparison_errors_do_not_emit_success_artifacts(
    tmp_path: Path, arguments: list[str]
) -> None:
    response = runner.invoke(app, ["compare-demand", "--out", str(tmp_path), *arguments])
    assert response.exit_code != 0
    assert not list(tmp_path.iterdir())
