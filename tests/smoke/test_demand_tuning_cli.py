"""Persisted tuning evidence and offline operation without statistical extras."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    run_demand_selection,
)
from revenue_prediction.core.evaluation.demand_tuning import DemandTuningConfig, run_demand_tuning
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


@pytest.mark.parametrize(
    ("frequency", "pattern"),
    [
        ("MS", "intermittent"),
        ("W-MON", "long_zero_runs"),
        ("MS", "obsolescence"),
    ],
)
def test_tuned_cli_roundtrip_and_diagnostics(tmp_path: Path, frequency: str, pattern: str) -> None:
    out = tmp_path / "results"
    response = runner.invoke(
        app,
        [
            "compare-demand",
            "--models",
            "naive,seasonal_naive,croston,sba,tsb",
            "--frequency",
            frequency,
            "--pattern",
            pattern,
            "--skus",
            "1",
            "--channels",
            "1",
            "--geographies",
            "1",
            "--tuning-grid",
            "configs/examples/demand-tuning.json",
            "--out",
            str(out),
        ],
    )
    assert response.exit_code == 0, response.output
    actuals = read_dataset(out / "actuals.parquet")
    config = DemandTuningConfig.model_validate_json((out / "tuning_config.json").read_text())
    result = run_demand_tuning(actuals, config)
    assert (
        result.selected_trial
        == json.loads((out / "tuning_selection.json").read_text())["selected_trial"]
    )
    assert json.loads((out / "selection.json").read_text()) == json.loads(
        result.selection.decision.model_dump_json()
    )
    pd.testing.assert_frame_equal(read_dataset(out / "tuning_trials.csv"), result.trial_ranking)
    pd.testing.assert_frame_equal(read_dataset(out / "tuning_metrics.csv"), result.trial_metrics)
    frozen = DemandSelectionConfig.model_validate_json((out / "experiment_config.json").read_text())
    reproduced = run_demand_selection(actuals, frozen)
    assert reproduced.decision == result.selection.decision
    for phase, evaluation in (
        ("validation", reproduced.validation),
        ("holdout", reproduced.holdout),
    ):
        for suffix, frame in (
            ("predictions.parquet", evaluation.predictions),
            ("metrics.csv", evaluation.metrics),
            ("series_metrics.csv", evaluation.series_metrics),
            ("segment_metrics.csv", evaluation.segment_metrics),
        ):
            pd.testing.assert_frame_equal(read_dataset(out / f"{phase}_{suffix}"), frame)
    diagnostic = read_dataset(out / "training_diagnostics.csv")
    assert (
        pd.to_datetime(diagnostic.training_end)
        < pd.Timestamp(config.experiment.validation.first_origin)
    ).all()
    assert (out / "model_eligibility.csv").is_file()


def test_cli_failed_candidate_is_reported_and_persisted(tmp_path: Path) -> None:
    grid = tmp_path / "grid.json"
    grid.write_text('[{"average_periods": 999}]')
    out = tmp_path / "results"
    response = runner.invoke(
        app,
        [
            "compare-demand",
            "--models",
            "naive,seasonal_naive,moving_average",
            "--tuning-grid",
            str(grid),
            "--out",
            str(out),
        ],
    )
    assert response.exit_code == 0, response.output
    assert "Rejected full-cohort trials" in response.output
    rows = read_dataset(out / "tuning_trials.csv")
    failure = rows.loc[rows.model_id == "moving_average"].iloc[0]
    assert not failure.eligible
    assert failure.selection_reason == "fit_or_evaluation_failed"
    assert "999 available periods" in failure.failure


@pytest.mark.parametrize(
    "arguments",
    [
        ["--models", "naive,seasonal_naive,unknown"],
        ["--models", "naive,naive,seasonal_naive"],
        ["--pattern", "invalid"],
        ["--occurrence-probability", "0"],
        ["--tuning-grid", "DOES_NOT_EXIST.json"],
    ],
)
def test_new_cli_errors_do_not_write_artifacts(tmp_path: Path, arguments: list[str]) -> None:
    response = runner.invoke(app, ["compare-demand", "--out", str(tmp_path), *arguments])
    assert response.exit_code != 0
    assert not list(tmp_path.iterdir())


def test_core_cli_imports_and_runs_when_statsmodels_is_blocked() -> None:
    code = """
import importlib.abc
import sys
class BlockStatsmodels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'statsmodels' or fullname.startswith('statsmodels.'):
            raise ImportError('optional statistical dependency blocked')
sys.meta_path.insert(0, BlockStatsmodels())
from revenue_prediction.core.data.demand_schema import DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.data.demand_forecast import DemandForecastRequest
from revenue_prediction.core.models.demand_forecasters import forecast_demand_models, demand_model_capabilities
actuals = generate_synthetic_demand_dataset(DemandDataConfig(n_skus=1,n_channels=1,n_geographies=1))
request = DemandForecastRequest(forecast_origin=actuals.date.iloc[48].date(),horizon=3)
forecast = forecast_demand_models(actuals, request, models=('croston','sba','tsb'))
assert len(forecast) == 9
assert len(demand_model_capabilities()) == 13
from revenue_prediction.interfaces.cli.main import app
from typer.testing import CliRunner
assert CliRunner().invoke(app, ['demand-models']).exit_code == 0
"""
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stderr
