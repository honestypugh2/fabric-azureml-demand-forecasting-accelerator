"""Runnable synthetic-only CLI path and persisted-output reproducibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.demand_schema import (
    DemandDataConfig,
    validate_demand_actuals,
)
from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    run_demand_backtest,
)
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


@pytest.mark.parametrize(("frequency", "window"), [("MS", None), ("W-MON", 104)])
def test_synthetic_demand_cli_roundtrip(tmp_path: Path, frequency: str, window: int | None) -> None:
    arguments = ["backtest-demand", "--frequency", frequency, "--out", str(tmp_path)]
    if window is not None:
        arguments += ["--window", str(window)]
    response = runner.invoke(app, arguments)
    assert response.exit_code == 0, response.output
    assert "local, unapproved" in response.output
    actuals = validate_demand_actuals(read_dataset(tmp_path / "actuals.parquet"))
    predictions = read_dataset(tmp_path / "backtest_predictions.parquet")
    metrics = read_dataset(tmp_path / "metrics.csv")
    assert len(predictions) == 3 * 3 * 12 * 2
    assert predictions.approval_status.eq("unapproved").all()
    config = DemandBacktestConfig.model_validate_json(
        (tmp_path / "backtest_config.json").read_text()
    )
    data_config = DemandDataConfig.model_validate_json((tmp_path / "data_config.json").read_text())
    assert data_config.frequency == frequency
    repeated = run_demand_backtest(actuals, config)
    pd.testing.assert_frame_equal(predictions, repeated.predictions)
    pd.testing.assert_frame_equal(metrics, repeated.metrics)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--frequency", "D"],
        ["--horizon", "0"],
        ["--origins", "999"],
        ["--window", "3"],
        ["--seed", "-1"],
    ],
)
def test_invalid_cli_input_fails_without_writing(tmp_path: Path, arguments: list[str]) -> None:
    response = runner.invoke(app, ["backtest-demand", "--out", str(tmp_path), *arguments])
    assert response.exit_code != 0
    assert not list(tmp_path.iterdir())
