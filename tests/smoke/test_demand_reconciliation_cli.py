"""Lossless hierarchy/node replay, content lineage and no-overwrite failures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from revenue_prediction.core.data.demand_forecast import FORECAST_COLUMNS
from revenue_prediction.core.data.demand_hierarchy import DemandHierarchy
from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.inference.demand_artifacts import content_digest, dataframe_records
from revenue_prediction.core.models.demand_reconciliation import (
    HIERARCHY_FORECAST_COLUMNS,
    validate_reconciled_demand,
)
from revenue_prediction.interfaces.cli.main import app

pytestmark = pytest.mark.smoke
runner = CliRunner()


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_reconciliation_cli_persists_and_replays_exactly(tmp_path: Path, frequency: str) -> None:
    config = json.loads(Path("configs/examples/demand-reconciliation.json").read_text())
    if frequency == "W-MON":
        for block in [config["validation"], config["holdout"]]:
            block["frequency"] = frequency
            block["seasonal_periods"] = 4
            for field in ["first_origin", "evaluation_as_of"]:
                value = pd.Timestamp(block[field])
                index = (value.year - 2020) * 12 + value.month - 1
                block[field] = (
                    (pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=index)).date().isoformat()
                )
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "original"
    args = [
        "reconcile-demand",
        "--config",
        str(config_path),
        "--skus",
        "9",
        "--channels",
        "2",
        "--out",
        str(output),
    ]
    response = runner.invoke(app, args)
    assert response.exit_code == 0, response.output
    summary = json.loads(response.output)
    assert summary["method"] == "bottom_up"
    assert summary["aggregate_intervals"] == "unsupported"
    assert summary["approval_status"] == "unapproved"
    hierarchy = DemandHierarchy.model_validate_json((output / "hierarchy.json").read_text())
    assert summary["hierarchy_id"] == hierarchy.hierarchy_id
    assert len(hierarchy.series) == 18
    manifest = json.loads((output / "lineage.json").read_text())
    assert manifest["schema_version"] == "demand-reconciliation-lineage-v1"
    assert manifest["hierarchy_id"] == hierarchy.hierarchy_id
    for name in ["validation", "holdout"]:
        nodes = read_dataset(output / f"{name}_nodes.parquet")
        leaf = read_dataset(output / f"{name}_bottom_predictions.parquet")
        coherence = read_dataset(output / f"{name}_coherence.parquet")
        assert coherence.coherent.all()
        assert manifest[f"{name}_nodes_sha256"] == content_digest(dataframe_records(nodes))
        assert manifest[f"{name}_bottom_points_sha256"] == content_digest(
            dataframe_records(leaf[FORECAST_COLUMNS])
        )
        validate_reconciled_demand(
            nodes[HIERARCHY_FORECAST_COLUMNS], leaf[FORECAST_COLUMNS], hierarchy
        )
        metrics = read_dataset(output / f"{name}_metrics.csv")
        assert set(metrics.hierarchy_level) == {"sku", "product", "family", "brand"}
        assert set(metrics.horizon) == {0, 1, 2, 3}
        assert not any("interval" in column for column in nodes)
    assert runner.invoke(app, args).exit_code == 1
    replay = tmp_path / "replay"
    response = runner.invoke(
        app,
        [
            "reconcile-demand",
            "--config",
            str(output / "experiment_config.json"),
            "--actuals",
            str(output / "actuals.parquet"),
            "--hierarchy",
            str(output / "hierarchy.json"),
            "--out",
            str(replay),
        ],
    )
    assert response.exit_code == 0, response.output
    assert len(list(output.iterdir())) == 16
    for path in output.iterdir():
        if path.suffix == ".parquet":
            pd.testing.assert_frame_equal(
                read_dataset(path), read_dataset(replay / path.name), check_exact=True
            )
        else:
            assert path.read_bytes() == (replay / path.name).read_bytes(), path.name


@pytest.mark.parametrize("kind", ["json", "schema", "missing_actuals", "bad_hierarchy"])
def test_cli_invalid_inputs_create_no_output(tmp_path: Path, kind: str) -> None:
    config_path = Path("configs/examples/demand-reconciliation.json")
    extra: list[str] = []
    if kind in {"json", "schema"}:
        config_path = tmp_path / "bad.json"
        config_path.write_text("{invalid" if kind == "json" else "{}")
    elif kind == "missing_actuals":
        extra = ["--actuals", str(tmp_path / "missing.parquet")]
    elif kind == "bad_hierarchy":
        hierarchy = tmp_path / "hierarchy.json"
        hierarchy.write_text("{}")
        extra = ["--hierarchy", str(hierarchy)]
    output = tmp_path / "never-created"
    response = runner.invoke(
        app, ["reconcile-demand", "--config", str(config_path), *extra, "--out", str(output)]
    )
    assert response.exit_code == 1
    assert "Demand reconciliation failed" in response.output
    assert not output.exists()
