"""Hierarchy availability, complete bottom-up sums and aggregate time-series metrics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import FORECAST_COLUMNS, DemandForecastRequest
from revenue_prediction.core.data.demand_hierarchy import (
    HIERARCHY_LEVELS,
    NODE_COLUMNS,
    DemandHierarchy,
    build_demand_hierarchy,
    expand_demand_hierarchy,
    hierarchy_membership,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DemandDataConfig
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
from revenue_prediction.core.evaluation.demand_reconciliation import (
    hierarchy_training_scales,
    run_demand_reconciliation,
)
from revenue_prediction.core.evaluation.demand_selection import DemandSelectionConfig
from revenue_prediction.core.features.demand import available_demand_history
from revenue_prediction.core.inference.demand_artifacts import content_digest
from revenue_prediction.core.models.demand_reconciliation import (
    HIERARCHY_FORECAST_COLUMNS,
    demand_coherence_report,
    reconcile_demand_points,
    validate_reconciled_demand,
)
from revenue_prediction.core.training.demand_backtesting import DemandBacktestConfig

pytestmark = pytest.mark.unit


def experiment(frequency: str = "MS", n_skus: int = 9):
    frame = generate_synthetic_demand_dataset(
        DemandDataConfig.model_validate(
            {
                "frequency": frequency,
                "n_periods": 60,
                "n_skus": n_skus,
                "n_channels": 2,
                "n_geographies": 2,
            }
        )
    )
    dates = frame.date.drop_duplicates().sort_values()
    common = {
        "frequency": frequency,
        "models": ["naive", "seasonal_naive", "drift"],
        "horizon": 2,
        "seasonal_periods": 4,
        "n_origins": 2,
    }

    def block(start: int, cutoff: int):
        return DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[start].date(),
                "evaluation_as_of": dates.iloc[cutoff].date(),
            }
        )

    config = DemandSelectionConfig(validation=block(25, 29), holdout=block(35, 40))
    return frame, config


@pytest.fixture(scope="module")
def evaluated():
    actuals, config = experiment()
    return actuals, config, run_demand_reconciliation(actuals, config)


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_complete_bottom_up_workflow_and_scope(frequency: str) -> None:
    actuals, config = experiment(frequency)
    result = run_demand_reconciliation(actuals, config)
    hierarchy = result.hierarchy
    assert hierarchy.payload.known_at == config.validation.first_origin
    assert len(hierarchy.series) == 36
    assert hierarchy == DemandHierarchy.model_validate_json(hierarchy.model_dump_json())
    assert set(result.holdout.metrics.hierarchy_level) == set(HIERARCHY_LEVELS)
    assert set(result.holdout.metrics.horizon) == {0, 1, 2}
    assert result.validation.coherence.coherent.all() and result.holdout.coherence.coherent.all()
    assert not any("interval" in column for column in result.holdout.predictions)
    for backtest, nodes in [
        (result.selection.validation, result.validation),
        (result.selection.holdout, result.holdout),
    ]:
        validate_reconciled_demand(
            nodes.predictions[HIERARCHY_FORECAST_COLUMNS],
            backtest.predictions[FORECAST_COLUMNS],
            hierarchy,
        )
        leaves = nodes.predictions.loc[nodes.predictions.hierarchy_level.eq("sku")]
        expected = backtest.predictions.rename(columns={"sku_id": "node_id"})
        keys = ["forecast_origin", "model_id", "node_id", "channel_id", "geography_id", "horizon"]
        pd.testing.assert_series_equal(
            leaves.set_index(keys).predicted_demand_units.sort_index(),
            expected.set_index(keys).predicted_demand_units.sort_index(),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            leaves.set_index(keys).demand_units.sort_index(),
            expected.set_index(keys).demand_units.sort_index(),
            check_names=False,
        )
        assert leaves.n_bottom_series.eq(1).all()
        assert nodes.predictions.approval_status.eq("unapproved").all()
        assert (nodes.predictions.source_available_at <= nodes.predictions.forecast_origin).all()
        assert (
            nodes.predictions.actual_evaluated_available_at <= nodes.predictions.evaluation_as_of
        ).all()
        assert nodes.predictions.hierarchy_id.eq(hierarchy.hierarchy_id).all()
    assert result.lineage["aggregate_intervals"] == "unsupported"
    assert result.lineage["forecast_approval_status"] == "unapproved"


def test_scope_is_not_assumed_cartesian_and_dimensions_are_not_pooled():
    actuals, config = experiment()
    omitted = actuals.sku_id.eq("SKU-009") & actuals.channel_id.eq("CHANNEL-002")
    actuals = actuals.loc[~omitted].copy()
    result = run_demand_reconciliation(actuals, config)
    assert len(result.hierarchy.series) == 34
    selected = result.selection.decision.selected_model
    origin = pd.Timestamp(config.holdout.first_origin)
    leaf = result.selection.holdout.predictions
    nodes = result.holdout.predictions
    for channel, region in [
        ("CHANNEL-001", "REGION-001"),
        ("CHANNEL-002", "REGION-002"),
    ]:
        raw = leaf.loc[
            leaf.model_id.eq(selected)
            & leaf.forecast_origin.eq(origin)
            & leaf.horizon.eq(1)
            & leaf.channel_id.eq(channel)
            & leaf.geography_id.eq(region)
        ]
        brands = nodes.loc[
            nodes.hierarchy_level.eq("brand")
            & nodes.model_id.eq(selected)
            & nodes.forecast_origin.eq(origin)
            & nodes.horizon.eq(1)
            & nodes.channel_id.eq(channel)
            & nodes.geography_id.eq(region)
        ]
        assert brands.n_bottom_series.sum() == len(raw)
        np.testing.assert_allclose(
            brands.predicted_demand_units.sum(), raw.predicted_demand_units.sum(), rtol=1e-12
        )
        np.testing.assert_allclose(brands.demand_units.sum(), raw.demand_units.sum(), rtol=1e-12)
    assert set(nodes.hierarchy_level) == set(HIERARCHY_LEVELS)


def test_hierarchy_discovery_ignores_future_values_and_unarrived_launches(evaluated):
    actuals, config, result = evaluated
    request = DemandForecastRequest(forecast_origin=config.validation.first_origin)
    changed = actuals.copy()
    changed.loc[changed.date >= request.origin, "demand_units"] *= 10
    future = actuals.loc[actuals.sku_id.eq("SKU-001") & (actuals.date >= request.origin)].copy()
    future["sku_id"] = "SKU-010"
    future["product_id"] = "PRODUCT-010"
    future["product_family_id"] = "FAMILY-010"
    future["brand_id"] = "BRAND-010"
    changed = pd.concat([changed, future], ignore_index=True)
    assert build_demand_hierarchy(changed, request) == result.hierarchy
    launched = changed.loc[~changed.sku_id.eq("SKU-010") | (changed.date >= request.origin)]
    assert len(build_demand_hierarchy(launched, request).series) == 36
    reordered = actuals.sort_values(["date", *DEMAND_SERIES_COLUMNS])
    assert build_demand_hierarchy(reordered, request) == result.hierarchy


@pytest.mark.parametrize(
    "mutation", ["digest", "duplicate", "order", "sku_parent", "product_parent", "family_parent"]
)
def test_hierarchy_rejects_invalid_content_and_parent_relationships(evaluated, mutation):
    _, _, result = evaluated
    data = result.hierarchy.model_dump(mode="json")
    members = data["payload"]["members"]
    if mutation == "digest":
        data["hierarchy_id"] = "0" * 64
    elif mutation == "duplicate":
        members.append(members[0])
    elif mutation == "order":
        members.reverse()
    elif mutation == "sku_parent":
        members[1]["product_id"] = "PRODUCT-999"
    elif mutation == "product_parent":
        for item in members:
            if item["series"]["sku_id"] == "SKU-002":
                item["product_family_id"] = "FAMILY-999"
    elif mutation == "family_parent":
        for item in members:
            if item["series"]["sku_id"] in {"SKU-003", "SKU-004"}:
                item["brand_id"] = "BRAND-999"
    if mutation != "digest":
        data["hierarchy_id"] = content_digest(data["payload"])
    with pytest.raises(ValidationError):
        DemandHierarchy.model_validate(data)


def test_future_or_mismatched_hierarchy_is_not_usable(evaluated):
    actuals, config, result = evaluated
    payload = result.hierarchy.payload.model_dump(mode="json")
    payload["known_at"] = config.holdout.first_origin.isoformat()
    future = DemandHierarchy.model_validate(
        {
            "hierarchy_id": content_digest(payload),
            "payload": payload,
        }
    )
    with pytest.raises(ContractViolation, match="first validation origin"):
        run_demand_reconciliation(actuals, config, future)
    points = result.selection.validation.predictions[FORECAST_COLUMNS]
    with pytest.raises(ContractViolation, match="not known"):
        reconcile_demand_points(points, future)
    payload = result.hierarchy.payload.model_dump(mode="json")
    for item in payload["members"]:
        if item["series"]["sku_id"] == "SKU-001":
            item["product_id"] = "PRODUCT-999"
    mismatch = DemandHierarchy.model_validate(
        {
            "hierarchy_id": content_digest(payload),
            "payload": payload,
        }
    )
    with pytest.raises(ContractViolation, match="cohort/parents"):
        run_demand_reconciliation(actuals, config, mismatch)


@pytest.mark.parametrize(
    "mutation",
    ["missing_sku", "missing_model", "missing_horizon", "intervals", "empty", "mixed_frequency"],
)
def test_reconciliation_rejects_incomplete_or_nonpoint_inputs(evaluated, mutation):
    _, _, result = evaluated
    points = result.selection.validation.predictions[FORECAST_COLUMNS].copy()
    if mutation == "missing_sku":
        points = points.loc[~points.sku_id.eq("SKU-009")]
    elif mutation == "missing_model":
        points = points.loc[
            ~(points.forecast_origin.eq(points.forecast_origin.min()) & points.model_id.eq("naive"))
        ]
    elif mutation == "missing_horizon":
        points = points.drop(points.index[0])
    elif mutation == "intervals":
        points["lower_demand_units"] = 0.0
    elif mutation == "empty":
        points = points.iloc[:0]
    else:
        points.loc[points.index[0], "frequency"] = "W-MON"
    with pytest.raises(ContractViolation):
        reconcile_demand_points(points, result.hierarchy)


@pytest.mark.parametrize(
    "mutation", ["value", "leaf", "metadata", "missing_parent", "duplicate", "interval", "count"]
)
def test_output_contract_checks_values_provenance_and_complete_nodes(evaluated, mutation):
    _, _, result = evaluated
    points = result.selection.validation.predictions[FORECAST_COLUMNS]
    nodes = result.validation.predictions[HIERARCHY_FORECAST_COLUMNS].copy()
    if mutation == "value":
        nodes.loc[nodes.hierarchy_level.eq("brand"), "predicted_demand_units"] += 1
    elif mutation == "leaf":
        nodes.loc[nodes.hierarchy_level.eq("sku"), "predicted_demand_units"] += 1
    elif mutation == "metadata":
        nodes["hierarchy_id"] = "0" * 64
    elif mutation == "missing_parent":
        nodes = nodes.loc[~nodes.hierarchy_level.eq("product")]
    elif mutation == "duplicate":
        nodes = pd.concat([nodes, nodes.iloc[:1]], ignore_index=True)
    elif mutation == "interval":
        nodes["upper_demand_units"] = 1000.0
    else:
        nodes.loc[nodes.index[0], "n_bottom_series"] += 1
    with pytest.raises(ContractViolation, match="contract"):
        validate_reconciled_demand(nodes, points, result.hierarchy)


def test_point_row_order_does_not_change_sums_and_coherence(evaluated):
    _, _, result = evaluated
    points = result.selection.validation.predictions[FORECAST_COLUMNS]
    expected = reconcile_demand_points(points, result.hierarchy)
    pd.testing.assert_frame_equal(
        expected, reconcile_demand_points(points.sample(frac=1, random_state=7), result.hierarchy)
    )
    report = demand_coherence_report(expected, result.hierarchy)
    assert report.coherent.all()
    assert (report.absolute_residual <= 1e-9 + 1e-12 * report.predicted_demand_units).all()


def test_aggregate_scales_use_complete_calendar_pairs_not_sum_of_leaf_scales():
    frame, _ = experiment(n_skus=2)
    for key, series in frame.groupby(DEMAND_SERIES_COLUMNS):
        index = np.arange(len(series), dtype=float)
        frame.loc[series.index, "demand_units"] = 10 + index if key[0] == "SKU-001" else 100 - index
    request = DemandForecastRequest(forecast_origin=date(2020, 7, 1))
    hierarchy = build_demand_hierarchy(frame, request)
    history = available_demand_history(frame, request)
    scales = hierarchy_training_scales(history, hierarchy, "MS", 1)
    assert scales.loc[scales.hierarchy_level.eq("sku"), "mae_scale"].eq(1).all()
    assert scales.loc[~scales.hierarchy_level.eq("sku"), "mae_scale"].eq(0).all()
    assert (
        scales.loc[~scales.hierarchy_level.eq("sku"), "scaled_metrics_status"]
        .eq("zero_seasonal_scale")
        .all()
    )
    assert scales.n_scale_pairs.eq(5).all()
    duplicated = pd.concat([history, history.iloc[:1]], ignore_index=True)
    with pytest.raises(ContractViolation, match="unique"):
        hierarchy_training_scales(duplicated, hierarchy, "MS", 1)


def test_delayed_training_descendant_is_not_treated_as_zero_or_compressed():
    frame, _ = experiment(n_skus=2)
    for key, series in frame.groupby(DEMAND_SERIES_COLUMNS):
        frame.loc[series.index, "demand_units"] = (np.arange(len(series), dtype=float) + 10) * (
            1 if key[0] == "SKU-001" else 2
        )
    request = DemandForecastRequest(forecast_origin=date(2020, 7, 1))
    hierarchy = build_demand_hierarchy(frame, request)
    delayed = frame.sku_id.eq("SKU-002") & frame.date.eq(pd.Timestamp("2020-03-01"))
    frame.loc[delayed, "actual_available_at"] = pd.Timestamp("2020-08-01")
    history = available_demand_history(frame, request)
    scales = hierarchy_training_scales(history, hierarchy, "MS", 1)
    parent = scales.loc[scales.hierarchy_level.eq("product")]
    assert parent.n_scale_pairs.eq(3).all()
    assert parent.mae_scale.eq(3).all()
    assert parent.mse_scale.eq(9).all()


def test_no_complete_parent_history_reports_undefined_not_dropped():
    frame, _ = experiment(n_skus=2)
    request = DemandForecastRequest(forecast_origin=date(2020, 3, 1))
    hierarchy = build_demand_hierarchy(frame, request)
    history = available_demand_history(frame, request)
    sparse = history.loc[
        (history.sku_id.eq("SKU-001") & history.date.eq(pd.Timestamp("2020-01-01")))
        | (history.sku_id.eq("SKU-002") & history.date.eq(pd.Timestamp("2020-02-01")))
    ]
    scales = hierarchy_training_scales(sparse, hierarchy, "MS", 1)
    parent = scales.loc[scales.hierarchy_level.eq("product")]
    assert len(parent) == 4
    assert parent.scaled_metrics_status.eq("insufficient_complete_training_periods").all()
    assert parent.mae_scale.isna().all() and parent.n_scale_pairs.eq(0).all()


def test_level_metrics_are_on_aggregated_errors_not_pooled_leaf_errors():
    frame, config = experiment(n_skus=2)
    for key, series in frame.groupby(DEMAND_SERIES_COLUMNS):
        values = np.where(np.arange(len(series)) % 2 == 0, 20.0, 80.0)
        frame.loc[series.index, "demand_units"] = values if key[0] == "SKU-001" else 100 - values
    config = DemandSelectionConfig.model_validate({**config.model_dump(), "primary_metric": "wape"})
    result = run_demand_reconciliation(frame, config)
    metrics = result.holdout.metrics
    naive = metrics.loc[metrics.model_id.eq("naive") & metrics.horizon.eq(1)]
    assert naive.loc[naive.hierarchy_level.eq("sku"), "mae"].iloc[0] == 60
    assert naive.loc[naive.hierarchy_level.eq("product"), "mae"].iloc[0] == 0
    parent = metrics.loc[~metrics.hierarchy_level.eq("sku")]
    assert parent.mase.isna().all() and parent.rmsse.isna().all()
    assert parent.scaled_metrics_status.eq("undefined_training_scale").all()
    assert result.holdout.predictions.n_scale_pairs.gt(0).all()


def test_holdout_changes_cannot_reselect_or_change_first_origin_forecasts(evaluated):
    actuals, config, original = evaluated
    changed = actuals.copy()
    changed.loc[changed.date >= pd.Timestamp(config.holdout.first_origin), "demand_units"] *= 3
    result = run_demand_reconciliation(changed, config)
    assert result.hierarchy == original.hierarchy
    assert result.selection.decision == original.selection.decision
    pd.testing.assert_frame_equal(result.selection.ranking, original.selection.ranking)
    pd.testing.assert_frame_equal(result.validation.predictions, original.validation.predictions)
    cutoff = pd.Timestamp(config.holdout.first_origin)
    before = original.holdout.predictions.loc[
        original.holdout.predictions.forecast_origin.eq(cutoff)
    ]
    after = result.holdout.predictions.loc[result.holdout.predictions.forecast_origin.eq(cutoff)]
    pd.testing.assert_frame_equal(
        before[HIERARCHY_FORECAST_COLUMNS], after[HIERARCHY_FORECAST_COLUMNS]
    )
    np.testing.assert_array_equal(before.mae_scale, after.mae_scale)
    assert not result.holdout.metrics.equals(original.holdout.metrics)


def test_missing_holdout_truth_fails_without_partial_parent_totals(evaluated):
    actuals, config, _ = evaluated
    delayed = actuals.copy()
    row = delayed.index[
        delayed.sku_id.eq("SKU-001") & delayed.date.eq(pd.Timestamp(config.holdout.first_origin))
    ]
    delayed.loc[row, "actual_available_at"] = pd.Timestamp(
        config.holdout.evaluation_as_of
    ) + pd.Timedelta(days=1)
    with pytest.raises(ContractViolation, match="complete evaluation actuals"):
        run_demand_reconciliation(delayed, config)


def test_overflow_and_unmapped_members_fail_explicitly(evaluated):
    actuals, config, result = evaluated
    points = result.selection.validation.predictions[FORECAST_COLUMNS].copy()
    points["predicted_demand_units"] = 1e308
    with pytest.raises(ContractViolation, match="overflowed"):
        reconcile_demand_points(points, result.hierarchy)
    history = available_demand_history(
        actuals, DemandForecastRequest(forecast_origin=config.validation.first_origin)
    )
    history["demand_units"] = 1e308
    with pytest.raises(ContractViolation, match="overflowed"):
        hierarchy_training_scales(history, result.hierarchy, "MS", 4)
    mapping = hierarchy_membership(result.hierarchy)
    mapping.loc[mapping.index[0], "sku_id"] = "SKU-999"
    with pytest.raises(ContractViolation, match="outside"):
        expand_demand_hierarchy(mapping, result.hierarchy)
    with pytest.raises(ContractViolation, match="nonempty"):
        expand_demand_hierarchy(mapping.iloc[:0], result.hierarchy)


def test_sliding_window_and_explicit_cohort_define_complete_parent_totals(evaluated):
    actuals, config, _ = evaluated
    payload = config.model_dump(mode="json")
    scope = [
        {"sku_id": f"SKU-{index:03d}", "channel_id": "CHANNEL-001", "geography_id": "REGION-001"}
        for index in [1, 2]
    ]
    for block in [payload["validation"], payload["holdout"]]:
        block.update(strategy="sliding", training_window_periods=10, series=scope)
    config = DemandSelectionConfig.model_validate(payload)
    result = run_demand_reconciliation(actuals, config)
    assert len(result.hierarchy.series) == 2
    assert len(result.holdout.predictions[NODE_COLUMNS].drop_duplicates()) == 5
    assert result.holdout.predictions.n_scale_pairs.eq(6).all()
    assert (
        result.holdout.predictions.loc[
            result.holdout.predictions.hierarchy_level.eq("brand"), "n_bottom_series"
        ]
        .eq(2)
        .all()
    )
    replay = run_demand_reconciliation(actuals, config, result.hierarchy)
    assert replay.lineage == result.lineage
    pd.testing.assert_frame_equal(replay.holdout.predictions, result.holdout.predictions)


def test_new_launches_never_expand_the_frozen_evaluation_cohort(evaluated):
    actuals, config, original = evaluated
    future = actuals.loc[
        actuals.sku_id.eq("SKU-001")
        & (actuals.date >= pd.Timestamp(config.validation.first_origin))
    ].copy()
    future["sku_id"] = "SKU-010"
    future["product_id"] = "PRODUCT-010"
    future["product_family_id"] = "FAMILY-010"
    future["brand_id"] = "BRAND-010"
    changed = pd.concat([actuals, future], ignore_index=True)
    result = run_demand_reconciliation(changed, config)
    assert result.hierarchy == original.hierarchy
    assert result.selection.decision == original.selection.decision
    pd.testing.assert_frame_equal(result.holdout.predictions, original.holdout.predictions)
    assert result.lineage["source_actuals_sha256"] != original.lineage["source_actuals_sha256"]
    assert result.lineage["holdout_nodes_sha256"] == original.lineage["holdout_nodes_sha256"]


@pytest.mark.parametrize("mutation", ["parent", "leaf", "duplicate", "nonfinite"])
def test_standalone_coherence_report_cannot_hide_missing_or_invalid_nodes(evaluated, mutation):
    _, _, result = evaluated
    nodes = result.holdout.predictions.copy()
    if mutation == "parent":
        nodes = nodes.loc[~nodes.hierarchy_level.eq("product")]
    elif mutation == "leaf":
        nodes = nodes.drop(nodes.index[nodes.hierarchy_level.eq("sku")][0])
    elif mutation == "duplicate":
        nodes = pd.concat([nodes, nodes.iloc[:1]], ignore_index=True)
    else:
        nodes.loc[nodes.index[0], "predicted_demand_units"] = np.inf
    with pytest.raises(ContractViolation, match="Coherence"):
        demand_coherence_report(nodes, result.hierarchy)
