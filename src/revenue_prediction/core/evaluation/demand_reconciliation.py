"""Complete-cohort hierarchy evaluation with origin-only aggregate seasonal scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    DemandForecastRequest,
    DemandFrequency,
)
from revenue_prediction.core.data.demand_hierarchy import (
    NODE_COLUMNS,
    DemandHierarchy,
    build_demand_hierarchy,
    expand_demand_hierarchy,
    hierarchy_membership,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DEMAND_TARGET
from revenue_prediction.core.evaluation.demand_metrics import seasonal_training_scales
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    DemandSelectionResult,
    run_demand_selection,
)
from revenue_prediction.core.features.demand import DemandFeatureConfig, available_demand_history
from revenue_prediction.core.inference.demand_artifacts import (
    content_digest,
    dataframe_records,
    implementation_digest,
    runtime_versions,
)
from revenue_prediction.core.models.demand_reconciliation import (
    RECONCILIATION_KEYS,
    demand_coherence_report,
    reconcile_demand_points,
    validate_reconciled_demand,
)
from revenue_prediction.core.training.demand_backtesting import (
    DemandBacktestConfig,
    DemandBacktestResult,
    summarise_demand_predictions,
)


@dataclass(frozen=True)
class DemandHierarchyEvaluation:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    node_metrics: pd.DataFrame
    coherence: pd.DataFrame


@dataclass(frozen=True)
class DemandReconciliationResult:
    hierarchy: DemandHierarchy
    selection: DemandSelectionResult
    validation: DemandHierarchyEvaluation
    holdout: DemandHierarchyEvaluation
    lineage: dict[str, object]


def hierarchy_training_scales(
    history: pd.DataFrame,
    hierarchy: DemandHierarchy,
    frequency: DemandFrequency,
    seasonal_periods: int,
) -> pd.DataFrame:
    """Scale each aggregate node using dates with every fixed descendant present.

    A late or not-yet-launched descendant makes a date unavailable, not zero.
    Missing calendar dates are retained as gaps for exact seasonal pairs.
    """
    if (
        not {"date", *DEMAND_SERIES_COLUMNS, DEMAND_TARGET}.issubset(history.columns)
        or history.duplicated(["date", *DEMAND_SERIES_COLUMNS]).any()
        or not np.isfinite(history[DEMAND_TARGET].to_numpy(dtype=float)).all()
        or (history[DEMAND_TARGET].to_numpy(dtype=float) < 0).any()
    ):
        raise ContractViolation("Aggregate scales require unique finite nonnegative training rows.")
    members = expand_demand_hierarchy(hierarchy_membership(hierarchy), hierarchy)
    expected = (
        members.groupby(NODE_COLUMNS, sort=True).size().rename("expected_count").reset_index()
    )
    expanded = expand_demand_hierarchy(history, hierarchy)
    training = (
        expanded.groupby([*NODE_COLUMNS, "date"], sort=True, as_index=False)
        .agg(
            demand_units=(DEMAND_TARGET, "sum"),
            n_present=("sku_id", "size"),
        )
        .merge(expected, on=NODE_COLUMNS, validate="many_to_one")
    )
    training = training.loc[training["n_present"] == training["expected_count"]].copy()
    if not np.isfinite(training[DEMAND_TARGET].to_numpy()).all():
        raise ContractViolation("Aggregate training demand overflowed; rescale demand units.")
    pieces: list[pd.DataFrame] = []
    for level, group in training.groupby("hierarchy_level", sort=True):
        scales = seasonal_training_scales(
            group.rename(columns={"node_id": "sku_id"}),
            frequency,
            seasonal_periods,
        ).rename(columns={"sku_id": "node_id"})
        scales["hierarchy_level"] = level
        pieces.append(scales)
    if pieces:
        result = expected.merge(
            pd.concat(pieces, ignore_index=True), on=NODE_COLUMNS, how="left", validate="one_to_one"
        )
    else:
        result = expected.copy()
        result["mae_scale"] = np.nan
        result["mse_scale"] = np.nan
        result["n_scale_pairs"] = 0
        result["scaled_metrics_status"] = "insufficient_complete_training_periods"
    result["scaled_metrics_status"] = result["scaled_metrics_status"].fillna(
        "insufficient_complete_training_periods"
    )
    result["n_scale_pairs"] = result["n_scale_pairs"].fillna(0).astype(int)
    return result.drop(columns="expected_count")


def _evaluate(
    actuals: pd.DataFrame,
    backtest: DemandBacktestResult,
    config: DemandBacktestConfig,
    hierarchy: DemandHierarchy,
) -> DemandHierarchyEvaluation:
    leaf = backtest.predictions
    nodes = reconcile_demand_points(leaf[FORECAST_COLUMNS], hierarchy)
    validate_reconciled_demand(nodes, leaf[FORECAST_COLUMNS], hierarchy)
    expanded_truth = expand_demand_hierarchy(
        leaf[
            [
                "forecast_origin",
                "model_id",
                "date",
                "horizon",
                *DEMAND_SERIES_COLUMNS,
                DEMAND_TARGET,
                "actual_evaluated_available_at",
                "evaluation_as_of",
            ]
        ],
        hierarchy,
    )
    truth = expanded_truth.groupby(RECONCILIATION_KEYS, sort=True, as_index=False).agg(
        demand_units=(DEMAND_TARGET, "sum"),
        actual_evaluated_available_at=("actual_evaluated_available_at", "max"),
        evaluation_as_of=("evaluation_as_of", "first"),
    )
    if not np.isfinite(truth[DEMAND_TARGET].to_numpy()).all():
        raise ContractViolation("Aggregate evaluation demand overflowed; rescale demand units.")
    nodes = nodes.merge(truth, on=RECONCILIATION_KEYS, validate="one_to_one")
    all_scales: list[pd.DataFrame] = []
    for origin in sorted(nodes["forecast_origin"].unique()):
        request = DemandForecastRequest(
            forecast_origin=pd.Timestamp(origin).date(),
            frequency=config.frequency,
            horizon=config.horizon,
            series=hierarchy.series,
        )
        history = available_demand_history(
            actuals,
            request,
            DemandFeatureConfig(
                seasonal_periods=config.seasonal_periods,
                training_window_periods=config.training_window_periods,
            ),
        )
        scales = hierarchy_training_scales(
            history,
            hierarchy,
            config.frequency,
            config.seasonal_periods or (12 if config.frequency == "MS" else 52),
        )
        scales["forecast_origin"] = pd.Series(
            request.origin, index=scales.index, dtype="datetime64[ns]"
        )
        all_scales.append(scales)
    nodes = (
        nodes.merge(
            pd.concat(all_scales, ignore_index=True),
            on=["forecast_origin", *NODE_COLUMNS],
            validate="many_to_one",
        )
        .sort_values(RECONCILIATION_KEYS, kind="stable")
        .reset_index(drop=True)
    )
    nodes["error"] = nodes["predicted_demand_units"] - nodes[DEMAND_TARGET]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            metrics = summarise_demand_predictions(nodes, ["hierarchy_level"])
            node_metrics = summarise_demand_predictions(nodes, NODE_COLUMNS)
    except FloatingPointError as exc:
        raise ContractViolation(
            "Hierarchy evaluation metrics overflowed; rescale demand units."
        ) from exc
    return DemandHierarchyEvaluation(
        nodes, metrics, node_metrics, demand_coherence_report(nodes, hierarchy)
    )


def run_demand_reconciliation(
    actuals: pd.DataFrame,
    config: DemandSelectionConfig,
    hierarchy: DemandHierarchy | None = None,
) -> DemandReconciliationResult:
    """Keep bottom validation-only selection; report levels without holdout reselection."""
    first = config.validation
    discovered = build_demand_hierarchy(
        actuals,
        DemandForecastRequest(
            forecast_origin=first.first_origin,
            frequency=first.frequency,
            horizon=first.horizon,
            series=first.series,
        ),
        DemandFeatureConfig(
            seasonal_periods=first.seasonal_periods,
            training_window_periods=first.training_window_periods,
        ),
    )
    if hierarchy is None:
        hierarchy = discovered
    else:
        hierarchy = DemandHierarchy.model_validate(hierarchy.model_dump())
        if hierarchy.payload.known_at > first.first_origin:
            raise ContractViolation("Hierarchy was not available at the first validation origin.")
        if hierarchy.payload.members != discovered.payload.members:
            raise ContractViolation(
                "Hierarchy must match the complete origin-available experiment cohort/parents; "
                "declare an explicit experiment scope rather than shrinking totals."
            )
    selected = run_demand_selection(actuals, config)
    validation = _evaluate(actuals, selected.validation, config.validation, hierarchy)
    holdout = _evaluate(actuals, selected.holdout, config.holdout, hierarchy)
    lineage: dict[str, object] = {
        "schema_version": "demand-reconciliation-lineage-v1",
        "method": "bottom_up",
        "hierarchy_id": hierarchy.hierarchy_id,
        "hierarchy_known_at": hierarchy.payload.known_at.isoformat(),
        "source_actuals_sha256": content_digest(dataframe_records(actuals)),
        "experiment_sha256": content_digest(config.model_dump(mode="json")),
        "selection_sha256": content_digest(selected.decision.model_dump(mode="json")),
        "implementation_sha256": implementation_digest(),
        "runtime": runtime_versions(selected.decision.selected_model),
        "forecast_approval_status": "unapproved",
        "aggregate_intervals": "unsupported",
    }
    for name, source, result in [
        ("validation", selected.validation, validation),
        ("holdout", selected.holdout, holdout),
    ]:
        lineage[f"{name}_bottom_points_sha256"] = content_digest(
            dataframe_records(source.predictions[FORECAST_COLUMNS])
        )
        lineage[f"{name}_bottom_evidence_sha256"] = content_digest(
            dataframe_records(source.predictions)
        )
        for output, frame in [
            ("nodes", result.predictions),
            ("metrics", result.metrics),
            ("node_metrics", result.node_metrics),
            ("coherence", result.coherence),
        ]:
            lineage[f"{name}_{output}_sha256"] = content_digest(dataframe_records(frame))
    return DemandReconciliationResult(hierarchy, selected, validation, holdout, lineage)
