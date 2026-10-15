"""Point-only bottom-up reconciliation over a complete frozen hierarchy cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    DemandForecastRequest,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_hierarchy import (
    HIERARCHY_LEVELS,
    NODE_COLUMNS,
    DemandHierarchy,
    expand_demand_hierarchy,
    hierarchy_membership,
)

HIERARCHY_FORECAST_SCHEMA = "demand-hierarchy-point-v1"
RECONCILIATION_KEYS = [
    "forecast_origin",
    "model_id",
    "date",
    "horizon",
    "hierarchy_level",
    "node_id",
    "channel_id",
    "geography_id",
]
HIERARCHY_FORECAST_COLUMNS = [
    *RECONCILIATION_KEYS,
    "frequency",
    "predicted_demand_units",
    "source_date",
    "source_available_at",
    "training_start",
    "training_end",
    "n_bottom_series",
    "hierarchy_id",
    "hierarchy_known_at",
    "schema_version",
    "reconciliation_method",
    "approval_status",
]
COHERENCE_RTOL = 1e-12
COHERENCE_ATOL = 1e-9


def _validate_points(points: pd.DataFrame, hierarchy: DemandHierarchy) -> None:
    if (
        points.empty
        or set(points.columns) != set(FORECAST_COLUMNS)
        or points.columns.duplicated().any()
    ):
        raise ContractViolation(
            "Bottom-up reconciliation requires exactly demand-point-v1 columns; "
            "intervals and scored rows must not be silently aggregated."
        )
    DemandHierarchy.model_validate(hierarchy.model_dump())
    if (
        points["forecast_origin"].dtype != np.dtype("datetime64[ns]")
        or points["forecast_origin"].isna().to_numpy().any()
        or points["frequency"].nunique() != 1
    ):
        raise ContractViolation("Reconciliation requires dated origins and a single frequency.")
    scopes: list[tuple[set[str], int]] = []
    for origin, group in points.groupby("forecast_origin", sort=True):
        if origin < pd.Timestamp(hierarchy.payload.known_at):
            raise ContractViolation("Hierarchy was not known at this forecast origin.")
        request = DemandForecastRequest(
            forecast_origin=origin.date(),
            frequency=group["frequency"].iloc[0],
            horizon=int(group["horizon"].max()),
            series=hierarchy.series,
        )
        ordered = group.sort_values(["model_id", "sku_id", "channel_id", "geography_id", "horizon"])
        validate_demand_forecasts(ordered, request)
        scopes.append((set(group["model_id"]), request.horizon))
    if any(scope != scopes[0] for scope in scopes):
        raise ContractViolation("Every origin must use the same complete model/horizon scope.")


def _sum_points(points: pd.DataFrame, hierarchy: DemandHierarchy) -> pd.DataFrame:
    expanded = expand_demand_hierarchy(points, hierarchy)
    result = expanded.groupby(RECONCILIATION_KEYS, sort=True, as_index=False).agg(
        frequency=("frequency", "first"),
        predicted_demand_units=("predicted_demand_units", "sum"),
        source_date=("source_date", "max"),
        source_available_at=("source_available_at", "max"),
        training_start=("training_start", "min"),
        training_end=("training_end", "max"),
        n_bottom_series=("sku_id", "size"),
    )
    if not np.isfinite(result["predicted_demand_units"].to_numpy()).all():
        raise ContractViolation("Bottom-up demand sums overflowed; rescale demand units.")
    result["hierarchy_id"] = hierarchy.hierarchy_id
    result["hierarchy_known_at"] = pd.Series(
        pd.Timestamp(hierarchy.payload.known_at), index=result.index, dtype="datetime64[ns]"
    )
    result["schema_version"] = HIERARCHY_FORECAST_SCHEMA
    result["reconciliation_method"] = "bottom_up"
    result["approval_status"] = "unapproved"
    return result[HIERARCHY_FORECAST_COLUMNS]


def reconcile_demand_points(points: pd.DataFrame, hierarchy: DemandHierarchy) -> pd.DataFrame:
    """Sum leaves at SKU/product/family/brand levels; never fit or alter bottom forecasts."""
    _validate_points(points, hierarchy)
    return _sum_points(points, hierarchy)


def validate_reconciled_demand(
    nodes: pd.DataFrame, points: pd.DataFrame, hierarchy: DemandHierarchy
) -> pd.DataFrame:
    """Verify exact version, scope, leaf values, sums and provenance against source points."""
    expected = reconcile_demand_points(points, hierarchy)
    if list(nodes.columns) != HIERARCHY_FORECAST_COLUMNS or not nodes.equals(expected):
        raise ContractViolation(
            "Reconciled point contract differs from canonical complete bottom-up values/provenance."
        )
    report = demand_coherence_report(nodes, hierarchy)
    if not report["coherent"].to_numpy().all():
        raise ContractViolation("Reconciled parent forecasts differ from immediate child totals.")
    return nodes


def demand_coherence_report(nodes: pd.DataFrame, hierarchy: DemandHierarchy) -> pd.DataFrame:
    """Audit complete immediate-child sums; full provenance still needs source validation."""
    required = {*RECONCILIATION_KEYS, "predicted_demand_units"}
    if (
        nodes.empty
        or not required.issubset(nodes.columns)
        or nodes.columns.duplicated().any()
        or nodes.duplicated(RECONCILIATION_KEYS).to_numpy().any()
        or nodes[RECONCILIATION_KEYS].isna().to_numpy().any()
        or not np.isfinite(nodes["predicted_demand_units"].to_numpy(dtype=float)).all()
        or (nodes["predicted_demand_units"].to_numpy(dtype=float) < 0).any()
    ):
        raise ContractViolation("Coherence reporting requires unique, finite, nonempty node rows.")
    DemandHierarchy.model_validate(hierarchy.model_dump())
    membership = hierarchy_membership(hierarchy)
    expected_nodes = expand_demand_hierarchy(membership, hierarchy)[NODE_COLUMNS].drop_duplicates()
    groups = nodes[["forecast_origin", "model_id", "date", "horizon"]].drop_duplicates()
    expected = groups.merge(expected_nodes, how="cross")
    if set(nodes[RECONCILIATION_KEYS].itertuples(index=False, name=None)) != set(
        expected[RECONCILIATION_KEYS].itertuples(index=False, name=None)
    ):
        raise ContractViolation(
            "Coherence reporting requires the complete frozen hierarchy node scope."
        )
    keys = ["forecast_origin", "model_id", "date", "horizon", "channel_id", "geography_id"]
    records: list[pd.DataFrame] = []
    for child, parent, child_column, parent_column in zip(
        HIERARCHY_LEVELS[:-1],
        HIERARCHY_LEVELS[1:],
        ("sku_id", "product_id", "product_family_id"),
        ("product_id", "product_family_id", "brand_id"),
        strict=True,
    ):
        mapping = (
            membership[[child_column, parent_column]]
            .drop_duplicates()
            .rename(columns={child_column: "node_id", parent_column: "parent_id"})
        )
        children = nodes.loc[nodes["hierarchy_level"] == child].merge(
            mapping, on="node_id", validate="many_to_one"
        )
        totals = (
            children.groupby([*keys, "parent_id"], sort=True, as_index=False)
            .agg(
                child_total=("predicted_demand_units", "sum"),
                n_children=("node_id", "size"),
            )
            .rename(columns={"parent_id": "node_id"})
        )
        parents = nodes.loc[
            nodes["hierarchy_level"] == parent, [*keys, "node_id", "predicted_demand_units"]
        ]
        report = parents.merge(totals, on=[*keys, "node_id"], validate="one_to_one")
        report["hierarchy_level"] = parent
        if not np.isfinite(report["child_total"].to_numpy()).all():
            raise ContractViolation("Coherence child totals overflowed; rescale demand units.")
        report["absolute_residual"] = np.abs(
            report["predicted_demand_units"].to_numpy() - report["child_total"].to_numpy()
        )
        report["coherent"] = np.isclose(
            report["predicted_demand_units"].to_numpy(),
            report["child_total"].to_numpy(),
            rtol=COHERENCE_RTOL,
            atol=COHERENCE_ATOL,
        )
        records.append(report)
    return pd.concat(records, ignore_index=True)
