"""Content-versioned, origin-available static demand hierarchy snapshots."""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import DemandForecastRequest, DemandSeriesKey
from revenue_prediction.core.data.demand_records import (
    ArtifactRecord,
    Digest,
    content_digest,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS
from revenue_prediction.core.features.demand import DemandFeatureConfig, available_demand_history

HIERARCHY_LEVELS = ("sku", "product", "family", "brand")
HIERARCHY_PARENT_COLUMNS = ("product_id", "product_family_id", "brand_id")
NODE_COLUMNS = ["hierarchy_level", "node_id", "channel_id", "geography_id"]


class DemandHierarchyMember(ArtifactRecord):
    series: DemandSeriesKey
    product_id: str = Field(pattern=r"^PRODUCT-\d{3,}$")
    product_family_id: str = Field(pattern=r"^FAMILY-\d{3,}$")
    brand_id: str = Field(pattern=r"^BRAND-\d{3,}$")


class DemandHierarchyPayload(ArtifactRecord):
    schema_version: Literal["demand-hierarchy-v1"] = "demand-hierarchy-v1"
    known_at: date
    data_classification: Literal["synthetic"] = "synthetic"
    members: tuple[DemandHierarchyMember, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _stable_parents(self) -> DemandHierarchyPayload:
        keys = [item.series.key for item in self.members]
        if keys != sorted(set(keys)):
            raise ValueError("Hierarchy series must be unique and canonically sorted.")
        parents: dict[tuple[str, str], tuple[str, ...]] = {}
        for item in self.members:
            for level, child, ancestor in [
                (
                    "sku",
                    item.series.sku_id,
                    (item.product_id, item.product_family_id, item.brand_id),
                ),
                ("product", item.product_id, (item.product_family_id, item.brand_id)),
                ("family", item.product_family_id, (item.brand_id,)),
            ]:
                key = (level, child)
                if key in parents and parents[key] != ancestor:
                    raise ValueError(f"Hierarchy {level} must have exactly one stable parent.")
                parents[key] = ancestor
        return self


class DemandHierarchy(ArtifactRecord):
    hierarchy_id: Digest
    payload: DemandHierarchyPayload

    @model_validator(mode="after")
    def _identity(self) -> DemandHierarchy:
        if self.hierarchy_id != content_digest(self.payload.model_dump(mode="json")):
            raise ValueError("Hierarchy content digest mismatch.")
        return self

    @property
    def series(self) -> tuple[DemandSeriesKey, ...]:
        return tuple(item.series for item in self.payload.members)


def hierarchy_membership(hierarchy: DemandHierarchy) -> pd.DataFrame:
    """Return one neutral membership per explicit bottom-level series."""
    return pd.DataFrame(
        [
            {**item.series.model_dump(), **item.model_dump(exclude={"series"})}
            for item in hierarchy.payload.members
        ]
    )


def build_demand_hierarchy(
    actuals: pd.DataFrame,
    request: DemandForecastRequest,
    features: DemandFeatureConfig | None = None,
) -> DemandHierarchy:
    """Discover only members and labels observed in origin-available training history."""
    history = available_demand_history(actuals, request, features)
    columns = [*DEMAND_SERIES_COLUMNS, *HIERARCHY_PARENT_COLUMNS]
    memberships = history[columns].drop_duplicates().sort_values(DEMAND_SERIES_COLUMNS)
    payload = DemandHierarchyPayload(
        known_at=request.forecast_origin,
        members=tuple(
            DemandHierarchyMember(
                series=DemandSeriesKey(**{key: row[key] for key in DEMAND_SERIES_COLUMNS}),
                **{key: row[key] for key in HIERARCHY_PARENT_COLUMNS},
            )
            for row in memberships.to_dict(orient="records")
        ),
    )
    return DemandHierarchy(
        hierarchy_id=content_digest(payload.model_dump(mode="json")), payload=payload
    )


def expand_demand_hierarchy(frame: pd.DataFrame, hierarchy: DemandHierarchy) -> pd.DataFrame:
    """Map explicit bottom rows to each level without pooling channels/geographies."""
    if frame.empty or not set(DEMAND_SERIES_COLUMNS).issubset(frame.columns):
        raise ContractViolation("Hierarchy expansion requires nonempty bottom-level series rows.")
    columns = [column for column in HIERARCHY_PARENT_COLUMNS if column in frame]
    joined = frame.drop(columns=columns).merge(
        hierarchy_membership(hierarchy),
        on=DEMAND_SERIES_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if joined[list(HIERARCHY_PARENT_COLUMNS)].isna().any().any():
        raise ContractViolation("Bottom-level rows include series outside the frozen hierarchy.")
    order = [
        *DEMAND_SERIES_COLUMNS,
        *[
            column
            for column in ("forecast_origin", "model_id", "date", "horizon")
            if column in joined
        ],
    ]
    joined = joined.sort_values(order, kind="stable")
    return pd.concat(
        [
            joined.assign(hierarchy_level=level, node_id=joined[column])
            for level, column in zip(
                HIERARCHY_LEVELS, ("sku_id", *HIERARCHY_PARENT_COLUMNS), strict=True
            )
        ],
        ignore_index=True,
    )
