"""Shared immutable records and canonical content hashes for demand evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ActorId = Annotated[str, Field(pattern=r"^ACTOR-\d{3,}$")]


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def canonical_json(value: object) -> str:
    """One strict JSON representation for content IDs and audit-chain hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """JSON scalars preserving full float precision, with explicit missing metrics."""
    result = frame.copy()
    for column in result:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].map(lambda value: value.isoformat())
    return result.astype(object).where(pd.notna(result), None).to_dict(orient="records")
