"""Deterministic generation without changing the inherited revenue path."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from revenue_prediction.core.data.demand_schema import (
    DEMAND_COLUMN_AVAILABILITY,
    DEMAND_COLUMNS,
    DEMAND_KEY_COLUMNS,
    DEMAND_TARGET,
    DemandDataConfig,
    validate_demand_actuals,
)
from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("frequency", ["MS", "W-MON"])
def test_default_history_and_contract(frequency: str) -> None:
    config = DemandDataConfig.model_validate({"frequency": frequency})
    frame = generate_synthetic_demand_dataset(config)
    assert len(frame) == config.period_count * 3 * 2 * 2
    assert list(frame.columns) == DEMAND_COLUMNS
    assert set(DEMAND_COLUMN_AVAILABILITY) == set(DEMAND_COLUMNS)
    assert frame["period_end"].max() >= pd.Timestamp(config.first_period) + pd.DateOffset(
        years=5
    ) - pd.Timedelta(days=1)
    assert not frame.duplicated(DEMAND_KEY_COLUMNS).any()
    assert np.isfinite(frame[DEMAND_TARGET]).all()
    assert (frame[DEMAND_TARGET] >= 0).all()
    validate_demand_actuals(frame)


def test_same_seed_is_repeatable_even_when_reusing_config() -> None:
    config = DemandDataConfig(seed=123, n_periods=8)
    first = generate_synthetic_demand_dataset(config)
    pd.testing.assert_frame_equal(first, generate_synthetic_demand_dataset(config))
    first.loc[0, DEMAND_TARGET] = -100.0
    assert (generate_synthetic_demand_dataset(config)[DEMAND_TARGET] >= 0).all()


def test_different_seed_changes_values_not_keys() -> None:
    first = generate_synthetic_demand_dataset(DemandDataConfig(seed=123, n_periods=8))
    second = generate_synthetic_demand_dataset(DemandDataConfig(seed=124, n_periods=8))
    pd.testing.assert_frame_equal(first[DEMAND_KEY_COLUMNS], second[DEMAND_KEY_COLUMNS])
    assert not first[DEMAND_TARGET].equals(second[DEMAND_TARGET])


def test_custom_grain_and_close_latency() -> None:
    config = DemandDataConfig(
        frequency="W-MON",
        start_date=date(2024, 12, 30),
        n_periods=3,
        n_skus=10,
        n_channels=1,
        n_geographies=1,
        close_delay_days=3,
    )
    frame = generate_synthetic_demand_dataset(config)
    assert len(frame) == 30
    assert frame["date"].min() == pd.Timestamp("2024-12-30")
    assert frame["date"].max() == pd.Timestamp("2025-01-13")
    assert (frame["actual_available_at"] - frame["period_end"]).eq(pd.Timedelta(days=3)).all()
    assert frame["brand_id"].nunique() == 2
    assert frame["product_family_id"].nunique() == 3


def test_generation_does_not_mutate_global_random_state() -> None:
    state_before = np.random.get_state()
    generate_synthetic_demand_dataset(DemandDataConfig(n_periods=1))
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


@pytest.mark.parametrize(
    "values",
    [
        {"n_periods": 0},
        {"n_skus": 0},
        {"n_channels": -1},
        {"n_geographies": 0},
        {"seed": -1},
        {"seed": True},
        {"n_periods": 2.5},
        {"frequency": "D"},
        {"start_date": "2024-01-02"},
        {"frequency": "W-MON", "start_date": "2024-01-02"},
        {"close_delay_days": 0},
        {"noise_scale": -0.1},
        {"noise_scale": float("nan")},
        {"noise_scale": float("inf")},
        {"n_sku": 2},
    ],
)
def test_invalid_config_fails_explicitly(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DemandDataConfig.model_validate(values)
