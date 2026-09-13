"""Adapt a real snapshot extract to the accelerator's data contract.

Source systems rarely emit the contract's exact column names, identifier format,
or date format. This script maps an extract onto the contract without requiring
changes upstream:

* normalizes facility identifiers to ``FAC-NNN`` and writes the mapping out so
  the transformation is reproducible and reversible,
* converts ``accounting_month`` to ``YYYY-MM`` and ``snapshot_date`` to
  ``YYYY-MM-DD`` from any pandas-parseable input,
* derives ``snapshot_day``, ``days_elapsed``, ``business_days_elapsed``, and
  ``remaining_days`` from ``snapshot_date`` (they are a function of the as-of
  date, never measured values),
* renames source columns onto contract columns using an optional mapping file,
* reports which contract columns are unavailable rather than fabricating them,
* screens the result for target leakage before anything is trained.

Usage:
    uv run python scripts/adapt_snapshot_extract.py \\
        --input extract.csv \\
        --facility-col facility_id \\
        --snapshot-col snapshot_date \\
        --mapping column_map.yaml \\
        --out data/adapted/revenue_snapshots.parquet

The mapping file is optional and maps ``source_column: contract_column``:

    total_ip_obs_discharges: month_to_date_discharges
    avg_daily_census: month_to_date_average_daily_census
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from revenue_prediction.core.data.schema import (
    DIMENSION_COLUMNS,
    FEATURE_COLUMNS,
    HISTORICAL_FEATURES,
    OPERATIONAL_FEATURES,
    TARGET,
)

# Derived purely from snapshot_date; never sourced from the extract.
DERIVED_TIME_COLUMNS = ["days_elapsed", "business_days_elapsed", "remaining_days"]


def normalize_facility_ids(series: pd.Series, prefix: str = "FAC") -> tuple[pd.Series, dict]:
    """Map arbitrary facility codes to the contract's ``FAC-NNN`` format."""
    codes = sorted(series.astype(str).str.strip().unique())
    mapping = {code: f"{prefix}-{i + 1:03d}" for i, code in enumerate(codes)}
    return series.astype(str).str.strip().map(mapping), mapping


def add_time_features(frame: pd.DataFrame, snapshot_col: str) -> pd.DataFrame:
    """Derive the contract's calendar columns from the as-of date."""
    parsed = pd.to_datetime(frame[snapshot_col], errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"{bad} rows have an unparseable {snapshot_col}")

    days_in_month = parsed.dt.days_in_month
    frame["snapshot_date"] = parsed.dt.strftime("%Y-%m-%d")
    frame["accounting_month"] = parsed.dt.strftime("%Y-%m")
    frame["snapshot_day"] = parsed.dt.day.astype(int)
    frame["days_elapsed"] = parsed.dt.day.astype(float)
    frame["remaining_days"] = (days_in_month - parsed.dt.day).astype(float)
    frame["business_days_elapsed"] = [
        float(np.busday_count(d.replace(day=1).date(), (d + pd.Timedelta(days=1)).date()))
        for d in parsed
    ]

    frame["month"] = parsed.dt.month.astype(int)
    frame["quarter"] = parsed.dt.quarter.astype(int)
    frame["month_sin"] = np.sin(2 * np.pi * (parsed.dt.month - 1) / 12.0).round(6)
    frame["month_cos"] = np.cos(2 * np.pi * (parsed.dt.month - 1) / 12.0).round(6)
    return frame


def add_historical_features(frame: pd.DataFrame) -> list[str]:
    """Derive prior-month and rolling target features from closed months only.

    For each facility-month, only strictly earlier months contribute, so a row's
    history never contains its own (or any later) month-end value.
    """
    if TARGET not in frame.columns:
        return []

    monthly = (
        frame.groupby(["facility_id", "accounting_month"])[TARGET].first().reset_index()
    ).sort_values(["facility_id", "accounting_month"])

    derived: dict[str, dict[tuple[str, str], float]] = {
        name: {}
        for name in [
            "prior_month_net_revenue",
            "prior_year_same_month_net_revenue",
            "rolling_3_month_net_revenue",
            "rolling_6_month_net_revenue",
            "rolling_12_month_net_revenue",
        ]
    }

    for facility, group in monthly.groupby("facility_id"):
        months = group["accounting_month"].tolist()
        values = group[TARGET].tolist()
        for i, month in enumerate(months):
            history = values[:i]  # strictly prior closed months
            key = (facility, month)
            derived["prior_month_net_revenue"][key] = history[-1] if i >= 1 else np.nan
            derived["prior_year_same_month_net_revenue"][key] = values[i - 12] if i >= 12 else np.nan
            for window, name in [
                (3, "rolling_3_month_net_revenue"),
                (6, "rolling_6_month_net_revenue"),
                (12, "rolling_12_month_net_revenue"),
            ]:
                derived[name][key] = float(np.mean(history[-window:])) if history else np.nan

    keys = list(zip(frame["facility_id"], frame["accounting_month"], strict=False))
    for name, lookup in derived.items():
        frame[name] = [lookup.get(k, np.nan) for k in keys]
    return list(derived)


def screen_for_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag features that look like backfilled month-end values.

    A genuine as-of feature must move between snapshots of the same
    facility-month. One that is constant within the month, and correlates almost
    perfectly with the target, is the month-end answer copied backwards.
    """
    if TARGET not in frame.columns:
        return pd.DataFrame(columns=["feature", "constant_within_month", "corr_with_target"])

    candidates = [
        c
        for c in FEATURE_COLUMNS
        if c in frame.columns
        and c not in DERIVED_TIME_COLUMNS
        and c not in HISTORICAL_FEATURES
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not candidates:
        return pd.DataFrame(columns=["feature", "constant_within_month", "corr_with_target"])

    per_month = frame.groupby(["facility_id", "accounting_month"])[candidates].nunique()
    constant = (per_month <= 1).all()
    corr = frame[candidates].corrwith(frame[TARGET])

    report = pd.DataFrame(
        {
            "feature": candidates,
            "constant_within_month": [bool(constant[c]) for c in candidates],
            "corr_with_target": [float(corr[c]) if pd.notna(corr[c]) else float("nan") for c in candidates],
        }
    )
    report["suspect"] = report["constant_within_month"] & (report["corr_with_target"].abs() > 0.99)
    return report.sort_values(["suspect", "corr_with_target"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--facility-col", default="facility_id")
    parser.add_argument("--snapshot-col", default="snapshot_date")
    parser.add_argument("--target-col", default=None, help="Source month-end net revenue column")
    parser.add_argument("--mapping", type=Path, default=None, help="source->contract column map")
    parser.add_argument("--facility-prefix", default="FAC")
    args = parser.parse_args()

    frame = (
        pd.read_parquet(args.input)
        if args.input.suffix == ".parquet"
        else pd.read_csv(args.input)
    )
    print(f"input rows    : {len(frame)}")
    print(f"input columns : {len(frame.columns)}")

    if args.mapping is not None:
        column_map = yaml.safe_load(args.mapping.read_text()) or {}
        frame = frame.rename(columns={k: v for k, v in column_map.items() if k in frame.columns})
        print(f"renamed       : {len(column_map)} source columns")

    frame["facility_id"], facility_map = normalize_facility_ids(
        frame[args.facility_col], args.facility_prefix
    )
    print(f"facilities    : {len(facility_map)}")

    frame = add_time_features(frame, args.snapshot_col)

    if args.target_col and args.target_col in frame.columns:
        frame[TARGET] = pd.to_numeric(frame[args.target_col], errors="coerce")

    # Dimensions are required non-null strings; mark them explicitly rather than
    # inventing a value that implies a real grouping.
    for col in DIMENSION_COLUMNS:
        if col not in frame.columns:
            frame[col] = "unspecified"

    available = [c for c in OPERATIONAL_FEATURES + HISTORICAL_FEATURES if c in frame.columns]
    derived_history = add_historical_features(frame)
    if derived_history:
        print(f"derived history   : {len(derived_history)} feature(s) from closed months only")

    available = [c for c in OPERATIONAL_FEATURES + HISTORICAL_FEATURES if c in frame.columns]
    missing = [c for c in OPERATIONAL_FEATURES + HISTORICAL_FEATURES if c not in frame.columns]

    # The contract requires every feature column to exist. Absent measures are
    # filled with NaN rather than a value, so "unknown" stays distinguishable
    # from a real observation.
    for col in missing:
        frame[col] = np.nan

    keep = ["facility_id", "accounting_month", "snapshot_date", "snapshot_day"]
    keep += DIMENSION_COLUMNS + OPERATIONAL_FEATURES + HISTORICAL_FEATURES
    if TARGET in frame.columns:
        keep.append(TARGET)
    out_frame = frame[[c for c in dict.fromkeys(keep) if c in frame.columns]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_parquet(args.out, index=False)
    map_path = args.out.with_name(f"{args.out.stem}_facility_map.json")
    map_path.write_text(json.dumps(facility_map, indent=2))

    print(f"\nmapped features   : {len(available)}")
    print(f"empty (NaN) cols  : {len(missing)}")
    if missing:
        print("  " + ", ".join(missing))
        print("  These impute to a single constant at training time and carry no")
        print("  signal. Exclude them from the model rather than treating them as data.")

    report = screen_for_leakage(out_frame)
    suspects = report[report["suspect"]] if "suspect" in report.columns else report
    print(f"\nleakage screen    : {len(suspects)} suspect feature(s)")
    if not suspects.empty:
        print(suspects.to_string(index=False))
        print("  Constant within facility-month AND ~1.0 correlation with the target")
        print("  means a month-end value was copied into a feature. Remove before training.")

    print(f"\nwritten to        : {args.out}")
    print(f"facility map      : {map_path}")
    print("\nValidate with:")
    flag = "" if TARGET in out_frame.columns else " --no-require-target"
    print(f"  uv run revenue-prediction validate-data {args.out}{flag}")


if __name__ == "__main__":
    main()
