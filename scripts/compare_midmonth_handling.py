"""Compare strategies for handling mid-month snapshot rows.

Mid-month rows carry predictors but no observed target until the accounting
month closes, so the target is the (constant) month-end actual. That raises a
design question: do the extra snapshot rows add predictive value, or do they
teach the model patterns it cannot use at scoring time?

Three training strategies are compared, all scored on the **same** test rows
(the checkpoint finance actually uses), so the comparison is apples to apples:

``all_snapshots``
    Train on every snapshot row, including days later in the month than the
    checkpoint being scored.
``upto_cutoff``
    Train only on rows at or before the checkpoint. At day 15 a later day-21
    row does not exist yet, so this matches production availability.
``cutoff_only``
    Train only on rows at the checkpoint itself, one row per facility-month.

Usage:
    uv run python scripts/compare_midmonth_handling.py --env dev --cutoff-day 15
"""

from __future__ import annotations

import argparse

import pandas as pd

from revenue_prediction.config.loader import load_settings
from revenue_prediction.core.data.io import read_dataset
from revenue_prediction.core.data.schema import SNAPSHOT_DAY_COL
from revenue_prediction.core.training.splitting import TemporalSplit, blocked_temporal_split
from revenue_prediction.core.training.train import train_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dev")
    parser.add_argument("--data", default=None, help="Defaults to the configured synthetic parquet")
    parser.add_argument("--cutoff-day", type=int, default=None)
    parser.add_argument(
        "--model", default=None, help="Defaults to the configured primary candidate"
    )
    args = parser.parse_args()

    settings = load_settings(args.env)
    path = args.data or f"{settings.data.raw_dir}/revenue_snapshots.parquet"
    cutoff = args.cutoff_day or settings.data.demo_cutoff_day
    model_name = args.model or "gradient_boosting"

    frame = read_dataset(path)
    split = blocked_temporal_split(frame, settings.split)

    # Hold the evaluation set fixed: the checkpoint rows in the test months.
    test_at_cutoff = split.test[split.test[SNAPSHOT_DAY_COL] == cutoff]
    if test_at_cutoff.empty:
        raise SystemExit(
            f"No test rows at snapshot_day={cutoff}. Available: "
            f"{sorted(split.test[SNAPSHOT_DAY_COL].unique())}"
        )

    strategies = {
        "all_snapshots": split.train,
        "upto_cutoff": split.train[split.train[SNAPSHOT_DAY_COL] <= cutoff],
        "cutoff_only": split.train[split.train[SNAPSHOT_DAY_COL] == cutoff],
    }

    print(f"environment   : {args.env}")
    print(f"model         : {model_name}")
    print(f"cutoff day    : {cutoff}")
    print(f"test rows     : {len(test_at_cutoff)} (fixed across strategies)\n")

    rows = []
    for name, train_frame in strategies.items():
        if train_frame.empty:
            continue
        variant = TemporalSplit(
            train=train_frame,
            validation=split.validation,
            test=test_at_cutoff,
            train_months=split.train_months,
            validation_months=split.validation_months,
            test_months=split.test_months,
        )
        result = train_candidate(model_name, variant, settings.model, settings.features)
        rows.append(
            {
                "strategy": name,
                "train_rows": len(train_frame),
                "wape": result.metrics["wape"],
                "bias": result.metrics["bias"],
                "mae": result.metrics["mae"],
                "rmse": result.metrics["rmse"],
            }
        )

    table = pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)
    print(table.to_string(index=False))

    best = table.iloc[0]
    baseline = table[table["strategy"] == "all_snapshots"]
    print(f"\nbest strategy : {best['strategy']} (wape {best['wape']:.5f})")
    if not baseline.empty:
        delta = (baseline.iloc[0]["wape"] - best["wape"]) / baseline.iloc[0]["wape"]
        print(f"vs all_snapshots: {delta:+.2%} relative WAPE change")
        if abs(delta) < 0.02:
            print("Within the 2% challenger threshold, so treat the strategies as equivalent")
            print("and prefer the simplest (fewest training rows).")


if __name__ == "__main__":
    main()
