"""Typer-based CLI entry point.

All commands run fully offline by default. Azure/Fabric commands are opt-in and
fail gracefully with actionable messages when credentials or the optional
``azure`` / ``fabric`` extras are not present.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from revenue_prediction.config.loader import load_settings
from revenue_prediction.interfaces.cli.demand_lifecycle import app as demand_lifecycle_app

app = typer.Typer(
    help="Healthcare facility net-revenue prediction accelerator (synthetic data).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
app.add_typer(demand_lifecycle_app, name="demand-lifecycle")


@app.command("generate-data")
def generate_data(
    environment: str = typer.Option("dev", "--env", "-e", help="dev|test|prod"),
) -> None:
    """Generate and persist the synthetic, sample, and invalid datasets."""
    from revenue_prediction.core.data.io import materialise_default_datasets

    settings = load_settings(environment)
    outputs = materialise_default_datasets(settings.data)
    table = Table(title="Synthetic datasets written")
    table.add_column("dataset")
    table.add_column("path")
    for name, path in outputs.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("train-local")
def train_local(
    environment: str = typer.Option("dev", "--env", "-e"),
    output_dir: Path = typer.Option(Path("outputs"), "--out", "-o"),
    track_mlflow: bool = typer.Option(False, "--mlflow", help="Log runs to a local MLflow store"),
) -> None:
    """Train all code-first candidates offline and select champion/challenger."""
    from revenue_prediction.pipelines.local_pipeline import run_local_pipeline

    settings = load_settings(environment)
    result = run_local_pipeline(settings, output_dir=output_dir, track_mlflow=track_mlflow)

    table = Table(title=f"Model comparison (primary metric: {result.selection.metric})")
    for col in result.comparison.columns:
        table.add_column(str(col))
    for _, row in result.comparison.iterrows():
        table.add_row(*[f"{v:.4g}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)

    console.print(f"[bold green]Champion:[/] {result.selection.champion}")
    if result.selection.challenger:
        console.print(f"[bold]Challenger:[/] {result.selection.challenger}")
    if result.bundle_path:
        console.print(f"Champion bundle saved to: {result.bundle_path}")


@app.command("predict")
def predict(
    bundle_path: Path = typer.Argument(..., help="Path to a saved champion bundle"),
    data_path: Path = typer.Argument(..., help="Path to snapshot data (parquet/csv)"),
    output_path: Path = typer.Option(Path("outputs/predictions.csv"), "--out", "-o"),
    cutoff_day: int | None = typer.Option(None, "--cutoff-day"),
) -> None:
    """Score a dataset with a saved champion bundle (batch inference)."""
    from revenue_prediction.core.data.io import read_dataset, write_dataset
    from revenue_prediction.core.inference.predict import batch_predict, load_bundle

    bundle = load_bundle(bundle_path)
    frame = read_dataset(data_path)
    predictions = batch_predict(bundle, frame, cutoff_day=cutoff_day)
    write_dataset(predictions, output_path)
    console.print(f"[green]Wrote {len(predictions)} predictions to {output_path}[/]")


@app.command("scorecard")
def scorecard(
    predictions_path: Path = typer.Argument(..., help="Scored predictions (parquet/csv)"),
    actuals_path: Path = typer.Argument(..., help="Closed-month actuals with the target column"),
    baseline_wape: float | None = typer.Option(
        None,
        "--baseline-wape",
        help="Manual-analyst baseline WAPE (fraction) for the beat-baseline KPI",
    ),
) -> None:
    """Measure evaluation metrics against checkpoint targets and business KPIs."""
    from revenue_prediction.core.data.io import read_dataset
    from revenue_prediction.core.data.schema import KEY_COLUMNS, TARGET
    from revenue_prediction.core.evaluation.kpi import kpi_scorecard
    from revenue_prediction.core.evaluation.metrics import compute_metrics, metrics_by_snapshot_day

    prediction_col = "predicted_month_end_net_revenue"
    preds = read_dataset(predictions_path)
    actuals = read_dataset(actuals_path)
    keys = [c for c in KEY_COLUMNS if c in preds.columns and c in actuals.columns]
    merged = preds.merge(actuals[[*keys, TARGET]], on=keys, how="inner")
    if merged.empty:
        console.print("[red]No overlapping keys between predictions and actuals.[/]")
        raise typer.Exit(code=1)

    overall = compute_metrics(merged[TARGET], merged[prediction_col])
    by_day = metrics_by_snapshot_day(merged, TARGET, prediction_col)
    board = kpi_scorecard(overall, by_day, baseline_wape=baseline_wape)

    table = Table(title="KPI scorecard — metrics vs targets")
    for col in ("checkpoint", "metric", "value", "target", "met", "margin"):
        table.add_column(col)
    for _, row in board.iterrows():
        met = "[green]YES[/]" if bool(row["met"]) else "[red]NO[/]"
        table.add_row(
            str(row["checkpoint"]),
            str(row["metric"]),
            f"{float(row['value']):.4f}",
            f"{float(row['target']):.4f}",
            met,
            f"{float(row['margin']):+.4f}",
        )
    console.print(table)
    console.print(
        f"[bold]Overall WAPE:[/] {overall['wape']:.4f}  [bold]bias:[/] {overall['bias']:+.4f}"
    )


@app.command("validate-data")
def validate_data(
    data_path: Path = typer.Argument(..., help="Path to snapshot data (parquet/csv)"),
    require_target: bool = typer.Option(True, "--require-target/--no-require-target"),
) -> None:
    """Validate a dataset against the schema and leakage rules."""
    from revenue_prediction.core.data.contracts import (
        ContractViolation,
        validate_leakage_rules,
        validate_raw_snapshots,
    )
    from revenue_prediction.core.data.io import read_dataset

    frame = read_dataset(data_path)
    try:
        validate_raw_snapshots(frame, require_target=require_target)
        validate_leakage_rules(frame)
    except ContractViolation as exc:
        console.print(f"[red]Contract violation:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]OK[/] — {len(frame)} rows passed schema + leakage checks")


@app.command("info")
def info(environment: str = typer.Option("dev", "--env", "-e")) -> None:
    """Show resolved configuration (secrets are never printed)."""
    settings = load_settings(environment)
    console.print(f"[bold]Environment:[/] {settings.environment}")
    console.print(f"[bold]Facilities:[/] {settings.data.n_facilities}")
    console.print(f"[bold]Months:[/] {settings.data.n_months}")
    console.print(f"[bold]Snapshot days:[/] {settings.data.snapshot_days}")
    console.print(f"[bold]Candidates:[/] {settings.model.candidates}")
    console.print(f"[bold]Azure ML configured:[/] {settings.azure_ml.is_configured()}")
    console.print(f"[bold]Fabric configured:[/] {settings.fabric.is_configured()}")


@app.command("backtest-demand")
def backtest_demand(
    frequency: str = typer.Option("MS", "--frequency", help="MS (monthly) or W-MON (weekly)"),
    horizon: int = typer.Option(3, "--horizon", min=1),
    n_origins: int = typer.Option(3, "--origins", min=1),
    step_periods: int = typer.Option(1, "--step", min=1),
    training_window: int | None = typer.Option(None, "--window", min=1),
    seed: int = typer.Option(42, "--seed", min=0),
    output_dir: Path = typer.Option(Path("outputs/demand-backtest"), "--out", "-o"),
) -> None:
    """Run synthetic-only demand baselines locally; no cloud or forecast approval."""
    from pydantic import ValidationError

    from revenue_prediction.core.data.contracts import ContractViolation
    from revenue_prediction.core.data.demand_schema import DemandDataConfig
    from revenue_prediction.core.data.io import write_dataset
    from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
    from revenue_prediction.core.training.demand_backtesting import (
        DemandBacktestConfig,
        run_demand_backtest,
    )

    try:
        data_config = DemandDataConfig.model_validate({"frequency": frequency, "seed": seed})
        actuals = generate_synthetic_demand_dataset(data_config)
        dates = actuals["date"].drop_duplicates().sort_values()
        first_index = len(dates) - horizon - (n_origins - 1) * step_periods
        if first_index < 1:
            raise typer.BadParameter(
                "Requested origins/horizon exceed the synthetic history; reduce --origins, "
                "--horizon or --step."
            )
        config = DemandBacktestConfig(
            first_origin=dates.iloc[first_index].date(),
            evaluation_as_of=actuals["actual_available_at"].max().date(),
            frequency=data_config.frequency,
            horizon=horizon,
            n_origins=n_origins,
            step_periods=step_periods,
            strategy="sliding" if training_window is not None else "expanding",
            training_window_periods=training_window,
        )
        result = run_demand_backtest(actuals, config)
    except (ValidationError, ContractViolation) as exc:
        console.print(f"Demand backtest failed: {exc}", style="red", markup=False)
        raise typer.Exit(code=1) from exc

    write_dataset(actuals, output_dir / "actuals.parquet")
    write_dataset(result.predictions, output_dir / "backtest_predictions.parquet")
    write_dataset(result.metrics, output_dir / "metrics.csv")
    (output_dir / "data_config.json").write_text(
        data_config.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_dir / "backtest_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    table = Table(title="Synthetic demand baselines (local, unapproved)")
    for column in ("model_id", "horizon", "n_predictions", "mae", "wape", "bias_fraction"):
        table.add_column(column)
    for _, row in result.metrics.iterrows():
        table.add_row(
            str(row["model_id"]),
            str(row["horizon"]),
            str(row["n_predictions"]),
            f"{row['mae']:.4f}",
            f"{row['wape']:.4f}",
            f"{row['bias_fraction']:+.4f}",
        )
    console.print(table)
    console.print(
        f"Wrote synthetic actuals, backtest predictions, metrics and config to {output_dir}"
    )


@app.command("demand-models")
def demand_models() -> None:
    """List implemented time-series forecasters and their executable capabilities."""
    import json
    from dataclasses import asdict

    from revenue_prediction.core.models.demand_forecasters import demand_model_capabilities

    console.print(
        json.dumps([asdict(item) for item in demand_model_capabilities()], indent=2),
        markup=False,
    )


@app.command("compare-demand")
def compare_demand(
    frequency: str = typer.Option("MS", "--frequency", help="MS or W-MON"),
    horizon: int = typer.Option(3, "--horizon", min=1),
    n_origins: int = typer.Option(
        3, "--origins", min=1, help="Origins in each validation/holdout block"
    ),
    training_window: int | None = typer.Option(None, "--window", min=1),
    primary_metric: str = typer.Option(
        "mase", "--metric", help="mase, rmsse or wape (not regression R2/MAPE)"
    ),
    minimum_improvement: float = typer.Option(0.0, "--minimum-improvement", min=0),
    max_abs_bias: float | None = typer.Option(None, "--max-abs-bias", min=0),
    max_horizon_degradation: float | None = typer.Option(None, "--max-horizon-degradation", min=0),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Comma-separated registry IDs; defaults preserve the seven original models",
    ),
    tuning_grid: Path | None = typer.Option(
        None, "--tuning-grid", help="JSON array of predeclared TimeSeriesModelConfig objects"
    ),
    demand_pattern: str = typer.Option(
        "regular", "--pattern", help="regular, intermittent, long_zero_runs or obsolescence"
    ),
    occurrence_probability: float = typer.Option(0.2, "--occurrence-probability"),
    zero_run_periods: int = typer.Option(6, "--zero-run-periods", min=1),
    obsolete_after_fraction: float = typer.Option(0.7, "--obsolete-after-fraction"),
    n_skus: int = typer.Option(3, "--skus", min=1),
    n_channels: int = typer.Option(2, "--channels", min=1),
    n_geographies: int = typer.Option(2, "--geographies", min=1),
    seed: int = typer.Option(42, "--seed", min=0),
    output_dir: Path = typer.Option(Path("outputs/demand-comparison"), "--out", "-o"),
) -> None:
    """Compare synthetic time-series algorithms on validation, then fixed final holdout."""
    import json
    from dataclasses import asdict

    from pydantic import ValidationError

    from revenue_prediction.core.data.contracts import ContractViolation
    from revenue_prediction.core.data.demand_forecast import DemandForecastRequest
    from revenue_prediction.core.data.demand_schema import DemandDataConfig
    from revenue_prediction.core.data.io import write_dataset
    from revenue_prediction.core.data.synthetic import generate_synthetic_demand_dataset
    from revenue_prediction.core.evaluation.demand_diagnostics import demand_training_diagnostics
    from revenue_prediction.core.evaluation.demand_selection import (
        DemandSelectionConfig,
        run_demand_selection,
    )
    from revenue_prediction.core.evaluation.demand_tuning import (
        DemandTuningConfig,
        run_demand_tuning,
    )
    from revenue_prediction.core.features.demand import DemandFeatureConfig
    from revenue_prediction.core.models.demand_forecasters import (
        DEFAULT_DEMAND_MODELS,
        demand_model_capabilities,
    )
    from revenue_prediction.core.training.demand_backtesting import DemandBacktestConfig

    try:
        data_config = DemandDataConfig.model_validate(
            {
                "frequency": frequency,
                "seed": seed,
                "demand_pattern": demand_pattern,
                "occurrence_probability": occurrence_probability,
                "zero_run_periods": zero_run_periods,
                "obsolete_after_fraction": obsolete_after_fraction,
                "n_skus": n_skus,
                "n_channels": n_channels,
                "n_geographies": n_geographies,
            }
        )
        actuals = generate_synthetic_demand_dataset(data_config)
        dates = actuals["date"].drop_duplicates().sort_values()
        holdout_index = len(dates) - horizon - (n_origins - 1)
        validation_index = holdout_index - horizon - (n_origins - 1)
        if validation_index < 1:
            raise typer.BadParameter(
                "Reduce --origins/--horizon to leave training, validation and holdout history."
            )
        common = {
            "frequency": data_config.frequency,
            "horizon": horizon,
            "n_origins": n_origins,
            "models": tuple(item.strip() for item in models.split(","))
            if models is not None
            else DEFAULT_DEMAND_MODELS,
            "strategy": "sliding" if training_window is not None else "expanding",
            "training_window_periods": training_window,
        }
        validation = DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[validation_index].date(),
                "evaluation_as_of": dates.iloc[holdout_index].date(),
            }
        )
        holdout = DemandBacktestConfig.model_validate(
            {
                **common,
                "first_origin": dates.iloc[holdout_index].date(),
                "evaluation_as_of": actuals["actual_available_at"].max().date(),
            }
        )
        config = DemandSelectionConfig.model_validate(
            {
                "validation": validation,
                "holdout": holdout,
                "primary_metric": primary_metric,
                "minimum_relative_improvement": minimum_improvement,
                "max_abs_bias_fraction": max_abs_bias,
                "max_horizon_relative_degradation": max_horizon_degradation,
            }
        )
        tuned = None
        tuning_config = None
        if tuning_grid is not None:
            tuning_config = DemandTuningConfig.model_validate(
                {
                    "experiment": config,
                    "parameter_grid": json.loads(tuning_grid.read_text(encoding="utf-8")),
                }
            )
            tuned = run_demand_tuning(actuals, tuning_config)
            config, result = tuned.frozen_config, tuned.selection
        else:
            result = run_demand_selection(actuals, config)
        diagnostics, eligibility = demand_training_diagnostics(
            actuals,
            DemandForecastRequest(
                forecast_origin=config.validation.first_origin,
                frequency=config.validation.frequency,
            ),
            DemandFeatureConfig(
                seasonal_periods=config.validation.seasonal_periods,
                training_window_periods=config.validation.training_window_periods,
            ),
            models=tuning_config.experiment.validation.models
            if tuning_config is not None
            else config.validation.models,
            parameters=config.validation.model_parameters,
        )
    except (ValidationError, ContractViolation, ImportError, ValueError, OSError) as exc:
        console.print(f"Demand comparison failed: {exc}", style="red", markup=False)
        raise typer.Exit(code=1) from exc

    write_dataset(actuals, output_dir / "actuals.parquet")
    for phase, evaluation in (("validation", result.validation), ("holdout", result.holdout)):
        write_dataset(evaluation.predictions, output_dir / f"{phase}_predictions.parquet")
        write_dataset(evaluation.metrics, output_dir / f"{phase}_metrics.csv")
        write_dataset(evaluation.series_metrics, output_dir / f"{phase}_series_metrics.csv")
        write_dataset(evaluation.segment_metrics, output_dir / f"{phase}_segment_metrics.csv")
    write_dataset(diagnostics, output_dir / "training_diagnostics.csv")
    write_dataset(eligibility, output_dir / "model_eligibility.csv")
    if tuned is not None and tuning_config is not None:
        write_dataset(tuned.trial_ranking, output_dir / "tuning_trials.csv")
        write_dataset(tuned.trial_metrics, output_dir / "tuning_metrics.csv")
        (output_dir / "tuning_config.json").write_text(
            tuning_config.model_dump_json(indent=2), encoding="utf-8"
        )
        (output_dir / "tuning_selection.json").write_text(
            json.dumps({"selected_trial": tuned.selected_trial}), encoding="utf-8"
        )
        failures = tuned.trial_ranking.loc[
            tuned.trial_ranking["selection_reason"] == "fit_or_evaluation_failed"
        ]
        if not failures.empty:
            console.print("Rejected full-cohort trials (see tuning_trials.csv):", style="yellow")
            for row in failures.itertuples(index=False):
                console.print(f"Trial {row.trial}, {row.model_id}: {row.failure}", markup=False)
    write_dataset(result.ranking, output_dir / "validation_ranking.csv")
    for name, value in (
        ("data_config", data_config),
        ("experiment_config", config),
        ("selection", result.decision),
    ):
        (output_dir / f"{name}.json").write_text(value.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "model_capabilities.json").write_text(
        json.dumps([asdict(item) for item in demand_model_capabilities()], indent=2),
        encoding="utf-8",
    )
    table = Table(title="Time-series selection: validation only (local, unapproved)")
    for column in ("model_id", "mase", "rmsse", "wape", "eligible", "selection_reason"):
        table.add_column(column)
    for _, row in result.ranking.iterrows():
        table.add_row(
            str(row["model_id"]),
            f"{row['mase']:.4f}",
            f"{row['rmsse']:.4f}",
            f"{row['wape']:.4f}",
            str(row["eligible"]),
            str(row["selection_reason"]),
        )
    console.print(table)
    console.print(
        f"Validation-selected model: {result.decision.selected_model}; "
        f"final holdout evaluated without reselection. Wrote results to {output_dir}"
    )


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload for development"),
) -> None:
    """Serve the API (and built React UI, if present). Requires the 'api' extra."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        console.print("[red]FastAPI/uvicorn not installed. Run:[/] uv sync --extra api")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Serving on http://{host}:{port}[/]  (API docs at /docs)")
    uvicorn.run("revenue_prediction.interfaces.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
