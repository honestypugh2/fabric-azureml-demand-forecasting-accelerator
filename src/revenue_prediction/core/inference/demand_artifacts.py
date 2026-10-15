"""Data-only demand bundles, reproducible evaluation lineage and model cards."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from revenue_prediction.core.data.contracts import ContractViolation
from revenue_prediction.core.data.demand_forecast import (
    FORECAST_COLUMNS,
    FORECAST_SCHEMA_VERSION,
    DemandForecastRequest,
    DemandSeriesKey,
    demand_period_offset,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DEMAND_TARGET
from revenue_prediction.core.evaluation.demand_selection import (
    DemandSelectionConfig,
    DemandSelectionDecision,
    run_demand_selection,
)
from revenue_prediction.core.evaluation.demand_tuning import DemandTuningConfig, run_demand_tuning
from revenue_prediction.core.features.demand import DemandFeatureConfig, available_demand_history
from revenue_prediction.core.models.demand_forecasters import (
    DemandForecasterState,
    create_demand_forecaster,
    demand_model_capability,
    restore_demand_forecaster,
)

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


def runtime_versions(model_id: str) -> dict[str, str]:
    packages = ["numpy", "pandas", "pydantic"]
    if demand_model_capability(model_id).provider == "statsmodels":
        packages += ["statsmodels", "scipy"]
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def implementation_digest() -> str:
    """Fingerprint executable demand sources, including uncommitted changes."""
    root = Path(__file__).resolve().parents[2]
    paths = [
        *sorted((root / "core").rglob("demand*.py")),
        root / "core/data/contracts.py",
        root / "core/evaluation/metrics.py",
        root / "ops/demand_registry.py",
    ]
    return content_digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


class TrainingObservation(ArtifactRecord):
    date: datetime
    actual_available_at: datetime
    value: float = Field(ge=0)


class SeriesArtifact(ArtifactRecord):
    series: DemandSeriesKey
    observations: tuple[TrainingObservation, ...] = Field(min_length=1)
    state: DemandForecasterState

    @model_validator(mode="after")
    def _match_training(self) -> SeriesArtifact:
        if (
            tuple(row.date for row in self.observations) != self.state.dates
            or tuple(row.value for row in self.observations) != self.state.values
        ):
            raise ValueError("Fitted state must match its complete training observations.")
        return self


class EvaluationMetric(ArtifactRecord):
    model_id: str
    horizon: int = Field(ge=0)
    n_predictions: int = Field(gt=0)
    mase: float | None = Field(ge=0)
    rmsse: float | None = Field(ge=0)
    wape: float | None = Field(ge=0)
    bias_fraction: float | None
    scaled_metrics_status: str


class DemandLineage(ArtifactRecord):
    source_actuals_sha256: Digest
    training_sha256: Digest
    validation_predictions_sha256: Digest
    holdout_predictions_sha256: Digest
    experiment_sha256: Digest
    implementation_sha256: Digest
    tuning_config_sha256: Digest | None = None
    tuning_trials_sha256: Digest | None = None
    runtime: dict[str, str]


class DemandBundlePayload(ArtifactRecord):
    schema_version: Literal["demand-bundle-v1"] = "demand-bundle-v1"
    created_at: datetime
    created_by: ActorId
    origin: date
    frequency: Literal["MS", "W-MON"]
    maximum_horizon: int = Field(ge=1, strict=True)
    target: Literal["demand_units"] = "demand_units"
    data_classification: Literal["synthetic"] = "synthetic"
    decision: DemandSelectionDecision
    experiment: DemandSelectionConfig
    lineage: DemandLineage
    series: tuple[SeriesArtifact, ...] = Field(min_length=1)
    validation_metrics: tuple[EvaluationMetric, ...] = Field(min_length=1)
    holdout_metrics: tuple[EvaluationMetric, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_bundle(self) -> DemandBundlePayload:
        origin = pd.Timestamp(self.origin)
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != UTC.utcoffset(
            self.created_at
        ):
            raise ValueError("Artifact creation timestamp must be UTC-aware.")
        if self.origin < self.experiment.holdout.evaluation_as_of:
            raise ValueError("Artifact origin must follow all evaluation evidence availability.")
        if self.decision.selected_at != self.experiment.holdout.first_origin:
            raise ValueError("Selection timestamp differs from experiment.")
        if (
            self.frequency != self.experiment.validation.frequency
            or self.maximum_horizon != self.experiment.validation.horizon
        ):
            raise ValueError("Artifact frequency/horizon must match evaluated scope.")
        if (
            self.decision.selected_model not in self.experiment.validation.models
            or self.decision.selected_parameters != self.experiment.validation.model_parameters
        ):
            raise ValueError("Selected model/parameters differ from frozen experiment.")
        if self.decision.primary_metric != self.experiment.primary_metric:
            raise ValueError("Decision metric differs from experiment.")
        keys = [item.series.key for item in self.series]
        if len(keys) != len(set(keys)):
            raise ValueError("Artifact series must be unique.")
        if self.experiment.validation.series is not None and set(keys) != {
            item.key for item in self.experiment.validation.series
        }:
            raise ValueError("Artifact cohort differs from explicit experiment scope.")
        training = [item.model_dump(mode="json") for item in self.series]
        if content_digest(training) != self.lineage.training_sha256:
            raise ValueError("Training lineage digest mismatch.")
        if (
            content_digest(self.experiment.model_dump(mode="json"))
            != self.lineage.experiment_sha256
        ):
            raise ValueError("Experiment lineage digest mismatch.")
        for item in self.series:
            state = item.state
            if (
                state.model_id != self.decision.selected_model
                or state.parameters != self.decision.selected_parameters
                or state.frequency != self.frequency
            ):
                raise ValueError("Series model state differs from artifact selection.")
            if state.seasonal_periods != (
                self.experiment.validation.seasonal_periods
                or (12 if self.frequency == "MS" else 52)
            ):
                raise ValueError("Series seasonal length differs from experiment.")
            dates = pd.DatetimeIndex(state.dates)
            if not dates.is_unique or not dates.is_monotonic_increasing:
                raise ValueError("Training state must be chronological and unique.")
            for row in item.observations:
                available = pd.Timestamp(row.actual_available_at)
                start = pd.Timestamp(row.date)
                end = (
                    start + pd.offsets.MonthEnd(0)
                    if self.frequency == "MS"
                    else start + pd.Timedelta(days=6)
                )
                if (
                    start.tzinfo is not None
                    or available.tzinfo is not None
                    or not end < available <= origin
                ):
                    raise ValueError(
                        "Artifact training includes unclosed/unavailable observations."
                    )
                window = self.experiment.validation.training_window_periods
                if window is not None:
                    cutoff = DemandForecastRequest(
                        forecast_origin=self.origin, frequency=self.frequency
                    ).forecast_dates[0] - window * demand_period_offset(self.frequency)
                    if start < cutoff:
                        raise ValueError(
                            "Artifact observations exceed the configured training window."
                        )
        for metrics, models, n_origins in (
            (
                self.validation_metrics,
                set(self.experiment.validation.models),
                self.experiment.validation.n_origins,
            ),
            (
                self.holdout_metrics,
                {"naive", "seasonal_naive", self.decision.selected_model},
                self.experiment.holdout.n_origins,
            ),
        ):
            expected = {
                (model, horizon) for model in models for horizon in range(self.maximum_horizon + 1)
            }
            if {(row.model_id, row.horizon) for row in metrics} != expected or len(metrics) != len(
                expected
            ):
                raise ValueError("Model card metrics must cover all evaluated models/horizons.")
            for row in metrics:
                count = (
                    len(self.series) * n_origins * (self.maximum_horizon if row.horizon == 0 else 1)
                )
                if row.n_predictions != count:
                    raise ValueError(
                        "Evaluation counts differ from complete cohort/origins/horizons."
                    )
        selected = next(
            row
            for row in self.validation_metrics
            if row.model_id == self.decision.selected_model and row.horizon == 0
        )
        seasonal = next(
            row
            for row in self.validation_metrics
            if row.model_id == "seasonal_naive" and row.horizon == 0
        )
        if (
            getattr(selected, self.experiment.primary_metric) != self.decision.validation_metric
            or getattr(seasonal, self.experiment.primary_metric)
            != self.decision.seasonal_naive_metric
        ):
            raise ValueError("Selection scores differ from recorded validation evidence.")
        return self


class DemandModelBundle(ArtifactRecord):
    artifact_id: Digest
    payload: DemandBundlePayload

    @model_validator(mode="after")
    def _integrity(self) -> DemandModelBundle:
        if self.artifact_id != content_digest(self.payload.model_dump(mode="json")):
            raise ValueError("Artifact content digest mismatch.")
        return self


def build_demand_bundle(
    actuals: pd.DataFrame,
    experiment: DemandSelectionConfig,
    origin: date,
    created_by: str,
    tuning: DemandTuningConfig | None = None,
) -> DemandModelBundle:
    """Recompute evidence, freeze selected coefficients and bind all lineage hashes.

    Input must be synthetic under the caller's data contract. No customer data
    ingestion or production approval is implemented by this local factory.
    """
    if origin < experiment.holdout.evaluation_as_of:
        raise ContractViolation("Bundle origin must be on/after holdout evaluation_as_of.")
    tuned = run_demand_tuning(actuals, tuning) if tuning is not None else None
    if tuned is not None and tuned.frozen_config != experiment:
        raise ContractViolation("Tuning evidence does not reproduce the frozen experiment.")
    result = tuned.selection if tuned is not None else run_demand_selection(actuals, experiment)
    scope = tuple(
        DemandSeriesKey(**dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)))
        for key in result.validation.predictions[DEMAND_SERIES_COLUMNS]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    request = DemandForecastRequest(
        forecast_origin=origin,
        frequency=experiment.validation.frequency,
        horizon=experiment.validation.horizon,
        series=scope,
    )
    history = available_demand_history(
        actuals,
        request,
        DemandFeatureConfig(
            seasonal_periods=experiment.validation.seasonal_periods,
            training_window_periods=experiment.validation.training_window_periods,
        ),
    )
    fitted: list[SeriesArtifact] = []
    for key, series in history.groupby(DEMAND_SERIES_COLUMNS, sort=True):
        forecaster = create_demand_forecaster(
            result.decision.selected_model, result.decision.selected_parameters
        )
        forecaster.fit(
            series,
            request.frequency,
            experiment.validation.seasonal_periods or (12 if request.frequency == "MS" else 52),
        )
        forecaster.predict(request.forecast_dates)
        fitted.append(
            SeriesArtifact(
                series=DemandSeriesKey(**dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True))),
                observations=tuple(
                    TrainingObservation(
                        date=row.date.to_pydatetime(),
                        actual_available_at=row.actual_available_at.to_pydatetime(),
                        value=float(getattr(row, DEMAND_TARGET)),
                    )
                    for row in series.itertuples(index=False)
                ),
                state=forecaster.export_state(),
            )
        )
    lineage = DemandLineage(
        source_actuals_sha256=content_digest(dataframe_records(actuals)),
        training_sha256=content_digest([item.model_dump(mode="json") for item in fitted]),
        validation_predictions_sha256=content_digest(
            dataframe_records(result.validation.predictions)
        ),
        holdout_predictions_sha256=content_digest(dataframe_records(result.holdout.predictions)),
        experiment_sha256=content_digest(experiment.model_dump(mode="json")),
        implementation_sha256=implementation_digest(),
        runtime=runtime_versions(result.decision.selected_model),
        tuning_config_sha256=content_digest(tuning.model_dump(mode="json")) if tuning else None,
        tuning_trials_sha256=content_digest(dataframe_records(tuned.trial_ranking))
        if tuned
        else None,
    )
    columns = list(EvaluationMetric.model_fields)
    payload = DemandBundlePayload(
        created_at=datetime.now(UTC),
        created_by=created_by,
        origin=origin,
        frequency=request.frequency,
        maximum_horizon=request.horizon,
        decision=result.decision,
        experiment=experiment,
        lineage=lineage,
        series=tuple(fitted),
        validation_metrics=tuple(
            EvaluationMetric.model_validate(row)
            for row in dataframe_records(result.validation.metrics[columns])
        ),
        holdout_metrics=tuple(
            EvaluationMetric.model_validate(row)
            for row in dataframe_records(result.holdout.metrics[columns])
        ),
    )
    return DemandModelBundle(
        artifact_id=content_digest(payload.model_dump(mode="json")), payload=payload
    )


def verify_demand_bundle(bundle: DemandModelBundle) -> None:
    """Revalidate content plus exact runtime and implementation compatibility."""
    bundle = DemandModelBundle.model_validate(bundle.model_dump())
    if bundle.payload.lineage.implementation_sha256 != implementation_digest():
        raise ContractViolation(
            "Bundle implementation differs; rebuild/review rather than silently upgrading."
        )
    if bundle.payload.lineage.runtime != runtime_versions(bundle.payload.decision.selected_model):
        raise ContractViolation(
            "Bundle runtime differs; restore the recorded environment or rebuild/review."
        )


def save_demand_bundle(bundle: DemandModelBundle, path: Path) -> None:
    verify_demand_bundle(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(bundle.model_dump(mode="json")) + "\n")


def load_demand_bundle(path: Path) -> DemandModelBundle:
    bundle = DemandModelBundle.model_validate_json(path.read_text(encoding="utf-8"))
    verify_demand_bundle(bundle)
    return bundle


def predict_demand_bundle(bundle: DemandModelBundle, horizon: int | None = None) -> pd.DataFrame:
    """Forecast from the frozen snapshot, without new actuals or optimization.

    Model approval is deliberately separate from forecast approval. The original
    point contract remains unapproved even for a locally approved model.
    """
    verify_demand_bundle(bundle)
    payload = bundle.payload
    horizon = payload.maximum_horizon if horizon is None else horizon
    if not 1 <= horizon <= payload.maximum_horizon:
        raise ContractViolation(
            "Forecast horizon must stay within the artifact's evaluated horizon."
        )
    request = DemandForecastRequest(
        forecast_origin=payload.origin,
        frequency=payload.frequency,
        horizon=horizon,
        series=tuple(item.series for item in payload.series),
    )
    records: list[dict[str, object]] = []
    for item in payload.series:
        forecaster = restore_demand_forecaster(item.state)
        predicted = forecaster.predict(request.forecast_dates)
        observations = {row.date: row for row in item.observations}
        for index, (target, value) in enumerate(
            zip(request.forecast_dates, predicted, strict=True)
        ):
            source = item.observations[-1]
            if payload.decision.selected_model == "seasonal_naive":
                source_date = target - (
                    (index // item.state.seasonal_periods) + 1
                ) * item.state.seasonal_periods * (demand_period_offset(payload.frequency))
                if source_date.to_pydatetime() not in observations:
                    raise ContractViolation(
                        "Frozen seasonal model lacks the requested calendar source."
                    )
                source = observations[source_date.to_pydatetime()]
            available = (
                source.actual_available_at
                if payload.decision.selected_model in {"naive", "seasonal_naive"}
                else max(row.actual_available_at for row in item.observations)
            )
            records.append(
                {
                    **item.series.model_dump(),
                    "date": target,
                    "frequency": payload.frequency,
                    "forecast_origin": request.origin,
                    "horizon": index + 1,
                    "model_id": payload.decision.selected_model,
                    "predicted_demand_units": float(value),
                    "source_date": source.date,
                    "source_available_at": available,
                    "training_start": item.observations[0].date,
                    "training_end": item.observations[-1].date,
                    "n_training_periods": len(item.observations),
                    "schema_version": FORECAST_SCHEMA_VERSION,
                    "forecast_kind": demand_model_capability(
                        payload.decision.selected_model
                    ).family,
                    "approval_status": "unapproved",
                }
            )
    frame = pd.DataFrame(records, columns=FORECAST_COLUMNS)
    for column in [
        "date",
        "forecast_origin",
        "source_date",
        "source_available_at",
        "training_start",
        "training_end",
    ]:
        frame[column] = frame[column].astype("datetime64[ns]")
    return validate_demand_forecasts(frame, request)


MODEL_LIMITATIONS = (
    "Synthetic-only local evidence; not customer accuracy or production approval.",
    "Univariate nonnegative point forecasts; no covariates, intervals or reconciliation.",
    "Validation selects algorithms/parameters; holdout is reported without reselection.",
    "Holdout refits at successive origins; this artifact freezes one later origin.",
    "Exact runtime/source compatibility required; no pickle, dynamic imports or optimizer on load.",
    "Local model approval never approves official forecasts or cloud deployment.",
)


def demand_model_card(bundle: DemandModelBundle) -> dict[str, object]:
    """Immutable machine-readable model card; registry events own mutable status."""
    verify_demand_bundle(bundle)
    payload = bundle.payload
    return {
        "schema_version": "demand-model-card-v1",
        "artifact_id": bundle.artifact_id,
        "created_at": payload.created_at.isoformat(),
        "created_by": payload.created_by,
        "target": payload.target,
        "data_classification": payload.data_classification,
        "model": asdict(demand_model_capability(payload.decision.selected_model)),
        "parameters": payload.decision.selected_parameters.model_dump(mode="json"),
        "frequency": payload.frequency,
        "forecast_origin": payload.origin.isoformat(),
        "maximum_horizon": payload.maximum_horizon,
        "series_training": [
            {
                **item.series.model_dump(),
                "training_start": item.observations[0].date.isoformat(),
                "training_end": item.observations[-1].date.isoformat(),
                "n_periods": len(item.observations),
                "last_actual_available_at": max(
                    row.actual_available_at for row in item.observations
                ).isoformat(),
            }
            for item in payload.series
        ],
        "selection": payload.decision.model_dump(mode="json"),
        "evaluation_design": payload.experiment.model_dump(mode="json"),
        "validation_metrics": [row.model_dump(mode="json") for row in payload.validation_metrics],
        "holdout_metrics": [row.model_dump(mode="json") for row in payload.holdout_metrics],
        "lineage": payload.lineage.model_dump(mode="json"),
        "limitations": list(MODEL_LIMITATIONS),
        "approval_authority": "Local registry audit events only; this card never grants approval.",
    }


def render_demand_model_card(bundle: DemandModelBundle) -> str:
    card = demand_model_card(bundle)
    return (
        "# Demand forecasting model card\n\n"
        f"Artifact: `{bundle.artifact_id}`\n\n"
        "## Scope and limitations\n\n"
        + "\n".join(f"- {item}" for item in MODEL_LIMITATIONS)
        + "\n\n## Versioned model, evaluation and lineage evidence\n\n```json\n"
        + json.dumps(card, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n```\n"
    )
