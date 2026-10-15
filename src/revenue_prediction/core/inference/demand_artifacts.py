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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

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
from revenue_prediction.core.evaluation.demand_uncertainty import (
    DemandIntervalCalibration,
    DemandUncertaintyConfig,
    apply_demand_intervals,
    run_demand_uncertainty,
)
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


class IntervalEvaluationMetric(ArtifactRecord):
    model_id: str
    nominal_coverage: float = Field(gt=0, lt=1)
    horizon: int = Field(ge=0, strict=True)
    n_predictions: int = Field(gt=0, strict=True)
    empirical_coverage: float = Field(ge=0, le=1)
    mean_interval_width: float = Field(ge=0)
    mean_interval_score: float = Field(ge=0)
    lower_pinball_loss: float = Field(ge=0)
    upper_pinball_loss: float = Field(ge=0)
    coverage_gap: float
    lower_quantile_level: float = Field(gt=0, lt=1)
    upper_quantile_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _metric_identity(self) -> IntervalEvaluationMetric:
        alpha = 1 - self.nominal_coverage
        if (
            self.coverage_gap != self.empirical_coverage - self.nominal_coverage
            or self.lower_quantile_level != alpha / 2
            or self.upper_quantile_level != 1 - alpha / 2
        ):
            raise ValueError("Interval metric coverage/tail levels differ from nominal coverage.")
        return self


class SeriesIntervalEvaluationMetric(IntervalEvaluationMetric):
    sku_id: str
    channel_id: str
    geography_id: str

    @property
    def series_key(self) -> tuple[str, str, str]:
        return DemandSeriesKey(
            sku_id=self.sku_id, channel_id=self.channel_id, geography_id=self.geography_id
        ).key


class DemandIntervalEvidence(ArtifactRecord):
    experiment: DemandUncertaintyConfig
    calibration: DemandIntervalCalibration
    calibration_predictions_sha256: Digest
    holdout_intervals_sha256: Digest
    calibration_sha256: Digest
    experiment_sha256: Digest
    holdout_metrics: tuple[IntervalEvaluationMetric, ...] = Field(min_length=1)
    series_holdout_metrics: tuple[SeriesIntervalEvaluationMetric, ...] = Field(min_length=1)


class DemandBundlePayload(ArtifactRecord):
    schema_version: Literal["demand-bundle-v1", "demand-bundle-v2"] = "demand-bundle-v1"
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
    uncertainty: DemandIntervalEvidence | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result = handler(self)
        if self.uncertainty is None:
            result.pop("uncertainty", None)
        return result

    @model_validator(mode="after")
    def _consistent_bundle(self) -> DemandBundlePayload:
        origin = pd.Timestamp(self.origin)
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != UTC.utcoffset(
            self.created_at
        ):
            raise ValueError("Artifact creation timestamp must be UTC-aware.")
        if self.origin < self.experiment.holdout.evaluation_as_of:
            raise ValueError("Artifact origin must follow all evaluation evidence availability.")
        uncertainty = self.uncertainty
        if (self.schema_version == "demand-bundle-v2") != (uncertainty is not None):
            raise ValueError("Only demand-bundle-v2 must contain interval evidence.")
        selected_at = (
            uncertainty.experiment.calibration.first_origin
            if uncertainty is not None
            else self.experiment.holdout.first_origin
        )
        if self.decision.selected_at != selected_at:
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
        if uncertainty is not None:
            self._validate_uncertainty(uncertainty)
        return self

    def _validate_uncertainty(self, evidence: DemandIntervalEvidence) -> None:
        config, calibration = evidence.experiment, evidence.calibration
        if (
            config.selection != self.experiment
            or calibration.frequency != self.frequency
            or calibration.nominal_coverage != config.nominal_coverage
            or calibration.calibration_as_of != config.calibration.evaluation_as_of
            or content_digest(calibration.model_dump(mode="json")) != evidence.calibration_sha256
            or content_digest(config.model_dump(mode="json")) != evidence.experiment_sha256
            or self.lineage.tuning_config_sha256 is not None
            or self.lineage.tuning_trials_sha256 is not None
        ):
            raise ValueError("Interval experiment/calibration lineage differs from point artifact.")
        models = {"naive", "seasonal_naive", self.decision.selected_model}
        keys = {item.series.key for item in self.series}
        expected_cells = {
            (model, *key, horizon)
            for model in models
            for key in keys
            for horizon in range(1, self.maximum_horizon + 1)
        }
        if {cell.key for cell in calibration.cells} != expected_cells or any(
            cell.n_predictions != config.calibration.n_origins for cell in calibration.cells
        ):
            raise ValueError(
                "Calibration differs from the complete evaluated model/cohort/horizon scope."
            )
        last_origin = pd.Timestamp(config.calibration.first_origin) + (
            (config.calibration.n_origins - 1)
            * config.calibration.step_periods
            * demand_period_offset(self.frequency)
        )
        last_target = (
            DemandForecastRequest(
                forecast_origin=last_origin.date(),
                frequency=self.frequency,
                horizon=self.maximum_horizon,
            )
            .forecast_dates[-1]
            .date()
        )
        if calibration.last_target_date != last_target:
            raise ValueError("Calibration final target differs from experiment.")
        for rows, per_series in [
            (evidence.holdout_metrics, False),
            (evidence.series_holdout_metrics, True),
        ]:
            expected = {
                (model, horizon, key)
                for model in models
                for horizon in range(self.maximum_horizon + 1)
                for key in (keys if per_series else {None})
            }
            observed = {
                (
                    row.model_id,
                    row.horizon,
                    row.series_key if isinstance(row, SeriesIntervalEvaluationMetric) else None,
                )
                for row in rows
            }
            if observed != expected or len(rows) != len(expected):
                raise ValueError("Interval metrics must cover the complete evaluated scope.")
            for row in rows:
                count = (
                    config.selection.holdout.n_origins
                    * (self.maximum_horizon if row.horizon == 0 else 1)
                    * (1 if per_series else len(keys))
                )
                if row.n_predictions != count or row.nominal_coverage != config.nominal_coverage:
                    raise ValueError("Interval metric counts/coverage differ from experiment.")


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
    *,
    uncertainty: DemandUncertaintyConfig | None = None,
) -> DemandModelBundle:
    """Recompute evidence, freeze selected coefficients and bind all lineage hashes.

    Input must be synthetic under the caller's data contract. No customer data
    ingestion or production approval is implemented by this local factory.
    """
    if origin < experiment.holdout.evaluation_as_of:
        raise ContractViolation("Bundle origin must be on/after holdout evaluation_as_of.")
    if uncertainty is not None and (uncertainty.selection != experiment or tuning is not None):
        raise ContractViolation(
            "Interval packaging requires the matching selection and no tuning grid."
        )
    interval_result = (
        run_demand_uncertainty(actuals, uncertainty) if uncertainty is not None else None
    )
    tuned = run_demand_tuning(actuals, tuning) if tuning is not None else None
    if tuned is not None and tuned.frozen_config != experiment:
        raise ContractViolation("Tuning evidence does not reproduce the frozen experiment.")
    result = (
        interval_result
        if interval_result is not None
        else tuned.selection
        if tuned is not None
        else run_demand_selection(actuals, experiment)
    )
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
        schema_version="demand-bundle-v2" if interval_result is not None else "demand-bundle-v1",
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
        uncertainty=DemandIntervalEvidence(
            experiment=uncertainty,
            calibration=interval_result.calibration,
            calibration_predictions_sha256=content_digest(
                dataframe_records(interval_result.calibration_backtest.predictions)
            ),
            holdout_intervals_sha256=content_digest(dataframe_records(interval_result.predictions)),
            calibration_sha256=content_digest(interval_result.calibration.model_dump(mode="json")),
            experiment_sha256=content_digest(uncertainty.model_dump(mode="json")),
            holdout_metrics=tuple(
                IntervalEvaluationMetric.model_validate(row)
                for row in dataframe_records(interval_result.metrics)
            ),
            series_holdout_metrics=tuple(
                SeriesIntervalEvaluationMetric.model_validate(row)
                for row in dataframe_records(interval_result.series_metrics)
            ),
        )
        if interval_result is not None and uncertainty is not None
        else None,
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
    if type(horizon) is not int or not 1 <= horizon <= payload.maximum_horizon:
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


def predict_demand_interval_bundle(
    bundle: DemandModelBundle, horizon: int | None = None
) -> pd.DataFrame:
    """Restore the selected model and only its bound calibration cells; never refit."""
    points = predict_demand_bundle(bundle, horizon)
    evidence = bundle.payload.uncertainty
    if evidence is None:
        raise ContractViolation("Point-only artifact has no approved interval capability.")
    maximum = int(points["horizon"].max())
    calibration = DemandIntervalCalibration.model_validate(
        {
            **evidence.calibration.model_dump(),
            "cells": [
                cell.model_dump()
                for cell in evidence.calibration.cells
                if cell.model_id == bundle.payload.decision.selected_model
                and cell.horizon <= maximum
            ],
        }
    )
    return apply_demand_intervals(points, calibration)


MODEL_LIMITATIONS = (
    "Synthetic-only local evidence; not customer accuracy or production approval.",
    "Univariate nonnegative point forecasts; no covariates, intervals or reconciliation.",
    "Validation selects algorithms/parameters; holdout is reported without reselection.",
    "Holdout refits at successive origins; this artifact freezes one later origin.",
    "Exact runtime/source compatibility required; no pickle, dynamic imports or optimizer on load.",
    "Local model approval never approves official forecasts or cloud deployment.",
)


def _model_limitations(payload: DemandBundlePayload) -> list[str]:
    if payload.uncertainty is None:
        return list(MODEL_LIMITATIONS)
    return [item for item in MODEL_LIMITATIONS if "no covariates, intervals" not in item] + [
        "Univariate residual intervals; no learned quantiles, covariates or reconciliation.",
        "Dependent rolling residuals do not guarantee time-series coverage.",
        "Frozen calibration and training policy; no adaptive recalibration or new origins.",
    ]


def demand_model_card(bundle: DemandModelBundle) -> dict[str, object]:
    """Immutable machine-readable model card; registry events own mutable status."""
    verify_demand_bundle(bundle)
    payload = bundle.payload
    card: dict[str, object] = {
        "schema_version": "demand-model-card-v2" if payload.uncertainty else "demand-model-card-v1",
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
        "limitations": _model_limitations(payload),
        "approval_authority": "Local registry audit events only; this card never grants approval.",
    }
    if payload.uncertainty is not None:
        card["uncertainty"] = payload.uncertainty.model_dump(mode="json")
    return card


def render_demand_model_card(bundle: DemandModelBundle) -> str:
    card = demand_model_card(bundle)
    return (
        "# Demand forecasting model card\n\n"
        f"Artifact: `{bundle.artifact_id}`\n\n"
        "## Scope and limitations\n\n"
        + "\n".join(f"- {item}" for item in _model_limitations(bundle.payload))
        + "\n\n## Versioned model, evaluation and lineage evidence\n\n```json\n"
        + json.dumps(card, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n```\n"
    )
