"""Discoverable univariate time-series forecasters, separate from regression."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
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
    ENSEMBLE_FORECAST_SCHEMA_VERSION,
    FORECAST_COLUMNS,
    FORECAST_SCHEMA_VERSION,
    DemandForecastRequest,
    DemandFrequency,
    demand_period_offset,
    validate_demand_forecasts,
)
from revenue_prediction.core.data.demand_schema import DEMAND_SERIES_COLUMNS, DEMAND_TARGET
from revenue_prediction.core.features.demand import DemandFeatureConfig, available_demand_history
from revenue_prediction.core.models.demand_baselines import forecast_demand_baselines


class EnsembleWeight(BaseModel):
    """One declared member, its out-of-sample error and frozen portfolio weight."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_id: str
    validation_error: float = Field(ge=0)
    weight: float = Field(ge=0, le=1)


def inverse_error_weights(errors: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize inverse errors stably; exact-zero members share all mass."""
    if len(errors) < 2 or any(not math.isfinite(error) or error < 0 for error in errors):
        raise ContractViolation("Ensemble needs at least two finite nonnegative validation errors.")
    zeros = errors.count(0.0)
    if zeros:
        return tuple(1.0 / zeros if error == 0 else 0.0 for error in errors)
    minimum = min(errors)
    scaled = tuple(minimum / error for error in errors)
    total = math.fsum(scaled)
    return tuple(value / total for value in scaled)


class TimeSeriesModelConfig(BaseModel):
    """Fixed, predeclared smoothing parameters; no holdout-based tuning."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    alpha: float = Field(default=0.2, gt=0, le=1)
    beta: float = Field(default=0.1, gt=0, le=1)
    damping: float = Field(default=0.9, gt=0, le=1)
    average_periods: int = Field(default=3, ge=1, strict=True)
    average_seasons: int = Field(default=2, ge=1, strict=True)
    ets_trend: Literal["add"] | None = "add"
    ets_seasonal: Literal["add"] | None = "add"
    ets_damped: bool = True
    arima_order: tuple[
        Annotated[int, Field(strict=True, ge=0, le=5)],
        Annotated[int, Field(strict=True, ge=0, le=5)],
        Annotated[int, Field(strict=True, ge=0, le=5)],
    ] = (1, 1, 0)
    sarima_order: tuple[
        Annotated[int, Field(strict=True, ge=0, le=5)],
        Annotated[int, Field(strict=True, ge=0, le=5)],
        Annotated[int, Field(strict=True, ge=0, le=5)],
    ] = (1, 0, 0)
    statistical_maxiter: int = Field(default=200, ge=1, le=2000, strict=True)
    ensemble_weights: tuple[EnsembleWeight, ...] = ()

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result = handler(self)
        if not self.ensemble_weights:
            result.pop("ensemble_weights", None)
        return result

    @model_validator(mode="after")
    def _statistical_parameters(self) -> TimeSeriesModelConfig:
        if self.ets_damped and self.ets_trend is None:
            raise ValueError("Damped ETS requires an additive trend.")
        for order in (self.arima_order, self.sarima_order):
            if any(type(value) is not int or value < 0 or value > 5 for value in order):
                raise ValueError("ARIMA orders must contain three integers between 0 and 5.")
        if self.ensemble_weights:
            names = [item.model_id for item in self.ensemble_weights]
            if names != sorted(set(names)) or "weighted_ensemble" in names:
                raise ValueError("Ensemble components must be unique, sorted and not nested.")
            for name in names:
                demand_model_capability(name)
            expected = inverse_error_weights(
                tuple(item.validation_error for item in self.ensemble_weights)
            )
            if tuple(item.weight for item in self.ensemble_weights) != expected:
                raise ValueError("Ensemble weights differ from the declared inverse-error rule.")
        return self


@dataclass(frozen=True)
class ForecasterCapability:
    """Truthful executable capabilities and history requirements."""

    model_id: str
    family: Literal["baseline", "statistical", "ensemble"]
    description: str
    minimum_periods: int
    requires_contiguous_history: bool
    frequencies: tuple[DemandFrequency, ...] = ("MS", "W-MON")
    provider: str = "original_numpy"
    version: str = "1"
    license: str = "MIT"
    required_extras: tuple[str, ...] = ()
    supported_covariates: tuple[str, ...] = ()
    supports_probabilistic_output: bool = False
    supports_hierarchical_reconciliation: bool = False
    local_execution: bool = True
    azure_execution_validated: bool = False
    supports_json_state: bool = True
    requires_validation_weights: bool = False


_CAPABILITIES = (
    ForecasterCapability("naive", "baseline", "Repeat the last available observation.", 1, False),
    ForecasterCapability("seasonal_naive", "baseline", "Repeat a prior calendar season.", 1, False),
    ForecasterCapability(
        "moving_average", "baseline", "Repeat a trailing calendar-window mean.", 1, True
    ),
    ForecasterCapability(
        "seasonal_moving_average",
        "baseline",
        "Average matching positions in trailing seasons.",
        1,
        True,
    ),
    ForecasterCapability(
        "drift", "baseline", "Extrapolate the first-to-last average change.", 2, True
    ),
    ForecasterCapability(
        "ses", "statistical", "Simple exponential smoothing: level, no trend or season.", 2, True
    ),
    ForecasterCapability(
        "holt_damped", "statistical", "Additive damped level/trend smoothing, no season.", 3, True
    ),
    ForecasterCapability(
        "croston", "statistical", "Smooth nonzero sizes and arrival intervals.", 2, True
    ),
    ForecasterCapability(
        "sba", "statistical", "Bias-corrected Croston intermittent forecast.", 2, True
    ),
    ForecasterCapability(
        "tsb", "statistical", "Smooth occurrence probability and positive size.", 2, True
    ),
    ForecasterCapability(
        "ets",
        "statistical",
        "Optimized additive Holt-Winters level/trend/seasonality.",
        4,
        True,
        provider="statsmodels",
        license="BSD-3-Clause",
        required_extras=("forecasting",),
    ),
    ForecasterCapability(
        "arima",
        "statistical",
        "Nonseasonal integrated autoregressive moving average.",
        8,
        True,
        provider="statsmodels",
        license="BSD-3-Clause",
        required_extras=("forecasting",),
    ),
    ForecasterCapability(
        "sarima",
        "statistical",
        "Seasonal integrated autoregressive moving average.",
        8,
        True,
        provider="statsmodels",
        license="BSD-3-Clause",
        required_extras=("forecasting",),
    ),
    ForecasterCapability(
        "weighted_ensemble",
        "ensemble",
        "Frozen nonnegative inverse-validation-error blend; no component fallback.",
        1,
        False,
        requires_validation_weights=True,
    ),
)

# Preserve the original offline comparison and avoid implicit optional dependencies.
DEFAULT_DEMAND_MODELS = (
    "naive",
    "seasonal_naive",
    "moving_average",
    "seasonal_moving_average",
    "drift",
    "ses",
    "holt_damped",
)


def demand_model_capabilities() -> tuple[ForecasterCapability, ...]:
    """Return immutable metadata for implemented models, not a research wish list."""
    return _CAPABILITIES


def demand_model_capability(model_id: str) -> ForecasterCapability:
    """Resolve an implemented model or raise an actionable configuration error."""
    for capability in _CAPABILITIES:
        if capability.model_id == model_id:
            return capability
    raise ValueError(
        f"Unknown demand forecaster {model_id!r}; inspect demand-models for supported IDs."
    )


class HoltWintersState(BaseModel):
    """Only numeric fitted coefficients/initialization, never executable objects."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    smoothing_level: float = Field(ge=0, le=1)
    smoothing_trend: float | None = Field(default=None, ge=0, le=1)
    smoothing_seasonal: float | None = Field(default=None, ge=0, le=1)
    damping_trend: float | None = Field(default=None, gt=0, le=1)
    initial_level: float
    initial_trend: float | None = None
    initial_seasons: tuple[float, ...] = ()


class DemandForecasterState(BaseModel):
    """Versioned JSON state; deterministic replay/filtering never reoptimizes."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["demand-state-v1", "demand-state-v2"] = "demand-state-v1"
    model_id: str
    parameters: TimeSeriesModelConfig
    frequency: DemandFrequency
    seasonal_periods: int = Field(ge=1, strict=True)
    dates: tuple[datetime, ...] = Field(min_length=1)
    values: tuple[Annotated[float, Field(ge=0)], ...] = Field(min_length=1)
    holt_winters: HoltWintersState | None = None
    arima_parameters: tuple[float, ...] | None = Field(default=None, min_length=1)
    components: tuple[DemandForecasterState, ...] = ()

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result = handler(self)
        if not self.components:
            result.pop("components", None)
        return result

    @model_validator(mode="after")
    def _consistent_state(self) -> DemandForecasterState:
        demand_model_capability(self.model_id)
        ensemble = self.model_id == "weighted_ensemble"
        if (self.schema_version == "demand-state-v2") != ensemble or bool(
            self.components
        ) != ensemble:
            raise ValueError("Only ensemble v2 state must contain component states.")
        if ensemble:
            weights = self.parameters.ensemble_weights
            parameters = self.parameters.model_dump(exclude={"ensemble_weights"})
            if not weights or tuple(row.model_id for row in self.components) != tuple(
                row.model_id for row in weights
            ):
                raise ValueError("Ensemble state differs from all declared components.")
            for component in self.components:
                if (
                    component.dates != self.dates
                    or component.values != self.values
                    or component.frequency != self.frequency
                    or component.seasonal_periods != self.seasonal_periods
                    or component.parameters.model_dump() != parameters
                    or component.model_id == "weighted_ensemble"
                ):
                    raise ValueError(
                        "Component state must match the complete shared training/configuration."
                    )
        if len(self.dates) != len(self.values):
            raise ValueError("State dates and values must have identical lengths.")
        if any(
            value.tzinfo is not None or value.time() != datetime.min.time() for value in self.dates
        ):
            raise ValueError("State dates must be timezone-naive midnight period starts.")
        if (self.model_id == "ets") != (self.holt_winters is not None):
            raise ValueError("Only ETS requires Holt-Winters coefficients.")
        if (self.model_id in {"arima", "sarima"}) != (self.arima_parameters is not None):
            raise ValueError("Only ARIMA/SARIMA require fitted ARIMA parameters.")
        if self.holt_winters is not None:
            state = self.holt_winters
            trend = self.parameters.ets_trend is not None
            if trend != (state.smoothing_trend is not None) or trend != (
                state.initial_trend is not None
            ):
                raise ValueError("ETS trend state differs from its configuration.")
            seasonal = self.parameters.ets_seasonal is not None
            if seasonal != (state.smoothing_seasonal is not None) or len(state.initial_seasons) != (
                self.seasonal_periods if seasonal else 0
            ):
                raise ValueError("ETS seasonal state differs from its configuration.")
            if self.parameters.ets_damped != (state.damping_trend is not None):
                raise ValueError("ETS damping state differs from its configuration.")
        validate_demand_model_history(
            self.model_id,
            pd.DataFrame({"date": pd.to_datetime(self.dates), DEMAND_TARGET: self.values}),
            self.frequency,
            self.seasonal_periods,
            self.parameters,
        )
        return self


class Forecaster(Protocol):
    """Minimal fitted time-series interface; unsupported capabilities stay explicit."""

    capability: ForecasterCapability

    def fit(self, history: pd.DataFrame, frequency: DemandFrequency, seasonal_periods: int) -> None:
        """Fit one origin-available series, never a randomly split tabular target."""
        ...

    def predict(self, dates: pd.DatetimeIndex) -> NDArray[np.float64]:
        """Forecast complete future dates without observing future targets."""
        ...

    def export_state(self) -> DemandForecasterState:
        """Export safe data-only state for restoration without coefficient optimization."""
        ...


class _LocalForecaster:
    def __init__(self, capability: ForecasterCapability, config: TimeSeriesModelConfig) -> None:
        self.capability = capability
        self.config = config
        self._history: pd.DataFrame | None = None

    def export_state(self) -> DemandForecasterState:
        if self._history is None:
            raise RuntimeError("Fit before exporting forecaster state.")
        return DemandForecasterState(
            model_id=self.capability.model_id,
            parameters=self.config,
            frequency=self._frequency,
            seasonal_periods=self._season,
            dates=tuple(pd.Timestamp(value).to_pydatetime() for value in self._history["date"]),
            values=tuple(float(value) for value in self._values),
        )

    def fit(self, history: pd.DataFrame, frequency: DemandFrequency, seasonal_periods: int) -> None:
        self._history = None
        validate_demand_model_history(
            self.capability.model_id, history, frequency, seasonal_periods, self.config
        )
        values = history[DEMAND_TARGET].to_numpy(dtype=float)
        self._history = history.copy(deep=True)
        self._frequency = frequency
        self._season = seasonal_periods
        self._values = values.copy()
        self._level = float(values[0])
        self._trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
        self._fit_states(values)

    def _fit_states(self, values: NDArray[np.float64]) -> None:
        if self.capability.model_id in {"ses", "holt_damped"}:
            for value in values[1:]:
                previous = self._level
                projected = previous
                if self.capability.model_id == "holt_damped":
                    projected += self.config.damping * self._trend
                self._level = self.config.alpha * value + (1 - self.config.alpha) * projected
                if self.capability.model_id == "holt_damped":
                    self._trend = self.config.beta * (self._level - previous) + (
                        (1 - self.config.beta) * self.config.damping * self._trend
                    )
        if self.capability.model_id in {"croston", "sba", "tsb"}:
            positives = np.flatnonzero(values > 0)
            if len(positives) == 0:
                self._intermittent_prediction = 0.0
            else:
                first = int(positives[0])
                size = float(values[first])
                interval = float(first + 1)
                probability = 1.0 / interval
                elapsed = 1
                for value in values[first + 1 :]:
                    occurs = value > 0
                    probability += self.config.beta * (float(occurs) - probability)
                    if occurs:
                        size += self.config.alpha * (value - size)
                        interval += self.config.alpha * (elapsed - interval)
                        elapsed = 1
                    else:
                        elapsed += 1
                self._intermittent_prediction = (
                    probability * size
                    if self.capability.model_id == "tsb"
                    else size
                    / interval
                    * (1 - self.config.alpha / 2 if self.capability.model_id == "sba" else 1)
                )

    def _calendar_leads(self, dates: pd.DatetimeIndex) -> NDArray[np.intp]:
        if self._history is None:
            raise RuntimeError("Fit the forecaster on origin-available history before predicting.")
        if len(dates) == 0:
            raise ValueError("Supply at least one future forecast date.")
        last_date = self._history["date"].iloc[-1]
        calendar = pd.date_range(start=last_date, end=dates[-1], freq=self._frequency)
        leads = calendar.get_indexer(dates)
        if np.any(leads <= 0) or not dates.equals(
            pd.date_range(start=dates[0], periods=len(dates), freq=self._frequency)
        ):
            raise ValueError(
                "Forecast dates must be unique, aligned, chronological and after training."
            )
        return leads

    def predict(self, dates: pd.DatetimeIndex) -> NDArray[np.float64]:
        leads = self._calendar_leads(dates)
        model_id = self.capability.model_id
        if model_id in {"croston", "sba", "tsb"}:
            prediction = np.full(len(dates), self._intermittent_prediction)
        elif model_id == "naive":
            prediction = np.full(len(dates), self._values[-1])
        elif model_id == "seasonal_naive":
            sources = [
                target
                - ((index // self._season) + 1)
                * self._season
                * demand_period_offset(self._frequency)
                for index, target in enumerate(dates)
            ]
            if self._history is None:
                raise RuntimeError("Fit before forecasting seasonal sources.")
            values = self._history.set_index("date")[DEMAND_TARGET].reindex(sources)
            if np.any(values.isna().to_numpy()):
                raise ContractViolation(
                    "seasonal_naive lacks available calendar sources; supply seasonal history."
                )
            prediction = values.to_numpy(dtype=float)
        elif model_id == "moving_average":
            prediction = np.full(len(dates), np.mean(self._values[-self.config.average_periods :]))
        elif model_id == "seasonal_moving_average":
            tail = self._values[-self._season * self.config.average_seasons :]
            cycle = tail.reshape(self.config.average_seasons, self._season).mean(axis=0)
            prediction = cycle[(leads - 1) % self._season]
        elif model_id == "drift":
            slope = (self._values[-1] - self._values[0]) / (len(self._values) - 1)
            prediction = self._values[-1] + leads * slope
        elif model_id == "ses":
            prediction = np.full(len(dates), self._level)
        elif model_id == "holt_damped":
            phi = self.config.damping
            damped_leads = leads if phi == 1 else phi * (1 - phi**leads) / (1 - phi)
            prediction = self._level + damped_leads * self._trend
        else:
            raise ValueError(f"No prediction implementation registered for {model_id!r}.")
        if not np.isfinite(prediction).all():
            raise ContractViolation(
                "Forecaster produced nonfinite predictions; check history magnitude/parameters."
            )
        return np.maximum(prediction, 0).astype(float)


def validate_demand_model_history(
    model_id: str,
    history: pd.DataFrame,
    frequency: DemandFrequency,
    seasonal_periods: int,
    config: TimeSeriesModelConfig,
) -> None:
    """Check structural eligibility without fitting or inspecting future actuals."""
    capability = demand_model_capability(model_id)
    if frequency not in capability.frequencies or seasonal_periods < 1:
        raise ValueError("Use supported MS/W-MON frequency and a positive seasonal length.")
    minimum = capability.minimum_periods
    if model_id == "moving_average":
        minimum = max(minimum, config.average_periods)
    if model_id == "seasonal_moving_average":
        minimum = max(minimum, seasonal_periods * config.average_seasons)
    if model_id == "ets" and config.ets_seasonal is not None:
        if seasonal_periods < 2:
            raise ContractViolation("Seasonal ETS requires a seasonal length of at least two.")
        minimum = max(minimum, 2 * seasonal_periods)
    if model_id in {"arima", "sarima"}:
        p, d, q = config.arima_order
        minimum = max(minimum, 2 * (p + d + q + 1))
        if model_id == "sarima":
            if seasonal_periods < 2:
                raise ContractViolation("SARIMA requires a seasonal length of at least two.")
            P, D, Q = config.sarima_order
            minimum = max(
                minimum, 2 * seasonal_periods, 2 * (p + d + q + (P + D + Q) * seasonal_periods + 1)
            )
    if len(history) < minimum:
        raise ContractViolation(
            f"{model_id} needs at least {minimum} available periods per series; "
            "enlarge history/window or explicitly choose another forecaster."
        )
    dates = pd.DatetimeIndex(history["date"])
    aligned = dates.day == 1 if frequency == "MS" else dates.dayofweek == 0
    if dates.tz is not None or not np.all(aligned) or not dates.equals(dates.normalize()):
        raise ContractViolation("Forecaster dates must be aligned timezone-naive period starts.")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ContractViolation("Forecaster history must have unique chronological period dates.")
    if capability.requires_contiguous_history and not dates.equals(
        pd.date_range(start=dates[0], periods=len(dates), freq=frequency)
    ):
        raise ContractViolation(
            f"{model_id} requires contiguous origin-available history; "
            "resolve arrival gaps explicitly rather than compressing the calendar."
        )
    values = history[DEMAND_TARGET].to_numpy(dtype=float)
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ContractViolation("Forecaster history requires finite nonnegative demand.")


class _EnsembleForecaster(_LocalForecaster):
    def __init__(self, capability: ForecasterCapability, config: TimeSeriesModelConfig) -> None:
        super().__init__(capability, config)
        if not config.ensemble_weights:
            raise ContractViolation("weighted_ensemble requires frozen validation-learned weights.")
        self._components: tuple[Forecaster, ...] = ()

    def fit(self, history: pd.DataFrame, frequency: DemandFrequency, seasonal_periods: int) -> None:
        self._components = ()
        super().fit(history, frequency, seasonal_periods)
        parameters = TimeSeriesModelConfig.model_validate(
            self.config.model_dump(exclude={"ensemble_weights"})
        )
        components: list[Forecaster] = []
        for member in self.config.ensemble_weights:
            component = create_demand_forecaster(member.model_id, parameters)
            component.fit(history, frequency, seasonal_periods)
            components.append(component)
        self._components = tuple(components)

    def predict(self, dates: pd.DatetimeIndex) -> NDArray[np.float64]:
        self._calendar_leads(dates)
        if len(self._components) != len(self.config.ensemble_weights):
            raise ContractViolation("Ensemble requires every fitted component; no fallback.")
        values = np.stack([component.predict(dates) for component in self._components])
        weights = np.asarray([item.weight for item in self.config.ensemble_weights])
        prediction = weights @ values
        if not np.isfinite(prediction).all():
            raise ContractViolation("Ensemble produced nonfinite predictions; no fallback.")
        return prediction.astype(float)

    def export_state(self) -> DemandForecasterState:
        if self._history is None:
            raise RuntimeError("Fit before exporting forecaster state.")
        return DemandForecasterState(
            schema_version="demand-state-v2",
            model_id=self.capability.model_id,
            parameters=self.config,
            frequency=self._frequency,
            seasonal_periods=self._season,
            dates=tuple(pd.Timestamp(value).to_pydatetime() for value in self._history["date"]),
            values=tuple(float(value) for value in self._values),
            components=tuple(component.export_state() for component in self._components),
        )


class _StatisticalForecaster(_LocalForecaster):
    """Guarded statsmodels adapters; fit failures never substitute another model."""

    def fit(self, history: pd.DataFrame, frequency: DemandFrequency, seasonal_periods: int) -> None:
        super().fit(history, frequency, seasonal_periods)
        try:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            self._history = None
            raise ImportError(
                "Install optional statistical adapters with uv sync --extra forecasting."
            ) from exc
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                if self.capability.model_id == "ets":
                    self._fitted = ExponentialSmoothing(
                        self._values,
                        trend=self.config.ets_trend,
                        damped_trend=self.config.ets_damped,
                        seasonal=self.config.ets_seasonal,
                        seasonal_periods=seasonal_periods if self.config.ets_seasonal else None,
                        initialization_method="estimated",
                    ).fit(
                        optimized=True,
                        minimize_kwargs={"options": {"maxiter": self.config.statistical_maxiter}},
                    )
                else:
                    seasonal_order = (
                        (*self.config.sarima_order, seasonal_periods)
                        if self.capability.model_id == "sarima"
                        else (0, 0, 0, 0)
                    )
                    self._fitted = SARIMAX(
                        self._values,
                        order=self.config.arima_order,
                        seasonal_order=seasonal_order,
                        trend="n",
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    ).fit(disp=False, maxiter=self.config.statistical_maxiter)
        except (ConvergenceWarning, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            self._history = None
            raise ContractViolation(f"{self.capability.model_id} fit failed: {exc}") from exc

    def predict(self, dates: pd.DatetimeIndex) -> NDArray[np.float64]:
        leads = self._calendar_leads(dates)
        prediction = np.asarray(self._fitted.forecast(int(leads[-1])), dtype=float)[leads - 1]
        if not np.isfinite(prediction).all():
            raise ContractViolation("Statistical forecaster produced nonfinite predictions.")
        return np.maximum(prediction, 0).astype(float)

    def export_state(self) -> DemandForecasterState:
        if self._history is None:
            raise RuntimeError("Fit before exporting forecaster state.")
        state: dict[str, object] = {
            "model_id": self.capability.model_id,
            "parameters": self.config,
            "frequency": self._frequency,
            "seasonal_periods": self._season,
            "dates": tuple(self._history["date"]),
            "values": tuple(float(value) for value in self._values),
        }
        if self.capability.model_id == "ets":
            fitted = self._fitted.params
            state["holt_winters"] = HoltWintersState(
                smoothing_level=float(fitted["smoothing_level"]),
                smoothing_trend=float(fitted["smoothing_trend"]) if self.config.ets_trend else None,
                smoothing_seasonal=float(fitted["smoothing_seasonal"])
                if self.config.ets_seasonal
                else None,
                damping_trend=float(fitted["damping_trend"]) if self.config.ets_damped else None,
                initial_level=float(fitted["initial_level"]),
                initial_trend=float(fitted["initial_trend"]) if self.config.ets_trend else None,
                initial_seasons=tuple(float(value) for value in fitted["initial_seasons"])
                if self.config.ets_seasonal
                else (),
            )
        else:
            state["arima_parameters"] = tuple(float(value) for value in self._fitted.params)
        return DemandForecasterState.model_validate(state)


def restore_demand_forecaster(state: DemandForecasterState) -> Forecaster:
    """Reconstruct numeric state with fixed coefficients; no pickle or optimizer."""
    state = DemandForecasterState.model_validate(state.model_dump())
    capability = demand_model_capability(state.model_id)
    if state.model_id == "weighted_ensemble":
        ensemble = _EnsembleForecaster(capability, state.parameters)
        history = pd.DataFrame({"date": pd.to_datetime(state.dates), DEMAND_TARGET: state.values})
        _LocalForecaster.fit(ensemble, history, state.frequency, state.seasonal_periods)
        ensemble._components = tuple(restore_demand_forecaster(item) for item in state.components)
        return ensemble
    model = (
        _StatisticalForecaster(capability, state.parameters)
        if capability.provider == "statsmodels"
        else _LocalForecaster(capability, state.parameters)
    )
    history = pd.DataFrame({"date": pd.to_datetime(state.dates), DEMAND_TARGET: state.values})
    _LocalForecaster.fit(model, history, state.frequency, state.seasonal_periods)
    if isinstance(model, _StatisticalForecaster):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError(
                "Restore statistical state with uv sync --extra forecasting."
            ) from exc
        try:
            if state.holt_winters is not None:
                fitted = state.holt_winters
                model._fitted = ExponentialSmoothing(
                    np.asarray(state.values),
                    trend=state.parameters.ets_trend,
                    damped_trend=state.parameters.ets_damped,
                    seasonal=state.parameters.ets_seasonal,
                    seasonal_periods=state.seasonal_periods
                    if state.parameters.ets_seasonal
                    else None,
                    initialization_method="known",
                    initial_level=fitted.initial_level,
                    initial_trend=fitted.initial_trend,
                    initial_seasonal=np.asarray(fitted.initial_seasons)
                    if state.parameters.ets_seasonal
                    else None,
                ).fit(
                    smoothing_level=fitted.smoothing_level,
                    smoothing_trend=fitted.smoothing_trend,
                    smoothing_seasonal=fitted.smoothing_seasonal,
                    damping_trend=fitted.damping_trend,
                    optimized=False,
                    remove_bias=False,
                )
            else:
                estimator = SARIMAX(
                    np.asarray(state.values),
                    order=state.parameters.arima_order,
                    seasonal_order=(*state.parameters.sarima_order, state.seasonal_periods)
                    if state.model_id == "sarima"
                    else (0, 0, 0, 0),
                    trend="n",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                if (
                    state.arima_parameters is None
                    or len(state.arima_parameters) != estimator.k_params
                ):
                    raise ContractViolation("ARIMA state parameter count differs from estimator.")
                model._fitted = estimator.filter(np.asarray(state.arima_parameters))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ContractViolation(f"Cannot restore statistical state: {exc}") from exc
    return model


def create_demand_forecaster(
    model_id: str, config: TimeSeriesModelConfig | None = None
) -> Forecaster:
    """Create an independent model instance from the capability registry."""
    capability = demand_model_capability(model_id)
    if model_id == "weighted_ensemble":
        return _EnsembleForecaster(capability, config or TimeSeriesModelConfig())
    constructor = (
        _StatisticalForecaster if capability.provider == "statsmodels" else _LocalForecaster
    )
    return constructor(capability, config or TimeSeriesModelConfig())


def forecast_demand_models(
    actuals: pd.DataFrame,
    request: DemandForecastRequest,
    features: DemandFeatureConfig | None = None,
    models: tuple[str, ...] = ("naive", "seasonal_naive", "ses", "holt_damped"),
    parameters: TimeSeriesModelConfig | None = None,
) -> pd.DataFrame:
    """Fit local time-series states per series/origin and forecast requested models."""
    if not models or len(set(models)) != len(models):
        raise ValueError("Demand models must be nonempty and unique.")
    for model_id in models:
        demand_model_capability(model_id)
    features = features or DemandFeatureConfig()
    history = available_demand_history(actuals, request, features)
    season = features.seasonal_periods or (12 if request.frequency == "MS" else 52)
    forecasts: list[pd.DataFrame] = []
    for model_id in models:
        if model_id in {"naive", "seasonal_naive"}:
            # Preserve the previously validated baseline contract and source semantics.
            for baseline in ("naive", "seasonal_naive"):
                if model_id == baseline:
                    forecasts.append(
                        forecast_demand_baselines(actuals, request, features, (baseline,))
                    )
            continue
        records: list[dict[str, object]] = []
        for key, series in history.groupby(DEMAND_SERIES_COLUMNS, sort=True):
            forecaster = create_demand_forecaster(model_id, parameters)
            forecaster.fit(series, request.frequency, season)
            prediction = forecaster.predict(request.forecast_dates)
            for horizon, (target_date, value) in enumerate(
                zip(request.forecast_dates, prediction, strict=True), start=1
            ):
                records.append(
                    {
                        **dict(zip(DEMAND_SERIES_COLUMNS, key, strict=True)),
                        "date": target_date,
                        "frequency": request.frequency,
                        "forecast_origin": request.origin,
                        "horizon": horizon,
                        "model_id": model_id,
                        "predicted_demand_units": float(value),
                        "source_date": series["date"].iloc[-1],
                        "source_available_at": series["actual_available_at"].max(),
                        "training_start": series["date"].iloc[0],
                        "training_end": series["date"].iloc[-1],
                        "n_training_periods": len(series),
                        "schema_version": ENSEMBLE_FORECAST_SCHEMA_VERSION
                        if model_id == "weighted_ensemble"
                        else FORECAST_SCHEMA_VERSION,
                        "forecast_kind": forecaster.capability.family,
                        "approval_status": "unapproved",
                    }
                )
        forecasts.append(pd.DataFrame(records, columns=FORECAST_COLUMNS))
    return validate_demand_forecasts(pd.concat(forecasts, ignore_index=True), request)
