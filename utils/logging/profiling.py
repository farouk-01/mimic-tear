from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, overload

import torch
from pydantic import BaseModel, ConfigDict, Field

from .logger import Logger
from .metrics.base import ProfileMetric
from .metrics.cpu import CPUMetric, CPUMetricConfig
from .metrics.cuda import CUDAMetric, CUDAMetricConfig
from .metrics.cuda_timer import CUDATimerMetric
from .metrics.ram import RAMMetric, RAMMetricConfig
from .metrics.timer import TimerMetric, TimerMetricConfig

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")

_profiler: Profiler | None = None


class Profilable(ABC):
    profiler: Profiler
    perf_logger: Logger


class FunctionProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    enabled: bool = False

    timer: bool = False
    cuda_timer: bool = False
    cpu: bool = False
    ram: bool = False
    cuda: bool = False

    aggregate: bool = False
    flush: bool = False


class ProfilerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    enabled: bool = True

    timer_enabled: bool = True
    cpu_enabled: bool = True
    ram_enabled: bool = True
    cuda_enabled: bool = True

    timer: TimerMetricConfig
    cpu: CPUMetricConfig
    ram: RAMMetricConfig
    cuda: CUDAMetricConfig

    profiles: dict[str, FunctionProfileConfig] = Field(default_factory=dict)


class Profiler:
    def __init__(
        self,
        config: ProfilerConfig,
        *,
        logger: Logger,
        device: str | torch.device,
    ) -> None:
        global _profiler

        self.config = config
        self.logger = logger
        self.device = torch.device(device)
        self.samples: dict[str, dict[str, list[float]]] = {}

        _profiler = self

    def get_profile(
        self, function: Callable[..., object]
    ) -> FunctionProfileConfig | None:
        return self.config.profiles.get(function.__qualname__)

    def create_metrics(
        self,
        *,
        timer: bool = True,
        cuda_timer: bool = True,
        cpu: bool = True,
        ram: bool = True,
        cuda: bool = True,
    ) -> tuple[ProfileMetric, ...]:
        metrics: list[ProfileMetric] = []

        if self.config.timer_enabled and timer:
            metrics.append(TimerMetric(**self.config.timer.model_dump()))

        if self.config.timer_enabled and cuda_timer and self.device.type == "cuda":
            metrics.append(CUDATimerMetric(**self.config.timer.model_dump()))

        if self.config.cpu_enabled and cpu:
            metrics.append(CPUMetric(**self.config.cpu.model_dump()))

        if self.config.ram_enabled and ram:
            metrics.append(RAMMetric(**self.config.ram.model_dump()))

        if self.config.cuda_enabled and cuda:
            metrics.append(
                CUDAMetric(**self.config.cuda.model_dump(), device=self.device)
            )

        return tuple(metrics)

    def start(self, metrics: tuple[ProfileMetric, ...]) -> None:
        if not self.config.enabled:
            return

        for metric in metrics:
            metric.start()

    def stop(
        self,
        metrics: tuple[ProfileMetric, ...],
    ) -> dict[str, dict[str, float]]:
        if not self.config.enabled:
            return {}

        result: dict[str, dict[str, float]] = {}

        for metric in metrics:
            result[metric.name] = metric.stop()

        return result

    def add_sample(
        self,
        name: str,
        results: dict[str, dict[str, float]],
    ) -> None:
        for metric_name, values in results.items():
            metric_samples = self.samples.setdefault(name, {}).setdefault(
                metric_name, []
            )
            metric_samples.extend(values.values())

    def log_samples(self) -> None:
        for name, metrics in self.samples.items():
            for metric_name, values in metrics.items():
                if not values:
                    continue

                average = sum(values) / len(values)

                self.logger.debug(
                    "%-26s | %-10s | %-8s | samples=%d",
                    name,
                    metric_name,
                    f"avg={average:.2f}",
                    len(values),
                )

        self.samples.clear()


def profile(
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    @wraps(function)
    def wrapper(
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> Result:
        if _profiler is None or not _profiler.config.enabled:
            return function(*args, **kwargs)

        config = _profiler.get_profile(function)

        if config is None or not config.enabled:
            return function(*args, **kwargs)

        metrics = _profiler.create_metrics(
            timer=config.timer,
            cuda_timer=config.cuda_timer,
            cpu=config.cpu,
            ram=config.ram,
            cuda=config.cuda,
        )

        _profiler.start(metrics)

        try:
            return function(*args, **kwargs)
        finally:
            results = _profiler.stop(metrics)

            if config.aggregate:
                _profiler.add_sample(function.__qualname__, results)
            else:
                for metric_name, values in results.items():
                    formatted_values = ", ".join(
                        f"{key}: {value:.2f}"
                        for key, value in values.items()
                    )

                    _profiler.logger.debug(
                        "%-26s | %-10s | {%s}",
                        function.__qualname__,
                        metric_name,
                        formatted_values,
                    )

            if config.flush:
                _profiler.log_samples()
                
    return wrapper
