from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
import torch
from typing import (
    Callable,
    TypeVar,
    ParamSpec,
)
from functools import wraps

from ..logging.logger import Logger
from .metrics.base import ProfileMetric
from .metrics.cpu import CPUMetric, CPUMetricConfig
from .metrics.cuda import CUDAMetric, CUDAMetricConfig
from .metrics.ram import RAMMetric, RAMMetricConfig
from .metrics.timer import TimerMetric, TimerMetricConfig


Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")

_profiler: Profiler | None = None


class Profilable(ABC):
    profiler: Profiler
    perf_logger: Logger


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

        metrics: list[ProfileMetric] = []

        if config.timer_enabled:
            metrics.append(TimerMetric(**config.timer.model_dump()))

        if config.cpu_enabled:
            metrics.append(CPUMetric(**config.cpu.model_dump()))

        if config.ram_enabled:
            metrics.append(RAMMetric(**config.ram.model_dump()))

        if config.cuda_enabled:
            metrics.append(CUDAMetric(**config.cuda.model_dump(), device=device))

        self.metrics = tuple(metrics)

        _profiler = self

    def start(self) -> None:
        if not self.config.enabled:
            return

        for metric in self.metrics:
            metric.start()

    def stop(self) -> dict[str, float]:
        if not self.config.enabled:
            return {}

        result: dict[str, float] = {}

        for metric in reversed(self.metrics):
            result.update(metric.stop())

        return result


def profile(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
    @wraps(function)
    def wrapper(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        if _profiler is None:
            return function(*args, **kwargs)

        _profiler.start()

        try:
            return function(*args, **kwargs)
        finally:
            metrics = _profiler.stop()

            _profiler.logger.debug("%s | %s", function.__qualname__, metrics)

    return wrapper
