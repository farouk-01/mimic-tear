from __future__ import annotations

from functools import wraps
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict
import torch
from rich.columns import Columns
from rich.table import Table

from ..logging.logger import Logger
from .metrics.base import MetricResult, ProfileMetric
from .metrics.cpu import CPUMetric, CPUMetricConfig
from .metrics.cuda import CUDAMetric, CUDAMetricConfig
from .metrics.ram import RAMMetric, RAMMetricConfig
from .metrics.timer import TimerMetric, TimerMetricConfig

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")

type ProfileResult = dict[str, MetricResult]

_profiler: Profiler | None = None


class ProfilerConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

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

    def stop(self) -> ProfileResult:
        if not self.config.enabled:
            return {}

        result: ProfileResult = {}

        for metric in self.metrics:
            result[metric.name] = metric.stop()

        return result

    def render(
        self,
        result: ProfileResult,
    ) -> Columns:
        tables: list[Table] = []

        for metric_name, values in result.items():
            if not values:
                continue

            table = Table(title=metric_name, show_header=False)

            table.add_column("Metric")
            table.add_column("Value", justify="right")

            for name, value in values.items():
                table.add_row(name, f"{value:.2f}")

            tables.append(table)

        return Columns(tables, equal=False, expand=False, padding=(0, 1))


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

        _profiler.start()

        try:
            return function(*args, **kwargs)

        finally:
            metrics = _profiler.stop()

            if metrics:
                _profiler.logger.debug("%s", function.__qualname__)
                _profiler.logger.print(_profiler.render(metrics))

    return wrapper
