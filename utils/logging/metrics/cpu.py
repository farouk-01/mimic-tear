# cpu.py

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import psutil
from pydantic import BaseModel, ConfigDict

from .base import FormattedMetricLines, MetricResult, ProfileMetric


class CPUMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cpu_percent: bool = True


class CPUMetric(ProfileMetric):
    name = "CPU"

    def __init__(self, cpu_percent: bool = True) -> None:
        self.cpu_percent = cpu_percent

        self._process = psutil.Process()
        self._started_at = 0.0
        self._cpu_start = None

    def start(self) -> None:
        self._started_at = perf_counter()
        self._cpu_start = self._process.cpu_times()

    def stop(self) -> MetricResult:
        if self._cpu_start is None:
            raise RuntimeError("CPUMetric was not started")

        elapsed_seconds = perf_counter() - self._started_at
        cpu_end = self._process.cpu_times()

        cpu_seconds = (
            cpu_end.user
            + cpu_end.system
            - self._cpu_start.user
            - self._cpu_start.system
        )

        result: MetricResult = {}

        if self.cpu_percent:
            result["cpu_percent"] = (
                cpu_seconds / elapsed_seconds * 100 if elapsed_seconds > 0 else 0.0
            )

        return result
