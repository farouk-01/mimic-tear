from __future__ import annotations

from collections.abc import Mapping

import psutil
from pydantic import BaseModel, ConfigDict

from .base import FormattedMetricLines, MetricResult, ProfileMetric


class RAMMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    usage: bool = True
    delta: bool = True


def _megabytes(value: int) -> float:
    return value / (1024 * 1024)


class RAMMetric(ProfileMetric):
    name = "RAM"
    
    def __init__(self, usage: bool = True, delta: bool = True) -> None:
        self._process = psutil.Process()
        self._ram_start = 0
        self.usage = usage
        self.delta = delta

    def start(self) -> None:
        self._ram_start = self._process.memory_info().rss

    def stop(self) -> MetricResult:
        ram_end = self._process.memory_info().rss

        result: MetricResult = {}

        if self.usage:
            result["ram_start_mb"] = _megabytes(self._ram_start)
            result["ram_end_mb"] = _megabytes(ram_end)

        if self.delta:
            result["ram_delta_mb"] = _megabytes(ram_end - self._ram_start)

        return result
