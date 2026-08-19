from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .base import FormattedMetricLines, MetricResult, ProfileMetric


class TimerMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    unit: Literal["ms", "s"] = "ms"
    precision: int = 2


class TimerMetric(ProfileMetric):
    name = "Timer"

    def __init__(
        self,
        unit: Literal["ms", "s"] = "ms",
        precision: int = 2,
    ) -> None:
        self._started_at = 0.0
        self.unit = unit
        self.precision = precision

    def start(self) -> None:
        self._started_at = perf_counter()

    def stop(self) -> MetricResult:
        return {
            "time_ms": (perf_counter() - self._started_at) * 1000,
        }
