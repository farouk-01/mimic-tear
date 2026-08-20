from __future__ import annotations

from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .base import MetricResult, ProfileMetric

TimeUnit = Literal["ms", "s", "min"]


class TimerMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    unit: TimeUnit = "ms"
    precision: int = 2


class TimerMetric(ProfileMetric):
    name = "Timer"

    def __init__(
        self,
        unit: TimeUnit = "ms",
        precision: int = 2,
    ) -> None:
        self._started_at = 0.0
        self.unit = unit
        self.precision = precision

    def start(self) -> None:
        self._started_at = perf_counter()

    def stop(self) -> MetricResult:
        elapsed_seconds = perf_counter() - self._started_at

        match self.unit:
            case "ms":
                elapsed = elapsed_seconds * 1000
            case "s":
                elapsed = elapsed_seconds
            case "min":
                elapsed = elapsed_seconds / 60
            case _:
                raise ValueError(f"Invalid time unit: {self.unit}")

        return {f"time_{self.unit}": round(elapsed, self.precision)}
