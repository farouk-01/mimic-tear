from __future__ import annotations

from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict
import torch

from .base import MetricResult, ProfileMetric

TimeUnit = Literal["ms", "s", "min"]

# CUDATimerMetric receives the same config as TimerMetricConfig

class CUDATimerMetric(ProfileMetric):
    name = "CUDATimer"

    def __init__(self, unit: TimeUnit = "ms", precision: int = 2) -> None:
        self.precision = precision
        self.unit = unit

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start_event.record()

    def stop(self) -> MetricResult:
        self._end_event.record()
        self._end_event.synchronize()

        elapsed_ms = self._start_event.elapsed_time(self._end_event)
        
        match self.unit:
            case "ms":
                elapsed = elapsed_ms
            case "s":
                elapsed = elapsed_ms / 1000 
            case "min":
                elapsed = elapsed_ms / 60_000
            case _:
                raise ValueError(f"Invalid time unit: {self.unit}")

        return {f"time_{self.unit}": round(elapsed, self.precision)}