from __future__ import annotations

from collections.abc import Mapping

import torch
from pydantic import BaseModel, ConfigDict

from .base import FormattedMetricLines, MetricResult, ProfileMetric


class CUDAMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    allocated: bool = True
    reserved: bool = True
    peak: bool = True


def _megabytes(value: int) -> float:
    return value / (1024 * 1024)


class CUDAMetric(ProfileMetric):
    name = "CUDA"

    def __init__(
        self,
        device: torch.device | str,
        allocated: bool = True,
        reserved: bool = True,
        peak: bool = True,
    ) -> None:
        self.device = torch.device(device)

        if self.device.type != "cuda":
            raise ValueError(f"CUDAMetric requires a CUDA device, got: {self.device}")

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available on : {device}")

        self.allocated = allocated
        self.reserved = reserved
        self.peak = peak

        self._vram_start = 0
        self._reserved_start = 0

    def start(self) -> None:
        torch.cuda.synchronize(self.device)

        if self.peak:
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.allocated:
            self._vram_start = torch.cuda.memory_allocated(self.device)

        if self.reserved:
            self._reserved_start = torch.cuda.memory_reserved(self.device)

    def stop(self) -> MetricResult:
        torch.cuda.synchronize(self.device)

        result: MetricResult = {}

        if self.allocated:
            result["vram_start_mb"] = _megabytes(self._vram_start)
            result["vram_end_mb"] = _megabytes(torch.cuda.memory_allocated(self.device))

            if self.peak:
                result["vram_peak_mb"] = _megabytes(
                    torch.cuda.max_memory_allocated(self.device)
                )

        if self.reserved:
            result["reserved_start_mb"] = _megabytes(self._reserved_start)
            result["reserved_end_mb"] = _megabytes(
                torch.cuda.memory_reserved(self.device)
            )

            if self.peak:
                result["reserved_peak_mb"] = _megabytes(
                    torch.cuda.max_memory_reserved(self.device)
                )

        return result
