from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from .base import ProfileMetric


class CUDAMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

def _megabytes(value: int) -> float:
    return value / (1024 * 1024)


class CUDAMetric(ProfileMetric):
    def __init__(self, device: torch.device | str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available on : {device}")

        self.device = torch.device(device)

        self._vram_start = 0
        self._reserved_start = 0

    def start(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        self._vram_start = torch.cuda.memory_allocated(self.device)
        self._reserved_start = torch.cuda.memory_reserved(self.device)

    def stop(self) -> dict[str, float]:
        torch.cuda.synchronize(self.device)

        vram_end = torch.cuda.memory_allocated(self.device)
        reserved_end = torch.cuda.memory_reserved(self.device)

        return {
            "vram_start_mb": _megabytes(self._vram_start),
            "vram_end_mb": _megabytes(vram_end),
            "vram_peak_mb": _megabytes(torch.cuda.max_memory_allocated(self.device)),
            "reserved_start_mb": _megabytes(self._reserved_start),
            "reserved_end_mb": _megabytes(reserved_end),
            "reserved_peak_mb": _megabytes(torch.cuda.max_memory_reserved(self.device)),
        }
