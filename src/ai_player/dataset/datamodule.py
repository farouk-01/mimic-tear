from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai_player.dataset.dataset import (
    EldenRingDataset,
    discover_sessions,
    load_recording_samples,
    partition_sessions_by_split,
    prepare_frame_caches,
)
from ai_player.dataset.transforms import (
    build_eval_transform,
    build_train_transform,
)


DEFAULT_NUM_WORKERS = max(0, min(4, (os.cpu_count() or 1) - 1))


@dataclass(frozen=True, slots=True)
class DataModuleConfig:
    recordings_directory: Path
    batch_size: int = 32
    width: int = 320
    height: int = 180
    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int = 2
    seed: int = 42
    augment_training: bool = True
    pin_memory: bool | None = None
    drop_last: bool = False
    frame_cache: bool = True
    rebuild_frame_cache: bool = False
    require_game_state: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than zero")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be greater than zero")

        if self.rebuild_frame_cache and not self.frame_cache:
            raise ValueError("cannot rebuild the frame cache when it is disabled")


class EldenRingDataModule:
    """
    Creates training and validation datasets/loaders.

    Entire recording sessions are assigned to either training or validation
    using config.split in each session's metadata.
    """

    def __init__(
        self,
        config: DataModuleConfig,
        *,
        sessions: Sequence[str | Path] | None = None,
    ) -> None:
        self.config = config
        self._session_paths = (
            [Path(session).expanduser().resolve() for session in sessions]
            if sessions is not None
            else None
        )

        self.train_dataset: EldenRingDataset | None = None
        self.validation_dataset: EldenRingDataset | None = None

    def setup(self) -> None:
        session_paths = self._session_paths

        if session_paths is None:
            session_paths = discover_sessions(
                self.config.recordings_directory
            )

        train_sessions, validation_sessions = partition_sessions_by_split(
            session_paths
        )
        train_samples = load_recording_samples(
            train_sessions,
            require_game_state=self.config.require_game_state,
        )
        validation_samples = load_recording_samples(
            validation_sessions,
            require_game_state=self.config.require_game_state,
        )
        frame_cache_paths = (
            prepare_frame_caches(
                [*train_samples, *validation_samples],
                width=self.config.width,
                height=self.config.height,
                rebuild=self.config.rebuild_frame_cache,
                progress=print,
            )
            if self.config.frame_cache
            else {}
        )

        self.train_dataset = EldenRingDataset(
            train_samples,
            transform=build_train_transform(
                width=self.config.width,
                height=self.config.height,
                augment=self.config.augment_training,
            ),
            frame_cache_paths=frame_cache_paths,
        )

        self.validation_dataset = EldenRingDataset(
            validation_samples,
            transform=build_eval_transform(
                width=self.config.width,
                height=self.config.height,
            ),
            frame_cache_paths=frame_cache_paths,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self._require_train_dataset()

        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self._pin_memory(),
            drop_last=self.config.drop_last,
            persistent_workers=self.config.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            **self._worker_options(),
        )

    def validation_dataloader(self) -> DataLoader:
        dataset = self._require_validation_dataset()

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self._pin_memory(),
            drop_last=False,
            persistent_workers=self.config.num_workers > 0,
            worker_init_fn=seed_worker,
            **self._worker_options(),
        )

    def close(self) -> None:
        if self.train_dataset is not None:
            self.train_dataset.close()

        if self.validation_dataset is not None:
            self.validation_dataset.close()

    def _pin_memory(self) -> bool:
        if self.config.pin_memory is not None:
            return self.config.pin_memory

        return torch.cuda.is_available()

    def _worker_options(self) -> dict[str, int]:
        if self.config.num_workers == 0:
            return {}
        return {"prefetch_factor": self.config.prefetch_factor}

    def _require_train_dataset(self) -> EldenRingDataset:
        if self.train_dataset is None:
            raise RuntimeError(
                "Data module has not been set up. Call setup() first."
            )

        return self.train_dataset

    def _require_validation_dataset(self) -> EldenRingDataset:
        if self.validation_dataset is None:
            raise RuntimeError(
                "Data module has not been set up. Call setup() first."
            )

        return self.validation_dataset


def seed_worker(worker_id: int) -> None:
    """
    Seed Python and NumPy from PyTorch's worker-specific seed.
    """

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
