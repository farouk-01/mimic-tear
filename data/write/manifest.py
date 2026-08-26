from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, ConfigDict

class RecordingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: int
    video_file: str
    controller_file: str
    game_state_file: str
    metadata_file: str

@dataclass(frozen=True, slots=True)
class Recording:
    root: Path
    video: Path
    controller: Path
    game_state: Path | None
    metadata: Path | None

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        config: RecordingConfig,
    ) -> Recording:
        root = Path(root)

        if not root.is_dir():
            raise ValueError(
                f"Recording directory does not exist: {root}"
            )

        video = root / config.video_file
        controller = root / config.controller_file
        game_state = root / config.game_state_file
        metadata = root / config.metadata_file

        if not video.is_file():
            raise FileNotFoundError(
                f"Missing recording video: {video}"
            )

        if not controller.is_file():
            raise FileNotFoundError(
                f"Missing controller data: {controller}"
            )

        return cls(
            root=root,
            video=video,
            controller=controller,
            game_state=(
                game_state
                if game_state.is_file()
                else None
            ),
            metadata=(
                metadata
                if metadata.is_file()
                else None
            ),
        )

    @property
    def has_game_state(self) -> bool:
        return self.game_state is not None

    @property
    def has_metadata(self) -> bool:
        return self.metadata is not None