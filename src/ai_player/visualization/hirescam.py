from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import HiResCAM
from torch import Tensor, nn

from ai_player.policy import load_policy_checkpoint


class _PolicyCamAdapter(nn.Module):
    """Present the policy's mixed outputs as one attribution vector."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.game_state: Tensor | None = None

    def forward(self, image: Tensor) -> Tensor:
        output = self.policy(image, self.game_state)
        return torch.cat(
            (output.analog, torch.sigmoid(output.button_logits)),
            dim=1,
        )


class _CurrentActionTarget:
    """Attribute the strength of the policy's current non-neutral action."""

    def __call__(self, output: Tensor) -> Tensor:
        sticks = output[..., :4]
        triggers = output[..., 4:6]
        active_buttons = torch.relu(output[..., 6:] - 0.5) * 2.0
        return (
            sticks.square().sum()
            + triggers.square().sum()
            + active_buttons.square().sum()
        )


def find_last_convolution(module: nn.Module) -> nn.Conv2d:
    target: nn.Conv2d | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            target = child
    if target is None:
        raise ValueError("HiResCAM requires a model containing a Conv2d layer")
    return target


class HiResCamVisualizer:
    """Generate HiResCAM maps for the policy's current controller action."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model.eval()
        self._adapter = _PolicyCamAdapter(self.model).eval()
        self._cam = HiResCAM(
            model=self._adapter,
            target_layers=[target_layer or find_last_convolution(self.model)],
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
    ) -> "HiResCamVisualizer":
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        model, _ = load_policy_checkpoint(path, device=device)
        return cls(model)

    def generate(
        self,
        image: Tensor,
        game_state: Tensor | None = None,
    ) -> np.ndarray:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "HiResCAM expects one BCHW image; received "
                f"{tuple(image.shape)}"
            )
        if getattr(self.model, "game_state_features", 0) > 0 and game_state is None:
            raise ValueError("This HiResCAM checkpoint requires game-state input")
        self._adapter.game_state = game_state
        with torch.enable_grad():
            heatmap = self._cam(
                input_tensor=image,
                targets=[_CurrentActionTarget()],
            )[0]
        self._adapter.game_state = None
        return np.asarray(heatmap, dtype=np.float32)

    def close(self) -> None:
        self._cam.activations_and_grads.release()

    def __enter__(self) -> "HiResCamVisualizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def blend_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    *,
    opacity: float = 0.45,
) -> np.ndarray:
    """Blend a normalized HiResCAM map onto a BGR video frame."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HWC BGR frame, received {frame.shape}")
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, received {heatmap.shape}")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")

    resized = cv2.resize(
        np.clip(heatmap, 0.0, 1.0),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    colored = cv2.applyColorMap(
        np.rint(resized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.addWeighted(frame, 1.0 - opacity, colored, opacity, 0.0)
