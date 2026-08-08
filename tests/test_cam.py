from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_player.visualization.hirescam import (  # noqa: E402
    HiResCamVisualizer,
    blend_heatmap,
)
from ai_player.policy import PolicyOutput  # noqa: E402


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        nn.init.ones_(self.encoder[0].weight)

    def forward(
        self,
        image: Tensor,
        game_state: Tensor | None = None,
    ) -> PolicyOutput:
        del game_state
        features = self.encoder(image).mean(dim=(-2, -1))
        action = features.mean(dim=1, keepdim=True)
        return PolicyOutput(
            analog=action.repeat(1, 6),
            button_logits=action.repeat(1, 2),
        )


class HiResCamTests(unittest.TestCase):
    def test_hirescam_generates_normalized_spatial_map(self) -> None:
        image = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
        image[:, :, 5:11, 5:11] = 1.0
        visualizer = HiResCamVisualizer(TinyPolicy())
        try:
            heatmap = visualizer.generate(image)
        finally:
            visualizer.close()

        self.assertEqual(heatmap.shape, (16, 16))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreater(float(heatmap.max()), 0.0)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_blend_heatmap_preserves_frame_shape(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        heatmap = np.zeros((5, 8), dtype=np.float32)
        heatmap[2, 4] = 1.0

        blended = blend_heatmap(frame, heatmap, opacity=0.5)

        self.assertEqual(blended.shape, frame.shape)
        self.assertEqual(blended.dtype, np.uint8)
        self.assertGreater(int(blended.sum()), 0)

    def test_blend_heatmap_rejects_invalid_opacity(self) -> None:
        with self.assertRaises(ValueError):
            blend_heatmap(
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.float32),
                opacity=1.1,
            )


if __name__ == "__main__":
    unittest.main()
