from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from mimic_tear.recording.schema import BUTTON_COLUMNS


class PolicyOutput(NamedTuple):
    analog: Tensor
    button_logits: Tensor


class EldenRingPolicy(nn.Module):
    def __init__(
        self,
        *,
        analog_outputs: int = 6,
        button_outputs: int = len(BUTTON_COLUMNS),
        game_state_features: int = 0,
    ) -> None:
        super().__init__()

        if analog_outputs != 6:
            raise ValueError(
                "This policy expects 6 analog outputs: "
                "4 stick axes and 2 triggers."
            )

        if button_outputs <= 0:
            raise ValueError("button_outputs must be greater than zero")
        if game_state_features < 0:
            raise ValueError("game_state_features cannot be negative")
        self.game_state_features = game_state_features

        #
        #
        # stride is the factor by which the input image is downsampled at each convolutional layer
        # 
        # -> assume input is 360 x 640 (height x width)
        #
        self.encoder = nn.Sequential(
            # layer 1
            # learns 32 basic features from the RGB channels of the input image
            # e.g color regions, hud shapes, large objects, etc.
            nn.Conv2d(
                # TODO Maybe : frame stacking to give model temporal context
                in_channels=3, # RGB channels
                out_channels=32, # this layer will learn 32 filters (feature maps)
                kernel_size=3, # each 32 filters is a 3x3 pixels (learned weights = in_channels x kernel_size x kernel_size)
                stride=2, # filter moves 2 pixels at a time (e.g pixels 0-2, 2-5, 4-7, etc.), this reduces the resolution of the feature maps by a factor of 2

                # add a n-pixel zero border around every edge of the input image.
                # 
                # - + ------------ + -
                # 0 | image pixels | 0
                # 0 | image pixels | 0
                # 0 | image pixels | 0
                # - + ------------ + -
                #
                # without padding, edge pixels would appear in fewer convolution windows
                # than pixels near the center of the image.
                padding=1, 
            ), # -> produces 32 x 180 x 320 feature maps (out_channels x height x width)
            # normalize the 32 feature maps
            nn.BatchNorm2d(32),
            # replace negative values with zero
            nn.ReLU(inplace=True),

            # layer 2
            # combines basic features into 64 more detailed patterns
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ), # -> produces 64 x 90 x 160 feature maps (out_channels x height x width)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # layer 3
            # learns 128 even more complex features combinations from the 64 previous ones
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ), # -> produces 128 x 45 x 80 feature maps (out_channels x height x width)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # layer 4
            # learns 256 even more complex features combinations from the 128 previous ones
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ), # -> produces 256 x 23 x 40 feature maps (out_channels x height x width)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 5: Keeps spatial structure while reducing the number of parameters
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # -> 256 x 12 x 20

            # before pooling, the feature maps is 256 x 12 x 20
            # after adaptive average pooling, the feature maps will be 256 x 4 x 4
            # it computes the average of each 12x20 feature map
            # keeps rough spatial structure (left vs right side of screen, top vs bottom of screen)
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # flatten size = 256 x 4 x 4 = 4096 data points
        self.shared = nn.Sequential(
            nn.Flatten(),
            # linear layer reduces the 4096 data points to 512
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),

            # randomly dropout 20% of the 512 data points during training
            # this forces the network to learn multiple to understand what is happening
            nn.Dropout(p=0.20),

            # linear layer reduces the 512 data points to 256
            # going from 4096 -> 512 -> 256 smooths the transition from the convolutional layers
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        if game_state_features > 0:
            self.game_state_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(game_state_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            self.fusion: nn.Module | None = nn.Sequential(
                nn.Linear(256 + 64, 256),
                nn.ReLU(inplace=True),
            )
        else:
            self.game_state_encoder = None
            self.fusion = None

        self.analog_head = nn.Linear(256, analog_outputs)
        self.button_head = nn.Linear(256, button_outputs)

        self._initialize_weights()

    def forward(
        self,
        image: Tensor,
        game_state: Tensor | None = None,
    ) -> PolicyOutput:
        features = self.encoder(image)
        features = self.shared(features)

        if self.game_state_features > 0:
            if game_state is None:
                raise ValueError("This policy checkpoint requires game-state input")
            if game_state.ndim != 2 or game_state.shape != (
                image.shape[0],
                self.game_state_features,
            ):
                raise ValueError(
                    "Expected game state with shape "
                    f"({image.shape[0]}, {self.game_state_features}), received "
                    f"{tuple(game_state.shape)}"
                )
            assert self.game_state_encoder is not None
            assert self.fusion is not None
            state_features = self.game_state_encoder(game_state)
            features = self.fusion(torch.cat((features, state_features), dim=1))
        elif game_state is not None and game_state.shape[0] != image.shape[0]:
            raise ValueError("Image and game-state batch sizes do not match")

        raw_analog = self.analog_head(features)

        sticks = torch.tanh(raw_analog[:, :4])
        triggers = torch.sigmoid(raw_analog[:, 4:6])

        analog = torch.cat((sticks, triggers), dim=1)

        button_logits = self.button_head(features)

        return PolicyOutput(
            analog=analog,
            button_logits=button_logits,
        )

    def predict(
        self,
        image: Tensor,
        game_state: Tensor | None = None,
        *,
        stick_deadzone: float = 0.08,
        trigger_deadzone: float = 0.10,
        button_threshold: float = 0.5,
    ) -> tuple[Tensor, Tensor]:
        if not 0.0 <= stick_deadzone <= 1.0:
            raise ValueError("stick_deadzone must be in [0, 1]")

        if not 0.0 <= trigger_deadzone <= 1.0:
            raise ValueError("trigger_deadzone must be in [0, 1]")

        output = self(image, game_state)

        sticks = output.analog[:, :4]
        triggers = output.analog[:, 4:6]

        sticks = torch.where(
            sticks.abs() < stick_deadzone,
            torch.zeros_like(sticks),
            sticks,
        )

        triggers = torch.where(
            triggers < trigger_deadzone,
            torch.zeros_like(triggers),
            triggers,
        )

        analog = torch.cat((sticks, triggers), dim=1)

        button_probabilities = torch.sigmoid(output.button_logits)
        buttons = button_probabilities >= button_threshold

        return analog, buttons

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
