# from __future__ import annotations

# from threading import Event
# from time import perf_counter_ns, sleep

# import torch
# from torch import Tensor

# from capture.screen import ScreenReader
# from controller import (
#     ANALOG_INPUTS,
#     BUTTON_INPUTS,
#     AnalogState,
#     ButtonState,
#     ControllerState,
# )
# from data.transforms.frames import FrameTransform
# from data.transforms.game_state import GameStateTransform
# from game_state import GameStateReader
# from mimic_tear.model.components.controller import ControllerOutput
# from mimic_tear.model.components.temporal import LSTMState
# from mimic_tear.model.policy import EldenRingPolicy
# from recording.writers.gamepad import GamepadWriter
# from utils.logging import Logger


# class Player:
#     def __init__(
#         self,
#         *,
#         model: EldenRingPolicy,
#         screen: ScreenReader,
#         gamepad: GamepadWriter,
#         frame_transform: FrameTransform,
#         device: torch.device | str,
#         game_state_reader: GameStateReader | None = None,
#         game_state_transform: GameStateTransform | None = None,
#         game_state_features: tuple[str, ...] | None = None,
#         button_threshold: float = 0.5,
#         fps: float = 30.0,
#         logger: Logger,
#         log_every: int = 30,
#     ) -> None:
#         if fps <= 0:
#             raise ValueError("fps must be greater than zero")

#         if not 0.0 <= button_threshold <= 1.0:
#             raise ValueError("button_threshold must be in [0, 1]")

#         if game_state_reader is not None and game_state_features is None:
#             raise ValueError(
#                 "game_state_features must be provided when "
#                 "game_state_reader is provided"
#             )

#         self.model = model
#         self.screen = screen
#         self.gamepad = gamepad
#         self.frame_transform = frame_transform
#         self.device = torch.device(device)

#         self.game_state_reader = game_state_reader
#         self.game_state_transform = game_state_transform
#         self.game_state_features = game_state_features

#         self.button_threshold = button_threshold
#         self.fps = fps

#         self.logger = logger
#         self.log_every = log_every
#         self._step = 0

#     def _prepare_frame(self, frame: Tensor) -> Tensor:
#         frame = frame.to(self.device)
#         frame = self.frame_transform(frame)

#         # [C, H, W] -> [B=1, T=1, C, H, W]
#         return frame.unsqueeze(0).unsqueeze(0)

#     def _prepare_game_state(self) -> Tensor | None:
#         if self.game_state_reader is None:
#             return None

#         snapshot = self.game_state_reader.read()

#         features = self.game_state_features or ()

#         missing = [feature for feature in features if feature not in snapshot.values]

#         if missing:
#             raise ValueError(f"Game-state snapshot is missing features: {missing}")

#         state = torch.tensor(
#             [float(snapshot.values[feature]) for feature in features],
#             dtype=torch.float32,
#             device=self.device,
#         )

#         if self._step % self.log_every == 0:
#             self.logger.debug(
#                 "step=%d | game_state=%s",
#                 self._step,
#                 {
#                     feature: round(float(value), 3)
#                     for feature, value in zip(features, state.tolist(), strict=True)
#                 },
#             )

#         if self.game_state_transform is not None:
#             state = self.game_state_transform(state)

#         return state.unsqueeze(0).unsqueeze(0)

#     def _handle_output(self, output: ControllerOutput) -> None:
#         analog = output.analog[0, 0].detach().cpu()
#         button_probs = torch.sigmoid(output.button_logits[0, 0]).detach().cpu()

#         analog_values = {
#             name: float(value)
#             for name, value in zip(ANALOG_INPUTS, analog.tolist(), strict=True)
#         }

#         button_probabilities = {
#             name: float(value)
#             for name, value in zip(BUTTON_INPUTS, button_probs.tolist(), strict=True)
#         }

#         if self._step % self.log_every == 0:
#             self.logger.debug(
#                 "step=%d | analog=%s | buttons=%s",
#                 self._step,
#                 {name: round(value, 3) for name, value in analog_values.items()},
#                 {name: round(value, 3) for name, value in button_probabilities.items()},
#             )

#         buttons = button_probs >= self.button_threshold

#         button_values = {
#             name: bool(value)
#             for name, value in zip(BUTTON_INPUTS, buttons.tolist(), strict=True)
#         }

#         state = ControllerState(
#             analog=AnalogState(**analog_values),
#             buttons=ButtonState(**button_values),
#         )

#         state.validate()
#         self.gamepad.write(state)

#         self._step += 1

#     def run(self, *, stop_event: Event | None = None) -> None:
#         self.model.eval()
#         state: LSTMState | None = None

#         period_ns = round(1_000_000_000 / self.fps)
#         next_tick_ns = perf_counter_ns()

#         self.gamepad.connect()

#         try:
#             with torch.inference_mode():
#                 while stop_event is None or not stop_event.is_set():
#                     captured = self.screen.read()

#                     frame = torch.from_numpy(captured.image).permute(2, 0, 1)
#                     images = self._prepare_frame(frame)
#                     game_state = self._prepare_game_state()

#                     output, next_state = self.model(
#                         images,
#                         game_state=game_state,
#                         state=state,
#                     )

#                     state = self.model.detach_state(next_state)

#                     self._handle_output(output)

#                     next_tick_ns += period_ns
#                     now_ns = perf_counter_ns()
#                     remaining_ns = next_tick_ns - now_ns

#                     if remaining_ns > 0:
#                         sleep(remaining_ns / 1_000_000_000)
#                         continue

#                     missed_ticks = ((now_ns - next_tick_ns) // period_ns) + 1
#                     next_tick_ns += missed_ticks * period_ns
#         finally:
#             self.gamepad.reset()
