# Native SDL3 controller module

`ai_controller.Controller` reads a physical gamepad. The new
`ai_controller.VirtualController` accepts AI-generated `ControllerState`
objects and exposes them as an SDL3 virtual gamepad.

```python
import ai_controller

output = ai_controller.VirtualController()
state = ai_controller.ControllerState()
state.left_y = -1.0
state.south = True
output.apply(state)

# Release every control, especially during shutdown or error handling.
output.reset()
```

Stick values must be in `[-1, 1]`; trigger values must be in `[0, 1]`.
`apply()` replaces the complete state, so callers should populate every held
control on every inference step.

## Important SDL limitation

An SDL virtual joystick exists only inside the process that creates it. It is
useful for testing the AI, replaying inputs, and controlling an SDL game running
in the same process. It does **not** create a Windows-wide Xbox controller, so a
separate game such as Elden Ring will not see it. Cross-process game control on
Windows uses the HIDMaestro backend in `bridges/hidmaestro/`; the SDL virtual
controller remains useful for isolated native tests.
