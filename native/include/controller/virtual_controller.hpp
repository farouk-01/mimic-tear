#pragma once

#include "controller/controller_state.hpp"

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>

#include <string>

namespace ai_player::controller {

// An SDL gamepad whose inputs are supplied by the AI instead of hardware.
// SDL virtual devices are visible only inside this process.
class VirtualController {
public:
    explicit VirtualController(
        std::string name = "AI Player Virtual Controller"
    );
    ~VirtualController();

    VirtualController(const VirtualController&) = delete;
    VirtualController& operator=(const VirtualController&) = delete;

    VirtualController(VirtualController&& other) noexcept;
    VirtualController& operator=(VirtualController&& other) noexcept;

    // Replaces the entire controller state. Values are committed together on
    // the next SDL joystick update, avoiding a frame with partially updated
    // AI output.
    void apply(const ControllerState& state);
    void reset();

    [[nodiscard]] std::string name() const;
    [[nodiscard]] bool connected() const noexcept;
    [[nodiscard]] SDL_JoystickID instance_id() const noexcept;

private:
    static Sint16 encode_stick(float value);
    static Sint16 encode_trigger(float value);

    void close() noexcept;

    SDL_Gamepad* gamepad_ = nullptr;
    SDL_Joystick* joystick_ = nullptr;  // Owned by gamepad_.
    SDL_JoystickID instance_id_ = 0;
    bool initialized_ = false;
};

}  // namespace ai_player::controller
