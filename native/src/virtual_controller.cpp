#include "controller/virtual_controller.hpp"

#include <SDL3/SDL.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimic_tear::controller {
namespace {

[[noreturn]] void throw_sdl_error(const char* operation) {
    throw std::runtime_error(
        std::string(operation) + ": " + SDL_GetError()
    );
}

void set_axis(
    SDL_Joystick* joystick,
    const SDL_GamepadAxis axis,
    const Sint16 value
) {
    if (!SDL_SetJoystickVirtualAxis(joystick, axis, value)) {
        throw_sdl_error("Failed to set virtual controller axis");
    }
}

void set_button(
    SDL_Joystick* joystick,
    const SDL_GamepadButton button,
    const bool down
) {
    if (!SDL_SetJoystickVirtualButton(joystick, button, down)) {
        throw_sdl_error("Failed to set virtual controller button");
    }
}

}  // namespace

VirtualController::VirtualController(std::string name) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Virtual controller name cannot be empty."
        );
    }

    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        throw_sdl_error("SDL gamepad initialization failed");
    }
    initialized_ = true;

    SDL_VirtualJoystickDesc description{};
    SDL_INIT_INTERFACE(&description);
    description.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    description.naxes = SDL_GAMEPAD_AXIS_COUNT;
    description.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    description.name = name.c_str();

    instance_id_ = SDL_AttachVirtualJoystick(&description);
    if (instance_id_ == 0) {
        const std::string error = SDL_GetError();
        close();
        throw std::runtime_error(
            "Failed to attach virtual controller: " + error
        );
    }

    gamepad_ = SDL_OpenGamepad(instance_id_);
    if (gamepad_ == nullptr) {
        const std::string error = SDL_GetError();
        close();
        throw std::runtime_error(
            "Failed to open virtual controller as a gamepad: " + error
        );
    }

    joystick_ = SDL_GetGamepadJoystick(gamepad_);
    if (joystick_ == nullptr) {
        const std::string error = SDL_GetError();
        close();
        throw std::runtime_error(
            "Failed to access virtual controller joystick: " + error
        );
    }

    reset();
}

VirtualController::~VirtualController() {
    close();
}

VirtualController::VirtualController(
    VirtualController&& other
) noexcept
    : gamepad_(std::exchange(other.gamepad_, nullptr)),
      joystick_(std::exchange(other.joystick_, nullptr)),
      instance_id_(std::exchange(other.instance_id_, 0)),
      initialized_(std::exchange(other.initialized_, false)) {}

VirtualController& VirtualController::operator=(
    VirtualController&& other
) noexcept {
    if (this == &other) {
        return *this;
    }

    close();
    gamepad_ = std::exchange(other.gamepad_, nullptr);
    joystick_ = std::exchange(other.joystick_, nullptr);
    instance_id_ = std::exchange(other.instance_id_, 0);
    initialized_ = std::exchange(other.initialized_, false);
    return *this;
}

void VirtualController::apply(const ControllerState& state) {
    if (joystick_ == nullptr) {
        throw std::runtime_error(
            "Cannot update a closed virtual controller."
        );
    }

    // Validate all analog values before mutating any SDL state.
    const Sint16 left_x = encode_stick(state.left_x);
    const Sint16 left_y = encode_stick(state.left_y);
    const Sint16 right_x = encode_stick(state.right_x);
    const Sint16 right_y = encode_stick(state.right_y);
    const Sint16 left_trigger = encode_trigger(state.left_trigger);
    const Sint16 right_trigger = encode_trigger(state.right_trigger);

    set_axis(joystick_, SDL_GAMEPAD_AXIS_LEFTX, left_x);
    set_axis(joystick_, SDL_GAMEPAD_AXIS_LEFTY, left_y);
    set_axis(joystick_, SDL_GAMEPAD_AXIS_RIGHTX, right_x);
    set_axis(joystick_, SDL_GAMEPAD_AXIS_RIGHTY, right_y);
    set_axis(
        joystick_,
        SDL_GAMEPAD_AXIS_LEFT_TRIGGER,
        left_trigger
    );
    set_axis(
        joystick_,
        SDL_GAMEPAD_AXIS_RIGHT_TRIGGER,
        right_trigger
    );

    set_button(joystick_, SDL_GAMEPAD_BUTTON_SOUTH, state.south);
    set_button(joystick_, SDL_GAMEPAD_BUTTON_EAST, state.east);
    set_button(joystick_, SDL_GAMEPAD_BUTTON_WEST, state.west);
    set_button(joystick_, SDL_GAMEPAD_BUTTON_NORTH, state.north);
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
        state.left_bumper
    );
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
        state.right_bumper
    );
    set_button(joystick_, SDL_GAMEPAD_BUTTON_BACK, state.back);
    set_button(joystick_, SDL_GAMEPAD_BUTTON_START, state.start);
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_LEFT_STICK,
        state.left_stick
    );
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_RIGHT_STICK,
        state.right_stick
    );
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_DPAD_UP,
        state.dpad_up
    );
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_DPAD_DOWN,
        state.dpad_down
    );
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_DPAD_LEFT,
        state.dpad_left
    );
    set_button(
        joystick_,
        SDL_GAMEPAD_BUTTON_DPAD_RIGHT,
        state.dpad_right
    );

    SDL_UpdateJoysticks();
}

void VirtualController::reset() {
    apply(ControllerState{});
}

std::string VirtualController::name() const {
    if (gamepad_ == nullptr) {
        return "Disconnected virtual controller";
    }

    const char* gamepad_name = SDL_GetGamepadName(gamepad_);
    return gamepad_name == nullptr
        ? "Unknown virtual controller"
        : gamepad_name;
}

bool VirtualController::connected() const noexcept {
    return gamepad_ != nullptr &&
           SDL_GamepadConnected(gamepad_);
}

SDL_JoystickID VirtualController::instance_id() const noexcept {
    return instance_id_;
}

Sint16 VirtualController::encode_stick(const float value) {
    if (!std::isfinite(value) || value < -1.0f || value > 1.0f) {
        throw std::invalid_argument(
            "Stick values must be finite and in the range [-1, 1]."
        );
    }

    return value >= 0.0f
        ? static_cast<Sint16>(
              std::lround(value * std::numeric_limits<Sint16>::max())
          )
        : static_cast<Sint16>(
              std::lround(-value * std::numeric_limits<Sint16>::min())
          );
}

Sint16 VirtualController::encode_trigger(const float value) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
        throw std::invalid_argument(
            "Trigger values must be finite and in the range [0, 1]."
        );
    }

    constexpr auto minimum = std::numeric_limits<Sint16>::min();
    constexpr auto span =
        static_cast<long>(std::numeric_limits<Uint16>::max());
    return static_cast<Sint16>(
        minimum + std::lround(value * span)
    );
}

void VirtualController::close() noexcept {
    joystick_ = nullptr;

    if (gamepad_ != nullptr) {
        SDL_CloseGamepad(gamepad_);
        gamepad_ = nullptr;
    }

    if (instance_id_ != 0) {
        SDL_DetachVirtualJoystick(instance_id_);
        instance_id_ = 0;
    }

    if (initialized_) {
        SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
        initialized_ = false;
    }
}

}  // namespace mimic_tear::controller
