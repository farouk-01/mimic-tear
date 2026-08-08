#include "controller/controller.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ai_player::controller {

Controller::Controller(const float stick_deadzone)
    : stick_deadzone_(stick_deadzone) {
    if (stick_deadzone < 0.0f || stick_deadzone >= 1.0f) {
        throw std::invalid_argument(
            "Stick deadzone must be in the range [0, 1)."
        );
    }

    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        throw std::runtime_error(
            std::string("SDL gamepad initialization failed: ") +
            SDL_GetError()
        );
    }

    int gamepad_count = 0;
    SDL_JoystickID* gamepad_ids =
        SDL_GetGamepads(&gamepad_count);

    if (gamepad_ids == nullptr || gamepad_count == 0) {
        SDL_free(gamepad_ids);
        SDL_QuitSubSystem(SDL_INIT_GAMEPAD);

        throw std::runtime_error(
            std::string("No SDL-compatible gamepad detected: ") +
            SDL_GetError()
        );
    }

    gamepad_ = SDL_OpenGamepad(gamepad_ids[0]);
    SDL_free(gamepad_ids);

    if (gamepad_ == nullptr) {
        const std::string error = SDL_GetError();
        SDL_QuitSubSystem(SDL_INIT_GAMEPAD);

        throw std::runtime_error(
            "Failed to open gamepad: " + error
        );
    }
}

Controller::~Controller() {
    close();
}

Controller::Controller(Controller&& other) noexcept
    : gamepad_(std::exchange(other.gamepad_, nullptr)),
      stick_deadzone_(other.stick_deadzone_) {}

Controller& Controller::operator=(Controller&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    close();

    gamepad_ = std::exchange(other.gamepad_, nullptr);
    stick_deadzone_ = other.stick_deadzone_;

    return *this;
}

ControllerState Controller::poll() const {
    if (gamepad_ == nullptr) {
        throw std::runtime_error(
            "Cannot poll a closed controller."
        );
    }

    /*
     * Pump pending events so SDL refreshes the current controller state.
     */
    SDL_PumpEvents();

    return ControllerState{
        .left_x = normalize_stick(
            SDL_GetGamepadAxis(
                gamepad_,
                SDL_GAMEPAD_AXIS_LEFTX
            ),
            stick_deadzone_
        ),
        .left_y = normalize_stick(
            SDL_GetGamepadAxis(
                gamepad_,
                SDL_GAMEPAD_AXIS_LEFTY
            ),
            stick_deadzone_
        ),
        .right_x = normalize_stick(
            SDL_GetGamepadAxis(
                gamepad_,
                SDL_GAMEPAD_AXIS_RIGHTX
            ),
            stick_deadzone_
        ),
        .right_y = normalize_stick(
            SDL_GetGamepadAxis(
                gamepad_,
                SDL_GAMEPAD_AXIS_RIGHTY
            ),
            stick_deadzone_
        ),
        .left_trigger = normalize_trigger(
            SDL_GetGamepadAxis(
                gamepad_,
                SDL_GAMEPAD_AXIS_LEFT_TRIGGER
            )
        ),
        .right_trigger = normalize_trigger(
            SDL_GetGamepadAxis(
                gamepad_,
                SDL_GAMEPAD_AXIS_RIGHT_TRIGGER
            )
        ),
        .south = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_SOUTH
        ),
        .east = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_EAST
        ),
        .west = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_WEST
        ),
        .north = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_NORTH
        ),
        .left_bumper = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_LEFT_SHOULDER
        ),
        .right_bumper = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
        ),
        .back = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_BACK
        ),
        .start = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_START
        ),
        .left_stick = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_LEFT_STICK
        ),
        .right_stick = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_RIGHT_STICK
        ),
        .dpad_up = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_DPAD_UP
        ),
        .dpad_down = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_DPAD_DOWN
        ),
        .dpad_left = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_DPAD_LEFT
        ),
        .dpad_right = SDL_GetGamepadButton(
            gamepad_,
            SDL_GAMEPAD_BUTTON_DPAD_RIGHT
        ),
    };
}

std::string Controller::name() const {
    if (gamepad_ == nullptr) {
        return "Disconnected controller";
    }

    const char* gamepad_name = SDL_GetGamepadName(gamepad_);

    return gamepad_name == nullptr
        ? "Unknown controller"
        : gamepad_name;
}

bool Controller::connected() const noexcept {
    return gamepad_ != nullptr &&
           SDL_GamepadConnected(gamepad_);
}

float Controller::normalize_stick(
    const Sint16 value,
    const float deadzone
) noexcept {
    /*
     * Sint16 reaches -32768 on the negative side but only 32767
     * on the positive side, so handle each side separately.
     */
    const float normalized = value >= 0
        ? static_cast<float>(value) / 32767.0f
        : static_cast<float>(value) / 32768.0f;

    if (std::abs(normalized) <= deadzone) {
        return 0.0f;
    }

    /*
     * Remove the dead-zone portion and rescale the remaining
     * range so the final output can still reach -1 and +1.
     */
    const float magnitude =
        (std::abs(normalized) - deadzone) /
        (1.0f - deadzone);

    return std::copysign(magnitude, normalized);
}

float Controller::normalize_trigger(
    const Sint16 value
) noexcept {
    return std::clamp(
        static_cast<float>(value) / 32767.0f,
        0.0f,
        1.0f
    );
}

void Controller::close() noexcept {
    if (gamepad_ != nullptr) {
        SDL_CloseGamepad(gamepad_);
        gamepad_ = nullptr;

        SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
    }
}

}  // namespace ai_player::controller