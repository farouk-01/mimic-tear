#pragma once

#include "controller/controller_state.hpp"

#include <SDL3/SDL_gamepad.h>

#include <string>

namespace mimic_tear::controller {

class Controller {
public:
    explicit Controller(float stick_deadzone = 0.12f);
    ~Controller();

    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    Controller(Controller&& other) noexcept;
    Controller& operator=(Controller&& other) noexcept;

    [[nodiscard]] ControllerState poll() const;

    [[nodiscard]] std::string name() const;
    [[nodiscard]] bool connected() const noexcept;

private:
    static float normalize_stick(
        Sint16 value,
        float deadzone
    ) noexcept;

    static float normalize_trigger(Sint16 value) noexcept;

    void close() noexcept;

    SDL_Gamepad* gamepad_ = nullptr;
    float stick_deadzone_ = 0.12f;
};

}  // namespace mimic_tear::controller