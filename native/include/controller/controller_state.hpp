#pragma once

#include <string>

namespace ai_player::controller {

struct ControllerState {
    float left_x = 0.0f;
    float left_y = 0.0f;

    float right_x = 0.0f;
    float right_y = 0.0f;

    float left_trigger = 0.0f;
    float right_trigger = 0.0f;

    bool south = false;
    bool east = false;
    bool west = false;
    bool north = false;

    bool left_bumper = false;
    bool right_bumper = false;

    bool back = false;
    bool start = false;

    bool left_stick = false;
    bool right_stick = false;

    bool dpad_up = false;
    bool dpad_down = false;
    bool dpad_left = false;
    bool dpad_right = false;

    [[nodiscard]] std::string to_string() const;
};

}  // namespace ai_player::controller