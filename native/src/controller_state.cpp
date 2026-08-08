#include "controller/controller_state.hpp"

#include <iomanip>
#include <sstream>

namespace ai_player::controller {

std::string ControllerState::to_string() const {
    std::ostringstream output;

    output
        << std::fixed
        << std::setprecision(2)

        << "L=("
        << std::showpos
        << left_x
        << ", "
        << left_y
        << ") "

        << "R=("
        << right_x
        << ", "
        << right_y
        << ") "

        << std::noshowpos
        << "LT="
        << left_trigger
        << " "
        << "RT="
        << right_trigger
        << " | "

        << "face["
        << "S=" << static_cast<int>(south) << " "
        << "E=" << static_cast<int>(east) << " "
        << "W=" << static_cast<int>(west) << " "
        << "N=" << static_cast<int>(north)
        << "] "

        << "bumper["
        << "L=" << static_cast<int>(left_bumper) << " "
        << "R=" << static_cast<int>(right_bumper)
        << "] "

        << "stick["
        << "L=" << static_cast<int>(left_stick) << " "
        << "R=" << static_cast<int>(right_stick)
        << "] "

        << "dpad["
        << "U=" << static_cast<int>(dpad_up) << " "
        << "D=" << static_cast<int>(dpad_down) << " "
        << "L=" << static_cast<int>(dpad_left) << " "
        << "R=" << static_cast<int>(dpad_right)
        << "] "

        << "menu["
        << "back=" << static_cast<int>(back) << " "
        << "start=" << static_cast<int>(start)
        << "]";

    return output.str();
}

}  // namespace ai_player::controller