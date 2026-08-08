#include "controller/controller_state.hpp"
#include "controller/virtual_controller.hpp"

#include <SDL3/SDL.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

using ai_player::controller::ControllerState;
using ai_player::controller::VirtualController;

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        VirtualController controller("AI Player Test Controller");
        require(controller.connected(), "Virtual controller is disconnected");

        SDL_Gamepad* gamepad =
            SDL_GetGamepadFromID(controller.instance_id());
        require(gamepad != nullptr, "SDL could not find the virtual gamepad");

        ControllerState state;
        state.left_x = 1.0f;
        state.left_y = -1.0f;
        state.right_trigger = 1.0f;
        state.south = true;
        state.dpad_left = true;
        controller.apply(state);

        require(
            SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTX) ==
                std::numeric_limits<Sint16>::max(),
            "Positive stick endpoint was not applied"
        );
        require(
            SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTY) ==
                std::numeric_limits<Sint16>::min(),
            "Negative stick endpoint was not applied"
        );
        require(
            SDL_GetGamepadAxis(
                gamepad,
                SDL_GAMEPAD_AXIS_RIGHT_TRIGGER
            ) == std::numeric_limits<Sint16>::max(),
            "Trigger endpoint was not applied"
        );
        require(
            SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_SOUTH),
            "South button was not applied"
        );
        require(
            SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_DPAD_LEFT),
            "D-pad button was not applied"
        );

        controller.reset();
        require(
            SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTX) == 0,
            "Reset did not center the stick"
        );
        require(
            SDL_GetGamepadAxis(
                gamepad,
                SDL_GAMEPAD_AXIS_RIGHT_TRIGGER
            ) == 0,
            "Reset did not release the trigger"
        );
        require(
            !SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_SOUTH),
            "Reset did not release the south button"
        );

        std::cout << "SDL3 virtual controller test passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "SDL3 virtual controller test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
