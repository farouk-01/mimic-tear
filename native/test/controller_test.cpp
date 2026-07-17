#include "controller/controller.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <exception>
#include <iostream>

using ai_player::controller::Controller;

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    try {
        Controller controller;

        std::cout
            << "Connected: "
            << controller.name()
            << '\n'
            << "Press Ctrl+C to stop."
            << '\n';

        while (controller.connected()) {
            const auto state = controller.poll();

            std::cout
                << '\r'
                << state.to_string()
                << "          "
                << std::flush;

            SDL_Delay(16);
        }

        std::cout << "\nController disconnected.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Controller test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}