#include "controller/controller.hpp"
#include "controller/controller_state.hpp"
#include "controller/virtual_controller.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using ai_player::controller::Controller;
using ai_player::controller::ControllerState;
using ai_player::controller::VirtualController;

PYBIND11_MODULE(ai_controller, module) {
    module.doc() = "Native SDL3 physical and virtual controllers";

    py::class_<ControllerState>(module, "ControllerState")
        .def(py::init<>())
        .def_readwrite("left_x", &ControllerState::left_x)
        .def_readwrite("left_y", &ControllerState::left_y)
        .def_readwrite("right_x", &ControllerState::right_x)
        .def_readwrite("right_y", &ControllerState::right_y)
        .def_readwrite(
            "left_trigger",
            &ControllerState::left_trigger
        )
        .def_readwrite(
            "right_trigger",
            &ControllerState::right_trigger
        )
        .def_readwrite("south", &ControllerState::south)
        .def_readwrite("east", &ControllerState::east)
        .def_readwrite("west", &ControllerState::west)
        .def_readwrite("north", &ControllerState::north)
        .def_readwrite(
            "left_bumper",
            &ControllerState::left_bumper
        )
        .def_readwrite(
            "right_bumper",
            &ControllerState::right_bumper
        )
        .def_readwrite("back", &ControllerState::back)
        .def_readwrite("start", &ControllerState::start)
        .def_readwrite(
            "left_stick",
            &ControllerState::left_stick
        )
        .def_readwrite(
            "right_stick",
            &ControllerState::right_stick
        )
        .def_readwrite("dpad_up", &ControllerState::dpad_up)
        .def_readwrite(
            "dpad_down",
            &ControllerState::dpad_down
        )
        .def_readwrite(
            "dpad_left",
            &ControllerState::dpad_left
        )
        .def_readwrite(
            "dpad_right",
            &ControllerState::dpad_right
        )
        .def("__repr__", [](const ControllerState& state) {
            return state.to_string();
        });

    py::class_<Controller>(module, "Controller")
        .def(
            py::init<float>(),
            py::arg("stick_deadzone") = 0.12f
        )
        .def("poll", &Controller::poll)
        .def_property_readonly("name", &Controller::name)
        .def_property_readonly(
            "connected",
            &Controller::connected
        );

    py::class_<VirtualController>(module, "VirtualController")
        .def(
            py::init<std::string>(),
            py::arg("name") = "AI Player Virtual Controller"
        )
        .def("apply", &VirtualController::apply, py::arg("state"))
        .def("reset", &VirtualController::reset)
        .def_property_readonly("name", &VirtualController::name)
        .def_property_readonly(
            "connected",
            &VirtualController::connected
        )
        .def_property_readonly(
            "instance_id",
            &VirtualController::instance_id
        );
}
