cmake_minimum_required(VERSION 3.25)

project(
    ai_player_native
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    Python
    REQUIRED
    COMPONENTS Interpreter Development.Module
)

execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        -m
        pybind11
        --cmakedir
    OUTPUT_VARIABLE
        PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(
    external/SDL
    EXCLUDE_FROM_ALL
)

add_library(
    ai_player_controller
    STATIC
    src/controller.cpp
    src/controller_state.cpp
    src/virtual_controller.cpp
)

target_include_directories(
    ai_player_controller
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    ai_player_controller
    PUBLIC
        SDL3::SDL3
)

target_compile_features(
    ai_player_controller
    PUBLIC
        cxx_std_20
)

if(MSVC)
    target_compile_options(
        ai_player_controller
        PRIVATE
            /W4
            /permissive-
    )
else()
    target_compile_options(
        ai_player_controller
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

add_executable(
    controller_test
    test/controller_test.cpp
)

target_include_directories(
    controller_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    controller_test
    PRIVATE
        ai_player_controller
)

add_executable(
    virtual_controller_test
    test/virtual_controller_test.cpp
)

target_link_libraries(
    virtual_controller_test
    PRIVATE
        ai_player_controller
)

enable_testing()
add_test(
    NAME virtual_controller_test
    COMMAND virtual_controller_test
)

pybind11_add_module(
    ai_controller
    MODULE
    src/python_bindings.cpp
)

target_link_libraries(
    ai_controller
    PRIVATE
        ai_player_controller
)

add_custom_command(
    TARGET ai_controller
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL3::SDL3>
        $<TARGET_FILE_DIR:ai_controller>
)

set_target_properties(
    ai_player_controller
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
