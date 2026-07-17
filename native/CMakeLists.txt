cmake_minimum_required(VERSION 3.25)

project(
    ai_player_native
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(
    external/SDL
    EXCLUDE_FROM_ALL
)

add_library(
    ai_player_controller
    STATIC
    src/controller.cpp
    src/controller_state.cpp
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

target_link_libraries(
    controller_test
    PRIVATE
        ai_player_controller
)

add_custom_command(
    TARGET controller_test
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL3::SDL3>
        $<TARGET_FILE_DIR:controller_test>
    COMMENT
        "Copying SDL3 runtime beside controller_test"
)