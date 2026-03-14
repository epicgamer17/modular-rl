cmake_minimum_required(VERSION 3.18)
project(mcts_cpp_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(
    mcts_cpp_backend
    bindings.cpp
    backprop.cpp
    min_max_stats.cpp
    modular_search.cpp
    nodes.cpp
    scoring.cpp
    selection.cpp
)
target_compile_features(mcts_cpp_backend PRIVATE cxx_std_17)
