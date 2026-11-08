
cmake_minimum_required(VERSION 3.18)

project(puf_aff3ct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find the Python/C++ binding library.
find_package(pybind11 CONFIG REQUIRED)

# Find AFF3CT's installed CMake package.
# Pass -Daff3ct_DIR=... when configuring.
find_package(Threads REQUIRED)
find_package(aff3ct CONFIG REQUIRED)

# Build importable Python extension.
pybind11_add_module(
    puf_aff3ct
    sc_bindings.cpp
    bsc_monte_carlo.cpp
)

target_link_libraries(
    puf_aff3ct
    PRIVATE
    aff3ct::aff3ct-shared-lib
    Threads::Threads
)

target_compile_options(
    puf_aff3ct
    PRIVATE
    -O3
    -march=native
)

# Put the compiled Python extension in the project root.
set_target_properties(
    puf_aff3ct
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
