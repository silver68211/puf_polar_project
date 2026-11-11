
cmake_minimum_required(VERSION 3.18)

# ============================================================
# 1. Project configuration
# ============================================================

project(
    puf_aff3ct
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Required when linking static libraries into
# the Python shared extension.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release unless the user explicitly
# selects another build configuration.
if(
    NOT CMAKE_BUILD_TYPE
    AND NOT CMAKE_CONFIGURATION_TYPES
)
    set(
        CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type" FORCE
    )
endif()


# ============================================================
# 2. Architecture-specific optimization
# ============================================================

# OFF by default for portability across HPC nodes.
#
# Enable only when the build and execution CPUs have
# compatible instruction sets.
#
# Example:
#   -DPUF_ENABLE_NATIVE_ARCH=ON

option(
    PUF_ENABLE_NATIVE_ARCH
    "Enable architecture-specific compiler optimizations"
    OFF
)


# ============================================================
# 3. Python and pybind11
# ============================================================

# Use the Python interpreter from the active Conda
# environment when configuring the project.

find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

find_package(
    pybind11
    CONFIG
    REQUIRED
)


# ============================================================
# 4. AFF3CT and threading dependencies
# ============================================================

find_package(
    Threads
    REQUIRED
)

find_package(
    aff3ct
    CONFIG
    REQUIRED
)

# AFF3CT exports different CMake targets depending
# on whether it was compiled as a shared or static library.
#
# Prefer the shared library when available.
# Otherwise use the static library.

if(TARGET aff3ct::aff3ct-shared-lib)

    set(
        PUF_AFF3CT_TARGET
        aff3ct::aff3ct-shared-lib
    )

elseif(TARGET aff3ct::aff3ct-static-lib)

    set(
        PUF_AFF3CT_TARGET
        aff3ct::aff3ct-static-lib
    )

else()

    message(FATAL_ERROR
        "No supported AFF3CT library target was found. "
        "Build and install AFF3CT with static or shared "
        "library support."
    )

endif()


# ============================================================
# 5. Build Python extension
# ============================================================

# sc_bindings.cpp:
#
#   - PolarSC
#   - PolarDSCF
#   - PolarSCLPUF
#   - BSCMonteCarloPolar
#
# bsc_monte_carlo.cpp:
#
#   - Legacy BSCMonteCarlo (SC-only)
#
# Other C++ components are header-based and included
# through sc_bindings.cpp.

pybind11_add_module(
    puf_aff3ct

    sc_bindings.cpp
    bsc_monte_carlo.cpp
)


# ============================================================
# 6. Link libraries
# ============================================================

target_link_libraries(
    puf_aff3ct

    PRIVATE

    ${PUF_AFF3CT_TARGET}
    Threads::Threads
)


# ============================================================
# 7. Compiler optimizations
# ============================================================

# Release builds use the compiler's normal optimized
# configuration, typically -O3 for GCC/Clang.
#
# Architecture-specific optimization is optional.

if(PUF_ENABLE_NATIVE_ARCH)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

        target_compile_options(
            puf_aff3ct
            PRIVATE
            -march=native
        )

    else()

        message(WARNING
            "Native architecture optimization is not "
            "configured for this compiler."
        )

    endif()

endif()


# ============================================================
# 8. Output directory
# ============================================================

# Place the importable Python module in the project root:
#
# puf_polar_project/
#   puf_aff3ct.cpython-310-x86_64-linux-gnu.so
#   sim_aff3ct.py
#   config.py
#   cpp/
#
# This allows:
#
#   from puf_aff3ct import BSCMonteCarloPolar

set_target_properties(
    puf_aff3ct

    PROPERTIES

    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
)


# ============================================================
# 9. Configuration summary
# ============================================================

message(STATUS "========================================")
message(STATUS "PUF-AFF3CT configuration")
message(STATUS "========================================")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")

message(STATUS "AFF3CT target: ${PUF_AFF3CT_TARGET}")

message(STATUS "Native CPU optimization: ${PUF_ENABLE_NATIVE_ARCH}")

message(STATUS "========================================")
