cmake_minimum_required(VERSION 3.15)
project(GalaxySim C CXX)

set(WEBGPU_BACKEND "WGPU" CACHE STRING "Choose WebGPU backend: WGPU, DAWN, EMSCRIPTEN, or EMDAWNWEBGPU")
option(WEBGPU_BUILD_FROM_SOURCE "Build WebGPU distribution from source" OFF)

if (EMSCRIPTEN)
    set(CMAKE_EXE_LINKER_FLAGS "")
    set(CMAKE_SHARED_LINKER_FLAGS "")
    set(CMAKE_MODULE_LINKER_FLAGS "")
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_C_FLAGS "")
endif()

include(FetchContent)

FetchContent_Declare(WebGPU-distribution
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
    GIT_TAG main-v0.2.0
)
# Dawn workarounds
if (WEBGPU_BACKEND STREQUAL "DAWN")
    # Prevent building libprotobuf-mutator (needs protobuf)
    add_library(libprotobuf-mutator INTERFACE)
    # Fix Abseil passing x86 SSE flags on arm64 Apple builds
    if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ABSL_RANDOM_RANDEN_COPTS "-march=armv8-a+crypto" CACHE STRING "" FORCE)
    endif()
endif()
FetchContent_MakeAvailable(WebGPU-distribution)

# GLFW must be available before glfw3webgpu
if(NOT EMSCRIPTEN AND NOT TARGET glfw)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG 3.4
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

FetchContent_Declare(glfw3webgpu
    GIT_REPOSITORY https://github.com/eliemichel/glfw3webgpu/
    GIT_TAG v1.2.0
)
FetchContent_MakeAvailable(glfw3webgpu)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.90.9
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.2
)
FetchContent_MakeAvailable(glm)

add_subdirectory(src)
