# ImGui static library
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_wgpu.cpp)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
if(NOT EMSCRIPTEN)
    target_compile_definitions(imgui_lib PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_WGPU)
endif()
if(EMSCRIPTEN)
    target_compile_options(imgui_lib PRIVATE -Wno-error)
endif()
target_link_libraries(imgui_lib PUBLIC glfw webgpu glfw3webgpu)

# GLFW handling (native: FetchContent from root; Emscripten: interface lib)
if(EMSCRIPTEN)
    add_library(glfw INTERFACE)
    target_link_options(glfw INTERFACE -sUSE_GLFW=3)
else()
    find_package(OpenGL REQUIRED)
endif()

# wgpu_utils library
add_library(wgpu_utils wgpu_utils.cpp wgpu_utils.hpp)
target_link_libraries(wgpu_utils PUBLIC webgpu spdlog)

# graphics library
add_library(graphics_lib graphics.cpp graphics.hpp)
target_link_libraries(graphics_lib PUBLIC webgpu spdlog glfw glfw3webgpu glm::glm)
if(NOT EMSCRIPTEN)
    target_link_libraries(graphics_lib PUBLIC OpenGL::GL)
endif()

# camera library
add_library(camera_lib camera.cpp camera.hpp)
target_link_libraries(camera_lib PUBLIC glm::glm)

# gui library
add_library(gui_lib gui.cpp gui.hpp)
target_link_libraries(gui_lib PUBLIC wgpu_utils graphics_lib imgui_lib glm::glm)

# config library
add_library(config_lib config.cpp config.hpp)
target_link_libraries(config_lib PUBLIC spdlog)

# diagnostics library
add_library(diagnostics_lib diagnostics.cpp diagnostics.hpp)
target_link_libraries(diagnostics_lib PUBLIC glm::glm)

# exporter library
add_library(exporter_lib exporter.cpp exporter.hpp)
target_link_libraries(exporter_lib PUBLIC diagnostics_lib)

# octree library
add_library(octree_lib octree.cpp octree.hpp)
target_link_libraries(octree_lib PUBLIC glm::glm)

# gpu tree builder library
add_library(gpu_tree_builder_lib gpu_tree_builder.cpp gpu_tree_builder.hpp)
target_link_libraries(gpu_tree_builder_lib PUBLIC wgpu_utils glm::glm spdlog)

# simulation library
add_library(simulation_lib simulation.cpp simulation.hpp)
target_link_libraries(simulation_lib PUBLIC wgpu_utils octree_lib gpu_tree_builder_lib config_lib exporter_lib glm::glm spdlog)

# particle renderer library
add_library(particle_renderer_lib particle_renderer.cpp particle_renderer.hpp)
target_link_libraries(particle_renderer_lib PUBLIC wgpu_utils glm::glm spdlog)

# Main executable
add_executable(galaxysim main.cpp)
set_target_properties(galaxysim PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR OFF)

target_copy_webgpu_binaries(galaxysim)

target_link_libraries(galaxysim PRIVATE
    wgpu_utils
    graphics_lib
    camera_lib
    gui_lib
    octree_lib
    gpu_tree_builder_lib
    simulation_lib
    particle_renderer_lib
    config_lib
    diagnostics_lib
    exporter_lib
    imgui_lib
    glm::glm
    spdlog)

if(EMSCRIPTEN)
    set_target_properties(galaxysim PROPERTIES SUFFIX ".html")
    target_link_options(galaxysim PRIVATE
        -sASYNCIFY
        -sALLOW_MEMORY_GROWTH=1
        -sFORCE_FILESYSTEM=1
        -sUSE_GLFW=3)
endif()
