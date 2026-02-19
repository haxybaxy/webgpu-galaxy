#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#ifdef WEBGPU_BACKEND_WGPU
#include <webgpu/wgpu.h>
#endif
#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include <functional>
#include <tuple>
#include <utility>
#include <webgpu/webgpu.h>

namespace graphics {

using Window = GLFWwindow;

enum class Key { W, A, S, D, Space, LeftShift };

using MouseMoveCallback = std::function<void(double xpos, double ypos)>;
using ScrollCallback = std::function<void(double xoffset, double yoffset)>;
using MouseButtonCallback = std::function<void(int button, int action, int mods)>;

std::pair<WGPUSurfaceTexture, WGPUTextureView>
getNextSurfaceViewData(WGPUSurface surface);

std::tuple<WGPUSurface, WGPUTextureFormat>
createAndConfigureSurface(WGPUInstance instance, WGPUAdapter adapter,
                          WGPUDevice device, Window *window);

Window *initializeWindow(int width = 1280, int height = 720,
                         const char *title = "Galaxy Simulation");

void pollEvents();
bool windowShouldClose(Window *window);
double getTimeSeconds();
bool isKeyPressed(Window *window, Key key);
std::pair<int, int> getWindowSize(Window *window);
std::pair<int, int> getFramebufferSize(Window *window);

void setMouseMoveCallback(Window *window, MouseMoveCallback callback);
void setScrollCallback(Window *window, ScrollCallback callback);
void setMouseButtonCallback(Window *window, MouseButtonCallback callback);

} // namespace graphics
