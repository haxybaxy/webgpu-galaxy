#include "graphics.hpp"
#include <spdlog/spdlog.h>

namespace graphics {

namespace {
    MouseMoveCallback g_mouseMoveCallback;
    ScrollCallback g_scrollCallback;
    MouseButtonCallback g_mouseButtonCallback;
}

std::pair<WGPUSurfaceTexture, WGPUTextureView>
getNextSurfaceViewData(WGPUSurface surface) {
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
    if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_Success) {
        return {surfaceTexture, nullptr};
    }
    WGPUTextureViewDescriptor viewDescriptor;
    viewDescriptor.nextInChain = nullptr;
    viewDescriptor.label = "Surface texture view";
    viewDescriptor.format = wgpuTextureGetFormat(surfaceTexture.texture);
    viewDescriptor.dimension = WGPUTextureViewDimension_2D;
    viewDescriptor.baseMipLevel = 0;
    viewDescriptor.mipLevelCount = 1;
    viewDescriptor.baseArrayLayer = 0;
    viewDescriptor.arrayLayerCount = 1;
    viewDescriptor.aspect = WGPUTextureAspect_All;
    WGPUTextureView targetView =
        wgpuTextureCreateView(surfaceTexture.texture, &viewDescriptor);
    return {surfaceTexture, targetView};
}

std::tuple<WGPUSurface, WGPUTextureFormat>
createAndConfigureSurface(WGPUInstance instance, WGPUAdapter adapter,
                          WGPUDevice device, Window *window) {
    WGPUSurface surface = glfwGetWGPUSurface(instance, window);
    spdlog::info("Created WGPU surface: {:#x}", size_t(surface));
    WGPUSurfaceConfiguration config = {};
    config.nextInChain = nullptr;
    int fbw, fbh;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    config.width = fbw;
    config.height = fbh;
    WGPUTextureFormat surfaceFormat = wgpuSurfaceGetPreferredFormat(surface, adapter);
    config.format = surfaceFormat;
    config.viewFormatCount = 0;
    config.viewFormats = nullptr;
    config.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
    config.device = device;
    config.presentMode = WGPUPresentMode_Fifo;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(surface, &config);
    return std::make_tuple(surface, surfaceFormat);
}

Window *initializeWindow(int width, int height, const char *title) {
    spdlog::debug("Initializing GLFW...");
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetCursorPosCallback(window, [](GLFWwindow *, double x, double y) {
        if (g_mouseMoveCallback) g_mouseMoveCallback(x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow *, double xoff, double yoff) {
        if (g_scrollCallback) g_scrollCallback(xoff, yoff);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow *, int button, int action, int mods) {
        if (g_mouseButtonCallback) g_mouseButtonCallback(button, action, mods);
    });

    spdlog::info("GLFW window created: {}x{}", width, height);
    return window;
}

void pollEvents() { glfwPollEvents(); }

bool windowShouldClose(Window *window) { return glfwWindowShouldClose(window); }

double getTimeSeconds() { return glfwGetTime(); }

bool isKeyPressed(Window *window, Key key) {
    int glfwKey = GLFW_KEY_UNKNOWN;
    switch (key) {
    case Key::W: glfwKey = GLFW_KEY_W; break;
    case Key::A: glfwKey = GLFW_KEY_A; break;
    case Key::S: glfwKey = GLFW_KEY_S; break;
    case Key::D: glfwKey = GLFW_KEY_D; break;
    case Key::Space: glfwKey = GLFW_KEY_SPACE; break;
    case Key::LeftShift: glfwKey = GLFW_KEY_LEFT_SHIFT; break;
    }
    return glfwGetKey(window, glfwKey) == GLFW_PRESS;
}

std::pair<int, int> getWindowSize(Window *window) {
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    return {w, h};
}

std::pair<int, int> getFramebufferSize(Window *window) {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return {w, h};
}

void setMouseMoveCallback(Window *, MouseMoveCallback callback) {
    g_mouseMoveCallback = std::move(callback);
}

void setScrollCallback(Window *, ScrollCallback callback) {
    g_scrollCallback = std::move(callback);
}

void setMouseButtonCallback(Window *, MouseButtonCallback callback) {
    g_mouseButtonCallback = std::move(callback);
}

} // namespace graphics
