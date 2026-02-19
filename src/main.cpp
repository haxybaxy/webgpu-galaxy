#include "camera.hpp"
#include "graphics.hpp"
#include "gui.hpp"
#include "particle_renderer.hpp"
#include "simulation.hpp"
#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace wgpu_utils;
using namespace graphics;

class Application {
    Window *window_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    WGPUSurface surface_ = nullptr;
    WGPUTextureFormat surfaceFormat_;

    Camera camera_;
    GUI gui_;
    ParticleRenderer renderer_;
    Simulation simulation_;

    bool running_ = false;
    double lastTime_ = 0.0;
    float fpsAccum_ = 0.0f;
    int fpsFrames_ = 0;
    float currentFPS_ = 0.0f;
    int lastParticleCount_ = 0;

public:
    void initialize() {
        auto [instance, device, adapter] = initializeWebGPU();
        device_ = device;
        queue_ = wgpuDeviceGetQueue(device_);
        window_ = initializeWindow(1280, 720, "Galaxy Simulation - Barnes-Hut N-Body");
        running_ = true;

        std::tie(surface_, surfaceFormat_) =
            createAndConfigureSurface(instance, adapter, device_, window_);

        auto [fbw, fbh] = getFramebufferSize(window_);
        renderer_.initialize(device_, surfaceFormat_, fbw, fbh);

        SimParams &params = gui_.getParams();
        simulation_.initialize(device_, params.numParticles);
        lastParticleCount_ = params.numParticles;

        gui_.initialize(device_, surfaceFormat_, window_);

        camera_.setDistance(80.0f);
        camera_.setTarget(glm::vec3(0.0f, 0.0f, 0.0f));

        setMouseMoveCallback(window_, [this](double x, double y) {
            camera_.onMouseMove(x, y);
        });
        setScrollCallback(window_, [this](double /*xoff*/, double yoff) {
            camera_.onScroll(yoff);
        });
        setMouseButtonCallback(window_, [this](int button, int action, int mods) {
            camera_.onMouseButton(button, action, mods);
        });

        lastTime_ = getTimeSeconds();
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        spdlog::info("Application initialized");
    }

    void loop() {
        double now = getTimeSeconds();
        float dt = static_cast<float>(now - lastTime_);
        lastTime_ = now;

        if (windowShouldClose(window_)) {
            running_ = false;
            return;
        }

        pollEvents();

        // Camera input
        camera_.update(dt,
            isKeyPressed(window_, Key::W),
            isKeyPressed(window_, Key::A),
            isKeyPressed(window_, Key::S),
            isKeyPressed(window_, Key::D),
            isKeyPressed(window_, Key::Space),
            isKeyPressed(window_, Key::LeftShift));

        SimParams &params = gui_.getParams();

        // Handle particle count change
        if (params.numParticles != lastParticleCount_) {
            simulation_.reinitialize(device_, params.numParticles);
            lastParticleCount_ = params.numParticles;
        }

        // Simulation step
        bool shouldStep = !params.paused || params.stepOnce;
        if (shouldStep) {
            simulation_.step(device_, queue_, params.dt, params.softening, params.theta);
            params.stepOnce = false;
        }

        // FPS counter
        fpsAccum_ += dt;
        fpsFrames_++;
        if (fpsAccum_ >= 1.0f) {
            currentFPS_ = static_cast<float>(fpsFrames_) / fpsAccum_;
            fpsAccum_ = 0.0f;
            fpsFrames_ = 0;
        }

        gui_.setFPS(currentFPS_);
        gui_.setParticleCount(simulation_.getParticleCount());
        gui_.setNodeCount(simulation_.getNodeCount());
        gui_.beginFrame(dt);

        auto [surfaceTexture, targetView] = getNextSurfaceViewData(surface_);
        if (!targetView) return;

        // Get aspect ratio
        auto [winW, winH] = getWindowSize(window_);
        float aspect = (winH > 0) ? static_cast<float>(winW) / static_cast<float>(winH) : 1.0f;

        renderer_.render(device_, queue_, targetView,
                        simulation_.getPositionBuffer(),
                        simulation_.getColorBuffer(),
                        simulation_.getParticleCount(),
                        camera_.getViewMatrix(),
                        camera_.getProjectionMatrix(aspect));

        gui_.render(targetView);

        wgpuTextureViewRelease(targetView);
#ifndef __EMSCRIPTEN__
        wgpuSurfacePresent(surface_);
#endif
#ifdef WEBGPU_BACKEND_WGPU
        wgpuTextureRelease(surfaceTexture.texture);
#endif
        wgpuPollEvents(device_, false);
    }

    bool isRunning() const { return running_; }

    ~Application() {
        spdlog::info("Cleaning up...");
        gui_.shutdown();
        renderer_.cleanup();
        running_ = false;
        if (surface_) {
            wgpuSurfaceUnconfigure(surface_);
            wgpuSurfaceRelease(surface_);
        }
        if (window_) {
            glfwDestroyWindow(window_);
            glfwTerminate();
        }
        if (queue_) wgpuQueueRelease(queue_);
        if (device_) wgpuDeviceRelease(device_);
    }
};

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--verbose") {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    Application app;
    app.initialize();

#ifdef __EMSCRIPTEN__
    auto callback = [](void *arg) {
        Application *pApp = reinterpret_cast<Application *>(arg);
        pApp->loop();
    };
    emscripten_set_main_loop_arg(callback, &app, 0, true);
#else
    while (app.isRunning()) {
        app.loop();
    }
#endif
    return 0;
}
