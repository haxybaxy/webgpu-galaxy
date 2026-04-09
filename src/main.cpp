#include "camera.hpp"
#include "config.hpp"
#include "diagnostics.hpp"
#include "exporter.hpp"
#include "graphics.hpp"
#include "gui.hpp"
#include "particle_renderer.hpp"
#include "screenshot.hpp"
#include "simulation.hpp"
#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace wgpu_utils;
using namespace graphics;

// ============================================================
// Interactive (windowed) application
// ============================================================

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
    DiagnosticsCalculator diagnostics_;
    Exporter exporter_;
    ScreenshotCapture screenshotCapture_;
    Config config_;

    bool running_ = false;
    double lastTime_ = 0.0;
    float fpsAccum_ = 0.0f;
    int fpsFrames_ = 0;
    float currentFPS_ = 0.0f;
    int lastParticleCount_ = 0;
    int stepCount_ = 0;
    double simTime_ = 0.0;
    int diagInterval_ = 60;
    Diagnostics lastDiag_{};

public:
    void initialize(const Config &config) {
        config_ = config;

        auto [instance, device, adapter] = initializeWebGPU();
        device_ = device;
        queue_ = wgpuDeviceGetQueue(device_);
        window_ = initializeWindow(1280, 720,
                                   "Galaxy Simulation - Barnes-Hut N-Body");
        running_ = true;

        std::tie(surface_, surfaceFormat_) =
            createAndConfigureSurface(instance, adapter, device_, window_);

        auto [fbw, fbh] = getFramebufferSize(window_);
        renderer_.initialize(device_, surfaceFormat_, fbw, fbh);

        simulation_.initialize(device_, queue_, config_);
        lastParticleCount_ = config_.numParticles;

        // Initialize GUI with config values
        SimParams &params = gui_.getParams();
        params.dt = config_.dt;
        params.softening = config_.softening;
        params.theta = config_.theta;
        params.numParticles = config_.numParticles;

        gui_.initialize(device_, surfaceFormat_, window_);
        gui_.setScenarioName(scenarioName(config_.scenario));
        gui_.setIntegratorName(integratorName(config_.integrator));
        gui_.setTreeMethodName(treeMethodName(config_.treeMethod));
        gui_.setForceMethodName(forceMethodName(config_.forceMethod));

        camera_.setDistance(80.0f);
        camera_.setTarget(glm::vec3(0.0f, 0.0f, 0.0f));

        setMouseMoveCallback(window_, [this](double x, double y) {
            if (!ImGui::GetIO().WantCaptureMouse)
                camera_.onMouseMove(x, y);
        });
        setScrollCallback(window_, [this](double /*xoff*/, double yoff) {
            if (!ImGui::GetIO().WantCaptureMouse)
                camera_.onScroll(yoff);
        });
        setMouseButtonCallback(
            window_, [this](int button, int action, int mods) {
                if (!ImGui::GetIO().WantCaptureMouse)
                    camera_.onMouseButton(button, action, mods);
            });

        // Open exporter if path specified
        if (!config_.exportPath.empty()) {
            if (exporter_.open(config_.exportPath)) {
                spdlog::info("Exporting to {}", config_.exportPath);
            } else {
                spdlog::error("Failed to open export file: {}",
                              config_.exportPath);
            }
        }

        // Initialize screenshot capture if targets specified
        if (!config_.screenshotTimes.empty()) {
            screenshotCapture_.setTargets(config_.screenshotTimes);
            auto [fbw2, fbh2] = getFramebufferSize(window_);
            screenshotCapture_.initialize(device_, surfaceFormat_, fbw2, fbh2);
        }

        // Capture t=0 screenshot if requested
        if (!config_.screenshotTimes.empty() &&
            config_.screenshotTimes.front() == 0.0) {
            auto [fbw3, fbh3] = getFramebufferSize(window_);
            float aspect0 = (fbh3 > 0)
                ? static_cast<float>(fbw3) / static_cast<float>(fbh3)
                : 1.0f;
            screenshotCapture_.capture(
                device_, queue_, renderer_,
                simulation_.getPositionBuffer(),
                simulation_.getColorBuffer(),
                simulation_.getParticleCount(),
                camera_.getViewMatrix(),
                camera_.getProjectionMatrix(aspect0),
                0.0);
        }

        lastTime_ = getTimeSeconds();
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        spdlog::info("Application initialized");
    }

    void loop() {
        if (!running_) return;

        double now = getTimeSeconds();
        float dt = static_cast<float>(now - lastTime_);
        lastTime_ = now;

        if (windowShouldClose(window_)) {
            running_ = false;
            return;
        }

        pollEvents();

        // Camera input (skip when ImGui wants keyboard, e.g. typing in InputInt)
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            camera_.update(dt, isKeyPressed(window_, Key::W),
                           isKeyPressed(window_, Key::A),
                           isKeyPressed(window_, Key::S),
                           isKeyPressed(window_, Key::D),
                           isKeyPressed(window_, Key::Space),
                           isKeyPressed(window_, Key::LeftShift));
        }

        SimParams &params = gui_.getParams();

        // Handle particle count change
        if (params.numParticles != lastParticleCount_) {
            config_.numParticles = params.numParticles;
            simulation_.reinitialize(device_, queue_, config_);
            diagnostics_.reset();
            lastParticleCount_ = params.numParticles;
            stepCount_ = 0;
            simTime_ = 0.0;
        }

        // Simulation step
        bool shouldStep = !params.paused || params.stepOnce;
        if (shouldStep) {
            simulation_.step(device_, queue_, params.dt, params.softening,
                             params.theta);
            params.stepOnce = false;
            stepCount_++;
            simTime_ += params.dt;

            // Capture screenshots at target times
            while (screenshotCapture_.shouldCapture(simTime_, params.dt)) {
                auto [winW2, winH2] = getWindowSize(window_);
                float aspect2 = (winH2 > 0)
                    ? static_cast<float>(winW2) / static_cast<float>(winH2)
                    : 1.0f;
                screenshotCapture_.capture(
                    device_, queue_, renderer_,
                    simulation_.getPositionBuffer(),
                    simulation_.getColorBuffer(),
                    simulation_.getParticleCount(),
                    camera_.getViewMatrix(),
                    camera_.getProjectionMatrix(aspect2),
                    simTime_);
            }

            const StepTiming &timing = simulation_.getLastTiming();
            gui_.setTreeBuildMs(timing.treeBuildMs);
            gui_.setForceMs(timing.forceMs);
            gui_.setIntegrateMs(timing.integrateMs);

            bool shouldDiag = (stepCount_ % diagInterval_ == 0);

            if (shouldDiag) {
                bool computePotential = (params.numParticles <= 5000);
                std::vector<glm::vec4> pos, vel;
                simulation_.readbackState(device_, queue_, pos, vel);
                lastDiag_ = diagnostics_.compute(
                    pos, vel, params.softening, computePotential);
            }

            gui_.setKineticEnergy(lastDiag_.kineticEnergy);
            gui_.setPotentialEnergy(lastDiag_.potentialEnergy);
            gui_.setTotalEnergy(lastDiag_.totalEnergy);
            gui_.setEnergyDrift(lastDiag_.energyDrift);
            gui_.setMomentumMagnitude(lastDiag_.momentumMagnitude);

            // Export if open
            if (exporter_.isOpen()) {
                exporter_.writeRow(stepCount_, simTime_, lastDiag_, timing);
            }
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

        auto [winW, winH] = getWindowSize(window_);
        float aspect =
            (winH > 0) ? static_cast<float>(winW) / static_cast<float>(winH)
                       : 1.0f;

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

#ifdef __EMSCRIPTEN__
        // Flush GPU queue every 4 frames to prevent command buffer
        // accumulation in the browser
        if (stepCount_ > 0 && stepCount_ % 4 == 0) {
            wgpu_utils::flushGpuQueue(device_, queue_);
        } else {
            wgpuPollEvents(device_, true);
        }
#else
        wgpuPollEvents(device_, true);
#endif

        // Check step limit for interactive mode
        if (config_.maxSteps > 0 && stepCount_ >= config_.maxSteps) {
            spdlog::info("Reached step limit ({})", config_.maxSteps);
            running_ = false;
#ifdef __EMSCRIPTEN__
            emscripten_cancel_main_loop();
#endif
        }
    }

    bool isRunning() const { return running_; }

    ~Application() {
        spdlog::info("Cleaning up...");
        screenshotCapture_.cleanup();
        exporter_.close();
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

// ============================================================
// Headless (batch) mode
// ============================================================

static void runHeadless(const Config &config) {
    spdlog::info("Running headless: {} steps, {} particles, {}, {}",
                 config.maxSteps, config.numParticles,
                 scenarioName(config.scenario),
                 integratorName(config.integrator));

    auto [instance, device, adapter] = initializeWebGPU();
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    Simulation simulation;
    simulation.initialize(device, queue, config);

    // Screenshot support in headless mode
    bool hasScreenshots = !config.screenshotTimes.empty();
    ParticleRenderer renderer;
    ScreenshotCapture screenshotCapture;
    const uint32_t ssWidth = 1280, ssHeight = 720;
    WGPUTextureFormat ssFormat = WGPUTextureFormat_BGRA8Unorm;

    if (hasScreenshots) {
        renderer.initialize(device, ssFormat, ssWidth, ssHeight);
        screenshotCapture.setTargets(config.screenshotTimes);
        screenshotCapture.initialize(device, ssFormat, ssWidth, ssHeight);
    }

    // Default camera for headless screenshots
    auto headlessView = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 80.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));
    float headlessAspect = static_cast<float>(ssWidth) / static_cast<float>(ssHeight);
    auto headlessProj = glm::perspective(
        glm::radians(45.0f), headlessAspect, 0.1f, 1000.0f);

    DiagnosticsCalculator diagnostics;
    Exporter exporter;
    if (!config.exportPath.empty()) {
        if (!exporter.open(config.exportPath)) {
            spdlog::error("Failed to open export file: {}", config.exportPath);
        }
    }

    // Capture t=0 screenshot if requested
    if (hasScreenshots && config.screenshotTimes.front() == 0.0) {
        screenshotCapture.capture(
            device, queue, renderer,
            simulation.getPositionBuffer(),
            simulation.getColorBuffer(),
            simulation.getParticleCount(),
            headlessView, headlessProj, 0.0);
    }

    double simTime = 0.0;
    bool computePotential = (config.numParticles <= 5000);
    int diagInterval = exporter.isOpen() ? 1 : 50;
    Diagnostics lastDiag{};

    for (int step = 0; step < config.maxSteps; step++) {
        simulation.step(device, queue, config.dt, config.softening,
                        config.theta);
#ifdef __EMSCRIPTEN__
        // Flush GPU queue every 4 steps to prevent command buffer
        // accumulation that starves the macOS WindowServer compositor
        if ((step + 1) % 4 == 0) {
            wgpu_utils::flushGpuQueue(device, queue);
        } else {
            wgpuPollEvents(device, true);
        }
#else
        wgpuPollEvents(device, true);
#endif
        simTime += config.dt;

        // Capture screenshots at target times
        while (hasScreenshots &&
               screenshotCapture.shouldCapture(simTime, config.dt)) {
            screenshotCapture.capture(
                device, queue, renderer,
                simulation.getPositionBuffer(),
                simulation.getColorBuffer(),
                simulation.getParticleCount(),
                headlessView, headlessProj, simTime);
        }

        const StepTiming &timing = simulation.getLastTiming();

        bool isProgressStep = config.maxSteps >= 10 &&
                              (step + 1) % (config.maxSteps / 10) == 0;
        bool shouldDiag = ((step + 1) % diagInterval == 0) || isProgressStep;

        if (shouldDiag) {
            std::vector<glm::vec4> pos, vel;
            simulation.readbackState(device, queue, pos, vel);
            lastDiag = diagnostics.compute(pos, vel, config.softening,
                                           computePotential);
        }

        if (exporter.isOpen() && shouldDiag) {
            exporter.writeRow(step + 1, simTime, lastDiag, timing);
        }

        if (isProgressStep) {
            spdlog::info("Step {}/{} — E_drift={:.2e}, |P|={:.6f}, tree={:.3f}ms, force={:.3f}ms, integrate={:.3f}ms, total={:.3f}ms",
                         step + 1, config.maxSteps, lastDiag.energyDrift,
                         lastDiag.momentumMagnitude,
                         timing.treeBuildMs, timing.forceMs, timing.integrateMs,
                         timing.treeBuildMs + timing.forceMs + timing.integrateMs);
        }
    }

    exporter.close();

    // Final diagnostics
    std::vector<glm::vec4> finalPos, finalVel;
    simulation.readbackState(device, queue, finalPos, finalVel);
    Diagnostics final_diag = diagnostics.compute(
        finalPos, finalVel, config.softening, computePotential);
    spdlog::info("Finished. Final energy drift: {:.6e}",
                 final_diag.energyDrift);
    spdlog::info("Final |momentum|: {:.6e}", final_diag.momentumMagnitude);

    if (hasScreenshots) {
        screenshotCapture.cleanup();
        renderer.cleanup();
    }

    wgpuQueueRelease(queue);
    wgpuAdapterRelease(adapter);
    wgpuDeviceRelease(device);
    wgpuInstanceRelease(instance);
}

// ============================================================
// Entry Point
// ============================================================

int main(int argc, char **argv) {
    // Check for --verbose before parsing everything
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose") {
            spdlog::set_level(spdlog::level::debug);
            break;
        }
    }
    if (spdlog::get_level() != spdlog::level::debug) {
        spdlog::set_level(spdlog::level::info);
    }

    Config config = parseArgs(argc, argv);

    spdlog::info("Config: scenario={}, integrator={}, tree={}, force={}, N={}, "
                 "dt={}, softening={}, theta={}, seed={}",
                 scenarioName(config.scenario),
                 integratorName(config.integrator),
                 treeMethodName(config.treeMethod),
                 forceMethodName(config.forceMethod),
                 config.numParticles,
                 config.dt, config.softening, config.theta, config.seed);

    if (config.headless) {
        runHeadless(config);
        return 0;
    }

    Application app;
    app.initialize(config);

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
