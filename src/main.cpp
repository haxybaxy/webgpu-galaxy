#include "camera.hpp"
#include "config.hpp"
#include "diagnostics.hpp"
#include "exporter.hpp"
#include "graphics.hpp"
#include "gui.hpp"
#include "particle_renderer.hpp"
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

            const StepTiming &timing = simulation_.getLastTiming();
            gui_.setTreeBuildMs(timing.treeBuildMs);
            gui_.setForceMs(timing.forceMs);
            gui_.setIntegrateMs(timing.integrateMs);

            bool useGpuTree =
                simulation_.getTreeMethod() == TreeMethod::GPU &&
                simulation_.getIntegrator() == Integrator::Leapfrog;

            bool shouldDiag = !useGpuTree ||
                              (stepCount_ % diagInterval_ == 0);

            if (shouldDiag) {
                bool computePotential = (params.numParticles <= 5000);
                if (useGpuTree) {
                    std::vector<glm::vec4> pos, vel;
                    simulation_.readbackState(device_, queue_, pos, vel);
                    lastDiag_ = diagnostics_.compute(
                        pos, vel, params.softening, computePotential);
                } else {
                    lastDiag_ = diagnostics_.compute(
                        simulation_.getCpuPositions(),
                        simulation_.getCpuVelocities(),
                        params.softening, computePotential);
                }
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
        wgpuPollEvents(device_, true);

        // Check step limit for interactive mode
        if (config_.maxSteps > 0 && stepCount_ >= config_.maxSteps) {
            spdlog::info("Reached step limit ({})", config_.maxSteps);
            running_ = false;
        }
    }

    bool isRunning() const { return running_; }

    ~Application() {
        spdlog::info("Cleaning up...");
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

    DiagnosticsCalculator diagnostics;
    Exporter exporter;
    if (!config.exportPath.empty()) {
        if (!exporter.open(config.exportPath)) {
            spdlog::error("Failed to open export file: {}", config.exportPath);
        }
    }

    double simTime = 0.0;
    bool computePotential = (config.numParticles <= 5000);
    bool useGpuTree = config.treeMethod == TreeMethod::GPU &&
                      config.integrator == Integrator::Leapfrog;
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

        const StepTiming &timing = simulation.getLastTiming();

        bool isProgressStep = config.maxSteps >= 10 &&
                              (step + 1) % (config.maxSteps / 10) == 0;
        bool shouldDiag = !useGpuTree ||
                          ((step + 1) % diagInterval == 0) ||
                          isProgressStep;

        if (shouldDiag) {
            if (useGpuTree) {
                std::vector<glm::vec4> pos, vel;
                simulation.readbackState(device, queue, pos, vel);
                lastDiag = diagnostics.compute(pos, vel,
                                               config.softening,
                                               computePotential);
            } else {
                lastDiag = diagnostics.compute(
                    simulation.getCpuPositions(),
                    simulation.getCpuVelocities(),
                    config.softening, computePotential);
            }
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
    Diagnostics final_diag;
    if (useGpuTree) {
        std::vector<glm::vec4> pos, vel;
        simulation.readbackState(device, queue, pos, vel);
        final_diag = diagnostics.compute(pos, vel,
                                          config.softening, computePotential);
    } else {
        final_diag = diagnostics.compute(
            simulation.getCpuPositions(), simulation.getCpuVelocities(),
            config.softening, computePotential);
    }
    spdlog::info("Finished. Final energy drift: {:.6e}",
                 final_diag.energyDrift);
    spdlog::info("Final |momentum|: {:.6e}", final_diag.momentumMagnitude);

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
