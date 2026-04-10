#pragma once

#include "graphics.hpp"
#include <webgpu/webgpu.h>

struct SimParams {
    float dt = 0.001f;
    float softening = 0.5f;
    float theta = 0.75f;
    int numParticles = 10000;
    bool paused = true;
    bool stepOnce = false;
    bool showTrails = false;
    float trailLength = 0.92f;
};

class GUI {
public:
    GUI();
    ~GUI();

    void initialize(WGPUDevice device, WGPUTextureFormat surfaceFormat,
                    graphics::Window *window);
    void beginFrame(float deltaTime);
    void render(WGPUTextureView targetView);
    void shutdown();

    SimParams &getParams() { return params_; }
    const SimParams &getParams() const { return params_; }

    void setFPS(float fps) { fps_ = fps; }
    void setParticleCount(int count) { particleCount_ = count; }
    void setNodeCount(int count) { nodeCount_ = count; }

    // Diagnostics display
    void setKineticEnergy(double ke) { kineticEnergy_ = ke; }
    void setPotentialEnergy(double pe) { potentialEnergy_ = pe; }
    void setTotalEnergy(double te) { totalEnergy_ = te; }
    void setEnergyDrift(double drift) { energyDrift_ = drift; }
    void setMomentumMagnitude(double mag) { momentumMag_ = mag; }

    // Timing display
    void setTreeBuildMs(double ms) { treeBuildMs_ = ms; }
    void setForceMs(double ms) { forceMs_ = ms; }
    void setIntegrateMs(double ms) { integrateMs_ = ms; }

    // Config display
    void setScenarioName(const char *name) { scenarioName_ = name; }
    void setForceMethodName(const char *name) { forceMethodName_ = name; }

private:
    void initImGui();
    void shutdownImGui();

    WGPUDevice device_ = nullptr;
    WGPUTextureFormat surfaceFormat_;
    graphics::Window *window_ = nullptr;
    bool imguiReady_ = false;

    SimParams params_;
    float fps_ = 0.0f;
    int particleCount_ = 0;
    int nodeCount_ = 0;

    // Diagnostics
    double kineticEnergy_ = 0.0;
    double potentialEnergy_ = 0.0;
    double totalEnergy_ = 0.0;
    double energyDrift_ = 0.0;
    double momentumMag_ = 0.0;

    // Timing
    double treeBuildMs_ = 0.0;
    double forceMs_ = 0.0;
    double integrateMs_ = 0.0;

    // Config info
    const char *scenarioName_ = "Unknown";
    const char *forceMethodName_ = "Unknown";
};
