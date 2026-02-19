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
    bool showOctree = false;
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
};
