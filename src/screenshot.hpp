#pragma once

#include "particle_renderer.hpp"
#include <vector>
#include <webgpu/webgpu.h>

class ScreenshotCapture {
public:
    void initialize(WGPUDevice device, WGPUTextureFormat format,
                    uint32_t width, uint32_t height);

    bool shouldCapture(double simTime, double dt) const;

    void capture(WGPUDevice device, WGPUQueue queue,
                 ParticleRenderer &renderer,
                 WGPUBuffer posBuffer, WGPUBuffer colorBuffer,
                 int particleCount,
                 const glm::mat4 &viewMatrix, const glm::mat4 &projMatrix,
                 double simTime);

    bool done() const { return nextTarget_ >= targets_.size(); }

    void setTargets(std::vector<double> targets);

    void cleanup();

private:
    WGPUTexture offscreenTexture_ = nullptr;
    WGPUTextureView offscreenView_ = nullptr;
    WGPUTextureFormat format_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<double> targets_;
    size_t nextTarget_ = 0;
    bool initialized_ = false;
};
