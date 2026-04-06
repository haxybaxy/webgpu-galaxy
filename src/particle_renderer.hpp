#pragma once

#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <webgpu/webgpu.h>

class ParticleRenderer {
public:
    void initialize(WGPUDevice device, WGPUTextureFormat surfaceFormat,
                    int width, int height);

    void render(WGPUDevice device, WGPUQueue queue, WGPUTextureView targetView,
                WGPUBuffer positionBuffer, WGPUBuffer colorBuffer,
                int particleCount,
                const glm::mat4 &viewMatrix, const glm::mat4 &projMatrix);

    void cleanup();

private:
    void createDepthTexture(WGPUDevice device, int width, int height);

    WGPURenderPipeline pipeline_ = nullptr;
    WGPUBindGroupLayout bindGroupLayout_ = nullptr;
    wgpu_utils::Buffer uniformBuffer_;
    WGPUTextureView depthTextureView_ = nullptr;
    WGPUTexture depthTexture_ = nullptr;
    WGPUTextureFormat surfaceFormat_;
    bool initialized_ = false;

    // Cached bind group
    WGPUBindGroup cachedBindGroup_ = nullptr;
    WGPUBuffer cachedPosBuffer_ = nullptr;
    WGPUBuffer cachedColorBuffer_ = nullptr;
    int cachedParticleCount_ = 0;
};
