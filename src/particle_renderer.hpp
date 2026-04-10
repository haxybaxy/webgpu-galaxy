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
                const glm::mat4 &viewMatrix, const glm::mat4 &projMatrix,
                bool showTrails = false, float trailLength = 0.92f,
                WGPUTexture targetTexture = nullptr);

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

    // Fade quad pipeline (for trails)
    WGPURenderPipeline fadePipeline_ = nullptr;
    WGPUBindGroupLayout fadeBindGroupLayout_ = nullptr;
    WGPUBindGroup fadeBindGroup_ = nullptr;
    wgpu_utils::Buffer fadeUniformBuffer_;

    // Ping-pong trail accumulation textures
    WGPUTexture trailTextures_[2] = {nullptr, nullptr};
    WGPUTextureView trailTextureViews_[2] = {nullptr, nullptr};
    int trailPingPong_ = 0;
    uint32_t trailWidth_ = 0;
    uint32_t trailHeight_ = 0;

    // Trail reprojection (camera-independent trails)
    WGPURenderPipeline reprojPipeline_ = nullptr;
    WGPUBindGroupLayout reprojBindGroupLayout_ = nullptr;
    WGPUBindGroup reprojBindGroups_[2] = {nullptr, nullptr};
    WGPUSampler reprojSampler_ = nullptr;
    wgpu_utils::Buffer reprojUniformBuffer_;
    glm::mat4 prevViewProj_ = glm::mat4(1.0f);
    bool hasPrevViewProj_ = false;
};
