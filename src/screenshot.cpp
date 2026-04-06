#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "screenshot.hpp"
#include "wgpu_utils.hpp"
#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

void ScreenshotCapture::setTargets(std::vector<double> targets) {
    targets_ = std::move(targets);
    std::sort(targets_.begin(), targets_.end());
    nextTarget_ = 0;
}

void ScreenshotCapture::initialize(WGPUDevice device, WGPUTextureFormat format,
                                   uint32_t width, uint32_t height) {
    format_ = format;
    width_ = width;
    height_ = height;

    WGPUTextureDescriptor desc{};
    desc.dimension = WGPUTextureDimension_2D;
    desc.format = format;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.size = {width, height, 1};
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    desc.viewFormatCount = 0;
    offscreenTexture_ = wgpuDeviceCreateTexture(device, &desc);

    WGPUTextureViewDescriptor viewDesc{};
    viewDesc.aspect = WGPUTextureAspect_All;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = 1;
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = 1;
    viewDesc.dimension = WGPUTextureViewDimension_2D;
    viewDesc.format = format;
    offscreenView_ = wgpuTextureCreateView(offscreenTexture_, &viewDesc);

    initialized_ = true;
    spdlog::info("Screenshot capture initialized ({}x{}, {} targets)",
                 width, height, targets_.size());
}

bool ScreenshotCapture::shouldCapture(double simTime, double dt) const {
    if (!initialized_ || nextTarget_ >= targets_.size()) return false;
    double target = targets_[nextTarget_];
    double prevTime = simTime - dt;
    return prevTime < target && simTime >= target;
}

void ScreenshotCapture::capture(WGPUDevice device, WGPUQueue queue,
                                ParticleRenderer &renderer,
                                WGPUBuffer posBuffer, WGPUBuffer colorBuffer,
                                int particleCount,
                                const glm::mat4 &viewMatrix,
                                const glm::mat4 &projMatrix,
                                double simTime) {
    if (!initialized_) return;

    // Render particles to offscreen texture
    renderer.render(device, queue, offscreenView_,
                    posBuffer, colorBuffer, particleCount,
                    viewMatrix, projMatrix);

    // Calculate row alignment (WebGPU requires bytesPerRow multiple of 256)
    uint32_t bytesPerPixel = 4;
    uint32_t unalignedBytesPerRow = width_ * bytesPerPixel;
    uint32_t bytesPerRow = (unalignedBytesPerRow + 255) & ~255u;
    uint32_t bufferSize = bytesPerRow * height_;

    // Create staging buffer
    WGPUBufferDescriptor stagingDesc{};
    stagingDesc.size = bufferSize;
    stagingDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    stagingDesc.mappedAtCreation = false;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &stagingDesc);

    // Copy texture to staging buffer
    WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);

    WGPUImageCopyTexture srcInfo{};
    srcInfo.texture = offscreenTexture_;
    srcInfo.mipLevel = 0;
    srcInfo.origin = {0, 0, 0};
    srcInfo.aspect = WGPUTextureAspect_All;

    WGPUImageCopyBuffer dstInfo{};
    dstInfo.buffer = staging;
    dstInfo.layout.offset = 0;
    dstInfo.layout.bytesPerRow = bytesPerRow;
    dstInfo.layout.rowsPerImage = height_;

    WGPUExtent3D extent = {width_, height_, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &srcInfo, &dstInfo, &extent);
    wgpu_utils::finishCommandEncoder(queue, encoder);

    // Map and read back
    const void *data =
        wgpu_utils::mapBufferSync(device, staging, bufferSize);
    if (data) {
        // Copy to tightly-packed buffer, swizzle BGRA→RGBA if needed
        std::vector<uint8_t> pixels(width_ * height_ * 4);
        bool isBGRA = (format_ == WGPUTextureFormat_BGRA8Unorm);

        for (uint32_t y = 0; y < height_; y++) {
            const uint8_t *srcRow =
                static_cast<const uint8_t *>(data) + y * bytesPerRow;
            uint8_t *dstRow = pixels.data() + y * width_ * 4;

            if (isBGRA) {
                for (uint32_t x = 0; x < width_; x++) {
                    dstRow[x * 4 + 0] = srcRow[x * 4 + 2]; // R <- B
                    dstRow[x * 4 + 1] = srcRow[x * 4 + 1]; // G
                    dstRow[x * 4 + 2] = srcRow[x * 4 + 0]; // B <- R
                    dstRow[x * 4 + 3] = srcRow[x * 4 + 3]; // A
                }
            } else {
                std::memcpy(dstRow, srcRow, width_ * 4);
            }
        }

        wgpuBufferUnmap(staging);

        // Generate filename
        char filename[128];
        snprintf(filename, sizeof(filename), "screenshot_t%.3f.png", simTime);

        stbi_write_png(filename, static_cast<int>(width_),
                       static_cast<int>(height_), 4, pixels.data(),
                       static_cast<int>(width_ * 4));
        spdlog::info("Screenshot saved: {}", filename);
    } else {
        spdlog::error("Failed to map staging buffer for screenshot");
    }

    // Cleanup staging buffer
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);

    // Advance to next target
    nextTarget_++;
}

void ScreenshotCapture::cleanup() {
    if (offscreenView_) {
        wgpuTextureViewRelease(offscreenView_);
        offscreenView_ = nullptr;
    }
    if (offscreenTexture_) {
        wgpuTextureRelease(offscreenTexture_);
        offscreenTexture_ = nullptr;
    }
    initialized_ = false;
}
