#include "particle_renderer.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>

static const char *kRenderShaderSource = R"(
struct Uniforms {
    viewProjection: mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> colors: array<vec4f>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) pointCoord: vec2f,
}

fn getQuadVertex(index: u32) -> vec2f {
    // Two triangles forming a quad: 0,1,2 and 3,4,5
    switch(index) {
        case 0u: { return vec2f(-1.0, -1.0); }
        case 1u: { return vec2f( 1.0, -1.0); }
        case 2u: { return vec2f(-1.0,  1.0); }
        case 3u: { return vec2f(-1.0,  1.0); }
        case 4u: { return vec2f( 1.0, -1.0); }
        case 5u: { return vec2f( 1.0,  1.0); }
        default: { return vec2f(0.0, 0.0); }
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32,
           @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    let qv = getQuadVertex(vertexIndex);

    let pos = positions[instanceIndex];
    let worldPos = vec4f(pos.xyz, 1.0);
    let clipPos = uniforms.viewProjection * worldPos;

    // Billboard size in clip space
    let pointSize: f32 = 0.15;
    let offset = qv * pointSize;

    var output: VertexOutput;
    output.position = vec4f(clipPos.xy + offset * clipPos.w * 0.01, clipPos.z, clipPos.w);
    output.color = colors[instanceIndex];
    output.pointCoord = qv;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let dist = length(input.pointCoord);
    if (dist > 1.0) { discard; }
    let alpha = input.color.a * (1.0 - dist * dist);
    return vec4f(input.color.rgb * alpha, alpha);
}
)";

static const char *kFadeShaderSource = R"(
struct FadeParams {
    alpha: f32,
}

@group(0) @binding(0) var<uniform> params: FadeParams;

@vertex
fn vs_fade(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_fade() -> @location(0) vec4f {
    return vec4f(0.01, 0.01, 0.02, params.alpha);
}
)";

void ParticleRenderer::createDepthTexture(WGPUDevice device, int width, int height) {
    if (depthTexture_) {
        wgpuTextureViewRelease(depthTextureView_);
        wgpuTextureRelease(depthTexture_);
    }

    WGPUTextureDescriptor depthDesc{};
    depthDesc.dimension = WGPUTextureDimension_2D;
    depthDesc.format = WGPUTextureFormat_Depth24Plus;
    depthDesc.mipLevelCount = 1;
    depthDesc.sampleCount = 1;
    depthDesc.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    depthDesc.usage = WGPUTextureUsage_RenderAttachment;
    depthDesc.viewFormatCount = 0;
    depthTexture_ = wgpuDeviceCreateTexture(device, &depthDesc);

    WGPUTextureViewDescriptor viewDesc{};
    viewDesc.aspect = WGPUTextureAspect_All;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = 1;
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = 1;
    viewDesc.dimension = WGPUTextureViewDimension_2D;
    viewDesc.format = WGPUTextureFormat_Depth24Plus;
    depthTextureView_ = wgpuTextureCreateView(depthTexture_, &viewDesc);
}

void ParticleRenderer::initialize(WGPUDevice device, WGPUTextureFormat surfaceFormat,
                                  int width, int height) {
    surfaceFormat_ = surfaceFormat;

    createDepthTexture(device, width, height);

    // Bind group layout: colors only used in vertex shader
    WGPUBindGroupLayoutEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Vertex;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.minBindingSize = 64;

    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Vertex;
    entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[1].buffer.minBindingSize = 0;

    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Vertex;
    entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[2].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor layoutDesc{};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;
    bindGroupLayout_ = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout_;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);

    WGPUShaderModule shader = wgpu_utils::createShaderModule(device, kRenderShaderSource);

    // Render pipeline
    WGPURenderPipelineDescriptor pipelineDesc{};
    pipelineDesc.layout = pipelineLayout;

    // Vertex state (no vertex buffers - using storage buffers)
    pipelineDesc.vertex.module = shader;
    pipelineDesc.vertex.entryPoint = "vs_main";
    pipelineDesc.vertex.bufferCount = 0;

    // Primitive state
    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    pipelineDesc.primitive.cullMode = WGPUCullMode_None;

    // Fragment state with additive blending
    WGPUBlendState blendState{};
    blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_One;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_One;
    blendState.alpha.dstFactor = WGPUBlendFactor_One;
    blendState.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState colorTarget{};
    colorTarget.format = surfaceFormat;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState{};
    fragmentState.module = shader;
    fragmentState.entryPoint = "fs_main";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;
    pipelineDesc.fragment = &fragmentState;

    // Depth stencil - fully initialize all fields to avoid Undefined enums
    WGPUDepthStencilState depthStencil{};
    depthStencil.format = WGPUTextureFormat_Depth24Plus;
    depthStencil.depthWriteEnabled = false;
    depthStencil.depthCompare = WGPUCompareFunction_LessEqual;
    // Stencil front
    depthStencil.stencilFront.compare = WGPUCompareFunction_Always;
    depthStencil.stencilFront.failOp = WGPUStencilOperation_Keep;
    depthStencil.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    depthStencil.stencilFront.passOp = WGPUStencilOperation_Keep;
    // Stencil back
    depthStencil.stencilBack.compare = WGPUCompareFunction_Always;
    depthStencil.stencilBack.failOp = WGPUStencilOperation_Keep;
    depthStencil.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    depthStencil.stencilBack.passOp = WGPUStencilOperation_Keep;
    // Stencil masks
    depthStencil.stencilReadMask = 0xFFFFFFFF;
    depthStencil.stencilWriteMask = 0xFFFFFFFF;
    depthStencil.depthBias = 0;
    depthStencil.depthBiasSlopeScale = 0.0f;
    depthStencil.depthBiasClamp = 0.0f;
    pipelineDesc.depthStencil = &depthStencil;

    // Multisample
    pipelineDesc.multisample.count = 1;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;

    pipeline_ = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
    if (!pipeline_) {
        spdlog::error("Failed to create render pipeline!");
    } else {
        spdlog::info("Render pipeline created successfully");
    }
    wgpuShaderModuleRelease(shader);
    wgpuPipelineLayoutRelease(pipelineLayout);

    // Uniform buffer for viewProjection matrix
    uniformBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 64);

    // --- Fade pipeline for trails ---
    WGPUBindGroupLayoutEntry fadeEntry{};
    fadeEntry.binding = 0;
    fadeEntry.visibility = WGPUShaderStage_Fragment;
    fadeEntry.buffer.type = WGPUBufferBindingType_Uniform;
    fadeEntry.buffer.minBindingSize = 16;

    WGPUBindGroupLayoutDescriptor fadeBGLDesc{};
    fadeBGLDesc.entryCount = 1;
    fadeBGLDesc.entries = &fadeEntry;
    fadeBindGroupLayout_ = wgpuDeviceCreateBindGroupLayout(device, &fadeBGLDesc);

    WGPUPipelineLayoutDescriptor fadePLDesc{};
    fadePLDesc.bindGroupLayoutCount = 1;
    fadePLDesc.bindGroupLayouts = &fadeBindGroupLayout_;
    WGPUPipelineLayout fadePipelineLayout = wgpuDeviceCreatePipelineLayout(device, &fadePLDesc);

    WGPUShaderModule fadeShader = wgpu_utils::createShaderModule(device, kFadeShaderSource);

    WGPURenderPipelineDescriptor fadePipelineDesc{};
    fadePipelineDesc.layout = fadePipelineLayout;

    fadePipelineDesc.vertex.module = fadeShader;
    fadePipelineDesc.vertex.entryPoint = "vs_fade";
    fadePipelineDesc.vertex.bufferCount = 0;

    fadePipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    fadePipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    fadePipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    fadePipelineDesc.primitive.cullMode = WGPUCullMode_None;

    // Standard alpha blending for fade effect
    WGPUBlendState fadeBlendState{};
    fadeBlendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    fadeBlendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    fadeBlendState.color.operation = WGPUBlendOperation_Add;
    fadeBlendState.alpha.srcFactor = WGPUBlendFactor_One;
    fadeBlendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    fadeBlendState.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState fadeColorTarget{};
    fadeColorTarget.format = surfaceFormat;
    fadeColorTarget.blend = &fadeBlendState;
    fadeColorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fadeFragmentState{};
    fadeFragmentState.module = fadeShader;
    fadeFragmentState.entryPoint = "fs_fade";
    fadeFragmentState.targetCount = 1;
    fadeFragmentState.targets = &fadeColorTarget;
    fadePipelineDesc.fragment = &fadeFragmentState;

    // Depth state: does nothing, but required for render pass compatibility
    WGPUDepthStencilState fadeDepthStencil{};
    fadeDepthStencil.format = WGPUTextureFormat_Depth24Plus;
    fadeDepthStencil.depthWriteEnabled = false;
    fadeDepthStencil.depthCompare = WGPUCompareFunction_Always;
    fadeDepthStencil.stencilFront.compare = WGPUCompareFunction_Always;
    fadeDepthStencil.stencilFront.failOp = WGPUStencilOperation_Keep;
    fadeDepthStencil.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    fadeDepthStencil.stencilFront.passOp = WGPUStencilOperation_Keep;
    fadeDepthStencil.stencilBack = fadeDepthStencil.stencilFront;
    fadeDepthStencil.stencilReadMask = 0;
    fadeDepthStencil.stencilWriteMask = 0;
    fadeDepthStencil.depthBias = 0;
    fadeDepthStencil.depthBiasSlopeScale = 0.0f;
    fadeDepthStencil.depthBiasClamp = 0.0f;
    fadePipelineDesc.depthStencil = &fadeDepthStencil;

    fadePipelineDesc.multisample.count = 1;
    fadePipelineDesc.multisample.mask = ~0u;
    fadePipelineDesc.multisample.alphaToCoverageEnabled = false;

    fadePipeline_ = wgpuDeviceCreateRenderPipeline(device, &fadePipelineDesc);
    wgpuShaderModuleRelease(fadeShader);
    wgpuPipelineLayoutRelease(fadePipelineLayout);

    fadeUniformBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 16);

    WGPUBindGroupEntry fadeBGEntry{};
    fadeBGEntry.binding = 0;
    fadeBGEntry.buffer = fadeUniformBuffer_.get();
    fadeBGEntry.offset = 0;
    fadeBGEntry.size = 16;

    WGPUBindGroupDescriptor fadeBGDesc{};
    fadeBGDesc.layout = fadeBindGroupLayout_;
    fadeBGDesc.entryCount = 1;
    fadeBGDesc.entries = &fadeBGEntry;
    fadeBindGroup_ = wgpuDeviceCreateBindGroup(device, &fadeBGDesc);

    // Trail accumulation texture (persistent across frames)
    WGPUTextureDescriptor trailDesc{};
    trailDesc.dimension = WGPUTextureDimension_2D;
    trailDesc.format = surfaceFormat;
    trailDesc.mipLevelCount = 1;
    trailDesc.sampleCount = 1;
    trailDesc.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    trailDesc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    trailTexture_ = wgpuDeviceCreateTexture(device, &trailDesc);
    trailWidth_ = static_cast<uint32_t>(width);
    trailHeight_ = static_cast<uint32_t>(height);

    WGPUTextureViewDescriptor trailViewDesc{};
    trailViewDesc.aspect = WGPUTextureAspect_All;
    trailViewDesc.baseArrayLayer = 0;
    trailViewDesc.arrayLayerCount = 1;
    trailViewDesc.baseMipLevel = 0;
    trailViewDesc.mipLevelCount = 1;
    trailViewDesc.dimension = WGPUTextureViewDimension_2D;
    trailViewDesc.format = surfaceFormat;
    trailTextureView_ = wgpuTextureCreateView(trailTexture_, &trailViewDesc);

    initialized_ = true;
    spdlog::info("Particle renderer initialized");
}

void ParticleRenderer::render(WGPUDevice device, WGPUQueue queue,
                              WGPUTextureView targetView,
                              WGPUBuffer positionBuffer, WGPUBuffer colorBuffer,
                              int particleCount,
                              const glm::mat4 &viewMatrix,
                              const glm::mat4 &projMatrix,
                              bool showTrails, float trailLength,
                              WGPUTexture targetTexture) {
    if (!initialized_ || particleCount <= 0 || !pipeline_) return;

    glm::mat4 viewProj = projMatrix * viewMatrix;
    uniformBuffer_.upload(queue, glm::value_ptr(viewProj), 64);

    // Cache bind group (recreate only when buffers change)
    if (!cachedBindGroup_ || cachedPosBuffer_ != positionBuffer ||
        cachedColorBuffer_ != colorBuffer || cachedParticleCount_ != particleCount) {
        if (cachedBindGroup_) wgpuBindGroupRelease(cachedBindGroup_);

        WGPUBindGroupEntry bindEntries[3] = {};
        bindEntries[0].binding = 0;
        bindEntries[0].buffer = uniformBuffer_.get();
        bindEntries[0].offset = 0;
        bindEntries[0].size = 64;
        bindEntries[1].binding = 1;
        bindEntries[1].buffer = positionBuffer;
        bindEntries[1].offset = 0;
        bindEntries[1].size = particleCount * sizeof(glm::vec4);
        bindEntries[2].binding = 2;
        bindEntries[2].buffer = colorBuffer;
        bindEntries[2].offset = 0;
        bindEntries[2].size = particleCount * sizeof(glm::vec4);

        WGPUBindGroupDescriptor bindGroupDesc{};
        bindGroupDesc.layout = bindGroupLayout_;
        bindGroupDesc.entryCount = 3;
        bindGroupDesc.entries = bindEntries;
        cachedBindGroup_ = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
        cachedPosBuffer_ = positionBuffer;
        cachedColorBuffer_ = colorBuffer;
        cachedParticleCount_ = particleCount;
    }

    // Choose render target: trail texture (persistent) or swapchain (direct)
    WGPUTextureView renderTarget = (showTrails && trailTextureView_) ? trailTextureView_ : targetView;

    WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);

    WGPURenderPassColorAttachment colorAttachment{};
    colorAttachment.view = renderTarget;
    colorAttachment.resolveTarget = nullptr;
    colorAttachment.storeOp = WGPUStoreOp_Store;
#ifndef WEBGPU_BACKEND_WGPU
    colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    if (showTrails && trailTextureView_) {
        colorAttachment.loadOp = WGPULoadOp_Load;
        colorAttachment.clearValue = {0.0, 0.0, 0.0, 0.0};
    } else {
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.clearValue = {0.01, 0.01, 0.02, 1.0};
    }

    WGPURenderPassDepthStencilAttachment depthAttachment{};
    depthAttachment.view = depthTextureView_;
    depthAttachment.depthLoadOp = WGPULoadOp_Clear;
    depthAttachment.depthStoreOp = WGPUStoreOp_Store;
    depthAttachment.depthClearValue = 1.0f;
    depthAttachment.depthReadOnly = false;
    depthAttachment.stencilLoadOp = WGPULoadOp_Undefined;
    depthAttachment.stencilStoreOp = WGPUStoreOp_Undefined;

    WGPURenderPassDescriptor passDesc{};
    passDesc.colorAttachmentCount = 1;
    passDesc.colorAttachments = &colorAttachment;
    passDesc.depthStencilAttachment = &depthAttachment;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);

    if (showTrails && trailTextureView_) {
        float fadeAlpha = 1.0f - trailLength;
        float fadeData[4] = {fadeAlpha, 0.0f, 0.0f, 0.0f};
        fadeUniformBuffer_.upload(queue, fadeData, 16);

        wgpuRenderPassEncoderSetPipeline(pass, fadePipeline_);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, fadeBindGroup_, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }

    wgpuRenderPassEncoderSetPipeline(pass, pipeline_);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, cachedBindGroup_, 0, nullptr);
    wgpuRenderPassEncoderDraw(pass, 6, static_cast<uint32_t>(particleCount), 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    // Copy trail texture to swapchain surface
    if (showTrails && trailTexture_ && targetTexture) {
        WGPUImageCopyTexture src{};
        src.texture = trailTexture_;
        src.mipLevel = 0;
        src.origin = {0, 0, 0};
        src.aspect = WGPUTextureAspect_All;

        WGPUImageCopyTexture dst{};
        dst.texture = targetTexture;
        dst.mipLevel = 0;
        dst.origin = {0, 0, 0};
        dst.aspect = WGPUTextureAspect_All;

        WGPUExtent3D copySize = {trailWidth_, trailHeight_, 1};
        wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &copySize);
    }

    wgpu_utils::finishCommandEncoder(queue, encoder);
}

void ParticleRenderer::cleanup() {
    if (cachedBindGroup_) {
        wgpuBindGroupRelease(cachedBindGroup_);
        cachedBindGroup_ = nullptr;
    }
    if (fadeBindGroup_) {
        wgpuBindGroupRelease(fadeBindGroup_);
        fadeBindGroup_ = nullptr;
    }
    if (fadePipeline_) {
        wgpuRenderPipelineRelease(fadePipeline_);
        fadePipeline_ = nullptr;
    }
    if (fadeBindGroupLayout_) {
        wgpuBindGroupLayoutRelease(fadeBindGroupLayout_);
        fadeBindGroupLayout_ = nullptr;
    }
    if (trailTextureView_) {
        wgpuTextureViewRelease(trailTextureView_);
        trailTextureView_ = nullptr;
    }
    if (trailTexture_) {
        wgpuTextureRelease(trailTexture_);
        trailTexture_ = nullptr;
    }
    if (depthTextureView_) {
        wgpuTextureViewRelease(depthTextureView_);
        depthTextureView_ = nullptr;
    }
    if (depthTexture_) {
        wgpuTextureRelease(depthTexture_);
        depthTexture_ = nullptr;
    }
    if (pipeline_) {
        wgpuRenderPipelineRelease(pipeline_);
        pipeline_ = nullptr;
    }
    if (bindGroupLayout_) {
        wgpuBindGroupLayoutRelease(bindGroupLayout_);
        bindGroupLayout_ = nullptr;
    }
    initialized_ = false;
}
