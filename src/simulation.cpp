#include "simulation.hpp"
#include <cmath>
#include <random>
#include <spdlog/spdlog.h>

static const char *kForceShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
}

struct Node {
    centerOfMass: vec4f,
    bounds: vec4f,
    c0: i32, c1: i32, c2: i32, c3: i32,
    c4: i32, c5: i32, c6: i32, c7: i32,
}

fn getChild(node: Node, idx: i32) -> i32 {
    switch(idx) {
        case 0: { return node.c0; }
        case 1: { return node.c1; }
        case 2: { return node.c2; }
        case 3: { return node.c3; }
        case 4: { return node.c4; }
        case 5: { return node.c5; }
        case 6: { return node.c6; }
        case 7: { return node.c7; }
        default: { return -1; }
    }
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read_write> accelerations: array<vec4f>;

const G: f32 = 1.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let pos = positions[i].xyz;
    var acc = vec3f(0.0);

    // Stack-based tree traversal
    var stack: array<i32, 64>;
    stack[0] = 0;
    var top: i32 = 1;

    let thetaSq = params.theta * params.theta;
    let softSq = params.softening * params.softening;

    while (top > 0) {
        top = top - 1;
        let nodeIdx = stack[top];
        if (nodeIdx < 0 || u32(nodeIdx) >= params.nodeCount) { continue; }

        let node = nodes[nodeIdx];
        let diff = node.centerOfMass.xyz - pos;
        let distSq = dot(diff, diff) + softSq;
        let halfWidth = node.bounds.w;
        let mass = node.centerOfMass.w;

        // Check if this is a leaf (all children are -1)
        let hasChildren = (node.c0 >= 0) || (node.c1 >= 0) || (node.c2 >= 0) || (node.c3 >= 0)
                       || (node.c4 >= 0) || (node.c5 >= 0) || (node.c6 >= 0) || (node.c7 >= 0);

        if (!hasChildren || (halfWidth * halfWidth / distSq < thetaSq)) {
            // Use this node's mass approximation
            if (mass > 0.0 && distSq > softSq * 0.01) {
                let invDist = inverseSqrt(distSq);
                let invDist3 = invDist * invDist * invDist;
                acc += G * mass * diff * invDist3;
            }
        } else {
            // Push non-empty children (unrolled to avoid dynamic array indexing)
            if (node.c0 >= 0 && top < 64) { stack[top] = node.c0; top = top + 1; }
            if (node.c1 >= 0 && top < 64) { stack[top] = node.c1; top = top + 1; }
            if (node.c2 >= 0 && top < 64) { stack[top] = node.c2; top = top + 1; }
            if (node.c3 >= 0 && top < 64) { stack[top] = node.c3; top = top + 1; }
            if (node.c4 >= 0 && top < 64) { stack[top] = node.c4; top = top + 1; }
            if (node.c5 >= 0 && top < 64) { stack[top] = node.c5; top = top + 1; }
            if (node.c6 >= 0 && top < 64) { stack[top] = node.c6; top = top + 1; }
            if (node.c7 >= 0 && top < 64) { stack[top] = node.c7; top = top + 1; }
        }
    }

    accelerations[i] = vec4f(acc, 0.0);
}
)";

static const char *kIntegrateShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> accelerations: array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let acc = accelerations[i].xyz;
    var vel = velocities[i].xyz;
    var pos = positions[i].xyz;
    let mass = positions[i].w;

    vel += acc * params.dt;
    pos += vel * params.dt;

    velocities[i] = vec4f(vel, 0.0);
    positions[i] = vec4f(pos, mass);
}
)";

void Simulation::initParticles(int numParticles) {
    cpuPositions_.resize(numParticles);
    cpuVelocities_.resize(numParticles);
    cpuColors_.resize(numParticles);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    std::exponential_distribution<float> radiusDist(0.08f);
    std::normal_distribution<float> heightDist(0.0f, 0.3f);
    std::uniform_real_distribution<float> massDist(0.5f, 2.0f);

    float totalMass = 0.0f;

    for (int i = 0; i < numParticles; i++) {
        float angle = angleDist(rng);
        float r = radiusDist(rng);
        r = std::min(r, 50.0f); // clamp max radius
        float y = heightDist(rng) * (1.0f / (1.0f + r * 0.5f));

        float x = r * std::cos(angle);
        float z = r * std::sin(angle);
        float mass = massDist(rng);

        cpuPositions_[i] = glm::vec4(x, y, z, mass);
        totalMass += mass;

        // Approximate Keplerian velocity: v = sqrt(M_enclosed / r)
        float enclosedMass = totalMass * (r / 50.0f); // rough approximation
        float speed = (r > 0.1f) ? std::sqrt(enclosedMass / r) * 0.5f : 0.0f;

        // Tangential velocity (perpendicular to radius in XZ plane)
        float vx = -speed * std::sin(angle);
        float vz = speed * std::cos(angle);
        cpuVelocities_[i] = glm::vec4(vx, 0.0f, vz, 0.0f);

        // Color: inner = blue-white, outer = yellow-red
        float t = std::min(r / 30.0f, 1.0f);
        float red = 0.6f + 0.4f * t;
        float green = 0.7f - 0.3f * t;
        float blue = 1.0f - 0.7f * t;
        float alpha = 0.8f;
        cpuColors_[i] = glm::vec4(red, green, blue, alpha);
    }
}

void Simulation::createComputePipelines(WGPUDevice device) {
    if (pipelinesCreated_) return;

    // Force pipeline bind group layout
    WGPUBindGroupLayoutEntry forceEntries[4] = {};
    // Params uniform
    forceEntries[0].binding = 0;
    forceEntries[0].visibility = WGPUShaderStage_Compute;
    forceEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    forceEntries[0].buffer.minBindingSize = 0;
    // Positions storage (read)
    forceEntries[1].binding = 1;
    forceEntries[1].visibility = WGPUShaderStage_Compute;
    forceEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    forceEntries[1].buffer.minBindingSize = 0;
    // Octree nodes storage (read)
    forceEntries[2].binding = 2;
    forceEntries[2].visibility = WGPUShaderStage_Compute;
    forceEntries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    forceEntries[2].buffer.minBindingSize = 0;
    // Accelerations storage (read_write)
    forceEntries[3].binding = 3;
    forceEntries[3].visibility = WGPUShaderStage_Compute;
    forceEntries[3].buffer.type = WGPUBufferBindingType_Storage;
    forceEntries[3].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor forceLayoutDesc{};
    forceLayoutDesc.entryCount = 4;
    forceLayoutDesc.entries = forceEntries;
    forceBindGroupLayout_ = wgpuDeviceCreateBindGroupLayout(device, &forceLayoutDesc);

    WGPUPipelineLayoutDescriptor forcePipelineLayoutDesc{};
    forcePipelineLayoutDesc.bindGroupLayoutCount = 1;
    forcePipelineLayoutDesc.bindGroupLayouts = &forceBindGroupLayout_;
    WGPUPipelineLayout forcePipelineLayout = wgpuDeviceCreatePipelineLayout(device, &forcePipelineLayoutDesc);

    WGPUShaderModule forceShader = wgpu_utils::createShaderModule(device, kForceShaderSource);
    WGPUComputePipelineDescriptor forcePipelineDesc{};
    forcePipelineDesc.layout = forcePipelineLayout;
    forcePipelineDesc.compute.module = forceShader;
    forcePipelineDesc.compute.entryPoint = "main";
    forcePipeline_ = wgpuDeviceCreateComputePipeline(device, &forcePipelineDesc);
    if (!forcePipeline_) spdlog::error("Failed to create force compute pipeline!");
    else spdlog::info("Force compute pipeline created successfully");
    wgpuShaderModuleRelease(forceShader);
    wgpuPipelineLayoutRelease(forcePipelineLayout);

    // Integration pipeline bind group layout
    WGPUBindGroupLayoutEntry intEntries[4] = {};
    intEntries[0].binding = 0;
    intEntries[0].visibility = WGPUShaderStage_Compute;
    intEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    intEntries[0].buffer.minBindingSize = 0;
    intEntries[1].binding = 1;
    intEntries[1].visibility = WGPUShaderStage_Compute;
    intEntries[1].buffer.type = WGPUBufferBindingType_Storage;
    intEntries[1].buffer.minBindingSize = 0;
    intEntries[2].binding = 2;
    intEntries[2].visibility = WGPUShaderStage_Compute;
    intEntries[2].buffer.type = WGPUBufferBindingType_Storage;
    intEntries[2].buffer.minBindingSize = 0;
    intEntries[3].binding = 3;
    intEntries[3].visibility = WGPUShaderStage_Compute;
    intEntries[3].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    intEntries[3].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor intLayoutDesc{};
    intLayoutDesc.entryCount = 4;
    intLayoutDesc.entries = intEntries;
    integrateBindGroupLayout_ = wgpuDeviceCreateBindGroupLayout(device, &intLayoutDesc);

    WGPUPipelineLayoutDescriptor intPipelineLayoutDesc{};
    intPipelineLayoutDesc.bindGroupLayoutCount = 1;
    intPipelineLayoutDesc.bindGroupLayouts = &integrateBindGroupLayout_;
    WGPUPipelineLayout intPipelineLayout = wgpuDeviceCreatePipelineLayout(device, &intPipelineLayoutDesc);

    WGPUShaderModule intShader = wgpu_utils::createShaderModule(device, kIntegrateShaderSource);
    WGPUComputePipelineDescriptor intPipelineDesc{};
    intPipelineDesc.layout = intPipelineLayout;
    intPipelineDesc.compute.module = intShader;
    intPipelineDesc.compute.entryPoint = "main";
    integratePipeline_ = wgpuDeviceCreateComputePipeline(device, &intPipelineDesc);
    if (!integratePipeline_) spdlog::error("Failed to create integrate compute pipeline!");
    else spdlog::info("Integrate compute pipeline created successfully");
    wgpuShaderModuleRelease(intShader);
    wgpuPipelineLayoutRelease(intPipelineLayout);

    pipelinesCreated_ = true;
}

void Simulation::initialize(WGPUDevice device, int numParticles) {
    numParticles_ = numParticles;
    initParticles(numParticles);

    positions_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        numParticles * sizeof(glm::vec4));
    velocities_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        numParticles * sizeof(glm::vec4));
    colors_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        numParticles * sizeof(glm::vec4));
    accelerations_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        numParticles * sizeof(glm::vec4));

    // Upload initial data
    positions_.upload(cpuPositions_.data(), numParticles * sizeof(glm::vec4));
    velocities_.upload(cpuVelocities_.data(), numParticles * sizeof(glm::vec4));
    colors_.upload(cpuColors_.data(), numParticles * sizeof(glm::vec4));

    // Zero out accelerations
    std::vector<glm::vec4> zeroAccel(numParticles, glm::vec4(0.0f));
    accelerations_.upload(zeroAccel.data(), numParticles * sizeof(glm::vec4));

    // Initial octree node buffer (will be resized each step)
    octreeBuffer_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        sizeof(OctreeNode) * 8);

    // Params buffer: dt, softening, theta, numParticles, nodeCount (5 * 4 bytes, padded to 32)
    paramsBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        32);

    createComputePipelines(device);
    spdlog::info("Simulation initialized with {} particles", numParticles);
}

void Simulation::reinitialize(WGPUDevice device, int numParticles) {
    numParticles_ = numParticles;
    initParticles(numParticles);

    positions_.upload(cpuPositions_.data(), numParticles * sizeof(glm::vec4));
    velocities_.upload(cpuVelocities_.data(), numParticles * sizeof(glm::vec4));
    colors_.upload(cpuColors_.data(), numParticles * sizeof(glm::vec4));

    std::vector<glm::vec4> zeroAccel(numParticles, glm::vec4(0.0f));
    accelerations_.upload(zeroAccel.data(), numParticles * sizeof(glm::vec4));
}

void Simulation::step(WGPUDevice device, WGPUQueue queue, float dt, float softening, float theta) {
    // 1. Read positions back to CPU for octree build
    // For now, use CPU positions (updated each step from GPU would require mapBufferSync)
    // We'll use a simpler approach: run the octree on the CPU-side copy, which we update
    // after each integration step by reading back GPU positions.

    // Build octree on CPU
    octree_.build(cpuPositions_);
    size_t nodeCount = octree_.nodeCount();
    if (nodeCount == 0) return;

    // Upload octree nodes
    auto &nodes = octree_.getNodes();
    octreeBuffer_.upload(queue, (void *)nodes.data(), nodeCount * sizeof(OctreeNode));

    // Upload params
    struct alignas(16) Params {
        float dt;
        float softening;
        float theta;
        uint32_t numParticles;
        uint32_t nodeCount;
        float _pad[3];
    } params;
    params.dt = dt;
    params.softening = softening;
    params.theta = theta;
    params.numParticles = static_cast<uint32_t>(numParticles_);
    params.nodeCount = static_cast<uint32_t>(nodeCount);
    params._pad[0] = params._pad[1] = params._pad[2] = 0.0f;
    paramsBuffer_.upload(queue, &params, sizeof(params));

    // Create bind groups
    // Force bind group
    WGPUBindGroupEntry forceBindEntries[4] = {};
    forceBindEntries[0].binding = 0;
    forceBindEntries[0].buffer = paramsBuffer_.get();
    forceBindEntries[0].offset = 0;
    forceBindEntries[0].size = sizeof(params);
    forceBindEntries[1].binding = 1;
    forceBindEntries[1].buffer = positions_.get();
    forceBindEntries[1].offset = 0;
    forceBindEntries[1].size = numParticles_ * sizeof(glm::vec4);
    forceBindEntries[2].binding = 2;
    forceBindEntries[2].buffer = octreeBuffer_.get();
    forceBindEntries[2].offset = 0;
    forceBindEntries[2].size = nodeCount * sizeof(OctreeNode);
    forceBindEntries[3].binding = 3;
    forceBindEntries[3].buffer = accelerations_.get();
    forceBindEntries[3].offset = 0;
    forceBindEntries[3].size = numParticles_ * sizeof(glm::vec4);

    WGPUBindGroupDescriptor forceBindGroupDesc{};
    forceBindGroupDesc.layout = forceBindGroupLayout_;
    forceBindGroupDesc.entryCount = 4;
    forceBindGroupDesc.entries = forceBindEntries;
    WGPUBindGroup forceBindGroup = wgpuDeviceCreateBindGroup(device, &forceBindGroupDesc);

    // Integration bind group
    WGPUBindGroupEntry intBindEntries[4] = {};
    intBindEntries[0].binding = 0;
    intBindEntries[0].buffer = paramsBuffer_.get();
    intBindEntries[0].offset = 0;
    intBindEntries[0].size = sizeof(params);
    intBindEntries[1].binding = 1;
    intBindEntries[1].buffer = positions_.get();
    intBindEntries[1].offset = 0;
    intBindEntries[1].size = numParticles_ * sizeof(glm::vec4);
    intBindEntries[2].binding = 2;
    intBindEntries[2].buffer = velocities_.get();
    intBindEntries[2].offset = 0;
    intBindEntries[2].size = numParticles_ * sizeof(glm::vec4);
    intBindEntries[3].binding = 3;
    intBindEntries[3].buffer = accelerations_.get();
    intBindEntries[3].offset = 0;
    intBindEntries[3].size = numParticles_ * sizeof(glm::vec4);

    WGPUBindGroupDescriptor intBindGroupDesc{};
    intBindGroupDesc.layout = integrateBindGroupLayout_;
    intBindGroupDesc.entryCount = 4;
    intBindGroupDesc.entries = intBindEntries;
    WGPUBindGroup intBindGroup = wgpuDeviceCreateBindGroup(device, &intBindGroupDesc);

    // Dispatch compute passes
    WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);

    // Force pass
    WGPUComputePassDescriptor computePassDesc{};
    computePassDesc.timestampWrites = nullptr;
    WGPUComputePassEncoder forcePass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
    wgpuComputePassEncoderSetPipeline(forcePass, forcePipeline_);
    wgpuComputePassEncoderSetBindGroup(forcePass, 0, forceBindGroup, 0, nullptr);
    uint32_t forceWorkgroups = (numParticles_ + 63) / 64;
    wgpuComputePassEncoderDispatchWorkgroups(forcePass, forceWorkgroups, 1, 1);
    wgpuComputePassEncoderEnd(forcePass);
    wgpuComputePassEncoderRelease(forcePass);

    // Integration pass
    WGPUComputePassEncoder intPass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
    wgpuComputePassEncoderSetPipeline(intPass, integratePipeline_);
    wgpuComputePassEncoderSetBindGroup(intPass, 0, intBindGroup, 0, nullptr);
    uint32_t intWorkgroups = (numParticles_ + 255) / 256;
    wgpuComputePassEncoderDispatchWorkgroups(intPass, intWorkgroups, 1, 1);
    wgpuComputePassEncoderEnd(intPass);
    wgpuComputePassEncoderRelease(intPass);

    wgpu_utils::finishCommandEncoder(queue, encoder);

    // Update CPU positions for next octree build by doing a simple Euler step on CPU too
    // This avoids expensive GPU readback each frame
    for (int i = 0; i < numParticles_; i++) {
        glm::vec3 pos(cpuPositions_[i]);
        glm::vec3 vel(cpuVelocities_[i]);

        // We don't have the GPU accelerations on CPU, so we approximate
        // by building the octree force on CPU as well (already done above).
        // For simplicity, do a CPU-side force calculation using the octree
        glm::vec3 acc(0.0f);
        const auto &octreeNodes = octree_.getNodes();
        // Simple stack-based traversal matching the GPU shader
        int stack[64];
        stack[0] = 0;
        int top = 1;
        float thetaSq = theta * theta;
        float softSq = softening * softening;

        while (top > 0) {
            top--;
            int nodeIdx = stack[top];
            if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= nodeCount) continue;

            const OctreeNode &node = octreeNodes[nodeIdx];
            glm::vec3 diff(node.centerOfMass.x - pos.x,
                          node.centerOfMass.y - pos.y,
                          node.centerOfMass.z - pos.z);
            float distSq = glm::dot(diff, diff) + softSq;
            float halfWidth = node.bounds.w;
            float mass = node.centerOfMass.w;

            bool isLeaf = true;
            for (int c = 0; c < 8; c++) {
                if (node.children[c] >= 0) { isLeaf = false; break; }
            }

            if (isLeaf || (halfWidth * halfWidth / distSq < thetaSq)) {
                if (mass > 0.0f && distSq > softSq * 0.01f) {
                    float invDist = 1.0f / std::sqrt(distSq);
                    float invDist3 = invDist * invDist * invDist;
                    acc += 1.0f * mass * diff * invDist3;
                }
            } else {
                for (int c = 0; c < 8; c++) {
                    if (node.children[c] >= 0 && top < 64) {
                        stack[top] = node.children[c];
                        top++;
                    }
                }
            }
        }

        vel += acc * dt;
        pos += vel * dt;
        float m = cpuPositions_[i].w;
        cpuPositions_[i] = glm::vec4(pos, m);
        cpuVelocities_[i] = glm::vec4(vel, 0.0f);
    }

    // Release bind groups
    wgpuBindGroupRelease(forceBindGroup);
    wgpuBindGroupRelease(intBindGroup);
}
