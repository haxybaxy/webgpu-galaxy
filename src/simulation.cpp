#include "simulation.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <spdlog/spdlog.h>

static constexpr float PI = 3.14159265358979323846f;

// ============================================================
// WGSL Shader Sources
// ============================================================

// Kick shader: v += a * dt/2
static const char *kKickShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
    paddedN: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(2) var<storage, read> accelerations: array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let acc = accelerations[i].xyz;
    var vel = velocities[i];
    vel = vec4f(vel.xyz + acc * (params.dt * 0.5), vel.w);
    velocities[i] = vel;
}
)";

// Drift shader: x += v * dt
static const char *kDriftShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
    paddedN: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let vel = velocities[i].xyz;
    var pos = positions[i];
    let mass = pos.w;
    pos = vec4f(pos.xyz + vel * params.dt, mass);
    positions[i] = pos;
}
)";

// Direct summation O(N²) force shader
static const char *kDirectForceShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
    paddedN: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<vec4f>;

const G: f32 = 1.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let posI = positions[i].xyz;
    var acc = vec3f(0.0);
    let softSq = params.softening * params.softening;

    for (var j = 0u; j < params.numParticles; j = j + 1u) {
        if (j == i) { continue; }
        let posJ = positions[j];
        let diff = posJ.xyz - posI;
        let distSq = dot(diff, diff) + softSq;
        let invDist = inverseSqrt(distSq);
        let invDist3 = invDist * invDist * invDist;
        acc += G * posJ.w * diff * invDist3;
    }

    accelerations[i] = vec4f(acc, 0.0);
}
)";

// BVH force traversal shader (binary tree, float32 bounds)
static const char *kBvhForceShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
    paddedN: u32,
}

struct BVHNode {
    centerOfMass: vec4f,
    boundsMin: vec4f,
    boundsMax: vec4f,
    left: i32,
    right: i32,
    parent: i32,
    particleIdx: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(3) var<storage, read_write> accelerations: array<vec4f>;

const G: f32 = 1.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let pos = positions[i].xyz;
    var acc = vec3f(0.0);

    var stack: array<i32, 64>;
    stack[0] = 0;
    var top: i32 = 1;

    let thetaSq = params.theta * params.theta;
    let softSq = params.softening * params.softening;

    while (top > 0) {
        top = top - 1;
        let nodeIdx = stack[top];
        if (nodeIdx < 0 || u32(nodeIdx) >= params.nodeCount) { continue; }

        let node = bvhNodes[nodeIdx];
        let diff = node.centerOfMass.xyz - pos;
        let distSq = dot(diff, diff) + softSq;
        let mass = node.centerOfMass.w;

        // Compute bounding-sphere radius at COM from float AABB bounds.
        // BVH binary trees are deeper than octrees (log2 N vs log8 N), and
        // tight AABBs make the criterion more aggressive at every level.
        // Scale by log2(mass) to make high-level nodes (many particles)
        // proportionally more conservative, matching octree behavior.
        let boundsMin = node.boundsMin.xyz;
        let boundsMax = node.boundsMax.xyz;
        let comToCorner = max(abs(node.centerOfMass.xyz - boundsMin), abs(node.centerOfMass.xyz - boundsMax));
        let logMass = log2(max(mass, 1.0));
        let halfExtent = length(comToCorner) * (1.0 + logMass * 0.6);

        let isLeaf = (node.left < 0 && node.right < 0);

        if (isLeaf || (halfExtent * halfExtent / distSq < thetaSq)) {
            if (mass > 0.0 && distSq > softSq * 0.01) {
                let invDist = inverseSqrt(distSq);
                let invDist3 = invDist * invDist * invDist;
                acc += G * mass * diff * invDist3;
            }
        } else {
            if (node.left >= 0 && top < 64) { stack[top] = node.left; top = top + 1; }
            if (node.right >= 0 && top < 64) { stack[top] = node.right; top = top + 1; }
        }
    }

    accelerations[i] = vec4f(acc, 0.0);
}
)";

// ============================================================
// Particle Initialization — Scenario Dispatch
// ============================================================

void Simulation::initParticles(const Config &config) {
    cpuPositions_.resize(numParticles_);
    cpuVelocities_.resize(numParticles_);
    cpuColors_.resize(numParticles_);

    switch (config.scenario) {
    case Scenario::TwoBody:
        initTwoBody(config);
        break;
    case Scenario::PlummerSphere:
        initPlummerSphere(config);
        break;
    case Scenario::RotatingDisk:
        initRotatingDisk(config);
        break;
    }
}

// Scenario A: Two-body circular orbit
void Simulation::initTwoBody(const Config &config) {
    float mass = 1000.0f;
    float separation = 10.0f;
    float G = 1.0f;
    float r = separation / 2.0f;
    float eps = config.softening;

    // Softened circular orbit velocity:
    // F = G*m^2*d / (d^2+eps^2)^(3/2) = m*v^2/(d/2)
    // v^2 = G*m*d^2 / (2*(d^2+eps^2)^(3/2))
    float d2 = separation * separation;
    float d2e = d2 + eps * eps;
    float speed = std::sqrt(G * mass * d2 / (2.0f * d2e * std::sqrt(d2e)));

    cpuPositions_[0] = glm::vec4(-r, 0.0f, 0.0f, mass);
    cpuPositions_[1] = glm::vec4(r, 0.0f, 0.0f, mass);

    cpuVelocities_[0] = glm::vec4(0.0f, 0.0f, -speed, 0.0f);
    cpuVelocities_[1] = glm::vec4(0.0f, 0.0f, speed, 0.0f);

    cpuColors_[0] = glm::vec4(1.0f, 0.5f, 0.2f, 1.0f);
    cpuColors_[1] = glm::vec4(0.2f, 0.5f, 1.0f, 1.0f);
}

// Scenario B: Plummer sphere (Aarseth et al. 1974)
void Simulation::initPlummerSphere(const Config &config) {
    int N = numParticles_;
    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    float a = 5.0f; // Plummer scale length
    float G = 1.0f;
    float totalMass = static_cast<float>(N); // unit mass per particle

    for (int i = 0; i < N; i++) {
        // Sample radius from Plummer CDF: M(r)/M = r^3/(r^2+a^2)^(3/2)
        // Inverse: r = a / sqrt(u^(-2/3) - 1)
        float u = uniform01(rng);
        u = std::max(0.001f, std::min(u, 0.999f));
        float r = a / std::sqrt(std::pow(u, -2.0f / 3.0f) - 1.0f);

        // Isotropic direction
        float costheta = 2.0f * uniform01(rng) - 1.0f;
        float sintheta = std::sqrt(1.0f - costheta * costheta);
        float phi = 2.0f * PI * uniform01(rng);

        float x = r * sintheta * std::cos(phi);
        float y = r * sintheta * std::sin(phi);
        float z = r * costheta;
        float mass = 1.0f;
        cpuPositions_[i] = glm::vec4(x, y, z, mass);

        // Escape velocity: v_e = sqrt(2*G*M / sqrt(r^2+a^2))
        float v_escape =
            std::sqrt(2.0f * G * totalMass / std::sqrt(r * r + a * a));

        // Rejection sampling for speed: g(q) = q^2 * (1-q^2)^(7/2)
        // Maximum at q = sqrt(2/9), g_max ~ 0.092
        float g_max = 0.1f;
        float q, g_q;
        do {
            q = uniform01(rng);
            float q2 = q * q;
            g_q = q2 * std::pow(1.0f - q2, 3.5f);
        } while (g_max * uniform01(rng) > g_q);

        float v = q * v_escape;

        // Isotropic velocity direction
        float vcosth = 2.0f * uniform01(rng) - 1.0f;
        float vsinth = std::sqrt(1.0f - vcosth * vcosth);
        float vphi = 2.0f * PI * uniform01(rng);

        cpuVelocities_[i] = glm::vec4(v * vsinth * std::cos(vphi),
                                       v * vsinth * std::sin(vphi),
                                       v * vcosth, 0.0f);

        // Color: inner = white, outer = blue
        float t = std::min(r / (3.0f * a), 1.0f);
        cpuColors_[i] =
            glm::vec4(1.0f - 0.5f * t, 1.0f - 0.3f * t, 1.0f, 0.8f);
    }
}

// Scenario C: Rotating exponential disk (original implementation)
void Simulation::initRotatingDisk(const Config &config) {
    int N = numParticles_;
    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * PI);
    std::exponential_distribution<float> radiusDist(0.08f);
    std::normal_distribution<float> heightDist(0.0f, 0.3f);
    std::uniform_real_distribution<float> massDist(0.5f, 2.0f);

    float totalMass = 0.0f;

    for (int i = 0; i < N; i++) {
        float angle = angleDist(rng);
        float r = radiusDist(rng);
        r = std::min(r, 50.0f);
        float y = heightDist(rng) * (1.0f / (1.0f + r * 0.5f));

        float x = r * std::cos(angle);
        float z = r * std::sin(angle);
        float mass = massDist(rng);

        cpuPositions_[i] = glm::vec4(x, y, z, mass);
        totalMass += mass;

        float enclosedMass = totalMass * (r / 50.0f);
        float speed =
            (r > 0.1f) ? std::sqrt(enclosedMass / r) * 0.5f : 0.0f;

        float vx = -speed * std::sin(angle);
        float vz = speed * std::cos(angle);
        cpuVelocities_[i] = glm::vec4(vx, 0.0f, vz, 0.0f);

        float t = std::min(r / 30.0f, 1.0f);
        float red = 0.6f + 0.4f * t;
        float green = 0.7f - 0.3f * t;
        float blue = 1.0f - 0.7f * t;
        cpuColors_[i] = glm::vec4(red, green, blue, 0.8f);
    }
}

// ============================================================
// Compute Initial Forces (for leapfrog bootstrap, GPU dispatch)
// ============================================================

void Simulation::computeInitialForces(WGPUDevice device, WGPUQueue queue,
                                      float softening, float theta) {
    uint32_t N = static_cast<uint32_t>(numParticles_);

    if (forceMethod_ == ForceMethod::Direct) {
        // Direct O(N²) force pass
        struct alignas(16) Params {
            float dt; float softening; float theta;
            uint32_t numParticles; uint32_t nodeCount; uint32_t paddedN;
            float _pad[2];
        } params{0.0f, softening, theta, N, 0, 0, {0.0f, 0.0f}};
        paramsBuffer_.upload(queue, &params, sizeof(params));

        WGPUBindGroupEntry directBE[3] = {};
        directBE[0].binding = 0; directBE[0].buffer = paramsBuffer_.get(); directBE[0].size = sizeof(params);
        directBE[1].binding = 1; directBE[1].buffer = positions_.get(); directBE[1].size = N * sizeof(glm::vec4);
        directBE[2].binding = 2; directBE[2].buffer = accelerations_.get(); directBE[2].size = N * sizeof(glm::vec4);

        WGPUBindGroupDescriptor bgd{}; bgd.layout = directForceBindGroupLayout_; bgd.entryCount = 3; bgd.entries = directBE;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgd);

        WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
        WGPUComputePassDescriptor passDesc{}; passDesc.timestampWrites = nullptr;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, directForcePipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, (N + 63) / 64, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpu_utils::finishCommandEncoder(queue, encoder);
        wgpuBindGroupRelease(bg);

        spdlog::info("Initial forces computed (direct O(N^2))");
    } else {
        // BVH tree force pass
        uint32_t paddedN = gpuTreeBuilder_.getPaddedN(N);
        uint32_t nodeCount = 2 * N - 1;

        struct alignas(16) Params {
            float dt; float softening; float theta;
            uint32_t numParticles; uint32_t nodeCount; uint32_t paddedN;
            float _pad[2];
        } params{0.0f, softening, theta, N, nodeCount, paddedN, {0.0f, 0.0f}};
        paramsBuffer_.upload(queue, &params, sizeof(params));

        gpuTreeBuilder_.prepareUploads(queue, N);
        ensureBindGroupsCached(device);

        WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
        gpuTreeBuilder_.recordTreeBuild(device, encoder,
                                         positions_.get(), paramsBuffer_.get(), N);

        WGPUComputePassDescriptor passDesc{}; passDesc.timestampWrites = nullptr;
        WGPUComputePassEncoder forcePass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(forcePass, bvhForcePipeline_);
        wgpuComputePassEncoderSetBindGroup(forcePass, 0, cachedBvhForceBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(forcePass, (N + 63) / 64, 1, 1);
        wgpuComputePassEncoderEnd(forcePass);
        wgpuComputePassEncoderRelease(forcePass);
        wgpu_utils::finishCommandEncoder(queue, encoder);

        spdlog::info("Initial forces computed ({} BVH nodes)", nodeCount);
    }
}

// ============================================================
// Pipeline Creation
// ============================================================

void Simulation::createComputePipelines(WGPUDevice device) {
    if (pipelinesCreated_) return;

    // --- Kick pipeline: params(uniform), velocities(rw), accelerations(ro) ---
    WGPUBindGroupLayoutEntry kickEntries[3] = {};
    kickEntries[0].binding = 0;
    kickEntries[0].visibility = WGPUShaderStage_Compute;
    kickEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    kickEntries[0].buffer.minBindingSize = 0;
    kickEntries[1].binding = 1;
    kickEntries[1].visibility = WGPUShaderStage_Compute;
    kickEntries[1].buffer.type = WGPUBufferBindingType_Storage;
    kickEntries[1].buffer.minBindingSize = 0;
    kickEntries[2].binding = 2;
    kickEntries[2].visibility = WGPUShaderStage_Compute;
    kickEntries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    kickEntries[2].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor kickLayoutDesc{};
    kickLayoutDesc.entryCount = 3;
    kickLayoutDesc.entries = kickEntries;
    kickBindGroupLayout_ =
        wgpuDeviceCreateBindGroupLayout(device, &kickLayoutDesc);

    WGPUPipelineLayoutDescriptor kickPLDesc{};
    kickPLDesc.bindGroupLayoutCount = 1;
    kickPLDesc.bindGroupLayouts = &kickBindGroupLayout_;
    WGPUPipelineLayout kickPL =
        wgpuDeviceCreatePipelineLayout(device, &kickPLDesc);

    WGPUShaderModule kickShader =
        wgpu_utils::createShaderModule(device, kKickShaderSource);
    WGPUComputePipelineDescriptor kickPipeDesc{};
    kickPipeDesc.layout = kickPL;
    kickPipeDesc.compute.module = kickShader;
    kickPipeDesc.compute.entryPoint = "main";
    kickPipeline_ = wgpuDeviceCreateComputePipeline(device, &kickPipeDesc);
    if (!kickPipeline_)
        spdlog::error("Failed to create kick compute pipeline!");
    else
        spdlog::info("Kick compute pipeline created");
    wgpuShaderModuleRelease(kickShader);
    wgpuPipelineLayoutRelease(kickPL);

    // --- Drift pipeline: params(uniform), positions(rw), velocities(ro) ---
    WGPUBindGroupLayoutEntry driftEntries[3] = {};
    driftEntries[0].binding = 0;
    driftEntries[0].visibility = WGPUShaderStage_Compute;
    driftEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    driftEntries[0].buffer.minBindingSize = 0;
    driftEntries[1].binding = 1;
    driftEntries[1].visibility = WGPUShaderStage_Compute;
    driftEntries[1].buffer.type = WGPUBufferBindingType_Storage;
    driftEntries[1].buffer.minBindingSize = 0;
    driftEntries[2].binding = 2;
    driftEntries[2].visibility = WGPUShaderStage_Compute;
    driftEntries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    driftEntries[2].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor driftLayoutDesc{};
    driftLayoutDesc.entryCount = 3;
    driftLayoutDesc.entries = driftEntries;
    driftBindGroupLayout_ =
        wgpuDeviceCreateBindGroupLayout(device, &driftLayoutDesc);

    WGPUPipelineLayoutDescriptor driftPLDesc{};
    driftPLDesc.bindGroupLayoutCount = 1;
    driftPLDesc.bindGroupLayouts = &driftBindGroupLayout_;
    WGPUPipelineLayout driftPL =
        wgpuDeviceCreatePipelineLayout(device, &driftPLDesc);

    WGPUShaderModule driftShader =
        wgpu_utils::createShaderModule(device, kDriftShaderSource);
    WGPUComputePipelineDescriptor driftPipeDesc{};
    driftPipeDesc.layout = driftPL;
    driftPipeDesc.compute.module = driftShader;
    driftPipeDesc.compute.entryPoint = "main";
    driftPipeline_ = wgpuDeviceCreateComputePipeline(device, &driftPipeDesc);
    if (!driftPipeline_)
        spdlog::error("Failed to create drift compute pipeline!");
    else
        spdlog::info("Drift compute pipeline created");
    wgpuShaderModuleRelease(driftShader);
    wgpuPipelineLayoutRelease(driftPL);

    // --- Direct force pipeline: params(uniform), positions(ro), accelerations(rw) ---
    WGPUBindGroupLayoutEntry directEntries[3] = {};
    directEntries[0].binding = 0;
    directEntries[0].visibility = WGPUShaderStage_Compute;
    directEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    directEntries[0].buffer.minBindingSize = 0;
    directEntries[1].binding = 1;
    directEntries[1].visibility = WGPUShaderStage_Compute;
    directEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    directEntries[1].buffer.minBindingSize = 0;
    directEntries[2].binding = 2;
    directEntries[2].visibility = WGPUShaderStage_Compute;
    directEntries[2].buffer.type = WGPUBufferBindingType_Storage;
    directEntries[2].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor directLayoutDesc{};
    directLayoutDesc.entryCount = 3;
    directLayoutDesc.entries = directEntries;
    directForceBindGroupLayout_ =
        wgpuDeviceCreateBindGroupLayout(device, &directLayoutDesc);

    WGPUPipelineLayoutDescriptor directPLDesc{};
    directPLDesc.bindGroupLayoutCount = 1;
    directPLDesc.bindGroupLayouts = &directForceBindGroupLayout_;
    WGPUPipelineLayout directPL =
        wgpuDeviceCreatePipelineLayout(device, &directPLDesc);

    WGPUShaderModule directShader =
        wgpu_utils::createShaderModule(device, kDirectForceShaderSource);
    WGPUComputePipelineDescriptor directPipeDesc{};
    directPipeDesc.layout = directPL;
    directPipeDesc.compute.module = directShader;
    directPipeDesc.compute.entryPoint = "main";
    directForcePipeline_ =
        wgpuDeviceCreateComputePipeline(device, &directPipeDesc);
    if (!directForcePipeline_)
        spdlog::error("Failed to create direct force compute pipeline!");
    else
        spdlog::info("Direct force compute pipeline created (O(N^2))");
    wgpuShaderModuleRelease(directShader);
    wgpuPipelineLayoutRelease(directPL);

    // --- BVH force pipeline: params(uniform), positions(ro), bvhNodes(ro), accelerations(rw) ---
    WGPUBindGroupLayoutEntry bvhEntries[4] = {};
    bvhEntries[0].binding = 0;
    bvhEntries[0].visibility = WGPUShaderStage_Compute;
    bvhEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    bvhEntries[0].buffer.minBindingSize = 0;
    bvhEntries[1].binding = 1;
    bvhEntries[1].visibility = WGPUShaderStage_Compute;
    bvhEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    bvhEntries[1].buffer.minBindingSize = 0;
    bvhEntries[2].binding = 2;
    bvhEntries[2].visibility = WGPUShaderStage_Compute;
    bvhEntries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    bvhEntries[2].buffer.minBindingSize = 0;
    bvhEntries[3].binding = 3;
    bvhEntries[3].visibility = WGPUShaderStage_Compute;
    bvhEntries[3].buffer.type = WGPUBufferBindingType_Storage;
    bvhEntries[3].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor bvhLayoutDesc{};
    bvhLayoutDesc.entryCount = 4;
    bvhLayoutDesc.entries = bvhEntries;
    bvhForceBindGroupLayout_ =
        wgpuDeviceCreateBindGroupLayout(device, &bvhLayoutDesc);

    WGPUPipelineLayoutDescriptor bvhPLDesc{};
    bvhPLDesc.bindGroupLayoutCount = 1;
    bvhPLDesc.bindGroupLayouts = &bvhForceBindGroupLayout_;
    WGPUPipelineLayout bvhPL =
        wgpuDeviceCreatePipelineLayout(device, &bvhPLDesc);

    WGPUShaderModule bvhShader =
        wgpu_utils::createShaderModule(device, kBvhForceShaderSource);
    WGPUComputePipelineDescriptor bvhPipeDesc{};
    bvhPipeDesc.layout = bvhPL;
    bvhPipeDesc.compute.module = bvhShader;
    bvhPipeDesc.compute.entryPoint = "main";
    bvhForcePipeline_ =
        wgpuDeviceCreateComputePipeline(device, &bvhPipeDesc);
    if (!bvhForcePipeline_)
        spdlog::error("Failed to create BVH force compute pipeline!");
    else
        spdlog::info("BVH force compute pipeline created");
    wgpuShaderModuleRelease(bvhShader);
    wgpuPipelineLayoutRelease(bvhPL);

    pipelinesCreated_ = true;
}

// ============================================================
// Initialize / Reinitialize
// ============================================================

void Simulation::initialize(WGPUDevice device, WGPUQueue queue,
                            const Config &config) {
    numParticles_ = config.numParticles;
    scenario_ = config.scenario;
    forceMethod_ = config.forceMethod;
    initParticles(config);

    positions_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                    WGPUBufferUsage_CopySrc,
        numParticles_ * sizeof(glm::vec4));
    velocities_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                    WGPUBufferUsage_CopySrc,
        numParticles_ * sizeof(glm::vec4));
    colors_.initialize(device,
                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                       numParticles_ * sizeof(glm::vec4));
    accelerations_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        numParticles_ * sizeof(glm::vec4));

    positions_.upload(queue, cpuPositions_.data(),
                      numParticles_ * sizeof(glm::vec4));
    velocities_.upload(queue, cpuVelocities_.data(),
                       numParticles_ * sizeof(glm::vec4));
    colors_.upload(queue, cpuColors_.data(),
                   numParticles_ * sizeof(glm::vec4));

    paramsBuffer_.initialize(
        device, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 32);

    createComputePipelines(device);

    gpuTreeBuilder_.initialize(device, 100000u);

    // Compute initial accelerations (required for leapfrog bootstrap)
    computeInitialForces(device, queue, config.softening, config.theta);

    spdlog::info("Simulation initialized: {} particles, {}, {}",
                 numParticles_, scenarioName(scenario_),
                 forceMethodName(forceMethod_));
}

void Simulation::reinitialize(WGPUDevice device, WGPUQueue queue,
                              const Config &config) {
    numParticles_ = config.numParticles;
    scenario_ = config.scenario;
    forceMethod_ = config.forceMethod;
    initParticles(config);

    positions_.upload(queue, cpuPositions_.data(),
                      numParticles_ * sizeof(glm::vec4));
    velocities_.upload(queue, cpuVelocities_.data(),
                       numParticles_ * sizeof(glm::vec4));
    colors_.upload(queue, cpuColors_.data(),
                   numParticles_ * sizeof(glm::vec4));

    invalidateBindGroups();
    computeInitialForces(device, queue, config.softening, config.theta);
}

void Simulation::invalidateBindGroups() {
    if (cachedKickBG_) { wgpuBindGroupRelease(cachedKickBG_); cachedKickBG_ = nullptr; }
    if (cachedDriftBG_) { wgpuBindGroupRelease(cachedDriftBG_); cachedDriftBG_ = nullptr; }
    if (cachedBvhForceBG_) { wgpuBindGroupRelease(cachedBvhForceBG_); cachedBvhForceBG_ = nullptr; }
    cachedBGParticleCount_ = 0;
}

void Simulation::ensureBindGroupsCached(WGPUDevice device) {
    if (cachedBGParticleCount_ == numParticles_) return;
    invalidateBindGroups();

    uint32_t N = static_cast<uint32_t>(numParticles_);
    uint32_t nodeCount = 2 * N - 1;

    auto makeBG = [&](WGPUBindGroupLayout layout,
                      WGPUBindGroupEntry *entries, uint32_t count) -> WGPUBindGroup {
        WGPUBindGroupDescriptor desc{};
        desc.layout = layout;
        desc.entryCount = count;
        desc.entries = entries;
        return wgpuDeviceCreateBindGroup(device, &desc);
    };

    // Kick bind group (params, velocities, accelerations)
    {
        WGPUBindGroupEntry e[3] = {};
        e[0].binding = 0; e[0].buffer = paramsBuffer_.get(); e[0].size = 32;
        e[1].binding = 1; e[1].buffer = velocities_.get(); e[1].size = N * sizeof(glm::vec4);
        e[2].binding = 2; e[2].buffer = accelerations_.get(); e[2].size = N * sizeof(glm::vec4);
        cachedKickBG_ = makeBG(kickBindGroupLayout_, e, 3);
    }

    // Drift bind group (params, positions, velocities)
    {
        WGPUBindGroupEntry e[3] = {};
        e[0].binding = 0; e[0].buffer = paramsBuffer_.get(); e[0].size = 32;
        e[1].binding = 1; e[1].buffer = positions_.get(); e[1].size = N * sizeof(glm::vec4);
        e[2].binding = 2; e[2].buffer = velocities_.get(); e[2].size = N * sizeof(glm::vec4);
        cachedDriftBG_ = makeBG(driftBindGroupLayout_, e, 3);
    }

    // BVH force bind group (params, positions, bvhNodes, accelerations)
    {
        WGPUBindGroupEntry e[4] = {};
        e[0].binding = 0; e[0].buffer = paramsBuffer_.get(); e[0].size = 32;
        e[1].binding = 1; e[1].buffer = positions_.get(); e[1].size = N * sizeof(glm::vec4);
        e[2].binding = 2; e[2].buffer = gpuTreeBuilder_.getBvhNodesBuffer(); e[2].size = nodeCount * GpuTreeBuilder::kNodeSize;
        e[3].binding = 3; e[3].buffer = accelerations_.get(); e[3].size = N * sizeof(glm::vec4);
        cachedBvhForceBG_ = makeBG(bvhForceBindGroupLayout_, e, 4);
    }

    cachedBGParticleCount_ = numParticles_;
}

void Simulation::debugDumpTree(WGPUDevice device, WGPUQueue queue) {
    uint32_t N = static_cast<uint32_t>(numParticles_);
    uint32_t nc = 2 * N - 1;
    size_t treeSz = nc * sizeof(BVHNodeGPU);

    WGPUBufferDescriptor sd{};
    sd.size = treeSz;
    sd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    sd.mappedAtCreation = false;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &sd);

    WGPUCommandEncoder enc = wgpu_utils::createCommandEncoder(device);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gpuTreeBuilder_.getBvhNodesBuffer(), 0, staging, 0, treeSz);
    wgpu_utils::finishCommandEncoder(queue, enc);

    const void *data = wgpu_utils::mapBufferSync(device, staging, treeSz);
    if (data) {
        auto *nodes = reinterpret_cast<const BVHNodeGPU*>(data);
        for (uint32_t n = 0; n < std::min(nc, 20u); n++) {
            bool isInt = n < N - 1;
            spdlog::info("  n[{}]{}: COM=({:.3f},{:.3f},{:.3f},m={:.3f}) L={} R={} P={}",
                n, isInt ? "I" : "L",
                nodes[n].centerOfMass.x, nodes[n].centerOfMass.y,
                nodes[n].centerOfMass.z, nodes[n].centerOfMass.w,
                nodes[n].left, nodes[n].right, nodes[n].parent);
        }
        wgpuBufferUnmap(staging);
    } else {
        spdlog::error("Tree readback failed!");
    }
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
}

// ============================================================
// GPU Readback (for diagnostics when using GPU tree path)
// ============================================================

void Simulation::readbackState(WGPUDevice device, WGPUQueue queue,
                                std::vector<glm::vec4> &outPositions,
                                std::vector<glm::vec4> &outVelocities) {
    size_t bufSize = numParticles_ * sizeof(glm::vec4);

    // Create staging buffers with MapRead | CopyDst
    WGPUBufferDescriptor stagingDesc{};
    stagingDesc.size = bufSize;
    stagingDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    stagingDesc.mappedAtCreation = false;

    WGPUBuffer stagingPos = wgpuDeviceCreateBuffer(device, &stagingDesc);
    WGPUBuffer stagingVel = wgpuDeviceCreateBuffer(device, &stagingDesc);

    // Copy storage -> staging
    WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, positions_.get(), 0,
                                          stagingPos, 0, bufSize);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, velocities_.get(), 0,
                                          stagingVel, 0, bufSize);
    wgpu_utils::finishCommandEncoder(queue, encoder);

    // Map and read positions
    outPositions.resize(numParticles_);
    const void *posData =
        wgpu_utils::mapBufferSync(device, stagingPos, bufSize);
    if (posData) {
        std::memcpy(outPositions.data(), posData, bufSize);
        wgpuBufferUnmap(stagingPos);
    }

    // Map and read velocities
    outVelocities.resize(numParticles_);
    const void *velData =
        wgpu_utils::mapBufferSync(device, stagingVel, bufSize);
    if (velData) {
        std::memcpy(outVelocities.data(), velData, bufSize);
        wgpuBufferUnmap(stagingVel);
    }

    // Cleanup staging buffers
    wgpuBufferDestroy(stagingPos);
    wgpuBufferRelease(stagingPos);
    wgpuBufferDestroy(stagingVel);
    wgpuBufferRelease(stagingVel);
}

// ============================================================
// Step Dispatch
// ============================================================

void Simulation::step(WGPUDevice device, WGPUQueue queue, float dt,
                      float softening, float theta) {
    if (forceMethod_ == ForceMethod::Direct) {
        stepWithDirectForce(device, queue, dt, softening, theta);
    } else {
        stepLeapfrogGpuTree(device, queue, dt, softening, theta);
    }
}

// ============================================================
// Direct Summation O(N²) Step (Leapfrog with direct force)
// ============================================================

void Simulation::stepWithDirectForce(WGPUDevice device, WGPUQueue queue,
                                      float dt, float softening, float theta) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // Upload params
    struct alignas(16) Params {
        float dt;
        float softening;
        float theta;
        uint32_t numParticles;
        uint32_t nodeCount;
        uint32_t paddedN;
        float _pad[2];
    } params;
    params.dt = dt;
    params.softening = softening;
    params.theta = theta;
    params.numParticles = static_cast<uint32_t>(numParticles_);
    params.nodeCount = 0;
    params.paddedN = 0;
    params._pad[0] = params._pad[1] = 0.0f;
    paramsBuffer_.upload(queue, &params, sizeof(params));

    auto t1 = Clock::now();

    // GPU: half-kick, drift, direct force, second kick
    {
        // Kick bind group
        WGPUBindGroupEntry kickBE[3] = {};
        kickBE[0].binding = 0;
        kickBE[0].buffer = paramsBuffer_.get();
        kickBE[0].offset = 0;
        kickBE[0].size = sizeof(params);
        kickBE[1].binding = 1;
        kickBE[1].buffer = velocities_.get();
        kickBE[1].offset = 0;
        kickBE[1].size = numParticles_ * sizeof(glm::vec4);
        kickBE[2].binding = 2;
        kickBE[2].buffer = accelerations_.get();
        kickBE[2].offset = 0;
        kickBE[2].size = numParticles_ * sizeof(glm::vec4);

        WGPUBindGroupDescriptor kickBGD{};
        kickBGD.layout = kickBindGroupLayout_;
        kickBGD.entryCount = 3;
        kickBGD.entries = kickBE;
        WGPUBindGroup kickBG = wgpuDeviceCreateBindGroup(device, &kickBGD);

        // Drift bind group
        WGPUBindGroupEntry driftBE[3] = {};
        driftBE[0].binding = 0;
        driftBE[0].buffer = paramsBuffer_.get();
        driftBE[0].offset = 0;
        driftBE[0].size = sizeof(params);
        driftBE[1].binding = 1;
        driftBE[1].buffer = positions_.get();
        driftBE[1].offset = 0;
        driftBE[1].size = numParticles_ * sizeof(glm::vec4);
        driftBE[2].binding = 2;
        driftBE[2].buffer = velocities_.get();
        driftBE[2].offset = 0;
        driftBE[2].size = numParticles_ * sizeof(glm::vec4);

        WGPUBindGroupDescriptor driftBGD{};
        driftBGD.layout = driftBindGroupLayout_;
        driftBGD.entryCount = 3;
        driftBGD.entries = driftBE;
        WGPUBindGroup driftBG = wgpuDeviceCreateBindGroup(device, &driftBGD);

        // Direct force bind group
        WGPUBindGroupEntry directBE[3] = {};
        directBE[0].binding = 0;
        directBE[0].buffer = paramsBuffer_.get();
        directBE[0].offset = 0;
        directBE[0].size = sizeof(params);
        directBE[1].binding = 1;
        directBE[1].buffer = positions_.get();
        directBE[1].offset = 0;
        directBE[1].size = numParticles_ * sizeof(glm::vec4);
        directBE[2].binding = 2;
        directBE[2].buffer = accelerations_.get();
        directBE[2].offset = 0;
        directBE[2].size = numParticles_ * sizeof(glm::vec4);

        WGPUBindGroupDescriptor directBGD{};
        directBGD.layout = directForceBindGroupLayout_;
        directBGD.entryCount = 3;
        directBGD.entries = directBE;
        WGPUBindGroup directBG = wgpuDeviceCreateBindGroup(device, &directBGD);

        WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
        WGPUComputePassDescriptor passDesc{};
        passDesc.timestampWrites = nullptr;

        uint32_t wg256 = (numParticles_ + 255) / 256;
        uint32_t wg64 = (numParticles_ + 63) / 64;

        // Kick pass (half-kick)
        WGPUComputePassEncoder kickPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(kickPass, kickPipeline_);
        wgpuComputePassEncoderSetBindGroup(kickPass, 0, kickBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(kickPass, wg256, 1, 1);
        wgpuComputePassEncoderEnd(kickPass);
        wgpuComputePassEncoderRelease(kickPass);

        // Drift pass
        WGPUComputePassEncoder driftPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(driftPass, driftPipeline_);
        wgpuComputePassEncoderSetBindGroup(driftPass, 0, driftBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(driftPass, wg256, 1, 1);
        wgpuComputePassEncoderEnd(driftPass);
        wgpuComputePassEncoderRelease(driftPass);

        // Direct force pass
        WGPUComputePassEncoder forcePass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(forcePass, directForcePipeline_);
        wgpuComputePassEncoderSetBindGroup(forcePass, 0, directBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(forcePass, wg64, 1, 1);
        wgpuComputePassEncoderEnd(forcePass);
        wgpuComputePassEncoderRelease(forcePass);

        // Second kick pass
        WGPUComputePassEncoder kickPass2 =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(kickPass2, kickPipeline_);
        wgpuComputePassEncoderSetBindGroup(kickPass2, 0, kickBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(kickPass2, wg256, 1, 1);
        wgpuComputePassEncoderEnd(kickPass2);
        wgpuComputePassEncoderRelease(kickPass2);

        wgpu_utils::finishCommandEncoder(queue, encoder);
        wgpuBindGroupRelease(kickBG);
        wgpuBindGroupRelease(driftBG);
        wgpuBindGroupRelease(directBG);
    }

    auto t2 = Clock::now();

    lastTiming_.treeBuildMs = 0.0;
    lastTiming_.forceMs =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    lastTiming_.integrateMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================
// GPU Tree (LBVH) Leapfrog Step
// ============================================================

void Simulation::stepLeapfrogGpuTree(WGPUDevice device, WGPUQueue queue,
                                      float dt, float softening, float theta) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    uint32_t N = static_cast<uint32_t>(numParticles_);
    uint32_t paddedN = gpuTreeBuilder_.getPaddedN(N);
    uint32_t nodeCount = 2 * N - 1;

    // Upload params with BVH node count and paddedN
    struct alignas(16) Params {
        float dt;
        float softening;
        float theta;
        uint32_t numParticles;
        uint32_t nodeCount;
        uint32_t paddedN;
        float _pad[2];
    } params;
    params.dt = dt;
    params.softening = softening;
    params.theta = theta;
    params.numParticles = N;
    params.nodeCount = nodeCount;
    params.paddedN = paddedN;
    params._pad[0] = params._pad[1] = 0.0f;
    paramsBuffer_.upload(queue, &params, sizeof(params));

    auto t1 = Clock::now();

    // Upload tree builder data BEFORE creating the command encoder
    // to avoid wgpuQueueWriteBuffer + wgpuPollEvents mid-recording
    gpuTreeBuilder_.prepareUploads(queue, N);

    // Single command encoder for the entire GPU step
    WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
    WGPUComputePassDescriptor passDesc{};
    passDesc.timestampWrites = nullptr;

    uint32_t wg256 = (numParticles_ + 255) / 256;
    uint32_t wg64 = (numParticles_ + 63) / 64;

    // Ensure cached bind groups exist
    ensureBindGroupsCached(device);

    // --- Half-kick pass ---
    {
        WGPUComputePassEncoder kickPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(kickPass, kickPipeline_);
        wgpuComputePassEncoderSetBindGroup(kickPass, 0, cachedKickBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(kickPass, wg256, 1, 1);
        wgpuComputePassEncoderEnd(kickPass);
        wgpuComputePassEncoderRelease(kickPass);
    }

    // --- Drift pass ---
    {
        WGPUComputePassEncoder driftPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(driftPass, driftPipeline_);
        wgpuComputePassEncoderSetBindGroup(driftPass, 0, cachedDriftBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(driftPass, wg256, 1, 1);
        wgpuComputePassEncoderEnd(driftPass);
        wgpuComputePassEncoderRelease(driftPass);
    }

    auto t2 = Clock::now();

    // --- GPU tree build (~7+ compute passes) ---
    gpuTreeBuilder_.recordTreeBuild(device, encoder,
                                     positions_.get(), paramsBuffer_.get(), N);

    auto t3 = Clock::now();

    // --- BVH force pass ---
    {
        WGPUComputePassEncoder forcePass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(forcePass, bvhForcePipeline_);
        wgpuComputePassEncoderSetBindGroup(forcePass, 0, cachedBvhForceBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(forcePass, wg64, 1, 1);
        wgpuComputePassEncoderEnd(forcePass);
        wgpuComputePassEncoderRelease(forcePass);
    }

    // --- Second half-kick pass (reuses cached kick bind group) ---
    {
        WGPUComputePassEncoder kickPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(kickPass, kickPipeline_);
        wgpuComputePassEncoderSetBindGroup(kickPass, 0, cachedKickBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(kickPass, wg256, 1, 1);
        wgpuComputePassEncoderEnd(kickPass);
        wgpuComputePassEncoderRelease(kickPass);
    }

    // Submit everything
    wgpu_utils::finishCommandEncoder(queue, encoder);

    auto t4 = Clock::now();

    lastTiming_.integrateMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    lastTiming_.treeBuildMs =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    lastTiming_.forceMs =
        std::chrono::duration<double, std::milli>(t4 - t3).count();
}
