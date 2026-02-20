#include "simulation.hpp"
#include <chrono>
#include <cmath>
#include <random>
#include <spdlog/spdlog.h>

static constexpr float PI = 3.14159265358979323846f;

// ============================================================
// WGSL Shader Sources
// ============================================================

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

        let hasChildren = (node.c0 >= 0) || (node.c1 >= 0) || (node.c2 >= 0) || (node.c3 >= 0)
                       || (node.c4 >= 0) || (node.c5 >= 0) || (node.c6 >= 0) || (node.c7 >= 0);

        if (!hasChildren || (halfWidth * halfWidth / distSq < thetaSq)) {
            if (mass > 0.0 && distSq > softSq * 0.01) {
                let invDist = inverseSqrt(distSq);
                let invDist3 = invDist * invDist * invDist;
                acc += G * mass * diff * invDist3;
            }
        } else {
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

// Kick shader: v += a * dt/2
static const char *kKickShaderSource = R"(
struct Params {
    dt: f32,
    softening: f32,
    theta: f32,
    numParticles: u32,
    nodeCount: u32,
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

// Euler integrate shader (fallback)
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

// ============================================================
// Particle Initialization — Scenario Dispatch
// ============================================================

void Simulation::initParticles(const Config &config) {
    cpuPositions_.resize(numParticles_);
    cpuVelocities_.resize(numParticles_);
    cpuColors_.resize(numParticles_);
    cpuAccelerations_.resize(numParticles_, glm::vec4(0.0f));

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
// CPU Barnes-Hut Force Computation (mirror)
// ============================================================

void Simulation::cpuBarnesHut(float softening, float theta) {
    const auto &octreeNodes = octree_.getNodes();
    size_t nodeCount = octree_.nodeCount();
    float thetaSq = theta * theta;
    float softSq = softening * softening;

    for (int i = 0; i < numParticles_; i++) {
        glm::vec3 pos(cpuPositions_[i]);
        glm::vec3 acc(0.0f);

        int stack[64];
        stack[0] = 0;
        int top = 1;

        while (top > 0) {
            top--;
            int nodeIdx = stack[top];
            if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= nodeCount)
                continue;

            const OctreeNode &node = octreeNodes[nodeIdx];
            glm::vec3 diff(node.centerOfMass.x - pos.x,
                           node.centerOfMass.y - pos.y,
                           node.centerOfMass.z - pos.z);
            float distSq = glm::dot(diff, diff) + softSq;
            float halfWidth = node.bounds.w;
            float mass = node.centerOfMass.w;

            bool isLeaf = true;
            for (int c = 0; c < 8; c++) {
                if (node.children[c] >= 0) {
                    isLeaf = false;
                    break;
                }
            }

            if (isLeaf || (halfWidth * halfWidth / distSq < thetaSq)) {
                if (mass > 0.0f && distSq > softSq * 0.01f) {
                    float invDist = 1.0f / std::sqrt(distSq);
                    float invDist3 = invDist * invDist * invDist;
                    acc += mass * diff * invDist3;
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

        cpuAccelerations_[i] = glm::vec4(acc, 0.0f);
    }
}

// ============================================================
// Compute Initial Forces (for leapfrog bootstrap)
// ============================================================

void Simulation::computeInitialForces(WGPUQueue queue, float softening,
                                      float theta) {
    octree_.build(cpuPositions_);
    size_t nodeCount = octree_.nodeCount();
    if (nodeCount == 0) return;

    // Upload octree for GPU (used on first step)
    auto &nodes = octree_.getNodes();
    octreeBuffer_.upload(queue, (void *)nodes.data(),
                         nodeCount * sizeof(OctreeNode));

    // CPU force computation
    cpuBarnesHut(softening, theta);

    // Upload CPU-computed accelerations to GPU so both start synchronized
    accelerations_.upload(queue, cpuAccelerations_.data(),
                          numParticles_ * sizeof(glm::vec4));

    spdlog::info("Initial forces computed ({} octree nodes)", nodeCount);
}

// ============================================================
// Pipeline Creation
// ============================================================

void Simulation::createComputePipelines(WGPUDevice device) {
    if (pipelinesCreated_) return;

    // --- Force pipeline (unchanged) ---
    WGPUBindGroupLayoutEntry forceEntries[4] = {};
    forceEntries[0].binding = 0;
    forceEntries[0].visibility = WGPUShaderStage_Compute;
    forceEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    forceEntries[0].buffer.minBindingSize = 0;
    forceEntries[1].binding = 1;
    forceEntries[1].visibility = WGPUShaderStage_Compute;
    forceEntries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    forceEntries[1].buffer.minBindingSize = 0;
    forceEntries[2].binding = 2;
    forceEntries[2].visibility = WGPUShaderStage_Compute;
    forceEntries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    forceEntries[2].buffer.minBindingSize = 0;
    forceEntries[3].binding = 3;
    forceEntries[3].visibility = WGPUShaderStage_Compute;
    forceEntries[3].buffer.type = WGPUBufferBindingType_Storage;
    forceEntries[3].buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor forceLayoutDesc{};
    forceLayoutDesc.entryCount = 4;
    forceLayoutDesc.entries = forceEntries;
    forceBindGroupLayout_ =
        wgpuDeviceCreateBindGroupLayout(device, &forceLayoutDesc);

    WGPUPipelineLayoutDescriptor forcePLDesc{};
    forcePLDesc.bindGroupLayoutCount = 1;
    forcePLDesc.bindGroupLayouts = &forceBindGroupLayout_;
    WGPUPipelineLayout forcePL =
        wgpuDeviceCreatePipelineLayout(device, &forcePLDesc);

    WGPUShaderModule forceShader =
        wgpu_utils::createShaderModule(device, kForceShaderSource);
    WGPUComputePipelineDescriptor forcePipeDesc{};
    forcePipeDesc.layout = forcePL;
    forcePipeDesc.compute.module = forceShader;
    forcePipeDesc.compute.entryPoint = "main";
    forcePipeline_ = wgpuDeviceCreateComputePipeline(device, &forcePipeDesc);
    if (!forcePipeline_)
        spdlog::error("Failed to create force compute pipeline!");
    else
        spdlog::info("Force compute pipeline created");
    wgpuShaderModuleRelease(forceShader);
    wgpuPipelineLayoutRelease(forcePL);

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

    // --- Euler integrate pipeline (fallback, unchanged layout) ---
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
    integrateBindGroupLayout_ =
        wgpuDeviceCreateBindGroupLayout(device, &intLayoutDesc);

    WGPUPipelineLayoutDescriptor intPLDesc{};
    intPLDesc.bindGroupLayoutCount = 1;
    intPLDesc.bindGroupLayouts = &integrateBindGroupLayout_;
    WGPUPipelineLayout intPL =
        wgpuDeviceCreatePipelineLayout(device, &intPLDesc);

    WGPUShaderModule intShader =
        wgpu_utils::createShaderModule(device, kIntegrateShaderSource);
    WGPUComputePipelineDescriptor intPipeDesc{};
    intPipeDesc.layout = intPL;
    intPipeDesc.compute.module = intShader;
    intPipeDesc.compute.entryPoint = "main";
    integratePipeline_ =
        wgpuDeviceCreateComputePipeline(device, &intPipeDesc);
    if (!integratePipeline_)
        spdlog::error("Failed to create integrate compute pipeline!");
    else
        spdlog::info("Integrate compute pipeline created (Euler fallback)");
    wgpuShaderModuleRelease(intShader);
    wgpuPipelineLayoutRelease(intPL);

    pipelinesCreated_ = true;
}

// ============================================================
// Initialize / Reinitialize
// ============================================================

void Simulation::initialize(WGPUDevice device, WGPUQueue queue,
                            const Config &config) {
    numParticles_ = config.numParticles;
    integrator_ = config.integrator;
    scenario_ = config.scenario;
    initParticles(config);

    positions_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                    WGPUBufferUsage_CopySrc,
        numParticles_ * sizeof(glm::vec4));
    velocities_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        numParticles_ * sizeof(glm::vec4));
    colors_.initialize(device,
                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                       numParticles_ * sizeof(glm::vec4));
    accelerations_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        numParticles_ * sizeof(glm::vec4));

    positions_.upload(queue, cpuPositions_.data(),
                      numParticles_ * sizeof(glm::vec4));
    velocities_.upload(queue, cpuVelocities_.data(),
                       numParticles_ * sizeof(glm::vec4));
    colors_.upload(queue, cpuColors_.data(),
                   numParticles_ * sizeof(glm::vec4));

    octreeBuffer_.initialize(
        device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        sizeof(OctreeNode) * 8);

    paramsBuffer_.initialize(
        device, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 32);

    createComputePipelines(device);

    // Compute initial accelerations (required for leapfrog bootstrap)
    computeInitialForces(queue, config.softening, config.theta);

    spdlog::info("Simulation initialized: {} particles, {}, {}",
                 numParticles_, scenarioName(scenario_),
                 integratorName(integrator_));
}

void Simulation::reinitialize(WGPUDevice device, WGPUQueue queue,
                              const Config &config) {
    numParticles_ = config.numParticles;
    integrator_ = config.integrator;
    scenario_ = config.scenario;
    initParticles(config);

    positions_.upload(queue, cpuPositions_.data(),
                      numParticles_ * sizeof(glm::vec4));
    velocities_.upload(queue, cpuVelocities_.data(),
                       numParticles_ * sizeof(glm::vec4));
    colors_.upload(queue, cpuColors_.data(),
                   numParticles_ * sizeof(glm::vec4));

    computeInitialForces(queue, config.softening, config.theta);
}

// ============================================================
// Step Dispatch
// ============================================================

void Simulation::step(WGPUDevice device, WGPUQueue queue, float dt,
                      float softening, float theta) {
    if (integrator_ == Integrator::Leapfrog) {
        stepLeapfrog(device, queue, dt, softening, theta);
    } else {
        stepEuler(device, queue, dt, softening, theta);
    }
}

// ============================================================
// KDK Leapfrog Step
// ============================================================

void Simulation::stepLeapfrog(WGPUDevice device, WGPUQueue queue, float dt,
                              float softening, float theta) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // --- Upload params (nodeCount=0, unused by kick/drift) ---
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
    params.nodeCount = 0;
    params._pad[0] = params._pad[1] = params._pad[2] = 0.0f;
    paramsBuffer_.upload(queue, &params, sizeof(params));

    // --- Phase 1: Half-kick + Drift (CPU mirror) ---
    for (int i = 0; i < numParticles_; i++) {
        glm::vec3 vel(cpuVelocities_[i]);
        glm::vec3 acc(cpuAccelerations_[i]);
        vel += acc * (dt * 0.5f);
        cpuVelocities_[i] = glm::vec4(vel, 0.0f);
    }
    for (int i = 0; i < numParticles_; i++) {
        glm::vec3 pos(cpuPositions_[i]);
        glm::vec3 vel(cpuVelocities_[i]);
        float mass = cpuPositions_[i].w;
        pos += vel * dt;
        cpuPositions_[i] = glm::vec4(pos, mass);
    }

    // --- Phase 1: Half-kick + Drift (GPU) ---
    {
        // Create kick bind group
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

        // Create drift bind group
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

        WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
        WGPUComputePassDescriptor passDesc{};
        passDesc.timestampWrites = nullptr;
        uint32_t wg = (numParticles_ + 255) / 256;

        // Kick pass
        WGPUComputePassEncoder kickPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(kickPass, kickPipeline_);
        wgpuComputePassEncoderSetBindGroup(kickPass, 0, kickBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(kickPass, wg, 1, 1);
        wgpuComputePassEncoderEnd(kickPass);
        wgpuComputePassEncoderRelease(kickPass);

        // Drift pass
        WGPUComputePassEncoder driftPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(driftPass, driftPipeline_);
        wgpuComputePassEncoderSetBindGroup(driftPass, 0, driftBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(driftPass, wg, 1, 1);
        wgpuComputePassEncoderEnd(driftPass);
        wgpuComputePassEncoderRelease(driftPass);

        wgpu_utils::finishCommandEncoder(queue, encoder);
        wgpuBindGroupRelease(kickBG);
        wgpuBindGroupRelease(driftBG);
    }

    auto t1 = Clock::now();

    // --- Phase 2: Octree build (CPU) ---
    octree_.build(cpuPositions_);
    size_t nodeCount = octree_.nodeCount();
    if (nodeCount == 0) return;

    auto t2 = Clock::now();

    // --- Phase 3: Force computation ---

    // Upload octree
    auto &nodes = octree_.getNodes();
    octreeBuffer_.upload(queue, (void *)nodes.data(),
                         nodeCount * sizeof(OctreeNode));

    // Re-upload params with correct nodeCount
    params.nodeCount = static_cast<uint32_t>(nodeCount);
    paramsBuffer_.upload(queue, &params, sizeof(params));

    // CPU force computation
    cpuBarnesHut(softening, theta);

    // GPU: force + second kick
    {
        // Force bind group
        WGPUBindGroupEntry forceBE[4] = {};
        forceBE[0].binding = 0;
        forceBE[0].buffer = paramsBuffer_.get();
        forceBE[0].offset = 0;
        forceBE[0].size = sizeof(params);
        forceBE[1].binding = 1;
        forceBE[1].buffer = positions_.get();
        forceBE[1].offset = 0;
        forceBE[1].size = numParticles_ * sizeof(glm::vec4);
        forceBE[2].binding = 2;
        forceBE[2].buffer = octreeBuffer_.get();
        forceBE[2].offset = 0;
        forceBE[2].size = nodeCount * sizeof(OctreeNode);
        forceBE[3].binding = 3;
        forceBE[3].buffer = accelerations_.get();
        forceBE[3].offset = 0;
        forceBE[3].size = numParticles_ * sizeof(glm::vec4);

        WGPUBindGroupDescriptor forceBGD{};
        forceBGD.layout = forceBindGroupLayout_;
        forceBGD.entryCount = 4;
        forceBGD.entries = forceBE;
        WGPUBindGroup forceBG = wgpuDeviceCreateBindGroup(device, &forceBGD);

        // Kick bind group (for second half-kick, same layout)
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

        WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
        WGPUComputePassDescriptor passDesc{};
        passDesc.timestampWrites = nullptr;

        // Force pass
        WGPUComputePassEncoder forcePass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(forcePass, forcePipeline_);
        wgpuComputePassEncoderSetBindGroup(forcePass, 0, forceBG, 0, nullptr);
        uint32_t forceWG = (numParticles_ + 63) / 64;
        wgpuComputePassEncoderDispatchWorkgroups(forcePass, forceWG, 1, 1);
        wgpuComputePassEncoderEnd(forcePass);
        wgpuComputePassEncoderRelease(forcePass);

        // Second kick pass
        WGPUComputePassEncoder kickPass =
            wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(kickPass, kickPipeline_);
        wgpuComputePassEncoderSetBindGroup(kickPass, 0, kickBG, 0, nullptr);
        uint32_t kickWG = (numParticles_ + 255) / 256;
        wgpuComputePassEncoderDispatchWorkgroups(kickPass, kickWG, 1, 1);
        wgpuComputePassEncoderEnd(kickPass);
        wgpuComputePassEncoderRelease(kickPass);

        wgpu_utils::finishCommandEncoder(queue, encoder);
        wgpuBindGroupRelease(forceBG);
        wgpuBindGroupRelease(kickBG);
    }

    // CPU: second half-kick
    for (int i = 0; i < numParticles_; i++) {
        glm::vec3 vel(cpuVelocities_[i]);
        glm::vec3 acc(cpuAccelerations_[i]);
        vel += acc * (dt * 0.5f);
        cpuVelocities_[i] = glm::vec4(vel, 0.0f);
    }

    auto t3 = Clock::now();

    // Store timing
    lastTiming_.integrateMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    lastTiming_.treeBuildMs =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    lastTiming_.forceMs =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
}

// ============================================================
// Euler Step (original behavior, preserved as fallback)
// ============================================================

void Simulation::stepEuler(WGPUDevice device, WGPUQueue queue, float dt,
                           float softening, float theta) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // Build octree on CPU
    octree_.build(cpuPositions_);
    size_t nodeCount = octree_.nodeCount();
    if (nodeCount == 0) return;

    auto t1 = Clock::now();

    // Upload octree nodes
    auto &nodes = octree_.getNodes();
    octreeBuffer_.upload(queue, (void *)nodes.data(),
                         nodeCount * sizeof(OctreeNode));

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
    WGPUBindGroupEntry forceBE[4] = {};
    forceBE[0].binding = 0;
    forceBE[0].buffer = paramsBuffer_.get();
    forceBE[0].offset = 0;
    forceBE[0].size = sizeof(params);
    forceBE[1].binding = 1;
    forceBE[1].buffer = positions_.get();
    forceBE[1].offset = 0;
    forceBE[1].size = numParticles_ * sizeof(glm::vec4);
    forceBE[2].binding = 2;
    forceBE[2].buffer = octreeBuffer_.get();
    forceBE[2].offset = 0;
    forceBE[2].size = nodeCount * sizeof(OctreeNode);
    forceBE[3].binding = 3;
    forceBE[3].buffer = accelerations_.get();
    forceBE[3].offset = 0;
    forceBE[3].size = numParticles_ * sizeof(glm::vec4);

    WGPUBindGroupDescriptor forceBGD{};
    forceBGD.layout = forceBindGroupLayout_;
    forceBGD.entryCount = 4;
    forceBGD.entries = forceBE;
    WGPUBindGroup forceBG = wgpuDeviceCreateBindGroup(device, &forceBGD);

    WGPUBindGroupEntry intBE[4] = {};
    intBE[0].binding = 0;
    intBE[0].buffer = paramsBuffer_.get();
    intBE[0].offset = 0;
    intBE[0].size = sizeof(params);
    intBE[1].binding = 1;
    intBE[1].buffer = positions_.get();
    intBE[1].offset = 0;
    intBE[1].size = numParticles_ * sizeof(glm::vec4);
    intBE[2].binding = 2;
    intBE[2].buffer = velocities_.get();
    intBE[2].offset = 0;
    intBE[2].size = numParticles_ * sizeof(glm::vec4);
    intBE[3].binding = 3;
    intBE[3].buffer = accelerations_.get();
    intBE[3].offset = 0;
    intBE[3].size = numParticles_ * sizeof(glm::vec4);

    WGPUBindGroupDescriptor intBGD{};
    intBGD.layout = integrateBindGroupLayout_;
    intBGD.entryCount = 4;
    intBGD.entries = intBE;
    WGPUBindGroup intBG = wgpuDeviceCreateBindGroup(device, &intBGD);

    // Dispatch
    WGPUCommandEncoder encoder = wgpu_utils::createCommandEncoder(device);
    WGPUComputePassDescriptor passDesc{};
    passDesc.timestampWrites = nullptr;

    WGPUComputePassEncoder forcePass =
        wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderSetPipeline(forcePass, forcePipeline_);
    wgpuComputePassEncoderSetBindGroup(forcePass, 0, forceBG, 0, nullptr);
    uint32_t forceWG = (numParticles_ + 63) / 64;
    wgpuComputePassEncoderDispatchWorkgroups(forcePass, forceWG, 1, 1);
    wgpuComputePassEncoderEnd(forcePass);
    wgpuComputePassEncoderRelease(forcePass);

    WGPUComputePassEncoder intPass =
        wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
    wgpuComputePassEncoderSetPipeline(intPass, integratePipeline_);
    wgpuComputePassEncoderSetBindGroup(intPass, 0, intBG, 0, nullptr);
    uint32_t intWG = (numParticles_ + 255) / 256;
    wgpuComputePassEncoderDispatchWorkgroups(intPass, intWG, 1, 1);
    wgpuComputePassEncoderEnd(intPass);
    wgpuComputePassEncoderRelease(intPass);

    wgpu_utils::finishCommandEncoder(queue, encoder);

    auto t2 = Clock::now();

    // CPU mirror: BH traversal + Euler integration
    cpuBarnesHut(softening, theta);
    for (int i = 0; i < numParticles_; i++) {
        glm::vec3 pos(cpuPositions_[i]);
        glm::vec3 vel(cpuVelocities_[i]);
        glm::vec3 acc(cpuAccelerations_[i]);

        vel += acc * dt;
        pos += vel * dt;
        float m = cpuPositions_[i].w;
        cpuPositions_[i] = glm::vec4(pos, m);
        cpuVelocities_[i] = glm::vec4(vel, 0.0f);
    }

    auto t3 = Clock::now();

    wgpuBindGroupRelease(forceBG);
    wgpuBindGroupRelease(intBG);

    // Store timing
    lastTiming_.treeBuildMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    lastTiming_.forceMs =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    lastTiming_.integrateMs =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
}
