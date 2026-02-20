#include "gpu_tree_builder.hpp"
#include <spdlog/spdlog.h>
#include <cmath>
#include <cstring>

// ============================================================
// WGSL Shader Sources — GPU Tree Construction (LBVH)
// ============================================================

// --- Bbox Pass 1: per-workgroup reduction ---
static const char *kBboxPass1Shader = R"(
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
@group(0) @binding(2) var<storage, read_write> bboxPartial: array<vec4f>;

var<workgroup> sMin: array<vec3f, 256>;
var<workgroup> sMax: array<vec3f, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let i = gid.x;
    let localId = lid.x;

    if (i < params.numParticles) {
        let pos = positions[i].xyz;
        sMin[localId] = pos;
        sMax[localId] = pos;
    } else {
        sMin[localId] = vec3f(1e30);
        sMax[localId] = vec3f(-1e30);
    }

    workgroupBarrier();

    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (localId < stride) {
            sMin[localId] = min(sMin[localId], sMin[localId + stride]);
            sMax[localId] = max(sMax[localId], sMax[localId + stride]);
        }
        workgroupBarrier();
    }

    if (localId == 0u) {
        let wg = wid.x;
        bboxPartial[wg * 2u] = vec4f(sMin[0], 0.0);
        bboxPartial[wg * 2u + 1u] = vec4f(sMax[0], 0.0);
    }
}
)";

// --- Bbox Pass 2: final reduction (1 workgroup) ---
static const char *kBboxPass2Shader = R"(
@group(0) @binding(0) var<storage, read> bboxPartial: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> bboxResult: array<vec4f>;
@group(0) @binding(2) var<uniform> numWorkgroups: u32;

var<workgroup> sMin: array<vec3f, 256>;
var<workgroup> sMax: array<vec3f, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let localId = lid.x;

    if (localId < numWorkgroups) {
        sMin[localId] = bboxPartial[localId * 2u].xyz;
        sMax[localId] = bboxPartial[localId * 2u + 1u].xyz;
    } else {
        sMin[localId] = vec3f(1e30);
        sMax[localId] = vec3f(-1e30);
    }

    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (localId < stride) {
            sMin[localId] = min(sMin[localId], sMin[localId + stride]);
            sMax[localId] = max(sMax[localId], sMax[localId + stride]);
        }
        workgroupBarrier();
    }

    if (localId == 0u) {
        bboxResult[0] = vec4f(sMin[0], 0.0);
        bboxResult[1] = vec4f(sMax[0], 0.0);
    }
}
)";

// --- Morton code computation ---
static const char *kMortonShader = R"(
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
@group(0) @binding(2) var<storage, read> bboxResult: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> mortonCodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> sortIndices: array<u32>;

fn expandBits(v_in: u32) -> u32 {
    var v = v_in;
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

fn morton3D(x: u32, y: u32, z: u32) -> u32 {
    return (expandBits(x) << 2u) | (expandBits(y) << 1u) | expandBits(z);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.paddedN) { return; }

    if (i >= params.numParticles) {
        mortonCodes[i] = 0xFFFFFFFFu;
        sortIndices[i] = i;
        return;
    }

    let bboxMin = bboxResult[0].xyz;
    let bboxMax = bboxResult[1].xyz;
    let extent = bboxMax - bboxMin;
    let safeExtent = max(extent, vec3f(1e-10));

    let pos = positions[i].xyz;
    let norm = (pos - bboxMin) / safeExtent;
    let clamped = clamp(norm, vec3f(0.0), vec3f(1.0));

    let qx = u32(clamped.x * 1023.0);
    let qy = u32(clamped.y * 1023.0);
    let qz = u32(clamped.z * 1023.0);

    mortonCodes[i] = morton3D(qx, qy, qz);
    sortIndices[i] = i;
}
)";

// --- Bitonic sort (key-value on mortonCodes/sortIndices) ---
static const char *kBitonicSortShader = R"(
struct SortParams {
    k: u32,
    j: u32,
    paddedN: u32,
}

@group(0) @binding(0) var<uniform> sortParams: SortParams;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= sortParams.paddedN) { return; }

    let j = sortParams.j;
    let k = sortParams.k;

    let partner = i ^ j;
    if (partner <= i) { return; }
    if (partner >= sortParams.paddedN) { return; }

    let keyI = keys[i];
    let keyP = keys[partner];
    let ascending = ((i & k) == 0u);

    var doSwap = false;
    if (ascending && keyI > keyP) { doSwap = true; }
    if (!ascending && keyI < keyP) { doSwap = true; }

    if (doSwap) {
        keys[i] = keyP;
        keys[partner] = keyI;
        let valI = values[i];
        let valP = values[partner];
        values[i] = valP;
        values[partner] = valI;
    }
}
)";

// --- Karras (2012) binary radix tree construction ---
static const char *kKarrasShader = R"(
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
@group(0) @binding(1) var<storage, read> mortonCodes: array<u32>;
@group(0) @binding(2) var<storage, read_write> bvhNodes: array<BVHNode>;
@group(0) @binding(3) var<storage, read_write> atomicCounters: array<atomic<u32>>;

fn delta(i: i32, j: i32, N: i32) -> i32 {
    if (j < 0 || j >= N) { return -1; }
    let keyI = mortonCodes[i];
    let keyJ = mortonCodes[j];
    if (keyI == keyJ) {
        return i32(32u + countLeadingZeros(u32(i ^ j)));
    }
    return i32(countLeadingZeros(keyI ^ keyJ));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = i32(gid.x);
    let N = i32(params.numParticles);
    if (idx >= N - 1) { return; }

    // Determine direction of range
    let dPos = delta(idx, idx + 1, N);
    let dNeg = delta(idx, idx - 1, N);
    let d = select(-1, 1, dPos > dNeg);

    // Compute upper bound for range length
    let dMin = delta(idx, idx - d, N);
    var lMax = 2;
    while (delta(idx, idx + lMax * d, N) > dMin) {
        lMax = lMax * 2;
    }

    // Binary search for actual range end
    var l = 0;
    var t = lMax / 2;
    while (t >= 1) {
        if (delta(idx, idx + (l + t) * d, N) > dMin) {
            l = l + t;
        }
        t = t / 2;
    }
    let j = idx + l * d;

    // Find split position
    let dNode = delta(idx, j, N);
    var s = 0;
    var div = 2;
    t = (l + div - 1) / div;
    while (t >= 1) {
        if (delta(idx, idx + (s + t) * d, N) > dNode) {
            s = s + t;
        }
        if (t <= 1) { break; }
        div = div * 2;
        t = (l + div - 1) / div;
    }
    let split = idx + s * d + min(d, 0);

    // Assign children
    let leftIdx = select(split + N - 1, split, min(idx, j) == split);
    let rightIdx = select(split + N, split + 1, max(idx, j) == split + 1);

    bvhNodes[idx].left = i32(leftIdx);
    bvhNodes[idx].right = i32(rightIdx);
    bvhNodes[idx].particleIdx = -1;
    bvhNodes[idx].centerOfMass = vec4f(0.0);
    bvhNodes[idx].boundsMin = vec4f(1e30);
    bvhNodes[idx].boundsMax = vec4f(-1e30);

    // Set parent pointers for children
    bvhNodes[leftIdx].parent = idx;
    bvhNodes[rightIdx].parent = idx;

    // Clear atomic counter for this internal node
    atomicStore(&atomicCounters[idx], 0u);

    // Root node (idx=0) has no parent
    if (idx == 0) {
        bvhNodes[0].parent = -1;
    }
}
)";

// --- Leaf initialization ---
static const char *kLeafInitShader = R"(
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
@group(0) @binding(2) var<storage, read> sortIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> bvhNodes: array<BVHNode>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let N = params.numParticles;
    if (i >= N) { return; }

    let leafIdx = i + N - 1u;
    let origIdx = sortIndices[i];
    let pos = positions[origIdx];

    bvhNodes[leafIdx].centerOfMass = vec4f(pos.xyz * pos.w, pos.w);
    bvhNodes[leafIdx].boundsMin = vec4f(pos.xyz, 0.0);
    bvhNodes[leafIdx].boundsMax = vec4f(pos.xyz, 0.0);
    bvhNodes[leafIdx].left = -1;
    bvhNodes[leafIdx].right = -1;
    bvhNodes[leafIdx].particleIdx = i32(origIdx);
}
)";

// --- Bottom-up aggregation ---
static const char *kAggregateShader = R"(
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
@group(0) @binding(1) var<storage, read_write> bvhNodes: array<BVHNode>;
@group(0) @binding(2) var<storage, read_write> atomicCounters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let N = params.numParticles;
    if (i >= N) { return; }

    let leafIdx = i + N - 1u;
    var current = bvhNodes[leafIdx].parent;

    // Walk up the tree
    while (current >= 0) {
        let old = atomicAdd(&atomicCounters[current], 1u);

        // First thread to arrive: stop (sibling not ready yet)
        if (old == 0u) { return; }

        // Second thread: both children are done, compute this node
        let leftIdx = bvhNodes[current].left;
        let rightIdx = bvhNodes[current].right;

        let leftCOM = bvhNodes[leftIdx].centerOfMass;
        let rightCOM = bvhNodes[rightIdx].centerOfMass;

        let totalMass = leftCOM.w + rightCOM.w;
        var com = vec3f(0.0);
        if (totalMass > 0.0) {
            com = (leftCOM.xyz * leftCOM.w + rightCOM.xyz * rightCOM.w) / totalMass;
        }
        bvhNodes[current].centerOfMass = vec4f(com, totalMass);

        bvhNodes[current].boundsMin = min(bvhNodes[leftIdx].boundsMin, bvhNodes[rightIdx].boundsMin);
        bvhNodes[current].boundsMax = max(bvhNodes[leftIdx].boundsMax, bvhNodes[rightIdx].boundsMax);

        current = bvhNodes[current].parent;
    }
}
)";

// ============================================================
// Helper: next power of 2
// ============================================================

uint32_t GpuTreeBuilder::getPaddedN(uint32_t N) const {
    if (N <= 1) return 2;
    uint32_t v = N - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// ============================================================
// Initialize
// ============================================================

void GpuTreeBuilder::initialize(WGPUDevice device, uint32_t maxParticles) {
    maxParticles_ = maxParticles;
    uint32_t paddedMax = getPaddedN(maxParticles);
    uint32_t maxNodes = 2 * maxParticles - 1;
    uint32_t maxWorkgroups = (maxParticles + 255) / 256;

    // Bounding box buffers
    bboxPartial_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        maxWorkgroups * 2 * sizeof(float) * 4);

    bboxResult_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        2 * sizeof(float) * 4);

    // Morton codes and sort indices
    mortonCodes_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        paddedMax * sizeof(uint32_t));

    sortIndices_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        paddedMax * sizeof(uint32_t));

    // BVH nodes
    bvhNodes_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        maxNodes * sizeof(BVHNodeGPU));

    // Atomic counters for internal nodes
    atomicCounters_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        maxParticles * sizeof(uint32_t));

    // Sort params: max steps for paddedMax
    // For paddedMax, steps = sum_{k=2..paddedMax, powers of 2} of log2(k)
    // = log2(paddedMax) * (log2(paddedMax)+1) / 2
    uint32_t log2N = 0;
    { uint32_t v = paddedMax; while (v > 1) { v >>= 1; log2N++; } }
    uint32_t maxSortSteps = log2N * (log2N + 1) / 2;
    sortParamsBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        maxSortSteps * 256);

    // Tiny uniform buffer for bbox pass 2 numWorkgroups
    numWorkgroupsBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        16);  // 16 bytes minimum for uniform buffer safety

    createPipelines(device);

    spdlog::info("GpuTreeBuilder initialized: maxParticles={}, paddedMax={}, maxNodes={}, maxSortSteps={}",
                 maxParticles, paddedMax, maxNodes, maxSortSteps);
}

// ============================================================
// Build sort params for bitonic sort
// ============================================================

void GpuTreeBuilder::buildSortParams(uint32_t paddedN) {
    if (paddedN == lastPaddedN_) return;
    lastPaddedN_ = paddedN;

    // Count total steps
    numSortSteps_ = 0;
    for (uint32_t k = 2; k <= paddedN; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            numSortSteps_++;
        }
    }
}

// ============================================================
// Pipeline Creation
// ============================================================

void GpuTreeBuilder::createPipelines(WGPUDevice device) {
    if (pipelinesCreated_) return;

    // Helper lambda to create a layout entry
    auto storageEntry = [](uint32_t binding, WGPUBufferBindingType type) -> WGPUBindGroupLayoutEntry {
        WGPUBindGroupLayoutEntry entry{};
        entry.binding = binding;
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer.type = type;
        entry.buffer.minBindingSize = 0;
        return entry;
    };

    auto makeLayout = [&](WGPUDevice dev, const WGPUBindGroupLayoutEntry* entries, uint32_t count) {
        WGPUBindGroupLayoutDescriptor desc{};
        desc.entryCount = count;
        desc.entries = entries;
        return wgpuDeviceCreateBindGroupLayout(dev, &desc);
    };

    auto makePipeline = [&](WGPUDevice dev, WGPUBindGroupLayout layout, const char* shaderSrc, const char* label) {
        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &layout;
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(dev, &plDesc);

        WGPUShaderModule shader = wgpu_utils::createShaderModule(dev, shaderSrc);
        WGPUComputePipelineDescriptor pipeDesc{};
        pipeDesc.layout = pl;
        pipeDesc.compute.module = shader;
        pipeDesc.compute.entryPoint = "main";
        WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(dev, &pipeDesc);

        if (!pipeline) spdlog::error("Failed to create {} pipeline!", label);
        else spdlog::debug("{} pipeline created", label);

        wgpuShaderModuleRelease(shader);
        wgpuPipelineLayoutRelease(pl);
        return pipeline;
    };

    // --- Bbox Pass 1: params(uniform), positions(ro), bboxPartial(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(2, WGPUBufferBindingType_Storage),
        };
        bboxPass1Layout_ = makeLayout(device, entries, 3);
        bboxPass1Pipeline_ = makePipeline(device, bboxPass1Layout_, kBboxPass1Shader, "BboxPass1");
    }

    // --- Bbox Pass 2: bboxPartial(ro), bboxResult(rw), numWorkgroups(uniform) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(1, WGPUBufferBindingType_Storage),
            storageEntry(2, WGPUBufferBindingType_Uniform),
        };
        bboxPass2Layout_ = makeLayout(device, entries, 3);
        bboxPass2Pipeline_ = makePipeline(device, bboxPass2Layout_, kBboxPass2Shader, "BboxPass2");
    }

    // --- Morton: params(uniform), positions(ro), bboxResult(ro), mortonCodes(rw), sortIndices(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[5] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(2, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(3, WGPUBufferBindingType_Storage),
            storageEntry(4, WGPUBufferBindingType_Storage),
        };
        mortonLayout_ = makeLayout(device, entries, 5);
        mortonPipeline_ = makePipeline(device, mortonLayout_, kMortonShader, "Morton");
    }

    // --- Bitonic sort: sortParams(uniform), keys(rw), values(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_Storage),
            storageEntry(2, WGPUBufferBindingType_Storage),
        };
        // Mark binding 0 as having dynamic offset
        entries[0].buffer.hasDynamicOffset = true;
        bitonicSortLayout_ = makeLayout(device, entries, 3);
        bitonicSortPipeline_ = makePipeline(device, bitonicSortLayout_, kBitonicSortShader, "BitonicSort");
    }

    // --- Karras: params(uniform), mortonCodes(ro), bvhNodes(rw), atomicCounters(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[4] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(2, WGPUBufferBindingType_Storage),
            storageEntry(3, WGPUBufferBindingType_Storage),
        };
        karrasLayout_ = makeLayout(device, entries, 4);
        karrasPipeline_ = makePipeline(device, karrasLayout_, kKarrasShader, "Karras");
    }

    // --- Leaf init: params(uniform), positions(ro), sortIndices(ro), bvhNodes(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[4] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(2, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(3, WGPUBufferBindingType_Storage),
        };
        leafInitLayout_ = makeLayout(device, entries, 4);
        leafInitPipeline_ = makePipeline(device, leafInitLayout_, kLeafInitShader, "LeafInit");
    }

    // --- Aggregate: params(uniform), bvhNodes(rw), atomicCounters(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_Storage),
            storageEntry(2, WGPUBufferBindingType_Storage),
        };
        aggregateLayout_ = makeLayout(device, entries, 3);
        aggregatePipeline_ = makePipeline(device, aggregateLayout_, kAggregateShader, "Aggregate");
    }

    pipelinesCreated_ = true;
    spdlog::info("GpuTreeBuilder: all 7 pipelines created");
}

// ============================================================
// Prepare Uploads — queue writes BEFORE encoder creation
// ============================================================

void GpuTreeBuilder::prepareUploads(WGPUQueue queue, uint32_t numParticles) {
    uint32_t paddedN = getPaddedN(numParticles);
    uint32_t numWorkgroups = (numParticles + 255) / 256;

    // Build and upload sort parameters (cached — only re-uploads when paddedN changes)
    if (paddedN != lastPaddedN_) {
        buildSortParams(paddedN);

        std::vector<uint8_t> sortData(numSortSteps_ * 256, 0);
        uint32_t stepIdx = 0;
        for (uint32_t k = 2; k <= paddedN; k <<= 1) {
            for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                struct SortParams { uint32_t k, j, paddedN; uint32_t pad; };
                SortParams sp{k, j, paddedN, 0};
                std::memcpy(sortData.data() + stepIdx * 256, &sp, sizeof(sp));
                stepIdx++;
            }
        }
        sortParamsBuffer_.upload(queue, sortData.data(), sortData.size());
    }

    // Upload numWorkgroups uniform for bbox pass 2
    {
        uint32_t nwg = numWorkgroups;
        numWorkgroupsBuffer_.upload(queue, &nwg, sizeof(uint32_t));
    }
}

// ============================================================
// Record Tree Build — all compute passes into an encoder
// ============================================================

void GpuTreeBuilder::recordTreeBuild(WGPUDevice device,
                                      WGPUCommandEncoder encoder,
                                      WGPUBuffer positionsBuffer,
                                      WGPUBuffer paramsBuffer,
                                      uint32_t numParticles) {
    uint32_t paddedN = getPaddedN(numParticles);
    uint32_t numNodes = 2 * numParticles - 1;
    uint32_t numWorkgroups = (numParticles + 255) / 256;
    uint32_t paddedWorkgroups = (paddedN + 255) / 256;

    // Explicitly zero atomic counters to prevent stale state across runs
    wgpuCommandEncoderClearBuffer(encoder, atomicCounters_.get(), 0,
                                   (numParticles - 1) * sizeof(uint32_t));

    // Helper to create bind groups
    auto makeBG = [&](WGPUBindGroupLayout layout,
                      std::initializer_list<std::tuple<uint32_t, WGPUBuffer, uint64_t, uint64_t>> bindings)
                      -> WGPUBindGroup {
        std::vector<WGPUBindGroupEntry> entries;
        for (auto& [binding, buffer, offset, size] : bindings) {
            WGPUBindGroupEntry e{};
            e.binding = binding;
            e.buffer = buffer;
            e.offset = offset;
            e.size = size;
            entries.push_back(e);
        }
        WGPUBindGroupDescriptor desc{};
        desc.layout = layout;
        desc.entryCount = static_cast<uint32_t>(entries.size());
        desc.entries = entries.data();
        return wgpuDeviceCreateBindGroup(device, &desc);
    };

    WGPUComputePassDescriptor passDesc{};
    passDesc.timestampWrites = nullptr;

    // ---- Pass 1: Bounding box reduction (per-workgroup) ----
    {
        WGPUBindGroup bg = makeBG(bboxPass1Layout_, {
            {0, paramsBuffer, 0, 32},
            {1, positionsBuffer, 0, numParticles * 16},
            {2, bboxPartial_.get(), 0, numWorkgroups * 2 * 16},
        });

        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, bboxPass1Pipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, numWorkgroups, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // ---- Pass 2: Bounding box final reduction ----
    {
        WGPUBindGroup bg = makeBG(bboxPass2Layout_, {
            {0, bboxPartial_.get(), 0, numWorkgroups * 2 * 16},
            {1, bboxResult_.get(), 0, 2 * 16},
            {2, numWorkgroupsBuffer_.get(), 0, sizeof(uint32_t)},
        });

        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, bboxPass2Pipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // ---- Pass 3: Morton code computation ----
    {
        WGPUBindGroup bg = makeBG(mortonLayout_, {
            {0, paramsBuffer, 0, 32},
            {1, positionsBuffer, 0, numParticles * 16},
            {2, bboxResult_.get(), 0, 2 * 16},
            {3, mortonCodes_.get(), 0, paddedN * 4},
            {4, sortIndices_.get(), 0, paddedN * 4},
        });

        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, mortonPipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, paddedWorkgroups, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // ---- Pass 4: Bitonic sort (~153 steps for N=100K) ----
    {
        // Create bind group with dynamic offset for sort params
        WGPUBindGroupEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].buffer = sortParamsBuffer_.get();
        entries[0].offset = 0;
        entries[0].size = 256;  // each slot is 256-byte aligned
        entries[1].binding = 1;
        entries[1].buffer = mortonCodes_.get();
        entries[1].offset = 0;
        entries[1].size = paddedN * 4;
        entries[2].binding = 2;
        entries[2].buffer = sortIndices_.get();
        entries[2].offset = 0;
        entries[2].size = paddedN * 4;

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.layout = bitonicSortLayout_;
        bgDesc.entryCount = 3;
        bgDesc.entries = entries;
        WGPUBindGroup sortBG = wgpuDeviceCreateBindGroup(device, &bgDesc);

        uint32_t sortWG = (paddedN + 255) / 256;
        for (uint32_t step = 0; step < numSortSteps_; step++) {
            uint32_t dynamicOffset = step * 256;

            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
            wgpuComputePassEncoderSetPipeline(pass, bitonicSortPipeline_);
            wgpuComputePassEncoderSetBindGroup(pass, 0, sortBG, 1, &dynamicOffset);
            wgpuComputePassEncoderDispatchWorkgroups(pass, sortWG, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        wgpuBindGroupRelease(sortBG);
    }

    // ---- Pass 5: Karras tree construction (N-1 internal nodes) ----
    {
        WGPUBindGroup bg = makeBG(karrasLayout_, {
            {0, paramsBuffer, 0, 32},
            {1, mortonCodes_.get(), 0, paddedN * 4},
            {2, bvhNodes_.get(), 0, numNodes * 64},
            {3, atomicCounters_.get(), 0, (numParticles - 1) * 4},
        });

        uint32_t internalWG = (numParticles - 1 + 255) / 256;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, karrasPipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, internalWG, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // ---- Pass 6: Leaf initialization ----
    {
        WGPUBindGroup bg = makeBG(leafInitLayout_, {
            {0, paramsBuffer, 0, 32},
            {1, positionsBuffer, 0, numParticles * 16},
            {2, sortIndices_.get(), 0, paddedN * 4},
            {3, bvhNodes_.get(), 0, numNodes * 64},
        });

        uint32_t leafWG = (numParticles + 255) / 256;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, leafInitPipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, leafWG, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }

    // ---- Pass 7: Bottom-up aggregation ----
    {
        WGPUBindGroup bg = makeBG(aggregateLayout_, {
            {0, paramsBuffer, 0, 32},
            {1, bvhNodes_.get(), 0, numNodes * 64},
            {2, atomicCounters_.get(), 0, (numParticles - 1) * 4},
        });

        uint32_t leafWG = (numParticles + 255) / 256;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, aggregatePipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, leafWG, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuBindGroupRelease(bg);
    }
}
