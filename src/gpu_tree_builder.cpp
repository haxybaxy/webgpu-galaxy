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

// --- Radix sort: histogram (per-workgroup digit counts) ---
static const char *kRadixHistogramShader = R"(
struct RadixParams {
    numElements: u32,
    bitOffset:   u32,
    numWorkgroups: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RadixParams;
@group(0) @binding(1) var<storage, read> keysIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<u32>;

var<workgroup> localHist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let localId = lid.x;
    let wgId = wid.x;

    // Clear local histogram
    atomicStore(&localHist[localId], 0u);
    workgroupBarrier();

    // Each thread processes 4 elements
    let base = wgId * 1024u;
    for (var t = 0u; t < 4u; t = t + 1u) {
        let idx = base + t * 256u + localId;
        if (idx < params.numElements) {
            let digit = (keysIn[idx] >> params.bitOffset) & 0xFFu;
            atomicAdd(&localHist[digit], 1u);
        }
    }

    workgroupBarrier();

    // Write local histogram to global in digit-major layout
    let count = atomicLoad(&localHist[localId]);
    histogram[localId * params.numWorkgroups + wgId] = count;
}
)";

// --- Radix sort: Blelloch exclusive prefix scan (blocks of 512) ---
static const char *kPrefixScanShader = R"(
struct ScanParams {
    numElements: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let localId = lid.x;
    let wgId = wid.x;
    let blockOffset = wgId * 512u;

    // Load two elements per thread into shared memory
    let ai = localId;
    let bi = localId + 256u;
    let globalAi = blockOffset + ai;
    let globalBi = blockOffset + bi;

    temp[ai] = select(0u, data[globalAi], globalAi < params.numElements);
    temp[bi] = select(0u, data[globalBi], globalBi < params.numElements);

    // Up-sweep (reduction) phase
    var offset = 1u;
    for (var d = 256u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (localId < d) {
            let ai2 = offset * (2u * localId + 1u) - 1u;
            let bi2 = offset * (2u * localId + 2u) - 1u;
            temp[bi2] = temp[bi2] + temp[ai2];
        }
        offset = offset << 1u;
    }

    // Save block total and clear last element
    workgroupBarrier();
    if (localId == 0u) {
        blockSums[wgId] = temp[511u];
        temp[511u] = 0u;
    }

    // Down-sweep phase
    for (var d = 1u; d < 512u; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if (localId < d) {
            let ai2 = offset * (2u * localId + 1u) - 1u;
            let bi2 = offset * (2u * localId + 2u) - 1u;
            let t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] = temp[bi2] + t;
        }
    }

    workgroupBarrier();

    // Write results back
    if (globalAi < params.numElements) { data[globalAi] = temp[ai]; }
    if (globalBi < params.numElements) { data[globalBi] = temp[bi]; }
}
)";

// --- Radix sort: propagate block sums into scanned data ---
static const char *kPrefixPropagateShader = R"(
struct ScanParams {
    numElements: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> blockSums: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let localId = lid.x;
    let wgId = wid.x;
    let blockOffset = wgId * 512u;
    let sum = blockSums[wgId];

    let ai = blockOffset + localId;
    let bi = blockOffset + localId + 256u;

    if (ai < params.numElements) { data[ai] = data[ai] + sum; }
    if (bi < params.numElements) { data[bi] = data[bi] + sum; }
}
)";

// --- Radix sort: scatter key-value pairs to sorted positions ---
// Uses cooperative digit-owned ranking: each thread owns one digit value
// and scans all 1024 elements left-to-right to assign stable ranks.
static const char *kRadixScatterShader = R"(
struct RadixParams {
    numElements: u32,
    bitOffset:   u32,
    numWorkgroups: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RadixParams;
@group(0) @binding(1) var<storage, read> keysIn: array<u32>;
@group(0) @binding(2) var<storage, read> valuesIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(4) var<storage, read_write> valuesOut: array<u32>;
@group(0) @binding(5) var<storage, read> prefixSums: array<u32>;

var<workgroup> sharedDigits: array<u32, 1024>;
var<workgroup> sharedRanks: array<u32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    let localId = lid.x;
    let wgId = wid.x;
    let base = wgId * 1024u;

    // Phase 1: Load digits into shared memory (4 elements per thread)
    for (var t = 0u; t < 4u; t = t + 1u) {
        let localPos = t * 256u + localId;
        let globalIdx = base + localPos;
        if (globalIdx < params.numElements) {
            sharedDigits[localPos] = (keysIn[globalIdx] >> params.bitOffset) & 0xFFu;
        } else {
            sharedDigits[localPos] = 256u;  // sentinel: no valid digit matches 256
        }
    }

    workgroupBarrier();

    // Phase 2: Cooperative digit-owned ranking
    // Thread localId owns digit value localId. It scans all 1024 elements
    // left-to-right, assigning sequential ranks to matching elements.
    // Guarantees stability (left-to-right = input order) with no atomics
    // and no write conflicts (each sharedRanks[p] written by exactly one thread).
    let myDigit = localId;
    var counter = 0u;
    for (var p = 0u; p < 1024u; p = p + 1u) {
        if (sharedDigits[p] == myDigit) {
            sharedRanks[p] = counter;
            counter = counter + 1u;
        }
    }

    workgroupBarrier();

    // Phase 3: Scatter using precomputed ranks
    for (var t = 0u; t < 4u; t = t + 1u) {
        let localPos = t * 256u + localId;
        let globalIdx = base + localPos;
        if (globalIdx >= params.numElements) { continue; }

        let digit = sharedDigits[localPos];
        let rank = sharedRanks[localPos];
        let dst = prefixSums[digit * params.numWorkgroups + wgId] + rank;
        keysOut[dst] = keysIn[globalIdx];
        valuesOut[dst] = valuesIn[globalIdx];
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

    // Assign children: single-key side → leaf, multi-key side → internal
    let leftIdx = select(split, split + N - 1, min(idx, j) == split);
    let rightIdx = select(split + 1, split + N, max(idx, j) == split + 1);

    bvhNodes[idx].left = i32(leftIdx);
    bvhNodes[idx].right = i32(rightIdx);
    bvhNodes[idx].particleIdx = -1;
    bvhNodes[idx].centerOfMass = vec4f(0.0);
    bvhNodes[idx].boundsMin = vec4f(1e30, 1e30, 1e30, 0.0);
    bvhNodes[idx].boundsMax = vec4f(-1e30, -1e30, -1e30, 0.0);

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

    bvhNodes[leafIdx].centerOfMass = vec4f(pos.xyz, pos.w);
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

        // Float min/max on AABB bounds
        bvhNodes[current].boundsMin = vec4f(min(bvhNodes[leftIdx].boundsMin.xyz, bvhNodes[rightIdx].boundsMin.xyz), 0.0);
        bvhNodes[current].boundsMax = vec4f(max(bvhNodes[leftIdx].boundsMax.xyz, bvhNodes[rightIdx].boundsMax.xyz), 0.0);

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

    // BVH nodes (float32 bounds: 64 bytes per node)
    bvhNodes_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        maxNodes * kNodeSize);

    // Atomic counters for internal nodes
    atomicCounters_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        maxParticles * sizeof(uint32_t));

    // Radix sort: ping-pong buffers
    mortonCodesAlt_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        paddedMax * sizeof(uint32_t));

    sortIndicesAlt_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        paddedMax * sizeof(uint32_t));

    // Radix sort: histogram (digit-major: 256 digits * maxHistWG workgroups)
    uint32_t maxHistWG = (paddedMax + 1023) / 1024;
    uint32_t maxHistElements = 256 * maxHistWG;
    histogram_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        maxHistElements * sizeof(uint32_t));

    // Radix sort: block sums for 2-level prefix scan
    uint32_t maxScanWG = (maxHistElements + 511) / 512;
    blockSums_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        maxScanWG * sizeof(uint32_t));

    blockSumsOfSums_.initialize(device,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        sizeof(uint32_t));

    // Radix sort: params (4 passes * 256-byte aligned entries)
    radixParamsBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        4 * 256);

    // Prefix scan params (2 entries * 256-byte aligned: level-1 and level-2)
    scanParamsBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        2 * 256);

    // Tiny uniform buffer for bbox pass 2 numWorkgroups
    numWorkgroupsBuffer_.initialize(device,
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        16);  // 16 bytes minimum for uniform buffer safety

    createPipelines(device);

    spdlog::info("GpuTreeBuilder initialized: maxParticles={}, paddedMax={}, maxNodes={}, maxHistWG={}",
                 maxParticles, paddedMax, maxNodes, maxHistWG);
}

// ============================================================
// Build radix sort params (caches lastPaddedN_)
// ============================================================

void GpuTreeBuilder::buildRadixParams(uint32_t paddedN) {
    if (paddedN == lastPaddedN_) return;
    lastPaddedN_ = paddedN;
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

    // --- Radix histogram: radixParams(uniform,dynamic), keysIn(ro), histogram(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(2, WGPUBufferBindingType_Storage),
        };
        entries[0].buffer.hasDynamicOffset = true;
        radixHistogramLayout_ = makeLayout(device, entries, 3);
        radixHistogramPipeline_ = makePipeline(device, radixHistogramLayout_,
                                                kRadixHistogramShader, "RadixHistogram");
    }

    // --- Prefix scan: scanParams(uniform,dynamic), data(rw), blockSums(rw) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_Storage),
            storageEntry(2, WGPUBufferBindingType_Storage),
        };
        entries[0].buffer.hasDynamicOffset = true;
        prefixScanLayout_ = makeLayout(device, entries, 3);
        prefixScanPipeline_ = makePipeline(device, prefixScanLayout_,
                                            kPrefixScanShader, "PrefixScan");
    }

    // --- Prefix propagate: scanParams(uniform,dynamic), data(rw), blockSums(ro) ---
    {
        WGPUBindGroupLayoutEntry entries[3] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_Storage),
            storageEntry(2, WGPUBufferBindingType_ReadOnlyStorage),
        };
        entries[0].buffer.hasDynamicOffset = true;
        prefixPropagateLayout_ = makeLayout(device, entries, 3);
        prefixPropagatePipeline_ = makePipeline(device, prefixPropagateLayout_,
                                                 kPrefixPropagateShader, "PrefixPropagate");
    }

    // --- Radix scatter: radixParams(uniform,dynamic), keysIn(ro), valsIn(ro),
    //                     keysOut(rw), valsOut(rw), prefixSums(ro) ---
    {
        WGPUBindGroupLayoutEntry entries[6] = {
            storageEntry(0, WGPUBufferBindingType_Uniform),
            storageEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(2, WGPUBufferBindingType_ReadOnlyStorage),
            storageEntry(3, WGPUBufferBindingType_Storage),
            storageEntry(4, WGPUBufferBindingType_Storage),
            storageEntry(5, WGPUBufferBindingType_ReadOnlyStorage),
        };
        entries[0].buffer.hasDynamicOffset = true;
        radixScatterLayout_ = makeLayout(device, entries, 6);
        radixScatterPipeline_ = makePipeline(device, radixScatterLayout_,
                                              kRadixScatterShader, "RadixScatter");
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
    spdlog::info("GpuTreeBuilder: all 10 pipelines created");
}

// ============================================================
// Prepare Uploads — queue writes BEFORE encoder creation
// ============================================================

void GpuTreeBuilder::prepareUploads(WGPUQueue queue, uint32_t numParticles) {
    uint32_t paddedN = getPaddedN(numParticles);
    uint32_t numWorkgroups = (numParticles + 255) / 256;

    // Build and upload radix sort parameters (cached — only re-uploads when paddedN changes)
    if (paddedN != lastPaddedN_) {
        buildRadixParams(paddedN);

        uint32_t numHistWG = (paddedN + 1023) / 1024;

        // Radix params: 4 passes, each at 256-byte offset
        std::vector<uint8_t> radixData(4 * 256, 0);
        for (uint32_t pass = 0; pass < 4; pass++) {
            struct RadixParams {
                uint32_t numElements;
                uint32_t bitOffset;
                uint32_t numWorkgroups;
                uint32_t pad;
            };
            RadixParams rp{paddedN, pass * 8, numHistWG, 0};
            std::memcpy(radixData.data() + pass * 256, &rp, sizeof(rp));
        }
        radixParamsBuffer_.upload(queue, radixData.data(), radixData.size());

        // Scan params: 2 entries at 256-byte offsets
        uint32_t histElements = 256 * numHistWG;
        uint32_t numScanWG = (histElements + 511) / 512;

        std::vector<uint8_t> scanData(2 * 256, 0);
        {
            struct ScanParams {
                uint32_t numElements;
                uint32_t pad[3];
            };
            ScanParams sp1{histElements, {0, 0, 0}};
            std::memcpy(scanData.data(), &sp1, sizeof(sp1));

            ScanParams sp2{numScanWG, {0, 0, 0}};
            std::memcpy(scanData.data() + 256, &sp2, sizeof(sp2));
        }
        scanParamsBuffer_.upload(queue, scanData.data(), scanData.size());
    }

    // Upload numWorkgroups uniform for bbox pass 2
    {
        uint32_t nwg = numWorkgroups;
        numWorkgroupsBuffer_.upload(queue, &nwg, sizeof(uint32_t));
    }
}

// ============================================================
// Bind Group Caching
// ============================================================

void GpuTreeBuilder::invalidateBindGroups() {
    auto release = [](WGPUBindGroup &bg) {
        if (bg) { wgpuBindGroupRelease(bg); bg = nullptr; }
    };
    release(cachedBboxPass1BG_);
    release(cachedBboxPass2BG_);
    release(cachedMortonBG_);
    release(cachedScanBG1_);
    release(cachedScanBG2_);
    release(cachedPropagateBG_);
    release(cachedRadixHistBG_[0]);
    release(cachedRadixHistBG_[1]);
    release(cachedRadixScatterBG_[0]);
    release(cachedRadixScatterBG_[1]);
    release(cachedKarrasBG_);
    release(cachedLeafInitBG_);
    release(cachedAggregateBG_);
    cachedBGParticleCount_ = 0;
    cachedPositionsBuffer_ = nullptr;
    cachedParamsBuffer_ = nullptr;
}

void GpuTreeBuilder::ensureBindGroupsCached(WGPUDevice device,
                                             WGPUBuffer positionsBuffer,
                                             WGPUBuffer paramsBuffer,
                                             uint32_t numParticles) {
    if (cachedBGParticleCount_ == numParticles &&
        cachedPositionsBuffer_ == positionsBuffer &&
        cachedParamsBuffer_ == paramsBuffer) return;

    invalidateBindGroups();

    uint32_t paddedN = getPaddedN(numParticles);
    uint32_t numNodes = 2 * numParticles - 1;
    uint32_t numWorkgroups = (numParticles + 255) / 256;
    uint32_t numHistWG = (paddedN + 1023) / 1024;
    uint32_t histElements = 256 * numHistWG;
    uint32_t histogramSize = histElements * sizeof(uint32_t);
    uint32_t numScanWG = (histElements + 511) / 512;

    auto makeBG = [&](WGPUBindGroupLayout layout,
                      WGPUBindGroupEntry *entries, uint32_t count) -> WGPUBindGroup {
        WGPUBindGroupDescriptor desc{};
        desc.layout = layout;
        desc.entryCount = count;
        desc.entries = entries;
        return wgpuDeviceCreateBindGroup(device, &desc);
    };

    auto entry = [](uint32_t binding, WGPUBuffer buffer, uint64_t size) -> WGPUBindGroupEntry {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = buffer;
        e.size = size;
        return e;
    };

    // Bbox pass 1
    {
        WGPUBindGroupEntry e[3] = {
            entry(0, paramsBuffer, 32),
            entry(1, positionsBuffer, numParticles * 16),
            entry(2, bboxPartial_.get(), numWorkgroups * 2 * 16),
        };
        cachedBboxPass1BG_ = makeBG(bboxPass1Layout_, e, 3);
    }

    // Bbox pass 2
    {
        WGPUBindGroupEntry e[3] = {
            entry(0, bboxPartial_.get(), numWorkgroups * 2 * 16),
            entry(1, bboxResult_.get(), 2 * 16),
            entry(2, numWorkgroupsBuffer_.get(), sizeof(uint32_t)),
        };
        cachedBboxPass2BG_ = makeBG(bboxPass2Layout_, e, 3);
    }

    // Morton
    {
        WGPUBindGroupEntry e[5] = {
            entry(0, paramsBuffer, 32),
            entry(1, positionsBuffer, numParticles * 16),
            entry(2, bboxResult_.get(), 2 * 16),
            entry(3, mortonCodes_.get(), paddedN * 4),
            entry(4, sortIndices_.get(), paddedN * 4),
        };
        cachedMortonBG_ = makeBG(mortonLayout_, e, 5);
    }

    // Scan BG1 (histogram -> blockSums)
    {
        WGPUBindGroupEntry e[3] = {
            entry(0, scanParamsBuffer_.get(), 256),
            entry(1, histogram_.get(), histogramSize),
            entry(2, blockSums_.get(), numScanWG * 4),
        };
        cachedScanBG1_ = makeBG(prefixScanLayout_, e, 3);
    }

    // Scan BG2 (blockSums -> blockSumsOfSums)
    {
        WGPUBindGroupEntry e[3] = {
            entry(0, scanParamsBuffer_.get(), 256),
            entry(1, blockSums_.get(), numScanWG * 4),
            entry(2, blockSumsOfSums_.get(), 4),
        };
        cachedScanBG2_ = makeBG(prefixScanLayout_, e, 3);
    }

    // Propagate
    {
        WGPUBindGroupEntry e[3] = {
            entry(0, scanParamsBuffer_.get(), 256),
            entry(1, histogram_.get(), histogramSize),
            entry(2, blockSums_.get(), numScanWG * 4),
        };
        cachedPropagateBG_ = makeBG(prefixPropagateLayout_, e, 3);
    }

    // Radix histogram bind groups (2 variants: even/odd pass)
    for (int p = 0; p < 2; p++) {
        WGPUBuffer srcKeys = (p == 0) ? mortonCodes_.get() : mortonCodesAlt_.get();
        WGPUBindGroupEntry e[3] = {
            entry(0, radixParamsBuffer_.get(), 256),
            entry(1, srcKeys, paddedN * 4),
            entry(2, histogram_.get(), histogramSize),
        };
        cachedRadixHistBG_[p] = makeBG(radixHistogramLayout_, e, 3);
    }

    // Radix scatter bind groups (2 variants: even/odd pass)
    for (int p = 0; p < 2; p++) {
        WGPUBuffer srcKeys = (p == 0) ? mortonCodes_.get() : mortonCodesAlt_.get();
        WGPUBuffer dstKeys = (p == 0) ? mortonCodesAlt_.get() : mortonCodes_.get();
        WGPUBuffer srcVals = (p == 0) ? sortIndices_.get() : sortIndicesAlt_.get();
        WGPUBuffer dstVals = (p == 0) ? sortIndicesAlt_.get() : sortIndices_.get();
        WGPUBindGroupEntry e[6] = {
            entry(0, radixParamsBuffer_.get(), 256),
            entry(1, srcKeys, paddedN * 4),
            entry(2, srcVals, paddedN * 4),
            entry(3, dstKeys, paddedN * 4),
            entry(4, dstVals, paddedN * 4),
            entry(5, histogram_.get(), histogramSize),
        };
        cachedRadixScatterBG_[p] = makeBG(radixScatterLayout_, e, 6);
    }

    // Karras
    {
        WGPUBindGroupEntry e[4] = {
            entry(0, paramsBuffer, 32),
            entry(1, mortonCodes_.get(), paddedN * 4),
            entry(2, bvhNodes_.get(), numNodes * kNodeSize),
            entry(3, atomicCounters_.get(), (numParticles - 1) * 4),
        };
        cachedKarrasBG_ = makeBG(karrasLayout_, e, 4);
    }

    // Leaf init (float bounds, no bboxResult needed)
    {
        WGPUBindGroupEntry e[4] = {
            entry(0, paramsBuffer, 32),
            entry(1, positionsBuffer, numParticles * 16),
            entry(2, sortIndices_.get(), paddedN * 4),
            entry(3, bvhNodes_.get(), numNodes * kNodeSize),
        };
        cachedLeafInitBG_ = makeBG(leafInitLayout_, e, 4);
    }

    // Aggregate
    {
        WGPUBindGroupEntry e[3] = {
            entry(0, paramsBuffer, 32),
            entry(1, bvhNodes_.get(), numNodes * kNodeSize),
            entry(2, atomicCounters_.get(), (numParticles - 1) * 4),
        };
        cachedAggregateBG_ = makeBG(aggregateLayout_, e, 3);
    }

    cachedBGParticleCount_ = numParticles;
    cachedPositionsBuffer_ = positionsBuffer;
    cachedParamsBuffer_ = paramsBuffer;
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

    // Ensure all bind groups are cached
    ensureBindGroupsCached(device, positionsBuffer, paramsBuffer, numParticles);

    WGPUComputePassDescriptor passDesc{};
    passDesc.timestampWrites = nullptr;

    // ---- Pass 1: Bounding box reduction (per-workgroup) ----
    {
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, bboxPass1Pipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, cachedBboxPass1BG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, numWorkgroups, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // ---- Pass 2: Bounding box final reduction ----
    {
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, bboxPass2Pipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, cachedBboxPass2BG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // ---- Pass 3: Morton code computation ----
    {
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, mortonPipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, cachedMortonBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, paddedWorkgroups, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // ---- Pass 4: Radix sort (4 digit passes x 5 dispatches = 20 dispatches) ----
    {
        uint32_t numHistWG = (paddedN + 1023) / 1024;
        uint32_t histElements = 256 * numHistWG;
        uint32_t histogramSize = histElements * sizeof(uint32_t);
        uint32_t numScanWG = (histElements + 511) / 512;

        for (uint32_t rpass = 0; rpass < 4; rpass++) {
            uint32_t radixDynOffset = rpass * 256;
            int pingPong = rpass % 2;

            // 1. Clear histogram buffer
            wgpuCommandEncoderClearBuffer(encoder, histogram_.get(), 0, histogramSize);

            // 2. Histogram dispatch
            {
                WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                wgpuComputePassEncoderSetPipeline(p, radixHistogramPipeline_);
                wgpuComputePassEncoderSetBindGroup(p, 0, cachedRadixHistBG_[pingPong], 1, &radixDynOffset);
                wgpuComputePassEncoderDispatchWorkgroups(p, numHistWG, 1, 1);
                wgpuComputePassEncoderEnd(p);
                wgpuComputePassEncoderRelease(p);
            }

            // 3. Prefix scan level 1
            {
                uint32_t scanDynOffset = 0;
                WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                wgpuComputePassEncoderSetPipeline(p, prefixScanPipeline_);
                wgpuComputePassEncoderSetBindGroup(p, 0, cachedScanBG1_, 1, &scanDynOffset);
                wgpuComputePassEncoderDispatchWorkgroups(p, numScanWG, 1, 1);
                wgpuComputePassEncoderEnd(p);
                wgpuComputePassEncoderRelease(p);
            }

            // 4. Prefix scan level 2
            {
                uint32_t scanDynOffset = 256;
                WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                wgpuComputePassEncoderSetPipeline(p, prefixScanPipeline_);
                wgpuComputePassEncoderSetBindGroup(p, 0, cachedScanBG2_, 1, &scanDynOffset);
                wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
                wgpuComputePassEncoderEnd(p);
                wgpuComputePassEncoderRelease(p);
            }

            // 5. Propagate block sums into histogram
            {
                uint32_t scanDynOffset = 0;
                WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                wgpuComputePassEncoderSetPipeline(p, prefixPropagatePipeline_);
                wgpuComputePassEncoderSetBindGroup(p, 0, cachedPropagateBG_, 1, &scanDynOffset);
                wgpuComputePassEncoderDispatchWorkgroups(p, numScanWG, 1, 1);
                wgpuComputePassEncoderEnd(p);
                wgpuComputePassEncoderRelease(p);
            }

            // 6. Scatter
            {
                WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                wgpuComputePassEncoderSetPipeline(p, radixScatterPipeline_);
                wgpuComputePassEncoderSetBindGroup(p, 0, cachedRadixScatterBG_[pingPong], 1, &radixDynOffset);
                wgpuComputePassEncoderDispatchWorkgroups(p, numHistWG, 1, 1);
                wgpuComputePassEncoderEnd(p);
                wgpuComputePassEncoderRelease(p);
            }
        }
    }

    // ---- Pass 5: Karras tree construction (N-1 internal nodes) ----
    {
        uint32_t internalWG = (numParticles - 1 + 255) / 256;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, karrasPipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, cachedKarrasBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, internalWG, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // ---- Pass 6: Leaf initialization ----
    {
        uint32_t leafWG = (numParticles + 255) / 256;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, leafInitPipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, cachedLeafInitBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, leafWG, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // ---- Pass 7: Bottom-up aggregation ----
    {
        uint32_t leafWG = (numParticles + 255) / 256;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, aggregatePipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0, cachedAggregateBG_, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, leafWG, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
}
