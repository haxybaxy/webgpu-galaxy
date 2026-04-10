#pragma once

#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct BVHNodeGPU {
    glm::vec4 centerOfMass;  // xyz = COM, w = mass
    glm::vec4 boundsMin;     // xyz = AABB min, w = unused
    glm::vec4 boundsMax;     // xyz = AABB max, w = unused
    int32_t left, right;
    int32_t parent, particleIdx;
};

static_assert(sizeof(BVHNodeGPU) == 64, "BVHNodeGPU must be 64 bytes");

struct TreePassTiming {
    double bboxReduceMs = 0.0;
    double mortonMs = 0.0;
    double radixSortMs = 0.0;
    double karrasMs = 0.0;
    double leafInitMs = 0.0;
    double aggregateMs = 0.0;
    double totalMs = 0.0;
};

class GpuTreeBuilder {
public:
    void initialize(WGPUDevice device, uint32_t maxParticles);

    void prepareUploads(WGPUQueue queue, uint32_t numParticles);

    void recordTreeBuild(WGPUDevice device,
                         WGPUCommandEncoder encoder,
                         WGPUBuffer positionsBuffer,
                         WGPUBuffer paramsBuffer,
                         uint32_t numParticles);

    TreePassTiming recordTreeBuildTimed(WGPUDevice device,
                                        WGPUQueue queue,
                                        WGPUBuffer positionsBuffer,
                                        WGPUBuffer paramsBuffer,
                                        uint32_t numParticles);

    WGPUBuffer getBvhNodesBuffer() const { return bvhNodes_.get(); }
    WGPUBuffer getBboxResultBuffer() const { return bboxResult_.get(); }
    int getNodeCount(int N) const { return (N <= 0) ? 0 : 2 * N - 1; }
    uint32_t getPaddedN(uint32_t N) const;
    uint32_t getMaxParticles() const { return maxParticles_; }
    static constexpr uint32_t kNodeSize = 64;

private:
    void createPipelines(WGPUDevice device);
    void buildRadixParams(uint32_t paddedN);

    uint32_t maxParticles_ = 0;
    uint32_t lastPaddedN_ = 0;

    // Buffers
    wgpu_utils::Buffer bboxPartial_;
    wgpu_utils::Buffer bboxResult_;
    wgpu_utils::Buffer mortonCodes_;
    wgpu_utils::Buffer sortIndices_;
    wgpu_utils::Buffer mortonCodesAlt_;
    wgpu_utils::Buffer sortIndicesAlt_;
    wgpu_utils::Buffer histogram_;
    wgpu_utils::Buffer blockSums_;
    wgpu_utils::Buffer blockSumsOfSums_;
    wgpu_utils::Buffer radixParamsBuffer_;
    wgpu_utils::Buffer scanParamsBuffer_;
    wgpu_utils::Buffer bvhNodes_;
    wgpu_utils::Buffer atomicCounters_;
    wgpu_utils::Buffer numWorkgroupsBuffer_;  // tiny uniform for bbox pass 2

    // Pipelines
    WGPUComputePipeline bboxPass1Pipeline_ = nullptr;
    WGPUComputePipeline bboxPass2Pipeline_ = nullptr;
    WGPUComputePipeline mortonPipeline_ = nullptr;
    WGPUComputePipeline radixHistogramPipeline_ = nullptr;
    WGPUComputePipeline prefixScanPipeline_ = nullptr;
    WGPUComputePipeline prefixPropagatePipeline_ = nullptr;
    WGPUComputePipeline radixScatterPipeline_ = nullptr;
    WGPUComputePipeline karrasPipeline_ = nullptr;
    WGPUComputePipeline leafInitPipeline_ = nullptr;
    WGPUComputePipeline aggregatePipeline_ = nullptr;

    // Bind group layouts
    WGPUBindGroupLayout bboxPass1Layout_ = nullptr;
    WGPUBindGroupLayout bboxPass2Layout_ = nullptr;
    WGPUBindGroupLayout mortonLayout_ = nullptr;
    WGPUBindGroupLayout radixHistogramLayout_ = nullptr;
    WGPUBindGroupLayout prefixScanLayout_ = nullptr;
    WGPUBindGroupLayout prefixPropagateLayout_ = nullptr;
    WGPUBindGroupLayout radixScatterLayout_ = nullptr;
    WGPUBindGroupLayout karrasLayout_ = nullptr;
    WGPUBindGroupLayout leafInitLayout_ = nullptr;
    WGPUBindGroupLayout aggregateLayout_ = nullptr;

    // Cached bind groups (invalidated when particle count changes)
    WGPUBindGroup cachedBboxPass1BG_ = nullptr;
    WGPUBindGroup cachedBboxPass2BG_ = nullptr;
    WGPUBindGroup cachedMortonBG_ = nullptr;
    WGPUBindGroup cachedScanBG1_ = nullptr;
    WGPUBindGroup cachedScanBG2_ = nullptr;
    WGPUBindGroup cachedPropagateBG_ = nullptr;
    WGPUBindGroup cachedRadixHistBG_[2] = {};   // even/odd pass (ping-pong)
    WGPUBindGroup cachedRadixScatterBG_[2] = {}; // even/odd pass
    WGPUBindGroup cachedKarrasBG_ = nullptr;
    WGPUBindGroup cachedLeafInitBG_ = nullptr;
    WGPUBindGroup cachedAggregateBG_ = nullptr;
    uint32_t cachedBGParticleCount_ = 0;
    WGPUBuffer cachedPositionsBuffer_ = nullptr;
    WGPUBuffer cachedParamsBuffer_ = nullptr;

    void invalidateBindGroups();
    void ensureBindGroupsCached(WGPUDevice device,
                                WGPUBuffer positionsBuffer,
                                WGPUBuffer paramsBuffer,
                                uint32_t numParticles);

    bool pipelinesCreated_ = false;
};
