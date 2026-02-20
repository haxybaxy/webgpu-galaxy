#pragma once

#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct BVHNodeGPU {
    glm::vec4 centerOfMass;  // xyz = COM, w = mass
    glm::vec4 boundsMin;     // xyz = AABB min, w = unused
    glm::vec4 boundsMax;     // xyz = AABB max, w = unused
    int32_t left, right, parent, particleIdx;
};

static_assert(sizeof(BVHNodeGPU) == 64, "BVHNodeGPU must be 64 bytes");

class GpuTreeBuilder {
public:
    void initialize(WGPUDevice device, uint32_t maxParticles);

    void prepareUploads(WGPUQueue queue, uint32_t numParticles);

    void recordTreeBuild(WGPUDevice device,
                         WGPUCommandEncoder encoder,
                         WGPUBuffer positionsBuffer,
                         WGPUBuffer paramsBuffer,
                         uint32_t numParticles);

    WGPUBuffer getBvhNodesBuffer() const { return bvhNodes_.get(); }
    int getNodeCount(int N) const { return (N <= 0) ? 0 : 2 * N - 1; }
    uint32_t getPaddedN(uint32_t N) const;
    uint32_t getMaxParticles() const { return maxParticles_; }

private:
    void createPipelines(WGPUDevice device);
    void buildSortParams(uint32_t paddedN);

    uint32_t maxParticles_ = 0;
    uint32_t lastPaddedN_ = 0;

    // Buffers
    wgpu_utils::Buffer bboxPartial_;
    wgpu_utils::Buffer bboxResult_;
    wgpu_utils::Buffer mortonCodes_;
    wgpu_utils::Buffer sortIndices_;
    wgpu_utils::Buffer sortParamsBuffer_;
    wgpu_utils::Buffer bvhNodes_;
    wgpu_utils::Buffer atomicCounters_;
    wgpu_utils::Buffer numWorkgroupsBuffer_;  // tiny uniform for bbox pass 2

    // Pipelines
    WGPUComputePipeline bboxPass1Pipeline_ = nullptr;
    WGPUComputePipeline bboxPass2Pipeline_ = nullptr;
    WGPUComputePipeline mortonPipeline_ = nullptr;
    WGPUComputePipeline bitonicSortPipeline_ = nullptr;
    WGPUComputePipeline karrasPipeline_ = nullptr;
    WGPUComputePipeline leafInitPipeline_ = nullptr;
    WGPUComputePipeline aggregatePipeline_ = nullptr;

    // Bind group layouts
    WGPUBindGroupLayout bboxPass1Layout_ = nullptr;
    WGPUBindGroupLayout bboxPass2Layout_ = nullptr;
    WGPUBindGroupLayout mortonLayout_ = nullptr;
    WGPUBindGroupLayout bitonicSortLayout_ = nullptr;
    WGPUBindGroupLayout karrasLayout_ = nullptr;
    WGPUBindGroupLayout leafInitLayout_ = nullptr;
    WGPUBindGroupLayout aggregateLayout_ = nullptr;

    // Sort step parameters
    uint32_t numSortSteps_ = 0;
    bool pipelinesCreated_ = false;
};
