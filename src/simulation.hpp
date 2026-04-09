#pragma once

#include "config.hpp"
#include "exporter.hpp"
#include "gpu_tree_builder.hpp"
#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <vector>

class Simulation {
public:
    void initialize(WGPUDevice device, WGPUQueue queue, const Config &config);
    void step(WGPUDevice device, WGPUQueue queue, float dt, float softening,
              float theta);
    void reinitialize(WGPUDevice device, WGPUQueue queue, const Config &config);
    void readbackState(WGPUDevice device, WGPUQueue queue,
                       std::vector<glm::vec4> &outPositions,
                       std::vector<glm::vec4> &outVelocities);

    WGPUBuffer getPositionBuffer() const { return positions_.get(); }
    WGPUBuffer getColorBuffer() const { return colors_.get(); }
    int getParticleCount() const { return numParticles_; }
    int getNodeCount() const {
        return gpuTreeBuilder_.getNodeCount(numParticles_);
    }
    Scenario getScenario() const { return scenario_; }
    ForceMethod getForceMethod() const { return forceMethod_; }

    const StepTiming &getLastTiming() const { return lastTiming_; }
    void debugDumpTree(WGPUDevice device, WGPUQueue queue);

private:
    void initParticles(const Config &config);
    void initTwoBody(const Config &config);
    void initPlummerSphere(const Config &config);
    void initRotatingDisk(const Config &config);
    void createComputePipelines(WGPUDevice device);
    void computeInitialForces(WGPUDevice device, WGPUQueue queue,
                              float softening, float theta);

    void stepWithDirectForce(WGPUDevice device, WGPUQueue queue, float dt,
                             float softening, float theta);
    void stepLeapfrogGpuTree(WGPUDevice device, WGPUQueue queue, float dt,
                             float softening, float theta);

    int numParticles_ = 0;
    Scenario scenario_ = Scenario::RotatingDisk;
    ForceMethod forceMethod_ = ForceMethod::Tree;

    wgpu_utils::Buffer positions_;
    wgpu_utils::Buffer velocities_;
    wgpu_utils::Buffer colors_;
    wgpu_utils::Buffer accelerations_;
    wgpu_utils::Buffer paramsBuffer_;

    std::vector<glm::vec4> cpuPositions_;
    std::vector<glm::vec4> cpuVelocities_;
    std::vector<glm::vec4> cpuColors_;

    GpuTreeBuilder gpuTreeBuilder_;
    StepTiming lastTiming_;

    // Kick/drift pipelines
    WGPUComputePipeline kickPipeline_ = nullptr;
    WGPUComputePipeline driftPipeline_ = nullptr;
    WGPUBindGroupLayout kickBindGroupLayout_ = nullptr;
    WGPUBindGroupLayout driftBindGroupLayout_ = nullptr;

    // Direct summation O(N²)
    WGPUComputePipeline directForcePipeline_ = nullptr;
    WGPUBindGroupLayout directForceBindGroupLayout_ = nullptr;

    // BVH force traversal
    WGPUComputePipeline bvhForcePipeline_ = nullptr;
    WGPUBindGroupLayout bvhForceBindGroupLayout_ = nullptr;

    // Cached bind groups (GPU tree path)
    WGPUBindGroup cachedKickBG_ = nullptr;
    WGPUBindGroup cachedDriftBG_ = nullptr;
    WGPUBindGroup cachedBvhForceBG_ = nullptr;
    int cachedBGParticleCount_ = 0;

    void invalidateBindGroups();
    void ensureBindGroupsCached(WGPUDevice device);

    bool pipelinesCreated_ = false;
};
