#pragma once

#include "config.hpp"
#include "exporter.hpp"
#include "gpu_tree_builder.hpp"
#include "octree.hpp"
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
        if (treeMethod_ == TreeMethod::GPU)
            return gpuTreeBuilder_.getNodeCount(numParticles_);
        return static_cast<int>(octree_.nodeCount());
    }
    Integrator getIntegrator() const { return integrator_; }
    Scenario getScenario() const { return scenario_; }
    TreeMethod getTreeMethod() const { return treeMethod_; }
    ForceMethod getForceMethod() const { return forceMethod_; }

    const StepTiming &getLastTiming() const { return lastTiming_; }

private:
    void initParticles(const Config &config);
    void initTwoBody(const Config &config);
    void initPlummerSphere(const Config &config);
    void initRotatingDisk(const Config &config);
    void createComputePipelines(WGPUDevice device);
    void computeInitialForces(WGPUQueue queue, float softening, float theta);
    void cpuBarnesHut(float softening, float theta);

    void stepLeapfrog(WGPUDevice device, WGPUQueue queue, float dt,
                      float softening, float theta);
    void stepEuler(WGPUDevice device, WGPUQueue queue, float dt,
                   float softening, float theta);
    void stepWithDirectForce(WGPUDevice device, WGPUQueue queue, float dt,
                             float softening, float theta);
    void stepLeapfrogGpuTree(WGPUDevice device, WGPUQueue queue, float dt,
                             float softening, float theta);

    int numParticles_ = 0;
    Integrator integrator_ = Integrator::Leapfrog;
    Scenario scenario_ = Scenario::RotatingDisk;
    TreeMethod treeMethod_ = TreeMethod::GPU;
    ForceMethod forceMethod_ = ForceMethod::Tree;

    wgpu_utils::Buffer positions_;
    wgpu_utils::Buffer velocities_;
    wgpu_utils::Buffer colors_;
    wgpu_utils::Buffer accelerations_;
    wgpu_utils::Buffer octreeBuffer_;
    wgpu_utils::Buffer paramsBuffer_;

    std::vector<glm::vec4> cpuPositions_;
    std::vector<glm::vec4> cpuVelocities_;
    std::vector<glm::vec4> cpuColors_;
    std::vector<glm::vec4> cpuAccelerations_;

    Octree octree_;
    GpuTreeBuilder gpuTreeBuilder_;
    StepTiming lastTiming_;

    // CPU octree force pipeline
    WGPUComputePipeline forcePipeline_ = nullptr;
    WGPUBindGroupLayout forceBindGroupLayout_ = nullptr;

    // Kick/drift pipelines (shared)
    WGPUComputePipeline kickPipeline_ = nullptr;
    WGPUComputePipeline driftPipeline_ = nullptr;
    WGPUBindGroupLayout kickBindGroupLayout_ = nullptr;
    WGPUBindGroupLayout driftBindGroupLayout_ = nullptr;

    // Euler fallback
    WGPUComputePipeline integratePipeline_ = nullptr;
    WGPUBindGroupLayout integrateBindGroupLayout_ = nullptr;

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
