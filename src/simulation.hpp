#pragma once

#include "octree.hpp"
#include "wgpu_utils.hpp"
#include <glm/glm.hpp>
#include <vector>

class Simulation {
public:
    void initialize(WGPUDevice device, int numParticles);
    void step(WGPUDevice device, WGPUQueue queue, float dt, float softening, float theta);
    void reinitialize(WGPUDevice device, int numParticles);

    WGPUBuffer getPositionBuffer() const { return positions_.get(); }
    WGPUBuffer getColorBuffer() const { return colors_.get(); }
    int getParticleCount() const { return numParticles_; }
    int getNodeCount() const { return static_cast<int>(octree_.nodeCount()); }

private:
    void initParticles(int numParticles);
    void createComputePipelines(WGPUDevice device);

    int numParticles_ = 0;
    wgpu_utils::Buffer positions_;
    wgpu_utils::Buffer velocities_;
    wgpu_utils::Buffer colors_;
    wgpu_utils::Buffer accelerations_;
    wgpu_utils::Buffer octreeBuffer_;
    wgpu_utils::Buffer paramsBuffer_;

    std::vector<glm::vec4> cpuPositions_;
    std::vector<glm::vec4> cpuVelocities_;
    std::vector<glm::vec4> cpuColors_;

    Octree octree_;

    WGPUComputePipeline forcePipeline_ = nullptr;
    WGPUComputePipeline integratePipeline_ = nullptr;
    WGPUBindGroupLayout forceBindGroupLayout_ = nullptr;
    WGPUBindGroupLayout integrateBindGroupLayout_ = nullptr;

    bool pipelinesCreated_ = false;
};
