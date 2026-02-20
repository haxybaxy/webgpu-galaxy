#pragma once

#include "config.hpp"
#include "exporter.hpp"
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

    WGPUBuffer getPositionBuffer() const { return positions_.get(); }
    WGPUBuffer getColorBuffer() const { return colors_.get(); }
    int getParticleCount() const { return numParticles_; }
    int getNodeCount() const {
        return static_cast<int>(octree_.nodeCount());
    }
    Integrator getIntegrator() const { return integrator_; }
    Scenario getScenario() const { return scenario_; }

    const std::vector<glm::vec4> &getCpuPositions() const {
        return cpuPositions_;
    }
    const std::vector<glm::vec4> &getCpuVelocities() const {
        return cpuVelocities_;
    }
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

    int numParticles_ = 0;
    Integrator integrator_ = Integrator::Leapfrog;
    Scenario scenario_ = Scenario::RotatingDisk;

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
    StepTiming lastTiming_;

    WGPUComputePipeline forcePipeline_ = nullptr;
    WGPUComputePipeline kickPipeline_ = nullptr;
    WGPUComputePipeline driftPipeline_ = nullptr;
    WGPUComputePipeline integratePipeline_ = nullptr; // Euler fallback
    WGPUBindGroupLayout forceBindGroupLayout_ = nullptr;
    WGPUBindGroupLayout kickBindGroupLayout_ = nullptr;
    WGPUBindGroupLayout driftBindGroupLayout_ = nullptr;
    WGPUBindGroupLayout integrateBindGroupLayout_ = nullptr;

    bool pipelinesCreated_ = false;
};
