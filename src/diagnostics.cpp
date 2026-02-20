#include "diagnostics.hpp"
#include <cmath>

Diagnostics DiagnosticsCalculator::compute(
    const std::vector<glm::vec4> &positions,
    const std::vector<glm::vec4> &velocities, float softening,
    bool computePotential) {
    Diagnostics diag;
    int N = static_cast<int>(positions.size());
    double softSq = static_cast<double>(softening) * softening;

    // Kinetic energy and momentum
    for (int i = 0; i < N; i++) {
        double mass = static_cast<double>(positions[i].w);
        glm::dvec3 vel(velocities[i].x, velocities[i].y, velocities[i].z);
        diag.kineticEnergy += 0.5 * mass * glm::dot(vel, vel);
        diag.totalMomentum += mass * vel;
    }
    diag.momentumMagnitude = glm::length(diag.totalMomentum);

    // Potential energy — O(N^2), only for N <= 5000
    if (computePotential && N <= 5000) {
        for (int i = 0; i < N; i++) {
            double mi = static_cast<double>(positions[i].w);
            for (int j = i + 1; j < N; j++) {
                double mj = static_cast<double>(positions[j].w);
                glm::dvec3 diff(
                    static_cast<double>(positions[i].x) - positions[j].x,
                    static_cast<double>(positions[i].y) - positions[j].y,
                    static_cast<double>(positions[i].z) - positions[j].z);
                double distSq = glm::dot(diff, diff) + softSq;
                diag.potentialEnergy -= mi * mj / std::sqrt(distSq);
            }
        }
    }

    diag.totalEnergy = diag.kineticEnergy + diag.potentialEnergy;

    if (!initialized_) {
        initialEnergy_ = diag.totalEnergy;
        initialized_ = true;
    }

    if (std::abs(initialEnergy_) > 1e-15) {
        diag.energyDrift = std::abs(diag.totalEnergy - initialEnergy_) /
                           std::abs(initialEnergy_);
    }

    return diag;
}
