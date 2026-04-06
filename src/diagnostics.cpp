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

    // Two-body metrics (only meaningful for N=2)
    if (N == 2) {
        glm::dvec3 r1(positions[0].x, positions[0].y, positions[0].z);
        glm::dvec3 r2(positions[1].x, positions[1].y, positions[1].z);
        glm::dvec3 v1(velocities[0].x, velocities[0].y, velocities[0].z);
        glm::dvec3 v2(velocities[1].x, velocities[1].y, velocities[1].z);
        double m1 = positions[0].w;
        double m2 = positions[1].w;

        glm::dvec3 dr = r2 - r1;
        glm::dvec3 dv = v2 - v1;
        diag.separation = glm::length(dr);

        // Reduced mass and total mass
        double M = m1 + m2;
        double mu = m1 * m2 / M;
        double G = 1.0;

        // Specific orbital energy and angular momentum (relative motion)
        double v_rel = glm::length(dv);
        double r_mag = diag.separation;
        double eps = softSq; // softening squared
        double specificEnergy = 0.5 * v_rel * v_rel - G * M / std::sqrt(r_mag * r_mag + eps);

        glm::dvec3 L = glm::cross(dr, dv);
        double L_mag = glm::length(L);

        // Semi-major axis: a = -G*M / (2*E_specific)
        if (std::abs(specificEnergy) > 1e-15) {
            double a = -G * M / (2.0 * specificEnergy);
            // Kepler period: T = 2*pi*sqrt(a^3 / (G*M))
            if (a > 0.0) {
                diag.orbitalPeriodEstimate = 2.0 * M_PI * std::sqrt(a * a * a / (G * M));
            }
            // Eccentricity: e = sqrt(1 + 2*E*L^2 / (G^2*M^2*mu))
            double eArg = 1.0 + 2.0 * specificEnergy * L_mag * L_mag / (G * G * M * M);
            diag.eccentricity = (eArg > 0.0) ? std::sqrt(eArg) : 0.0;
        }
    }

    return diag;
}
