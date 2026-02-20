#pragma once

#include <glm/glm.hpp>
#include <vector>

struct Diagnostics {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    double energyDrift = 0.0; // |E(t) - E(0)| / |E(0)|
    glm::dvec3 totalMomentum{0.0};
    double momentumMagnitude = 0.0;
};

class DiagnosticsCalculator {
public:
    Diagnostics compute(const std::vector<glm::vec4> &positions,
                        const std::vector<glm::vec4> &velocities,
                        float softening, bool computePotential = true);
    void reset() {
        initialEnergy_ = 0.0;
        initialized_ = false;
    }

private:
    double initialEnergy_ = 0.0;
    bool initialized_ = false;
};
