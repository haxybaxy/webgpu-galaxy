#include "exporter.hpp"
#include <iomanip>

bool Exporter::open(const std::string &path) {
    file_.open(path);
    if (!file_.is_open()) return false;

    file_ << "step,time,kinetic_energy,potential_energy,total_energy,"
              "energy_drift,px,py,pz,tree_build_ms,force_ms,integrate_ms,"
              "separation,orbital_period,eccentricity\n";
    return true;
}

void Exporter::writeRow(int step, double simTime, const Diagnostics &diag,
                        const StepTiming &timing) {
    if (!file_.is_open()) return;

    file_ << step << "," << std::fixed << std::setprecision(6) << simTime
           << "," << diag.kineticEnergy << "," << diag.potentialEnergy << ","
           << diag.totalEnergy << "," << std::scientific << std::setprecision(6)
           << diag.energyDrift << "," << std::fixed << std::setprecision(6)
           << diag.totalMomentum.x << "," << diag.totalMomentum.y << ","
           << diag.totalMomentum.z << "," << std::setprecision(3)
           << timing.treeBuildMs << "," << timing.forceMs << ","
           << timing.integrateMs << ","
           << diag.separation << "," << diag.orbitalPeriodEstimate << ","
           << diag.eccentricity << "\n";
}

void Exporter::close() {
    if (file_.is_open()) file_.close();
}
