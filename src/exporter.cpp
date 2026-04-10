#include "exporter.hpp"
#include <iomanip>

bool Exporter::open(const std::string &path, bool withPassBreakdown) {
    file_.open(path);
    if (!file_.is_open()) return false;

    withPassBreakdown_ = withPassBreakdown;
    file_ << "step,time,kinetic_energy,potential_energy,total_energy,"
              "energy_drift,px,py,pz,tree_build_ms,force_ms,integrate_ms";
    if (withPassBreakdown_) {
        file_ << ",bbox_ms,morton_ms,radix_sort_ms,karras_ms,leaf_init_ms,aggregate_ms";
    }
    file_ << ",separation,orbital_period,eccentricity\n";
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
           << timing.integrateMs;
    if (withPassBreakdown_) {
        file_ << "," << timing.bboxReduceMs << "," << timing.mortonMs
              << "," << timing.radixSortMs << "," << timing.karrasMs
              << "," << timing.leafInitMs << "," << timing.aggregateMs;
    }
    file_ << "," << diag.separation << "," << diag.orbitalPeriodEstimate << ","
           << diag.eccentricity << "\n";
}

void Exporter::close() {
    if (file_.is_open()) file_.close();
}
