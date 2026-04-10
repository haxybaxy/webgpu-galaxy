#pragma once

#include "diagnostics.hpp"
#include <fstream>
#include <string>

struct StepTiming {
    double treeBuildMs = 0.0;
    double forceMs = 0.0;
    double integrateMs = 0.0;
    // Per-pass breakdown (populated when --benchmark-passes)
    double bboxReduceMs = 0.0;
    double mortonMs = 0.0;
    double radixSortMs = 0.0;
    double karrasMs = 0.0;
    double leafInitMs = 0.0;
    double aggregateMs = 0.0;
    bool hasPassBreakdown = false;
};

class Exporter {
public:
    bool open(const std::string &path, bool withPassBreakdown = false);
    void writeRow(int step, double simTime, const Diagnostics &diag,
                  const StepTiming &timing);
    void close();
    bool isOpen() const { return file_.is_open(); }

private:
    std::ofstream file_;
    bool withPassBreakdown_ = false;
};
