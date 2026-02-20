#pragma once

#include "diagnostics.hpp"
#include <fstream>
#include <string>

struct StepTiming {
    double treeBuildMs = 0.0;
    double forceMs = 0.0;
    double integrateMs = 0.0;
};

class Exporter {
public:
    bool open(const std::string &path);
    void writeRow(int step, double simTime, const Diagnostics &diag,
                  const StepTiming &timing);
    void close();
    bool isOpen() const { return file_.is_open(); }

private:
    std::ofstream file_;
};
