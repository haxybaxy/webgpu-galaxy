#pragma once

#include <cstdint>
#include <string>

enum class Scenario { TwoBody, PlummerSphere, RotatingDisk };
enum class ForceMethod { Tree, Direct };

struct Config {
    Scenario scenario = Scenario::RotatingDisk;
    ForceMethod forceMethod = ForceMethod::Tree;
    int numParticles = 10000;
    float dt = 0.001f;
    float softening = 0.5f;
    float theta = 0.75f;
    uint32_t seed = 42;
    int maxSteps = 0;          // 0 = interactive (no limit)
    std::string exportPath;    // empty = no export
    bool headless = false;     // run without window (batch mode)
    bool syncTiming = false;   // flush GPU before recording timestamps
    bool benchmarkPasses = false; // per-pass LBVH timing via separate submissions
};

Config parseArgs(int argc, char **argv);
const char *scenarioName(Scenario s);
const char *forceMethodName(ForceMethod f);
