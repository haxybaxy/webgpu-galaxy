#pragma once

#include <cstdint>
#include <string>

enum class Scenario { TwoBody, PlummerSphere, RotatingDisk };
enum class Integrator { Euler, Leapfrog };

struct Config {
    Scenario scenario = Scenario::RotatingDisk;
    Integrator integrator = Integrator::Leapfrog;
    int numParticles = 10000;
    float dt = 0.001f;
    float softening = 0.5f;
    float theta = 0.75f;
    uint32_t seed = 42;
    int maxSteps = 0;          // 0 = interactive (no limit)
    std::string exportPath;    // empty = no export
    bool headless = false;     // run without window (batch mode)
};

Config parseArgs(int argc, char **argv);
const char *scenarioName(Scenario s);
const char *integratorName(Integrator i);
