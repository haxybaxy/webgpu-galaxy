#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class Scenario { TwoBody, PlummerSphere, RotatingDisk };
enum class Integrator { Euler, Leapfrog };
enum class TreeMethod { CPU, GPU };
enum class ForceMethod { Tree, Direct };

struct Config {
    Scenario scenario = Scenario::RotatingDisk;
    Integrator integrator = Integrator::Leapfrog;
    TreeMethod treeMethod = TreeMethod::GPU;
    ForceMethod forceMethod = ForceMethod::Tree;
    int numParticles = 10000;
    float dt = 0.001f;
    float softening = 0.5f;
    float theta = 0.75f;
    uint32_t seed = 42;
    int maxSteps = 0;          // 0 = interactive (no limit)
    std::string exportPath;    // empty = no export
    bool headless = false;     // run without window (batch mode)
    std::vector<double> screenshotTimes; // times at which to capture screenshots
};

Config parseArgs(int argc, char **argv);
const char *scenarioName(Scenario s);
const char *integratorName(Integrator i);
const char *treeMethodName(TreeMethod t);
const char *forceMethodName(ForceMethod f);
