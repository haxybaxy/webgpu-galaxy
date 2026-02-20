#include "config.hpp"
#include <spdlog/spdlog.h>
#include <string>

const char *scenarioName(Scenario s) {
    switch (s) {
    case Scenario::TwoBody: return "Two-Body Orbit";
    case Scenario::PlummerSphere: return "Plummer Sphere";
    case Scenario::RotatingDisk: return "Rotating Disk";
    }
    return "Unknown";
}

const char *integratorName(Integrator i) {
    switch (i) {
    case Integrator::Euler: return "Euler";
    case Integrator::Leapfrog: return "Leapfrog (KDK)";
    }
    return "Unknown";
}

Config parseArgs(int argc, char **argv) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        auto nextVal = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            spdlog::error("Missing value for {}", arg);
            return "";
        };

        if (arg == "--scenario") {
            std::string val = nextVal();
            if (val == "twobody") cfg.scenario = Scenario::TwoBody;
            else if (val == "plummer") cfg.scenario = Scenario::PlummerSphere;
            else if (val == "disk") cfg.scenario = Scenario::RotatingDisk;
            else spdlog::warn("Unknown scenario '{}', using default", val);
        } else if (arg == "--integrator") {
            std::string val = nextVal();
            if (val == "euler") cfg.integrator = Integrator::Euler;
            else if (val == "leapfrog") cfg.integrator = Integrator::Leapfrog;
            else spdlog::warn("Unknown integrator '{}', using default", val);
        } else if (arg == "--N") {
            cfg.numParticles = std::stoi(nextVal());
        } else if (arg == "--dt") {
            cfg.dt = std::stof(nextVal());
        } else if (arg == "--softening") {
            cfg.softening = std::stof(nextVal());
        } else if (arg == "--theta") {
            cfg.theta = std::stof(nextVal());
        } else if (arg == "--seed") {
            cfg.seed = static_cast<uint32_t>(std::stoul(nextVal()));
        } else if (arg == "--steps") {
            cfg.maxSteps = std::stoi(nextVal());
        } else if (arg == "--export") {
            cfg.exportPath = nextVal();
        } else if (arg == "--headless") {
            cfg.headless = true;
        } else if (arg == "--verbose") {
            // handled separately in main
        } else {
            spdlog::warn("Unknown argument: {}", arg);
        }
    }

    if (cfg.headless && cfg.maxSteps == 0) {
        spdlog::warn("Headless mode requires --steps, defaulting to 1000");
        cfg.maxSteps = 1000;
    }

    if (cfg.scenario == Scenario::TwoBody) {
        cfg.numParticles = 2;
    }

    return cfg;
}
