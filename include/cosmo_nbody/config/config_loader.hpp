// TOML parsing boundary: reject unknown keys, apply CLI overrides, validate
// SimulationParameters, and optionally bind the exact parsed source SHA-256.
#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"

#include <string>

namespace cosmo_nbody {
namespace config {

struct LoadedSimulationConfig {
    SimulationParameters parameters;
    std::string source_sha256;
};

class ConfigLoader {
public:
    static SimulationParameters load(
        const std::string& filepath,
        int argc = 0,
        const char* const* argv = nullptr);

    // SHA-256 covers the same source bytes that were parsed; overrides affect parameters.
    static LoadedSimulationConfig load_with_identity(
        const std::string& filepath,
        int argc = 0,
        const char* const* argv = nullptr);
};

} // namespace config
} // namespace cosmo_nbody
