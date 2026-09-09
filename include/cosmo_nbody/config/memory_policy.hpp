#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <string>
#include <string_view>

namespace cosmo_nbody {
namespace config {

enum class ScratchMode {
    Memory,
    Disk
};

std::string_view scratch_mode_name(ScratchMode mode) noexcept;
ScratchMode parse_scratch_mode(std::string value);

constexpr bool uses_file_backed_scratch(ScratchMode mode) noexcept {
    return mode == ScratchMode::Disk;
}

// Scratch placement only; actual allocation remains the runtime authority.
struct MemoryPolicyParams {
    // IC planner and LPT work arrays.
    ScratchMode ic_scratch_mode{ScratchMode::Memory};
    // Serial/replicated PM and deterministic CIC scratch; numeric method is unchanged.
    ScratchMode evolution_scratch_mode{ScratchMode::Memory};
    std::string scratch_directory{""};
};

} // namespace config
} // namespace cosmo_nbody
