#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>

namespace cosmo_nbody::analysis {

// Non-physical execution settings; they may change performance/storage placement
// but are excluded from numerical product identity.
struct AnalysisResourcePolicy {
    // Zero uses the analyzer host-thread policy; positive values are host-thread
    // counts in the same size_t domain as HostThreadContext.
    std::size_t requested_threads{0};
    // Optional ceiling for operators that explicitly admit bounded in-process
    // workspaces. It is not a global RSS/FFT cap; zero means no explicit ceiling.
    core::Real memory_budget_gib{0.0};
    // Scratch placement is independent of the bounded-workspace ceiling.
    config::MemoryPolicyParams memory{};
};

} // namespace cosmo_nbody::analysis
