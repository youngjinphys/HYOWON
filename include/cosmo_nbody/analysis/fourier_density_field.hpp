#pragma once

#include "cosmo_nbody/analysis/periodic_domain.hpp"
#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <complex>
#include <cstddef>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

struct FourierDensityField {
    int mesh_size{0};
    core::Real box_size{0.0};
    bool interlaced{false};
    std::size_t particle_count{0};
    // Dimensionless sum(w_i^2)/sum(w_i)^2, invariant to common mass scaling.
    core::Real mass_square_fraction{0.0};
    std::vector<std::complex<core::Real>> modes;
};

class FourierDensityBuilder {
public:
    // Memory policy affects storage only; analysis depends on the physical domain.
    static FourierDensityField build(
        const PeriodicDomain& domain,
        const core::ParticleStore& particles,
        int mesh_size,
        bool interlaced,
        config::MemoryPolicyParams memory_policy = {});
};

} // namespace analysis
} // namespace cosmo_nbody
