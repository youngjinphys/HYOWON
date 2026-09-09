// Exact-member Bullock spin on a periodic spherical-overdensity aperture.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace cosmo_nbody::analysis {

struct SpinResult {
    std::string status{"not_computed"};
    // Bullock et al. 2001: lambda'=J/(sqrt(2) M V R), V^2=GM/R.
    core::Real bullock_lambda{0.0};
    core::Real angular_momentum{0.0};
    core::Real circular_velocity{0.0};
    core::Real r_delta{0.0};
    core::Real mass_delta{0.0};
    std::size_t particle_count{0};

    bool computed() const noexcept { return status == "computed"; }
};

class HaloSpin final {
public:
    // Exact SO members are sorted in place by stable ID for layout-independent
    // arithmetic; their exact positive mass sum must match the published SO mass.
    static SpinResult compute_members(
        const core::ParticleStore& particles,
        const core::Vec3& center,
        core::Real r_delta,
        core::Real mass_delta,
        core::Real box_size,
        core::Real scale_factor,
        std::span<halo::PeriodicNeighbor> members);
};

} // namespace cosmo_nbody::analysis
