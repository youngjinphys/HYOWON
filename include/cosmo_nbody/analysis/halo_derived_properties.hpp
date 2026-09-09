// Offline halo bulk properties from snapshot phase space and explicit FoF membership.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"

#include <cstddef>

namespace cosmo_nbody::analysis {

struct HaloDerivedProperties {
    std::size_t halo_id{0};
    std::size_t particle_count{0};
    core::Real mass{0.0};
    core::Vec3 center_of_mass{};
    core::Vec3 mean_momentum{};
    core::Vec3 mean_velocity{};

    // Mass-weighted Cartesian peculiar-velocity dispersion and quadrature sum.
    core::Vec3 velocity_dispersion{};
    core::Real velocity_dispersion_3d{0.0};

    // Sum m*dx_comoving x (v_pec-mean_v); physical angular momentum adds factor a.
    core::Vec3 angular_momentum_comoving{};
    core::Vec3 specific_angular_momentum_comoving{};

    core::Real rms_radius_comoving{0.0};
    core::Real max_radius_comoving{0.0};
};

class HaloDerivedPropertyAnalyzer {
public:
    // FoF supplies connectivity only; bulk quantities come from retained phase space.
    static HaloDerivedProperties compute(
        const core::ParticleStore& particles,
        const halo::FoFMembership& membership,
        core::Real box_size,
        core::Real scale_factor);
};

} // namespace cosmo_nbody::analysis
