// Maximum circular velocity over exact periodic halo members.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"

#include <vector>

namespace cosmo_nbody::analysis {

struct VmaxResult {
    bool computed{false};
    core::Real vmax_km_s{0.0};
    core::Real r_vmax_comoving_Mpc_h{0.0};
};

VmaxResult compute_vmax(
    const core::ParticleStore& particles,
    core::Vec3 center,
    core::Real box_size,
    std::vector<halo::PeriodicNeighbor>& members,
    core::Real scale_factor);

} // namespace cosmo_nbody::analysis
