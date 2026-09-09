#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mpi_fft_allocation_layout.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace cosmo_nbody::mesh {

struct PotentialHaloPlane {
    std::size_t global_plane{0};
    int owner_rank{0};
    std::size_t owner_local_plane{0};
};

// Centered x-gradient at CIC support needs start-1, start+count, and
// start+count+1 modulo N; duplicate periodic planes are removed.
std::vector<PotentialHaloPlane> potential_halo_plan(
    const MpiFFTCommunicatorLayout& communicator,
    int rank);

// Collective exchange over MPI_COMM_WORLD; local requested planes are copied,
// remote planes transported, and output follows potential_halo_plan() order.
std::vector<core::Real> exchange_potential_halo_planes(
    const MpiFFTCommunicatorLayout& communicator,
    int rank,
    std::span<const core::Real> local_field);

std::span<const core::Real> potential_halo_plane(
    std::span<const core::Real> exchanged,
    std::size_t plane_cells,
    std::size_t plan_index);

} // namespace cosmo_nbody::mesh
