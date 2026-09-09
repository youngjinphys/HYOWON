#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>

namespace cosmo_nbody {
namespace domain {

class GhostExchange {
public:
    GhostExchange(core::Real box_size,
                  std::size_t mesh_size,
                  int rank,
                  int size);

    // Replaces the ghost tail with each remote owned particle within periodic
    // r_search of the local mesh-aligned x slab, exactly once. Stored positions
    // stay in [0,L); consumers apply minimum-image interaction geometry.
    void exchange(core::ParticleStore& particles,
                  core::Real r_search) const;

private:
    core::Real box_size_;
    std::size_t mesh_size_;
    int rank_;
    int size_;
};

} // namespace domain
} // namespace cosmo_nbody
