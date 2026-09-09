#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"

#include <cstddef>
#include <vector>

namespace cosmo_nbody::halo {

// Immutable batch aperture query: validate index membership once, then reuse it
// while particle positions and ownership remain unchanged.
class ExactPeriodicApertureQuery {
public:
    ExactPeriodicApertureQuery(
        const PeriodicNeighborIndex& index,
        const core::ParticleStore& particles,
        core::Real box_size);

    bool describes(
        const core::ParticleStore& particles,
        core::Real box_size) const noexcept;

    core::Real cell_width() const;

    void collect(
        const core::Vec3& center,
        core::Real radius,
        std::vector<PeriodicNeighbor>& output) const;

private:
    const PeriodicNeighborIndex* index_{nullptr};
    const core::ParticleStore* particles_{nullptr};
    core::Real box_size_{0.0};
    std::size_t particle_count_{0};
};

// One-shot path with index-identity/current-membership validation.
void collect_exact_periodic_aperture(
    const PeriodicNeighborIndex& index,
    const core::ParticleStore& particles,
    const core::Vec3& center,
    core::Real radius,
    core::Real box_size,
    std::vector<PeriodicNeighbor>& output);

} // namespace cosmo_nbody::halo
