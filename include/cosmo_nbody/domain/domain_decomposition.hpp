#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/domain/ghost_exchange.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cosmo_nbody {
namespace domain {

struct DomainBounds {
    core::Real x_min{0.0};
    core::Real x_max{0.0};
    core::Real y_min{0.0};
    core::Real y_max{0.0};
    core::Real z_min{0.0};
    core::Real z_max{0.0};
};

class DomainDecomposition {
public:
    explicit DomainDecomposition(
        const config::SimulationParameters& config,
        int rank = 0,
        int size = 1);

    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }

    std::ptrdiff_t get_local_slab(
        std::ptrdiff_t N,
        std::ptrdiff_t& local_n0,
        std::ptrdiff_t& local_0_start) const;

    void partition_domain(core::ParticleStore& local_particles);

    void exchange_ghosts(core::ParticleStore& local_particles,
                         core::Real r_search);

    DomainBounds get_local_bounds() const noexcept { return local_bounds_; }
    int owner_rank(core::Real x) const;

private:
    // Collect rank-consistent IC provenance before migration publishes ownership.
    std::string collect_initial_condition_provenance(
        const core::ParticleStore& local_particles) const;

    const config::SimulationParameters& config_;
    DomainBounds local_bounds_;
    int rank_{0};
    int size_{1};
    GhostExchange ghost_exchange_;
    std::size_t mesh_size_{0};
    core::Real mesh_cell_size_{0.0};
    // Exclusive end plane per rank from the PM slab layout; nondecreasing values
    // support empty ranks while upper_bound still gives unique ownership.
    std::vector<std::size_t> slab_end_planes_;
    bool initial_condition_provenance_synchronized_{false};
    // Established partition preserves strict stable-ID order across later migration.
    bool canonical_partition_established_{false};
};

} // namespace domain
} // namespace cosmo_nbody
