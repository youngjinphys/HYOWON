#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cosmo_nbody::halo {

struct PeriodicNeighbor {
    core::Real radius{0.0};
    std::size_t particle_index{0};
};

struct PeriodicNeighborIndexPlan {
    std::size_t particle_count{0};
    std::size_t cells_per_dimension{0};
    std::size_t cell_count{0};
    std::size_t index_bytes{0};
    std::size_t index_cap_bytes{0};
    bool compact_indices{true};
};

class PeriodicNeighborIndex {
public:
    static PeriodicNeighborIndexPlan make_plan(
        std::size_t particle_count,
        std::size_t index_cap_bytes);

    PeriodicNeighborIndex(
        const core::ParticleStore& particles,
        core::Real box_size,
        std::size_t index_cap_bytes);

    const PeriodicNeighborIndexPlan& plan() const noexcept { return plan_; }
    core::Real cell_width() const noexcept { return cell_width_; }

    // Validate object/count/box and current cell membership. Coordinates may move
    // within a cell because query distances read live positions; cell migration is stale.
    bool describes(
        const core::ParticleStore& particles,
        core::Real box_size) const noexcept;

    void collect_within(
        const core::Vec3& center,
        core::Real radius,
        std::vector<PeriodicNeighbor>& output) const;

private:
    const core::ParticleStore* particles_{nullptr};
    core::Real box_size_{0.0};
    core::Real cell_width_{0.0};
    PeriodicNeighborIndexPlan plan_;

    std::vector<std::uint32_t> compact_heads_;
    std::vector<std::uint32_t> compact_next_;
    std::vector<std::size_t> wide_heads_;
    std::vector<std::size_t> wide_next_;

    std::size_t cell_id_for_position(
        core::Real x,
        core::Real y,
        core::Real z) const;
    std::size_t cell_occupancy(std::size_t cell_id) const;
    void append_cell_neighbors(
        std::size_t cell_id,
        const core::Vec3& wrapped_center,
        core::Real radius,
        std::vector<PeriodicNeighbor>& output) const;
};

} // namespace cosmo_nbody::halo
