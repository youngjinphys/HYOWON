// FoF returns periodic connected-component membership only; bulk halo properties
// are derived separately from snapshot phase space.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody::halo {

struct FoFMembership {
    std::size_t id{0};
    std::vector<std::size_t> particle_indices;
};

struct FoFMembershipGridPlan {
    std::size_t particle_count{0};
    std::size_t cells_per_dimension{0};
    std::size_t head_cell_count{0};
    std::size_t link_index_bytes{0};
    std::size_t disjoint_set_index_bytes{0};
    std::size_t estimated_head_bytes{0};
    std::size_t estimated_link_bytes{0};
    std::size_t estimated_disjoint_set_bytes{0};
    core::Real linking_length{0.0};
    core::Real cell_size{0.0};
};

// FoF geometry requires only periodic box size and mean particle spacing.
class FoFGeometryContext {
public:
    FoFGeometryContext(core::Real box_size, core::Real mean_spacing)
        : box_{box_size}, mean_spacing_(mean_spacing) {
        if (!std::isfinite(box_.L) || box_.L <= 0.0
            || !std::isfinite(mean_spacing_) || mean_spacing_ <= 0.0
            || mean_spacing_ > box_.L) {
            throw std::invalid_argument(
                "FoF geometry requires finite positive box size and mean spacing");
        }
    }

    struct BoxView {
        core::Real L{0.0};
    };

    const BoxView& get_box() const noexcept { return box_; }
    core::Real d_mean() const noexcept { return mean_spacing_; }

private:
    BoxView box_;
    core::Real mean_spacing_{0.0};
};

class FoFMembershipFinder {
public:
    // Linking length and retained membership threshold are explicit estimator choices.
    FoFMembershipFinder(
        core::Real box_size,
        core::Real mean_spacing,
        core::Real linking_length_b,
        std::size_t min_particles)
        : config_(box_size, mean_spacing),
          linking_length_b_(linking_length_b),
          min_particles_(min_particles) {
        if (!std::isfinite(linking_length_b_) || linking_length_b_ <= 0.0) {
            throw std::invalid_argument(
                "FoF membership linking parameter b must be finite and positive");
        }
        if (min_particles_ == 0) {
            throw std::invalid_argument(
                "FoF membership min_particles must be positive");
        }
    }

    FoFMembershipGridPlan grid_plan_for_particle_count(
        std::size_t particle_count) const;

    std::vector<FoFMembership> find_memberships(
        const core::ParticleStore& particles) const;

private:
    FoFGeometryContext config_;
    core::Real linking_length_b_;
    std::size_t min_particles_;
};

} // namespace cosmo_nbody::halo
