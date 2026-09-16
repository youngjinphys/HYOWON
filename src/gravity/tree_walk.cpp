#include "cosmo_nbody/gravity/tree_walk.hpp"

#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/math/conservative_axis_extent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody {
namespace gravity {

namespace {

unsigned int octant_for_position(
    const core::Vec3& position,
    const core::Vec3& center) noexcept {
    unsigned int octant = 0;
    if (!(position.x < center.x)) octant |= 4U;
    if (!(position.y < center.y)) octant |= 2U;
    if (!(position.z < center.z)) octant |= 1U;
    return octant;
}

core::Real positive_upper_sum(core::Real lhs, core::Real rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)
        || lhs < 0.0 || rhs < 0.0) {
        return std::numeric_limits<core::Real>::infinity();
    }
    if (lhs == 0.0) return rhs;
    if (rhs == 0.0) return lhs;
    if (rhs > std::numeric_limits<core::Real>::max() - lhs) {
        return std::numeric_limits<core::Real>::infinity();
    }
    const core::Real sum = lhs + rhs;
    if (!std::isfinite(sum)) {
        return std::numeric_limits<core::Real>::infinity();
    }
    return std::nextafter(
        sum, std::numeric_limits<core::Real>::infinity());
}

core::Real absolute_difference_upper(core::Real lhs, core::Real rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return std::numeric_limits<core::Real>::infinity();
    }
    if (lhs == rhs) return 0.0;
    const core::Real difference = lhs - rhs;
    if (!std::isfinite(difference)) {
        return std::numeric_limits<core::Real>::infinity();
    }
    return std::nextafter(
        std::abs(difference),
        std::numeric_limits<core::Real>::infinity());
}

core::Real positive_upper_quotient(
    core::Real numerator,
    core::Real denominator) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator)
        || numerator < 0.0 || !(denominator > 0.0)) {
        return std::numeric_limits<core::Real>::infinity();
    }
    if (numerator == 0.0) return 0.0;
    const core::Real quotient = numerator / denominator;
    if (!std::isfinite(quotient)) {
        return std::numeric_limits<core::Real>::infinity();
    }
    return std::nextafter(
        quotient, std::numeric_limits<core::Real>::infinity());
}

// Conservative modified Barnes criterion
//
//     r > l/theta + delta,
//
// where r is the target-to-COM distance and delta is the COM displacement from
// the node's geometric center. Outward bounds are used so floating-point
// rounding can only cause an extra node opening, never a spuriously accepted
// multipole interaction.
bool offset_mac_accepts(
    const core::Vec3& target_to_center_of_mass,
    const core::Vec3& center_of_mass,
    const core::Vec3& geometric_center,
    core::Real side_length,
    core::Real theta) {
    if (!std::isfinite(target_to_center_of_mass.x)
        || !std::isfinite(target_to_center_of_mass.y)
        || !std::isfinite(target_to_center_of_mass.z)
        || !std::isfinite(center_of_mass.x)
        || !std::isfinite(center_of_mass.y)
        || !std::isfinite(center_of_mass.z)
        || !std::isfinite(geometric_center.x)
        || !std::isfinite(geometric_center.y)
        || !std::isfinite(geometric_center.z)
        || !std::isfinite(side_length)
        || !std::isfinite(theta)
        || !(side_length > 0.0)
        || !(theta > 0.0)) {
        return false;
    }

    // theta=0 reaches the early return above: open every non-pruned internal
    // node without dividing by zero or substituting a finite opening angle.
    const core::Real offset_x = absolute_difference_upper(
        center_of_mass.x, geometric_center.x);
    const core::Real offset_y = absolute_difference_upper(
        center_of_mass.y, geometric_center.y);
    const core::Real offset_z = absolute_difference_upper(
        center_of_mass.z, geometric_center.z);
    const core::Real offset_upper = positive_upper_sum(
        positive_upper_sum(offset_x, offset_y), offset_z);
    const core::Real aperture_upper = positive_upper_quotient(
        side_length, theta);
    const core::Real required_radius_upper = positive_upper_sum(
        aperture_upper, offset_upper);
    if (!std::isfinite(required_radius_upper)) return false;

    return !core::scale_safe_norm3_leq(
        target_to_center_of_mass.x,
        target_to_center_of_mass.y,
        target_to_center_of_mass.z,
        required_radius_upper);
}

} // namespace

TreeWalk::TreeWalk(
    const Octree& tree,
    const core::ParticleStore& particles,
    const config::SimulationParameters& config)
    : tree_(tree),
      particles_(particles),
      config_(config),
      softening_(config.eps()),
      split_(config.get_gravity().solver == "TreePM"
          ? std::optional<ForceSplitKernel>(
              std::in_place,
              config.r_s(),
              config.r_cut() / config.r_s())
          : std::nullopt),
      r_cut_(split_
          ? split_->cutoff_radius()
          : std::numeric_limits<core::Real>::infinity()),
      G_(cosmology::units::G) {
    if (!std::isfinite(G_) || G_ <= 0.0) {
        throw std::runtime_error(
            "TreeWalk gravitational constant is invalid");
    }
}

core::Vec3 TreeWalk::compute_force(
    std::size_t target_idx,
    core::Real theta) const {
    return compute_force_with_diagnostics(target_idx, theta).force;
}

TreeWalkResult TreeWalk::compute_force_with_diagnostics(
    std::size_t target_idx,
    core::Real theta) const {
    TreeWalkScratch scratch;
    return compute_force_with_diagnostics(target_idx, theta, scratch);
}

TreeWalkResult TreeWalk::compute_force_with_diagnostics(
    std::size_t target_idx,
    core::Real theta,
    TreeWalkScratch& scratch) const {
    if (!std::isfinite(theta) || theta < 0.0) {
        throw std::invalid_argument(
            "TreeWalk opening angle theta must be finite and non-negative");
    }
    if (target_idx >= particles_.num_owned_particles()) {
        throw std::out_of_range(
            "TreeWalk target index must refer to an owned particle");
    }

    TreeWalkResult empty_result;
    empty_result.diagnostics.target_count = 1;
    if (tree_.get_nodes().empty()) return empty_result;

    const core::Real L = config_.get_box().L;
    if (!std::isfinite(L) || L <= 0.0) {
        throw std::runtime_error("TreeWalk box size is invalid");
    }

    const core::Vec3 target_position{
        particles_.get_positions_x()[target_idx],
        particles_.get_positions_y()[target_idx],
        particles_.get_positions_z()[target_idx]};
    if (!std::isfinite(target_position.x)
        || !std::isfinite(target_position.y)
        || !std::isfinite(target_position.z)) {
        throw std::invalid_argument(
            "TreeWalk target position must be finite");
    }
    return compute_force_impl(
        target_position, target_idx, theta, scratch);
}

TreeWalkResult TreeWalk::compute_force_impl(
    const core::Vec3& target_pos,
    std::size_t self_index,
    core::Real theta,
    TreeWalkScratch& scratch) const {
    if (!std::isfinite(theta) || theta < 0.0) {
        throw std::invalid_argument(
            "TreeWalk opening angle theta must be finite and non-negative");
    }
    if (!std::isfinite(target_pos.x)
        || !std::isfinite(target_pos.y)
        || !std::isfinite(target_pos.z)) {
        throw std::invalid_argument(
            "TreeWalk target position must be finite");
    }

    TreeWalkResult result;
    core::Vec3& force = result.force;
    result.diagnostics.target_count = 1;

    const auto& nodes = tree_.get_nodes();
    if (nodes.empty()) return result;
    const auto& particle_indices = tree_.get_particle_indices();

    const core::Real L = config_.get_box().L;
    if (!std::isfinite(L) || L <= 0.0) {
        throw std::runtime_error("TreeWalk box size is invalid");
    }

    const bool has_uniform =
        particles_.get_uniform_mass().has_value();
    const core::Real uniform_mass = has_uniform
        ? particles_.get_uniform_mass().value()
        : 0.0;

    auto cutoff_relation = [&](const OctreeNode& node) -> int {
        if (!split_) return 1;

        const core::Vec3 center_delta = math::minimum_image_displacement(
            target_pos, node.geometric_center, L);
        const core::Real half = 0.5 * node.side_length;
        if (!std::isfinite(half) || half < 0.0) {
            throw std::runtime_error(
                "TreeWalk node half-width is invalid");
        }

        // Classify the exact cube against the open cutoff ball. Axis
        // remainders are never materialized, so a nearest-even |delta|-half
        // overshoot cannot prune an intersecting node. The exact leaf
        // predicate then decides pair inclusion.
        return math::aabb_cutoff_relation(
            center_delta.x,
            center_delta.y,
            center_delta.z,
            half,
            r_cut_);
    };

    auto plummer_force_factor = [&](core::Real radius) {
        if (std::isnan(radius) || radius < 0.0) {
            throw std::runtime_error(
                "TreeWalk pair separation is invalid");
        }
        const core::Real radius_squared = radius * radius;
        const core::Real factor = softening_.force_factor(radius_squared);
        if (!std::isfinite(factor) || factor < 0.0) {
            throw std::runtime_error(
                "TreeWalk Plummer force factor is invalid");
        }
        return factor;
    };

    auto pair_force_contribution = [&] (
        const core::Vec3& displacement,
        core::Real source_mass,
        core::Real radius) {
        if (split_) {
            // TreePM is a subtraction of two individually much larger force
            // coefficients. Compose the dimensionless short-force fraction and
            // dimensional factors together for every interaction to avoid a
            // cancellation-damaged residual.
            return split_->scale_safe_short_range_acceleration(
                displacement,
                source_mass,
                softening_.epsilon());
        }

        const core::Real factor = plummer_force_factor(radius);
        const core::Real coefficient = G_ * source_mass * factor;
        const core::Vec3 candidate{
            coefficient * displacement.x,
            coefficient * displacement.y,
            coefficient * displacement.z};
        if (!std::isfinite(coefficient)
            || !std::isfinite(candidate.x)
            || !std::isfinite(candidate.y)
            || !std::isfinite(candidate.z)) {
            throw std::overflow_error(
                "TreeWalk Plummer force contribution is non-finite");
        }
        return candidate;
    };

    auto& stack = scratch.stack_;
    stack.clear();
    const std::size_t required_stack = tree_.traversal_stack_bound();
    if (stack.capacity() < required_stack) {
        stack.reserve(required_stack);
    }
    stack.push_back(TreeWalkStackEntry{0, true});

    while (!stack.empty()) {
        const TreeWalkStackEntry entry = stack.back();
        stack.pop_back();
        ++result.diagnostics.nodes_visited;
        if (entry.node_index >= nodes.size()) {
            throw std::runtime_error(
                "TreeWalk encountered an invalid node index");
        }

        const OctreeNode& node = nodes[entry.node_index];
        if (node.particle_end <= node.particle_begin
            || node.particle_end > particle_indices.size()) {
            throw std::runtime_error(
                "TreeWalk node particle range is invalid");
        }

        const int cutoff_state = cutoff_relation(node);
        if (cutoff_state < 0) {
            ++result.diagnostics.nodes_cutoff_pruned;
            continue;
        }

        const core::Vec3 dx = math::minimum_image_displacement(
            target_pos, node.center_of_mass, L);
        const core::Real radius = dx.norm();
        if (std::isnan(radius) || radius < 0.0) {
            throw std::runtime_error(
                "TreeWalk node separation is invalid");
        }
        const bool node_direction_unique =
            math::minimum_image_displacement_is_directionally_unique(
                target_pos, node.center_of_mass, L);

        if (node.is_leaf_node()) {
            ++result.diagnostics.leaf_nodes_visited;
            for (std::size_t i = node.particle_begin;
                 i < node.particle_end;
                 ++i) {
                const std::size_t source_index = particle_indices[i];
                if (source_index == self_index) {
                    continue;
                }
                if (source_index >= particles_.size()) {
                    throw std::out_of_range(
                        "TreeWalk source index exceeds ParticleStore size");
                }

                const core::Vec3 source_pos{
                    particles_.get_positions_x()[source_index],
                    particles_.get_positions_y()[source_index],
                    particles_.get_positions_z()[source_index]};
                const core::Vec3 pair_dx =
                    math::minimum_image_displacement(
                        target_pos, source_pos, L);
                if (split_
                    && !core::scale_safe_norm3_less(
                        pair_dx.x,
                        pair_dx.y,
                        pair_dx.z,
                        r_cut_)) {
                    ++result.diagnostics.pair_cutoff_rejected;
                    continue;
                }
                const core::Real pair_radius = pair_dx.norm();
                if (!math::minimum_image_displacement_is_directionally_unique(
                        target_pos, source_pos, L)) {
                    throw std::invalid_argument(
                        "TreeWalk minimum-image force direction is undefined at an exact half-box separation");
                }

                const core::Real mass = has_uniform
                    ? uniform_mass
                    : particles_.get_masses()[source_index];
                if (!std::isfinite(mass) || mass <= 0.0) {
                    throw std::invalid_argument(
                        "TreeWalk source mass must be finite and positive");
                }
                const core::Vec3 contribution = pair_force_contribution(
                    pair_dx,
                    mass,
                    pair_radius);
                ++result.diagnostics.exact_pairs_evaluated;
                force.x += contribution.x;
                force.y += contribution.y;
                force.z += contribution.z;
            }
        } else {
            const bool straddles_cutoff =
                split_ && cutoff_state == 0;

            // A geometric-size MAC based only on the distance to the center of
            // mass can accept a highly lopsided node even when its nearest
            // represented source is too close. The modified Barnes criterion
            // accounts for the displacement between geometric center and COM.
            // Its arithmetic is evaluated fail-closed by offset_mac_accepts().
            const bool opening_accepts = node_direction_unique
                && offset_mac_accepts(
                    dx,
                    node.center_of_mass,
                    node.geometric_center,
                    node.side_length,
                    theta);

            if (!entry.contains_target
                && !straddles_cutoff
                && opening_accepts) {
                const core::Vec3 contribution = pair_force_contribution(
                    dx,
                    node.mass,
                    radius);
                ++result.diagnostics.multipole_nodes_accepted;
                force.x += contribution.x;
                force.y += contribution.y;
                force.z += contribution.z;
            } else {
                ++result.diagnostics.internal_nodes_opened;
                const unsigned int target_octant = entry.contains_target
                    ? octant_for_position(target_pos, node.geometric_center)
                    : 0U;
                if (entry.contains_target
                    && !node.has_child(target_octant)) {
                    throw std::logic_error(
                        "TreeWalk target-containing node has no target child");
                }
                const std::size_t child_count = node.occupied_child_count();
                // Enforce the precomputed mathematical stack bound rather than
                // relying on implementation-specific vector over-allocation.
                if (stack.size() > required_stack
                    || child_count > required_stack - stack.size()) {
                    throw std::logic_error(
                        "TreeWalk exceeded the traversal-stack bound");
                }
                for (unsigned int octant = 0; octant < 8; ++octant) {
                    if (!node.has_child(octant)) continue;
                    stack.push_back(TreeWalkStackEntry{
                        node.child_index(octant),
                        entry.contains_target && octant == target_octant});
                }
            }
        }

        if (!std::isfinite(force.x)
            || !std::isfinite(force.y)
            || !std::isfinite(force.z)) {
            throw std::overflow_error(
                "TreeWalk accumulated force is non-finite");
        }
    }

    return result;
}

} // namespace gravity
} // namespace cosmo_nbody
