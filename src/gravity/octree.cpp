#include "cosmo_nbody/gravity/octree.hpp"

#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/range_safe_weighted_mean.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody {
namespace gravity {

namespace {

bool finite_vec3(const core::Vec3& value) noexcept {
    return std::isfinite(value.x)
        && std::isfinite(value.y)
        && std::isfinite(value.z);
}

core::Real representable_real(long double value, const char* label) {
    const long double maximum = static_cast<long double>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || std::abs(value) > maximum) {
        throw std::overflow_error(
            std::string("Octree ") + label
            + " is not representable in core::Real");
    }
    return static_cast<core::Real>(value);
}

core::Real outward_enclosing_side(
    long double required_half,
    core::Real nominal_side,
    const char* label) {
    if (!std::isfinite(required_half) || required_half < 0.0L) {
        throw std::overflow_error(
            std::string("Octree ") + label
            + " half-width is not finite and non-negative");
    }
    const long double required_side = 2.0L * required_half;
    core::Real enclosing = representable_real(required_side, label);
    if (static_cast<long double>(enclosing) < required_side) {
        enclosing = std::nextafter(
            enclosing,
            std::numeric_limits<core::Real>::infinity());
    }
    if (!std::isfinite(enclosing)) {
        throw std::overflow_error(
            std::string("Octree ") + label
            + " cannot be rounded outward in core::Real");
    }
    return std::max(nominal_side, enclosing);
}

core::Vec3 child_center(
    const core::Vec3& parent_center,
    core::Real offset,
    unsigned int octant) {
    return {
        (octant & 4U)
            ? parent_center.x + offset : parent_center.x - offset,
        (octant & 2U)
            ? parent_center.y + offset : parent_center.y - offset,
        (octant & 1U)
            ? parent_center.z + offset : parent_center.z - offset};
}

struct ParticleRangeSummary {
    core::Real mass{0.0};
    core::Vec3 center_of_mass{};
};

ParticleRangeSummary summarize_particle_range(
    const core::ParticleStore& particles,
    std::span<const OctreeIndex> particle_indices,
    std::size_t begin,
    std::size_t end,
    std::string_view role) {
    if (end <= begin || end > particle_indices.size()) {
        throw std::logic_error(
            std::string(role) + " particle range is invalid");
    }

    math::ExactPositiveDoubleSum mass_sum;
    for (std::size_t index = begin; index < end; ++index) {
        const std::size_t particle_index = static_cast<std::size_t>(
            particle_indices[index]);
        mass_sum.add(particles.mass_at(particle_index));
    }
    const core::Real total_mass = mass_sum.value();
    if (!std::isfinite(total_mass) || !(total_mass > 0.0)) {
        throw std::overflow_error(
            std::string(role) + " total mass is not finite and positive");
    }

    const std::size_t particle_count = end - begin;
    const auto particle_at = [&](std::size_t local_index) -> std::size_t {
        return static_cast<std::size_t>(particle_indices[begin + local_index]);
    };
    const auto mass_at = [&](std::size_t local_index) -> core::Real {
        return particles.mass_at(particle_at(local_index));
    };
    const std::string role_prefix(role);
    ParticleRangeSummary summary;
    summary.mass = total_mass;
    summary.center_of_mass = {
        math::range_safe_weighted_mean(
            particle_count,
            [&](std::size_t local_index) {
                return particles.get_positions_x()[particle_at(local_index)];
            },
            mass_at,
            total_mass,
            role_prefix + " center x"),
        math::range_safe_weighted_mean(
            particle_count,
            [&](std::size_t local_index) {
                return particles.get_positions_y()[particle_at(local_index)];
            },
            mass_at,
            total_mass,
            role_prefix + " center y"),
        math::range_safe_weighted_mean(
            particle_count,
            [&](std::size_t local_index) {
                return particles.get_positions_z()[particle_at(local_index)];
            },
            mass_at,
            total_mass,
            role_prefix + " center z")};
    return summary;
}

} // namespace

std::size_t Octree::maximum_node_count(std::size_t particle_count) {
    if (particle_count == 0) return 0;

    // octree_no_child is reserved as the absent-child sentinel. A tree may use
    // at most that many nodes, with valid indices [0, octree_no_child-1].
    constexpr std::size_t compact_node_limit =
        static_cast<std::size_t>(octree_no_child);
    constexpr std::size_t compact_particle_limit =
        compact_node_limit / 2 + compact_node_limit % 2;
    if (particle_count > compact_particle_limit) {
        throw std::length_error(
            "Single-rank Octree particle count exceeds the compact 2*N-1 topology range");
    }
    return 2 * particle_count - 1;
}

std::size_t Octree::traversal_stack_bound() const noexcept {
    if (nodes_.empty()) return 0;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (maximum_subdivision_depth_ > (maximum - 1) / 7) {
        return nodes_.size();
    }
    return std::min(
        nodes_.size(),
        std::size_t{1} + 7 * maximum_subdivision_depth_);
}

void Octree::build(
    const core::ParticleStore& particles,
    const domain::DomainBounds& bounds) {
    if (!std::isfinite(bounds.x_min) || !std::isfinite(bounds.x_max)
        || !std::isfinite(bounds.y_min) || !std::isfinite(bounds.y_max)
        || !std::isfinite(bounds.z_min) || !std::isfinite(bounds.z_max)) {
        throw std::invalid_argument("Octree bounds must be finite");
    }

    const core::Real dx = bounds.x_max - bounds.x_min;
    const core::Real dy = bounds.y_max - bounds.y_min;
    const core::Real dz = bounds.z_max - bounds.z_min;
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)
        || dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument(
            "Octree bounds must have finite positive extents");
    }

    const core::Vec3 root_center{
        std::midpoint(bounds.x_min, bounds.x_max),
        std::midpoint(bounds.y_min, bounds.y_max),
        std::midpoint(bounds.z_min, bounds.z_max)};
    const core::Real root_side = std::max({dx, dy, dz});
    if (!finite_vec3(root_center) || !std::isfinite(root_side)
        || root_side <= 0.0) {
        throw std::overflow_error(
            "Octree root geometry is not representable");
    }

    const std::size_t num_particles = particles.size();
    const std::size_t maximum_nodes = maximum_node_count(num_particles);
    const auto uniform_mass = particles.get_uniform_mass();
    if (uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
        throw std::invalid_argument(
            "Octree uniform particle mass must be finite and positive");
    }
    for (std::size_t idx = 0; idx < num_particles; ++idx) {
        const core::Vec3 position{
            particles.get_positions_x()[idx],
            particles.get_positions_y()[idx],
            particles.get_positions_z()[idx]};
        if (!finite_vec3(position)) {
            throw std::invalid_argument(
                "Octree particle positions must be finite");
        }
        if (position.x < bounds.x_min || position.x >= bounds.x_max
            || position.y < bounds.y_min || position.y >= bounds.y_max
            || position.z < bounds.z_min || position.z >= bounds.z_max) {
            throw std::invalid_argument(
                "Octree particle positions must lie in the half-open root bounds");
        }
        const core::Real mass = uniform_mass.has_value()
            ? *uniform_mass : particles.get_masses()[idx];
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "Octree particle masses must be finite and positive");
        }
    }

    // Input validation precedes mutation, preserving the last valid tree for all
    // malformed-input failures. Once valid construction starts, existing
    // allocations are reused. A later allocation/arithmetic failure clears the
    // partial tree instead of retaining a second full tree for rollback.
    nodes_.clear();
    particle_indices_.clear();
    maximum_subdivision_depth_ = 0;

    try {
        particle_indices_.resize(num_particles);
        for (std::size_t idx = 0; idx < num_particles; ++idx) {
            particle_indices_[idx] = static_cast<OctreeIndex>(idx);
        }

        if (num_particles == 0) return;

        // Sparse occupied-child storage plus unary-path compression guarantees
        // at most 2*N-1 nodes. Reserving the admitted bound eliminates vector
        // growth copies and makes runtime allocation match resource planning.
        nodes_.reserve(maximum_nodes);

        OctreeNode root;
        root.geometric_center = root_center;
        root.side_length = root_side;
        root.particle_begin = 0;
        root.particle_end = static_cast<OctreeIndex>(num_particles);
        nodes_.push_back(root);
        build_recursive(particles, 0, 0);
    } catch (...) {
        nodes_.clear();
        particle_indices_.clear();
        maximum_subdivision_depth_ = 0;
        throw;
    }
}

void Octree::finalize_leaf(
    const core::ParticleStore& particles,
    OctreeIndex node_idx) {
    OctreeNode& node = nodes_.at(static_cast<std::size_t>(node_idx));
    node.first_child = octree_no_child;
    node.child_mask = 0;

    if (node.particle_begin == node.particle_end) {
        throw std::logic_error(
            "Sparse octree attempted to materialize an empty leaf");
    }

    long double required_half = 0.0L;
    for (std::size_t idx = node.particle_begin;
         idx < node.particle_end;
         ++idx) {
        const std::size_t particle_idx = static_cast<std::size_t>(
            particle_indices_[idx]);
        const long double position_x = static_cast<long double>(
            particles.get_positions_x()[particle_idx]);
        const long double position_y = static_cast<long double>(
            particles.get_positions_y()[particle_idx]);
        const long double position_z = static_cast<long double>(
            particles.get_positions_z()[particle_idx]);
        required_half = std::max({
            required_half,
            std::abs(position_x
                - static_cast<long double>(node.geometric_center.x)),
            std::abs(position_y
                - static_cast<long double>(node.geometric_center.y)),
            std::abs(position_z
                - static_cast<long double>(node.geometric_center.z))});
    }

    const ParticleRangeSummary summary = summarize_particle_range(
        particles,
        std::span<const OctreeIndex>(particle_indices_),
        static_cast<std::size_t>(node.particle_begin),
        static_cast<std::size_t>(node.particle_end),
        "Octree leaf");
    node.mass = summary.mass;
    node.center_of_mass = summary.center_of_mass;
    node.side_length = outward_enclosing_side(
        required_half,
        node.side_length,
        "leaf enclosing side");
}

void Octree::build_recursive(
    const core::ParticleStore& particles,
    OctreeIndex node_idx,
    std::size_t depth) {
    const std::size_t node_pos = static_cast<std::size_t>(node_idx);
    if (node_pos >= nodes_.size()) {
        throw std::out_of_range("Octree recursive node index is invalid");
    }

    while (true) {
        maximum_subdivision_depth_ = std::max(
            maximum_subdivision_depth_, depth);

        const std::size_t begin = nodes_[node_pos].particle_begin;
        const std::size_t end = nodes_[node_pos].particle_end;
        if (end <= begin || end > particle_indices_.size()) {
            throw std::logic_error("Octree node particle range is invalid");
        }
        const std::size_t count = end - begin;
        const core::Vec3 center = nodes_[node_pos].geometric_center;
        const core::Real side_length = nodes_[node_pos].side_length;

        if (count == 1) {
            finalize_leaf(particles, node_idx);
            return;
        }

        std::size_t octant_bounds[9];
        octant_bounds[0] = begin;
        octant_bounds[8] = end;

        auto partition_range = [&](std::size_t range_begin,
                                   std::size_t range_end,
                                   auto predicate) -> std::size_t {
            if (range_begin == range_end) return range_begin;
            auto it = std::partition(
                particle_indices_.begin()
                    + static_cast<std::ptrdiff_t>(range_begin),
                particle_indices_.begin()
                    + static_cast<std::ptrdiff_t>(range_end),
                predicate);
            return static_cast<std::size_t>(it - particle_indices_.begin());
        };

        auto pred_x = [&](OctreeIndex particle_idx) {
            return particles.get_positions_x()[particle_idx] < center.x;
        };
        auto pred_y = [&](OctreeIndex particle_idx) {
            return particles.get_positions_y()[particle_idx] < center.y;
        };
        auto pred_z = [&](OctreeIndex particle_idx) {
            return particles.get_positions_z()[particle_idx] < center.z;
        };

        octant_bounds[4] = partition_range(
            octant_bounds[0], octant_bounds[8], pred_x);
        octant_bounds[2] = partition_range(
            octant_bounds[0], octant_bounds[4], pred_y);
        octant_bounds[6] = partition_range(
            octant_bounds[4], octant_bounds[8], pred_y);
        octant_bounds[1] = partition_range(
            octant_bounds[0], octant_bounds[2], pred_z);
        octant_bounds[3] = partition_range(
            octant_bounds[2], octant_bounds[4], pred_z);
        octant_bounds[5] = partition_range(
            octant_bounds[4], octant_bounds[6], pred_z);
        octant_bounds[7] = partition_range(
            octant_bounds[6], octant_bounds[8], pred_z);

        std::uint8_t child_mask = 0;
        unsigned int occupied_count = 0;
        unsigned int sole_octant = 0;
        for (unsigned int octant = 0; octant < 8; ++octant) {
            if (octant_bounds[octant + 1] == octant_bounds[octant]) continue;
            child_mask = static_cast<std::uint8_t>(
                static_cast<unsigned int>(child_mask) | (1U << octant));
            sole_octant = octant;
            ++occupied_count;
        }
        if (occupied_count == 0) {
            throw std::logic_error(
                "Octree partition produced no occupied child");
        }

        const core::Real child_side = core::Real{0.5} * side_length;
        const core::Real offset = core::Real{0.5} * child_side;
        if (!(child_side > 0.0) || !(offset > 0.0)
            || (center.x + offset == center.x && center.x - offset == center.x)
            || (center.y + offset == center.y && center.y - offset == center.y)
            || (center.z + offset == center.z && center.z - offset == center.z)) {
            // The next dyadic cell is not distinct in the coordinate
            // representation. Keep the represented sources in one exact leaf;
            // exact particle interactions are preferable to an arbitrary depth
            // or epsilon-derived topology cutoff.
            finalize_leaf(particles, node_idx);
            return;
        }

        if (occupied_count == 1) {
            const std::size_t first_idx = static_cast<std::size_t>(
                particle_indices_[begin]);
            const core::Real first_x = particles.get_positions_x()[first_idx];
            const core::Real first_y = particles.get_positions_y()[first_idx];
            const core::Real first_z = particles.get_positions_z()[first_idx];
            bool identical = true;
            for (std::size_t idx = begin + 1; idx < end; ++idx) {
                const std::size_t particle_idx = static_cast<std::size_t>(
                    particle_indices_[idx]);
                if (particles.get_positions_x()[particle_idx] != first_x
                    || particles.get_positions_y()[particle_idx] != first_y
                    || particles.get_positions_z()[particle_idx] != first_z) {
                    identical = false;
                    break;
                }
            }
            if (identical) {
                finalize_leaf(particles, node_idx);
                return;
            }

            // A unary subdivision carries no multipole information and only
            // deepens traversal. Compress it in place while retaining the same
            // particle range and exact geometric cell of the conventional path.
            const core::Vec3 next_center = child_center(
                center, offset, sole_octant);
            if (!finite_vec3(next_center)) {
                throw std::overflow_error(
                    "Octree compressed child center is non-finite");
            }
            nodes_[node_pos].geometric_center = next_center;
            nodes_[node_pos].side_length = child_side;
            if (depth == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error(
                    "Octree subdivision depth overflows size_t");
            }
            ++depth;
            continue;
        }

        const std::size_t maximum_nodes = maximum_node_count(particles.size());
        if (occupied_count > maximum_nodes - nodes_.size()) {
            throw std::length_error(
                "Octree occupied children exceed the admitted 2*N-1 node bound");
        }

        const OctreeIndex first_child = static_cast<OctreeIndex>(nodes_.size());
        nodes_[node_pos].first_child = first_child;
        nodes_[node_pos].child_mask = child_mask;
        nodes_.resize(nodes_.size() + occupied_count);

        std::size_t packed_child = static_cast<std::size_t>(first_child);
        for (unsigned int octant = 0; octant < 8; ++octant) {
            if ((static_cast<unsigned int>(child_mask)
                 & (1U << octant)) == 0) {
                continue;
            }
            OctreeNode& child = nodes_[packed_child++];
            child.particle_begin = static_cast<OctreeIndex>(
                octant_bounds[octant]);
            child.particle_end = static_cast<OctreeIndex>(
                octant_bounds[octant + 1]);
            child.side_length = child_side;
            child.geometric_center = child_center(center, offset, octant);
            if (!finite_vec3(child.geometric_center)) {
                throw std::overflow_error(
                    "Octree child geometric center is non-finite");
            }
        }

        if (depth == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "Octree subdivision depth overflows size_t");
        }
        const std::size_t child_depth = depth + 1;
        const OctreeNode& topology = nodes_[node_pos];
        for (unsigned int octant = 0; octant < 8; ++octant) {
            if (!topology.has_child(octant)) continue;
            build_recursive(
                particles,
                topology.child_index(octant),
                child_depth);
        }

        OctreeNode& current = nodes_[node_pos];
        long double required_half = 0.0L;
        for (unsigned int octant = 0; octant < 8; ++octant) {
            if (!current.has_child(octant)) continue;
            const OctreeNode& child = nodes_[current.child_index(octant)];
            const long double child_half = 0.5L
                * static_cast<long double>(child.side_length);
            required_half = std::max({
                required_half,
                std::abs(
                    static_cast<long double>(child.geometric_center.x)
                    - static_cast<long double>(current.geometric_center.x))
                    + child_half,
                std::abs(
                    static_cast<long double>(child.geometric_center.y)
                    - static_cast<long double>(current.geometric_center.y))
                    + child_half,
                std::abs(
                    static_cast<long double>(child.geometric_center.z)
                    - static_cast<long double>(current.geometric_center.z))
                    + child_half});
        }
        const std::size_t child_count = current.occupied_child_count();
        if (child_count != static_cast<std::size_t>(occupied_count)) {
            throw std::logic_error(
                "Octree packed child count disagrees with its topology mask");
        }

        // A node multipole is defined by its source particles, not by a second
        // weighted average of already-rounded child masses and centers. The
        // latter makes the monopole/center depend on topology partitioning and
        // can move a representable center by one ulp. Recompute the final mass
        // and first moment from this node's unchanged particle range.
        const ParticleRangeSummary summary = summarize_particle_range(
            particles,
            std::span<const OctreeIndex>(particle_indices_),
            static_cast<std::size_t>(current.particle_begin),
            static_cast<std::size_t>(current.particle_end),
            "Octree internal");
        current.mass = summary.mass;
        current.center_of_mass = summary.center_of_mass;
        current.side_length = outward_enclosing_side(
            required_half,
            current.side_length,
            "internal enclosing side");
        return;
    }
}

} // namespace gravity
} // namespace cosmo_nbody
