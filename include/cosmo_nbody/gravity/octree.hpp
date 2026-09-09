#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/domain/domain_decomposition.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody {
namespace gravity {

using OctreeIndex = std::uint32_t;
inline constexpr OctreeIndex octree_no_child =
    std::numeric_limits<OctreeIndex>::max();

struct OctreeNode {
    core::Real mass{0.0};
    core::Vec3 center_of_mass{0.0, 0.0, 0.0};
    core::Vec3 geometric_center{0.0, 0.0, 0.0};

    // Outward-rounded subdivision width encloses every source; exact parent/child
    // factor-of-two width is not guaranteed after construction.
    core::Real side_length{0.0};

    OctreeIndex particle_begin{0};
    OctreeIndex particle_end{0};

    // Occupied children are packed contiguously in ascending octant order;
    // child_mask maps packed entries to geometric octants.
    OctreeIndex first_child{octree_no_child};
    std::uint8_t child_mask{0};

    bool is_leaf_node() const noexcept { return child_mask == 0; }

    bool has_child(unsigned int octant) const noexcept {
        return octant < 8
            && (static_cast<unsigned int>(child_mask)
                & (1U << octant)) != 0;
    }

    OctreeIndex child_index(unsigned int octant) const {
        if (first_child == octree_no_child || !has_child(octant)) {
            throw std::out_of_range("Octree child octant is not occupied");
        }
        const unsigned int lower_octants = octant == 0
            ? 0U : ((1U << octant) - 1U);
        const unsigned int packed_offset = std::popcount(
            static_cast<unsigned int>(child_mask) & lower_octants);
        return static_cast<OctreeIndex>(
            first_child + static_cast<OctreeIndex>(packed_offset));
    }

    std::size_t occupied_child_count() const noexcept {
        return static_cast<std::size_t>(std::popcount(
            static_cast<unsigned int>(child_mask)));
    }
};

class Octree {
public:
    Octree() = default;

    // Subdivide until one source, identical represented coordinates, or no
    // representable child cell remains; no hidden bucket/depth/epsilon criterion
    // participates in the TreeWalk opening approximation.
    void build(
        const core::ParticleStore& particles,
        const domain::DomainBounds& bounds);

    const std::vector<OctreeNode>& get_nodes() const { return nodes_; }
    const std::vector<OctreeIndex>& get_particle_indices() const {
        return particle_indices_;
    }

    // Compact non-empty topology has at most 2*N-1 nodes and must fit OctreeIndex.
    static std::size_t maximum_node_count(std::size_t particle_count);

    // Conservative DFS stack bound: at most one path plus seven unvisited siblings
    // per geometric depth, capped by actual node count.
    std::size_t traversal_stack_bound() const noexcept;

    std::size_t maximum_subdivision_depth() const noexcept {
        return maximum_subdivision_depth_;
    }

private:
    std::vector<OctreeNode> nodes_;
    std::vector<OctreeIndex> particle_indices_;
    std::size_t maximum_subdivision_depth_{0};

    void build_recursive(
        const core::ParticleStore& particles,
        OctreeIndex node_idx,
        std::size_t depth);
    void finalize_leaf(
        const core::ParticleStore& particles,
        OctreeIndex node_idx);
};

} // namespace gravity
} // namespace cosmo_nbody
