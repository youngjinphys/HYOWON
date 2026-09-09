#ifndef COSMO_NBODY_GRAVITY_TREE_WALK_HPP
#define COSMO_NBODY_GRAVITY_TREE_WALK_HPP

#include "cosmo_nbody/gravity/octree.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/gravity/softening_kernel.hpp"
#include "cosmo_nbody/gravity/force_split.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace cosmo_nbody {
namespace gravity {

struct TreeWalkDiagnostics {
    std::uint64_t target_count{0};
    std::uint64_t nodes_visited{0};
    std::uint64_t internal_nodes_opened{0};
    std::uint64_t multipole_nodes_accepted{0};
    std::uint64_t leaf_nodes_visited{0};
    std::uint64_t exact_pairs_evaluated{0};
    std::uint64_t pair_cutoff_rejected{0};
    std::uint64_t nodes_cutoff_pruned{0};

    TreeWalkDiagnostics& operator+=(const TreeWalkDiagnostics& other) noexcept {
        target_count += other.target_count;
        nodes_visited += other.nodes_visited;
        internal_nodes_opened += other.internal_nodes_opened;
        multipole_nodes_accepted += other.multipole_nodes_accepted;
        leaf_nodes_visited += other.leaf_nodes_visited;
        exact_pairs_evaluated += other.exact_pairs_evaluated;
        pair_cutoff_rejected += other.pair_cutoff_rejected;
        nodes_cutoff_pruned += other.nodes_cutoff_pruned;
        return *this;
    }
};

struct TreeWalkResult {
    core::Vec3 force{};
    TreeWalkDiagnostics diagnostics{};
};

struct TreeWalkStackEntry {
    OctreeIndex node_index{0};
    bool contains_target{false};
};

class TreeWalkScratch {
public:
    void reserve(std::size_t entries) { stack_.reserve(entries); }
    std::size_t capacity() const noexcept { return stack_.capacity(); }

private:
    friend class TreeWalk;
    std::vector<TreeWalkStackEntry> stack_;
};

class TreeWalk {
public:
    // Short-range operator; split and cutoff coordinates come from
    // SimulationParameters.
    TreeWalk(const Octree& tree, const core::ParticleStore& particles,
             const config::SimulationParameters& config);

    core::Vec3 compute_force(std::size_t target_idx, core::Real theta) const;
    TreeWalkResult compute_force_with_diagnostics(
        std::size_t target_idx, core::Real theta) const;
    TreeWalkResult compute_force_with_diagnostics(
        std::size_t target_idx, core::Real theta,
        TreeWalkScratch& scratch) const;

private:
    const Octree& tree_;
    const core::ParticleStore& particles_;
    const config::SimulationParameters& config_;
    SofteningKernel softening_;
    std::optional<ForceSplitKernel> split_;

    core::Real r_cut_;
    core::Real G_;

    TreeWalkResult compute_force_impl(
        const core::Vec3& target_position,
        std::size_t self_index,
        core::Real theta,
        TreeWalkScratch& scratch) const;
};

} // namespace gravity
} // namespace cosmo_nbody

#endif // COSMO_NBODY_GRAVITY_TREE_WALK_HPP
