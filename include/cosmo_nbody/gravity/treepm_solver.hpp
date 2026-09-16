#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/domain/domain_decomposition.hpp"
#include "cosmo_nbody/gravity/force_split.hpp"
#include "cosmo_nbody/gravity/pm_solver.hpp"
#include "cosmo_nbody/gravity/tree_walk.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace cosmo_nbody::gravity {

// Scheduling only: every target traverses the immutable tree in the same order.
inline constexpr std::size_t treepm_parallel_chunks_per_worker = 8;

inline std::size_t treepm_parallel_chunk_count(
    std::size_t particle_count,
    std::size_t worker_count) noexcept {
    if (particle_count == 0) return 0;
    if (worker_count == 0) worker_count = 1;
    if (worker_count
        > std::numeric_limits<std::size_t>::max()
            / treepm_parallel_chunks_per_worker) {
        return particle_count;
    }
    const std::size_t desired =
        worker_count * treepm_parallel_chunks_per_worker;
    return std::min(particle_count, desired);
}

struct TreePMChunkWorkspace {
    TreeWalkScratch traversal;
    TreeWalkDiagnostics diagnostics{};
};

class TreePMSolver {
public:
    TreePMSolver(
        const config::SimulationParameters& config,
        const mesh::MeshGeometry& geom,
        domain::DomainBounds local_bounds);

    // Writes PM-long + tree-short acceleration in place. Existing VALID force is
    // rejected; the new field is published VALID only after complete validation.
    void compute_forces(core::ParticleStore& particles);

    // Traversal decisions from the last successfully published force refresh.
    const TreeWalkDiagnostics& last_traversal_diagnostics() const noexcept {
        return last_traversal_diagnostics_;
    }

    // PM-long diagnostics from the last distributed refresh; serial TreePM leaves
    // this empty because its force path does not materialize them.
    const std::optional<PMForceDiagnostics>& last_pm_diagnostics() const noexcept {
        return last_pm_diagnostics_;
    }

    // Source tree from the last successful force refresh, reusable by diagnostics
    // at the unchanged particle state.
    const Octree& last_force_tree() const noexcept {
        return tree_;
    }

private:
    ForceSplitKernel split_;
    std::unique_ptr<PMSolver> pm_solver_;
    // Numerical configuration is immutable after construction. Own a value so
    // the solver does not depend on the caller object's lifetime; run-local
    // snapshot provenance remains shared by SimulationParameters copies.
    config::SimulationParameters config_;
    core::Real theta_;

    Octree tree_;
    std::vector<TreePMChunkWorkspace> chunk_workspaces_;
    TreeWalkDiagnostics last_traversal_diagnostics_{};
    std::optional<PMForceDiagnostics> last_pm_diagnostics_;

    // Diagnostic-only counter, excluded from numerical state.
    std::uint64_t diagnostic_force_refresh_index_{0};
};

} // namespace cosmo_nbody::gravity
