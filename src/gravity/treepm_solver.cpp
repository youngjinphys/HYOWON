#include "cosmo_nbody/gravity/treepm_solver.hpp"
#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/gravity/softening_kernel.hpp"
#include "cosmo_nbody/mesh/pm_force_method.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody::gravity {
namespace {

using ForceFingerprint = std::array<std::uint64_t, 8>;
constexpr std::size_t fingerprint_lanes = 4;
constexpr std::array<std::uint64_t, fingerprint_lanes> fingerprint_seeds{
    0x243f6a8885a308d3ULL,
    0x13198a2e03707344ULL,
    0xa4093822299f31d0ULL,
    0x082efa98ec4e6c89ULL};

static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<core::Real>::is_iec559);
static_assert(std::numeric_limits<core::Real>::radix == 2);
static_assert(std::numeric_limits<core::Real>::digits == 53);

core::Real configured_cutoff_multiplier(
    const config::SimulationParameters& config) {
    const core::Real split_scale = config.r_s();
    const core::Real cutoff = config.r_cut();
    if (!std::isfinite(split_scale) || split_scale <= 0.0
        || !std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument(
            "TreePM configured split scale and cutoff must be finite and positive");
    }
    const core::Real multiplier = cutoff / split_scale;
    if (!std::isfinite(multiplier) || multiplier <= 0.0) {
        throw std::invalid_argument(
            "TreePM configured cutoff multiplier is invalid");
    }
    return multiplier;
}

int treepm_mpi_size(const config::SimulationParameters& config) {
    if (!config.get_runtime().mpi_enabled) return 1;
#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        "TreePM MPI execution requested in a non-MPI build");
#else
    runtime::require_active_mpi_main_thread("TreePM force execution");
    int size = 1;
    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size < 1) {
        throw std::runtime_error(
            "TreePM failed to query the active MPI communicator size");
    }
    return size;
#endif
}

void synchronize_treepm_stage(
    std::exception_ptr local_exception,
    int mpi_size,
    const char* context) {
    if (mpi_size <= 1) {
        if (local_exception) std::rethrow_exception(local_exception);
        return;
    }
    runtime::synchronize_mpi_exception(
        local_exception, mpi_size, context);
}

std::size_t treepm_worker_capacity(
    const config::SimulationParameters& config) {
    runtime::HostThreadPolicyInput input;
    input.requested_threads = config.get_runtime().num_threads;
#ifdef COSMO_NBODY_HAS_OPENMP
    input.openmp_available = true;
    input.runtime_max_threads = static_cast<std::size_t>(
        std::max(1, omp_get_max_threads()));
    input.available_processors = static_cast<std::size_t>(
        std::max(1, omp_get_num_procs()));
    input.thread_limit = static_cast<std::size_t>(
        std::max(1, omp_get_thread_limit()));
#else
    input.openmp_available = false;
#endif
    const runtime::HostThreadPolicy policy =
        runtime::resolve_host_thread_policy(input);
    if (policy.effective_threads == 0) {
        throw std::logic_error(
            "Resolved TreePM host thread policy has zero effective threads");
    }
    return policy.effective_threads;
}

std::uint64_t checked_diagnostic_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string("TreePM traversal diagnostic overflow: ") + label);
    }
    return lhs + rhs;
}

void accumulate_diagnostics_checked(
    TreeWalkDiagnostics& destination,
    const TreeWalkDiagnostics& source) {
    destination.target_count = checked_diagnostic_add(
        destination.target_count, source.target_count, "target_count");
    destination.nodes_visited = checked_diagnostic_add(
        destination.nodes_visited, source.nodes_visited, "nodes_visited");
    destination.internal_nodes_opened = checked_diagnostic_add(
        destination.internal_nodes_opened,
        source.internal_nodes_opened,
        "internal_nodes_opened");
    destination.multipole_nodes_accepted = checked_diagnostic_add(
        destination.multipole_nodes_accepted,
        source.multipole_nodes_accepted,
        "multipole_nodes_accepted");
    destination.leaf_nodes_visited = checked_diagnostic_add(
        destination.leaf_nodes_visited, source.leaf_nodes_visited,
        "leaf_nodes_visited");
    destination.exact_pairs_evaluated = checked_diagnostic_add(
        destination.exact_pairs_evaluated,
        source.exact_pairs_evaluated,
        "exact_pairs_evaluated");
    destination.pair_cutoff_rejected = checked_diagnostic_add(
        destination.pair_cutoff_rejected,
        source.pair_cutoff_rejected,
        "pair_cutoff_rejected");
    destination.nodes_cutoff_pruned = checked_diagnostic_add(
        destination.nodes_cutoff_pruned,
        source.nodes_cutoff_pruned,
        "nodes_cutoff_pruned");
}

std::pair<std::size_t, std::size_t> contiguous_chunk_bounds(
    std::size_t count,
    std::size_t chunks,
    std::size_t chunk) {
    if (chunks == 0 || chunk >= chunks) {
        throw std::out_of_range("TreePM chunk index is invalid");
    }
    const std::size_t base = count / chunks;
    const std::size_t remainder = count % chunks;
    const std::size_t begin = chunk * base + std::min(chunk, remainder);
    const std::size_t end = begin + base + (chunk < remainder ? 1 : 0);
    return {begin, end};
}

void require_canonical_periodic_positions(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    core::Real box_size) {
    if (pos_x.size() != pos_y.size() || pos_x.size() != pos_z.size()) {
        throw std::logic_error(
            "TreePM position component sizes do not match");
    }

    const std::size_t count = pos_x.size();
    std::size_t first_nonfinite = count;
    std::size_t first_out_of_bounds = count;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for reduction(min:first_nonfinite, first_out_of_bounds) \
        schedule(static) if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t i = 0; i < count; ++i) {
        const bool finite = std::isfinite(pos_x[i])
            && std::isfinite(pos_y[i])
            && std::isfinite(pos_z[i]);
        if (!finite) {
            first_nonfinite = std::min(first_nonfinite, i);
            continue;
        }
        if (pos_x[i] < 0.0 || pos_x[i] >= box_size
            || pos_y[i] < 0.0 || pos_y[i] >= box_size
            || pos_z[i] < 0.0 || pos_z[i] >= box_size) {
            first_out_of_bounds = std::min(first_out_of_bounds, i);
        }
    }
    if (first_nonfinite <= first_out_of_bounds && first_nonfinite < count) {
        throw std::invalid_argument(
            "TreePM particle positions must be finite");
    }
    if (first_out_of_bounds < count) {
        throw std::invalid_argument(
            "TreePM particle positions must lie in the periodic half-open interval [0,L)");
    }
}

std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t real_bits(core::Real value) noexcept {
    return core::portable_bit_cast<std::uint64_t>(value);
}

void accumulate_force_fingerprint(
    ForceFingerprint& fingerprint,
    core::ParticleId id,
    core::Real x,
    core::Real y,
    core::Real z) noexcept {
    const std::uint64_t x_bits = real_bits(x);
    const std::uint64_t y_bits = real_bits(y);
    const std::uint64_t z_bits = real_bits(z);
    for (std::size_t lane = 0; lane < fingerprint_lanes; ++lane) {
        std::uint64_t value = mix64(
            static_cast<std::uint64_t>(id) ^ fingerprint_seeds[lane]);
        value = mix64(value ^ std::rotl(
            x_bits, static_cast<int>(7U + 5U * lane)));
        value = mix64(value ^ std::rotl(
            y_bits, static_cast<int>(13U + 7U * lane)));
        value = mix64(value ^ std::rotl(
            z_bits, static_cast<int>(19U + 9U * lane)));
        fingerprint[lane] += value;
        fingerprint[fingerprint_lanes + lane] ^= value;
    }
}

void combine_force_fingerprint(
    ForceFingerprint& destination,
    const ForceFingerprint& source) noexcept {
    for (std::size_t lane = 0; lane < fingerprint_lanes; ++lane) {
        destination[lane] += source[lane];
        destination[fingerprint_lanes + lane] ^=
            source[fingerprint_lanes + lane];
    }
}

std::string fingerprint_hex(
    const ForceFingerprint& fingerprint,
    std::size_t offset) {
    if (offset != 0 && offset != fingerprint_lanes) {
        throw std::invalid_argument("TreePM fingerprint offset is invalid");
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t lane = 0; lane < fingerprint_lanes; ++lane) {
        out << std::setw(16) << fingerprint[offset + lane];
    }
    return out.str();
}

std::uint64_t checked_size_to_u64(
    std::size_t value,
    const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                std::string("TreePM diagnostic ") + label
                + " exceeds uint64_t");
        }
    }
    return static_cast<std::uint64_t>(value);
}

std::uint64_t checked_byte_product(
    std::size_t count,
    std::size_t element_size,
    const char* label) {
    const std::uint64_t count_u64 = checked_size_to_u64(count, label);
    const std::uint64_t element_u64 = checked_size_to_u64(element_size, label);
    if (count_u64 != 0
        && element_u64 > std::numeric_limits<std::uint64_t>::max() / count_u64) {
        throw std::overflow_error(
            std::string("TreePM diagnostic ") + label
            + " byte count overflows uint64_t");
    }
    return count_u64 * element_u64;
}

std::uint64_t checked_byte_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string("TreePM diagnostic ") + label
            + " byte count overflows uint64_t");
    }
    return lhs + rhs;
}

std::array<std::uint64_t, 8> traversal_words(
    const TreeWalkDiagnostics& diagnostics) noexcept {
    return {
        diagnostics.target_count,
        diagnostics.nodes_visited,
        diagnostics.internal_nodes_opened,
        diagnostics.multipole_nodes_accepted,
        diagnostics.leaf_nodes_visited,
        diagnostics.exact_pairs_evaluated,
        diagnostics.pair_cutoff_rejected,
        diagnostics.nodes_cutoff_pruned};
}

TreeWalkDiagnostics diagnostics_from_words(
    const std::array<std::uint64_t, 8>& words) noexcept {
    TreeWalkDiagnostics diagnostics;
    diagnostics.target_count = words[0];
    diagnostics.nodes_visited = words[1];
    diagnostics.internal_nodes_opened = words[2];
    diagnostics.multipole_nodes_accepted = words[3];
    diagnostics.leaf_nodes_visited = words[4];
    diagnostics.exact_pairs_evaluated = words[5];
    diagnostics.pair_cutoff_rejected = words[6];
    diagnostics.nodes_cutoff_pruned = words[7];
    return diagnostics;
}

void emit_runtime_diagnostic(
    const config::SimulationParameters& config,
    const Octree& tree,
    const std::vector<TreePMChunkWorkspace>& chunk_workspaces,
    const TreeWalkDiagnostics& local_traversal,
    const ForceFingerprint& local_short_fingerprint,
    const ForceFingerprint& local_combined_fingerprint,
    std::size_t owned,
    std::size_t source_count,
    int mpi_size,
    std::uint64_t refresh_index) {
    constexpr std::size_t metric_count = 11;
    std::array<std::uint64_t, metric_count> local_metrics{};
    std::exception_ptr preparation_exception;
    try {
        if (source_count < owned) {
            throw std::logic_error(
                "TreePM diagnostic source count is smaller than owned count");
        }
        const auto& nodes = tree.get_nodes();
        const auto& indices = tree.get_particle_indices();
        const std::uint64_t node_bytes = checked_byte_product(
            nodes.capacity(), sizeof(OctreeNode), "node capacity");
        const std::uint64_t index_bytes = checked_byte_product(
            indices.capacity(), sizeof(OctreeIndex), "particle-index capacity");
        const std::uint64_t tree_reserved_bytes = checked_byte_add(
            node_bytes, index_bytes, "tree reserved storage");

        std::uint64_t scratch_reserved_bytes = checked_byte_product(
            chunk_workspaces.capacity(),
            sizeof(TreePMChunkWorkspace),
            "chunk workspace capacity");
        for (const auto& workspace : chunk_workspaces) {
            scratch_reserved_bytes = checked_byte_add(
                scratch_reserved_bytes,
                checked_byte_product(
                    workspace.traversal.capacity(),
                    sizeof(TreeWalkStackEntry),
                    "tree traversal stack capacity"),
                "tree traversal scratch storage");
        }

        local_metrics = {
            checked_size_to_u64(owned, "owned particles"),
            checked_size_to_u64(source_count - owned, "ghost particles"),
            checked_size_to_u64(source_count, "source particles"),
            checked_size_to_u64(nodes.size(), "node count"),
            checked_size_to_u64(nodes.capacity(), "node capacity"),
            checked_size_to_u64(indices.size(), "particle-index count"),
            checked_size_to_u64(indices.capacity(), "particle-index capacity"),
            checked_size_to_u64(
                tree.maximum_subdivision_depth(),
                "maximum subdivision depth"),
            checked_size_to_u64(
                tree.traversal_stack_bound(),
                "traversal stack bound"),
            tree_reserved_bytes,
            scratch_reserved_bytes};
    } catch (...) {
        preparation_exception = std::current_exception();
    }
    synchronize_treepm_stage(
        preparation_exception,
        mpi_size,
        "TreePM runtime diagnostic preparation");

    std::array<std::uint64_t, metric_count> minima = local_metrics;
    std::array<std::uint64_t, metric_count> maxima = local_metrics;
    std::array<std::uint64_t, 8> global_traversal_words =
        traversal_words(local_traversal);
    ForceFingerprint global_short_fingerprint = local_short_fingerprint;
    ForceFingerprint global_combined_fingerprint = local_combined_fingerprint;
    int rank = 0;

    if (mpi_size > 1) {
#ifndef COSMO_NBODY_HAS_MPI
        throw std::logic_error(
            "Multi-rank TreePM diagnostics requested in a non-MPI build");
#else
        runtime::require_active_mpi_main_thread("TreePM runtime diagnostics");
        if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS
            || rank < 0 || rank >= mpi_size) {
            throw std::runtime_error(
                "TreePM runtime diagnostics failed to query MPI rank");
        }
        const auto local_minima = minima;
        const auto local_maxima = maxima;
        const int minima_status = MPI_Allreduce(
            local_minima.data(), minima.data(),
            static_cast<int>(metric_count), MPI_UINT64_T,
            MPI_MIN, MPI_COMM_WORLD);
        const int maxima_status = MPI_Allreduce(
            local_maxima.data(), maxima.data(),
            static_cast<int>(metric_count), MPI_UINT64_T,
            MPI_MAX, MPI_COMM_WORLD);
        if (minima_status != MPI_SUCCESS || maxima_status != MPI_SUCCESS) {
            throw std::runtime_error(
                "TreePM runtime diagnostic metric reduction failed");
        }

        std::vector<std::uint64_t> gathered_traversal;
        std::vector<std::uint64_t> gathered_fingerprints;
        std::exception_ptr allocation_exception;
        try {
            const std::size_t ranks = static_cast<std::size_t>(mpi_size);
            if (ranks > std::numeric_limits<std::size_t>::max() / 16U) {
                throw std::overflow_error(
                    "TreePM runtime diagnostic gather size overflows size_t");
            }
            gathered_traversal.resize(ranks * 8U);
            gathered_fingerprints.resize(ranks * 16U);
        } catch (...) {
            allocation_exception = std::current_exception();
        }
        synchronize_treepm_stage(
            allocation_exception,
            mpi_size,
            "TreePM runtime diagnostic gather allocation");

        const auto local_traversal_words = traversal_words(local_traversal);
        std::array<std::uint64_t, 16> local_fingerprint_words{};
        std::copy(
            local_short_fingerprint.begin(),
            local_short_fingerprint.end(),
            local_fingerprint_words.begin());
        std::copy(
            local_combined_fingerprint.begin(),
            local_combined_fingerprint.end(),
            local_fingerprint_words.begin() + 8);

        const int traversal_status = MPI_Allgather(
            local_traversal_words.data(),
            8,
            MPI_UINT64_T,
            gathered_traversal.data(),
            8,
            MPI_UINT64_T,
            MPI_COMM_WORLD);
        const int fingerprint_status = MPI_Allgather(
            local_fingerprint_words.data(),
            16,
            MPI_UINT64_T,
            gathered_fingerprints.data(),
            16,
            MPI_UINT64_T,
            MPI_COMM_WORLD);
        if (traversal_status != MPI_SUCCESS
            || fingerprint_status != MPI_SUCCESS) {
            throw std::runtime_error(
                "TreePM runtime diagnostic rank gather failed");
        }

        global_traversal_words.fill(0);
        global_short_fingerprint.fill(0);
        global_combined_fingerprint.fill(0);
        for (int source_rank = 0; source_rank < mpi_size; ++source_rank) {
            const std::size_t traversal_offset =
                static_cast<std::size_t>(source_rank) * 8U;
            for (std::size_t field = 0; field < 8U; ++field) {
                global_traversal_words[field] = checked_diagnostic_add(
                    global_traversal_words[field],
                    gathered_traversal[traversal_offset + field],
                    "collective traversal count");
            }

            const std::size_t fingerprint_offset =
                static_cast<std::size_t>(source_rank) * 16U;
            ForceFingerprint rank_short{};
            ForceFingerprint rank_combined{};
            std::copy_n(
                gathered_fingerprints.begin()
                    + static_cast<std::ptrdiff_t>(fingerprint_offset),
                8,
                rank_short.begin());
            std::copy_n(
                gathered_fingerprints.begin()
                    + static_cast<std::ptrdiff_t>(fingerprint_offset + 8U),
                8,
                rank_combined.begin());
            combine_force_fingerprint(
                global_short_fingerprint, rank_short);
            combine_force_fingerprint(
                global_combined_fingerprint, rank_combined);
        }
#endif
    }

    if (rank != 0) return;
    const TreeWalkDiagnostics global_traversal =
        diagnostics_from_words(global_traversal_words);

    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<core::Real>::max_digits10);
    out << "TREEPM_RUNTIME_DIAGNOSTIC {"
        << "\"schema\":\"hyowon.treepm_runtime_diagnostic\""
        << ",\"measurement_only\":true"
        << ",\"force_refresh_index\":" << refresh_index
        << ",\"theta\":" << config.get_gravity().theta
        << ",\"rank_count\":" << mpi_size
        << ",\"owned_particles_min\":" << minima[0]
        << ",\"owned_particles_max\":" << maxima[0]
        << ",\"ghost_particles_min\":" << minima[1]
        << ",\"ghost_particles_max\":" << maxima[1]
        << ",\"source_particles_min\":" << minima[2]
        << ",\"source_particles_max\":" << maxima[2]
        << ",\"tree_nodes_min\":" << minima[3]
        << ",\"tree_nodes_max\":" << maxima[3]
        << ",\"tree_node_capacity_min\":" << minima[4]
        << ",\"tree_node_capacity_max\":" << maxima[4]
        << ",\"tree_particle_indices_min\":" << minima[5]
        << ",\"tree_particle_indices_max\":" << maxima[5]
        << ",\"tree_particle_index_capacity_min\":" << minima[6]
        << ",\"tree_particle_index_capacity_max\":" << maxima[6]
        << ",\"tree_max_subdivision_depth_min\":" << minima[7]
        << ",\"tree_max_subdivision_depth_max\":" << maxima[7]
        << ",\"tree_traversal_stack_bound_min\":" << minima[8]
        << ",\"tree_traversal_stack_bound_max\":" << maxima[8]
        << ",\"tree_reserved_bytes_min\":" << minima[9]
        << ",\"tree_reserved_bytes_max\":" << maxima[9]
        << ",\"tree_scratch_reserved_bytes_min\":" << minima[10]
        << ",\"tree_scratch_reserved_bytes_max\":" << maxima[10]
        << ",\"octree_node_size_bytes\":" << sizeof(OctreeNode)
        << ",\"octree_index_size_bytes\":" << sizeof(OctreeIndex)
        << ",\"traversal_target_count\":" << global_traversal.target_count
        << ",\"traversal_nodes_visited\":" << global_traversal.nodes_visited
        << ",\"traversal_internal_nodes_opened\":"
        << global_traversal.internal_nodes_opened
        << ",\"traversal_multipole_nodes_accepted\":"
        << global_traversal.multipole_nodes_accepted
        << ",\"traversal_leaf_nodes_visited\":"
        << global_traversal.leaf_nodes_visited
        << ",\"traversal_exact_pairs_evaluated\":"
        << global_traversal.exact_pairs_evaluated
        << ",\"traversal_pair_cutoff_rejected\":"
        << global_traversal.pair_cutoff_rejected
        << ",\"traversal_nodes_cutoff_pruned\":"
        << global_traversal.nodes_cutoff_pruned
        << ",\"short_range_pairwise_exact\":"
        << (global_traversal.multipole_nodes_accepted == 0 ? "true" : "false")
        << ",\"short_force_fingerprint_sum256\":\""
        << fingerprint_hex(global_short_fingerprint, 0) << "\""
        << ",\"short_force_fingerprint_xor256\":\""
        << fingerprint_hex(global_short_fingerprint, fingerprint_lanes) << "\""
        << ",\"combined_force_fingerprint_sum256\":\""
        << fingerprint_hex(global_combined_fingerprint, 0) << "\""
        << ",\"combined_force_fingerprint_xor256\":\""
        << fingerprint_hex(global_combined_fingerprint, fingerprint_lanes) << "\""
        << ",\"fingerprint_scope\":\"owned ParticleID plus exact binary64 acceleration bits; commutative across particles and MPI ranks\""
        << ",\"fingerprint_reduction_rank_partition_invariant\":true"
        << ",\"fingerprint_match_is_proof\":false"
        << "}\n";
    std::cout << out.str();
    std::cout.flush();
}

} // namespace

TreePMSolver::TreePMSolver(
    const config::SimulationParameters& config,
    const mesh::MeshGeometry& geom,
    domain::DomainBounds local_bounds)
    : split_(config.r_s(), configured_cutoff_multiplier(config)),
      config_(config),
      theta_(config.get_gravity().theta) {
    (void)local_bounds;

    if (!std::isfinite(theta_) || theta_ < 0.0) {
        throw std::invalid_argument(
            "TreePMSolver theta must be finite and non-negative");
    }
    const core::Real box_size = config.get_box().L;
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "TreePMSolver box size must be finite and positive");
    }
    if (!(split_.cutoff_radius() < 0.5 * box_size)) {
        throw std::invalid_argument(
            "TreePM requires the configured r_cut < L/2 so the minimum-image short-range neighborhood is unambiguous");
    }

    (void)SofteningKernel(config.eps()).force_factor(0.0);

    pm_solver_ = std::make_unique<PMSolver>(
        geom,
        mesh::PMForceMethod::treepm_long_range(split_.split_scale()),
        config.get_runtime().mpi_enabled,
        config.get_memory_policy());
}

void TreePMSolver::compute_forces(core::ParticleStore& particles) {
    last_pm_diagnostics_.reset();
    const int mpi_size = treepm_mpi_size(config_);
    const bool multi_rank = mpi_size > 1;
    const bool collect_runtime_diagnostics =
        config_.get_validation().write_diagnostics;
    const std::size_t owned = particles.num_owned_particles();
    const std::size_t source_count = particles.size();
    const auto pos_x = particles.get_positions_x().first(owned);
    const auto pos_y = particles.get_positions_y().first(owned);
    const auto pos_z = particles.get_positions_z().first(owned);
    const auto ids = particles.get_ids().first(owned);
    const auto masses = particles.get_uniform_mass().has_value()
        ? std::span<const core::Real>{}
        : particles.get_masses().first(owned);

    const core::Real L = config_.get_box().L;
    std::span<core::Real> ax;
    std::span<core::Real> ay;
    std::span<core::Real> az;
    std::exception_ptr setup_exception;
    try {
        if (!pm_solver_) {
            throw std::logic_error(
                "TreePMSolver has no configured long-range solver");
        }
        if (source_count < owned) {
            throw std::logic_error(
                "TreePM local source population is smaller than the owned target population");
        }
        if (multi_rank
            && particles.get_ghost_validity() != core::FieldValidity::VALID) {
            throw std::logic_error(
                "Distributed TreePM requires a current r_cut ghost exchange before force evaluation");
        }
        require_canonical_periodic_positions(
            particles.get_positions_x().first(source_count),
            particles.get_positions_y().first(source_count),
            particles.get_positions_z().first(source_count),
            L);
        if (particles.get_acceleration_validity() == core::FieldValidity::VALID) {
            throw std::logic_error(
                "TreePM direct force assembly requires the previous acceleration to be retired before force entry");
        }

        // The canonical leapfrog transaction retires the previous acceleration
        // before force entry. Reassert INVALID so a later PM/tree failure cannot
        // advertise a partially overwritten field as usable. Mutable access
        // materializes exactly the three owned components when migration released
        // them; no second persistent 3*N force buffer is needed.
        particles.set_acceleration_validity(core::FieldValidity::INVALID);
        ax = particles.mutable_accelerations_x().first(owned);
        ay = particles.mutable_accelerations_y().first(owned);
        az = particles.mutable_accelerations_z().first(owned);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) \
            if(runtime::should_use_host_parallel_team(owned))
#endif
        for (std::size_t i = 0; i < owned; ++i) {
            ax[i] = 0.0;
            ay[i] = 0.0;
            az[i] = 0.0;
        }
        pm_solver_->ensure_workspace(owned);
    } catch (...) {
        setup_exception = std::current_exception();
    }
    synchronize_treepm_stage(
        setup_exception, mpi_size, "TreePM local force setup");

    // Only the slab-resident distributed backend can expose PM potential/self
    // energy from the force path. Serial/replicated TreePM keeps diagnostics on
    // the existing conservation path while all force dispatch still goes
    // through the canonical PMSolver facade.
    const bool collect_pm_potential =
        collect_runtime_diagnostics && pm_solver_->using_distributed_backend();
    std::optional<PMForceDiagnostics> pm_diagnostics =
        pm_solver_->compute_forces_in_place(
            pos_x,
            pos_y,
            pos_z,
            masses,
            particles.get_uniform_mass(),
            ax,
            ay,
            az,
            collect_pm_potential);

    TreeWalkDiagnostics completed_traversal;
    ForceFingerprint local_short_fingerprint{};
    ForceFingerprint local_combined_fingerprint{};
    std::exception_ptr short_range_exception;
    try {
        const domain::DomainBounds tree_bounds{
            0.0, L,
            0.0, L,
            0.0, L};
        tree_.build(particles, tree_bounds);

        const auto& target_order = tree_.get_particle_indices();
        if (target_order.size() != source_count) {
            throw std::logic_error(
                "TreePM tree permutation does not match the local source population");
        }
        const bool has_ghost_sources = source_count != owned;
        TreeWalk tree_walk(tree_, particles, config_);

        const std::size_t chunks = treepm_parallel_chunk_count(
            owned, treepm_worker_capacity(config_));
        if (chunk_workspaces_.size() < chunks) {
            chunk_workspaces_.resize(chunks);
        }
        const std::size_t stack_bound = tree_.traversal_stack_bound();
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            chunk_workspaces_[chunk].traversal.reserve(stack_bound);
            chunk_workspaces_[chunk].diagnostics = {};
        }

        std::vector<ForceFingerprint> short_chunk_fingerprints;
        std::vector<ForceFingerprint> combined_chunk_fingerprints;
        if (collect_runtime_diagnostics) {
            short_chunk_fingerprints.resize(chunks);
            combined_chunk_fingerprints.resize(chunks);
        }

        std::atomic<bool> worker_failed{false};
        std::exception_ptr worker_exception;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            if (worker_failed.load(std::memory_order_relaxed)) continue;
            try {
                TreePMChunkWorkspace& workspace = chunk_workspaces_[chunk];
                const auto [begin, end] = contiguous_chunk_bounds(
                    owned, chunks, chunk);
                TreeWalkDiagnostics local_diagnostics;
                for (std::size_t order_position = begin;
                     order_position < end;
                     ++order_position) {
                    if (worker_failed.load(std::memory_order_relaxed)) break;
                    const std::size_t particle_index = has_ghost_sources
                        ? order_position
                        : static_cast<std::size_t>(
                              target_order[order_position]);
                    if (particle_index >= owned) {
                        throw std::out_of_range(
                            "TreePM target order exceeds the owned particle range");
                    }
                    const TreeWalkResult short_result =
                        tree_walk.compute_force_with_diagnostics(
                            particle_index,
                            theta_,
                            workspace.traversal);
                    accumulate_diagnostics_checked(
                        local_diagnostics, short_result.diagnostics);
                    const core::Vec3& short_correction = short_result.force;
                    const core::Real updated_x =
                        ax[particle_index] + short_correction.x;
                    const core::Real updated_y =
                        ay[particle_index] + short_correction.y;
                    const core::Real updated_z =
                        az[particle_index] + short_correction.z;
                    if (!std::isfinite(updated_x) || !std::isfinite(updated_y)
                        || !std::isfinite(updated_z)) {
                        throw std::overflow_error(
                            "TreePMSolver combined force is non-finite");
                    }
                    if (collect_runtime_diagnostics) {
                        accumulate_force_fingerprint(
                            short_chunk_fingerprints[chunk],
                            ids[particle_index],
                            short_correction.x,
                            short_correction.y,
                            short_correction.z);
                        accumulate_force_fingerprint(
                            combined_chunk_fingerprints[chunk],
                            ids[particle_index],
                            updated_x,
                            updated_y,
                            updated_z);
                    }
                    ax[particle_index] = updated_x;
                    ay[particle_index] = updated_y;
                    az[particle_index] = updated_z;
                }
                workspace.diagnostics = local_diagnostics;
            } catch (...) {
                worker_failed.store(true, std::memory_order_relaxed);
#ifdef COSMO_NBODY_HAS_OPENMP
                #pragma omp critical(cosmo_nbody_treepm_worker_failure)
#endif
                {
                    if (!worker_exception) {
                        worker_exception = std::current_exception();
                    }
                }
            }
        }
        if (worker_exception) std::rethrow_exception(worker_exception);

        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            accumulate_diagnostics_checked(
                completed_traversal,
                chunk_workspaces_[chunk].diagnostics);
            if (collect_runtime_diagnostics) {
                combine_force_fingerprint(
                    local_short_fingerprint,
                    short_chunk_fingerprints[chunk]);
                combine_force_fingerprint(
                    local_combined_fingerprint,
                    combined_chunk_fingerprints[chunk]);
            }
        }

        int invalid_force = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for reduction(|:invalid_force) schedule(static) \
            if(runtime::should_use_host_parallel_team(owned))
#endif
        for (std::size_t i = 0; i < owned; ++i) {
            invalid_force |= !std::isfinite(ax[i])
                || !std::isfinite(ay[i])
                || !std::isfinite(az[i]);
        }
        if (invalid_force != 0) {
            throw std::overflow_error(
                "TreePMSolver complete force is non-finite");
        }
    } catch (...) {
        short_range_exception = std::current_exception();
    }
    synchronize_treepm_stage(
        short_range_exception,
        mpi_size,
        "TreePM short-range force completion");

    if (collect_runtime_diagnostics) {
        if (diagnostic_force_refresh_index_
            == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "TreePM diagnostic force-refresh index exceeds uint64_t");
        }
        emit_runtime_diagnostic(
            config_,
            tree_,
            chunk_workspaces_,
            completed_traversal,
            local_short_fingerprint,
            local_combined_fingerprint,
            owned,
            source_count,
            mpi_size,
            diagnostic_force_refresh_index_);
        ++diagnostic_force_refresh_index_;
    }

    particles.set_acceleration_validity(core::FieldValidity::VALID);
    last_traversal_diagnostics_ = completed_traversal;
    last_pm_diagnostics_ = std::move(pm_diagnostics);
}

} // namespace cosmo_nbody::gravity
