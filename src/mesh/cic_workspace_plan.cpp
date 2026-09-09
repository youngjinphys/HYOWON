#include "cosmo_nbody/mesh/mass_assignment.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::mesh {
namespace {

std::uint64_t checked_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string("CIC workspace plan overflow: ") + label);
    }
    return lhs + rhs;
}

std::uint64_t checked_multiply(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* label) {
    if (lhs != 0
        && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(
            std::string("CIC workspace plan overflow: ") + label);
    }
    return lhs * rhs;
}

} // namespace

CICDepositMemoryPlan cic_deposit_memory_plan(
    std::uint64_t particle_count,
    std::uint64_t mesh_size,
    std::uint64_t plane_tasks,
    std::uint64_t maximum_threads) {
    if (mesh_size == 0) {
        throw std::invalid_argument(
            "CIC workspace planning requires a positive mesh size");
    }
    if (plane_tasks == 0 || plane_tasks > mesh_size) {
        throw std::invalid_argument(
            "CIC workspace planning requires plane tasks in [1, mesh size]");
    }
    if (maximum_threads == 0) {
        throw std::invalid_argument(
            "CIC workspace planning requires at least one available thread");
    }

    CICDepositMemoryPlan plan;
    if (particle_count == 0) return plan;

    const std::uint64_t int_limit = static_cast<std::uint64_t>(
        std::numeric_limits<int>::max());
    const std::uint64_t available_threads = std::min(
        maximum_threads, int_limit);

    // Parallel x-binning uses two mesh-sized size_t rows per worker. Admit a
    // worker only when its average particle work covers at least one mesh row.
    const std::uint64_t metadata_amortized_workers = std::max<std::uint64_t>(
        1,
        particle_count / mesh_size);
    const std::uint64_t binning_threads = std::min({
        available_threads,
        particle_count,
        metadata_amortized_workers});

    // Each particle visits exactly two x-planes, so non-empty plane tasks <= 2*N.
    const std::uint64_t maximum_nonempty_plane_tasks =
        particle_count > std::numeric_limits<std::uint64_t>::max() / 2
        ? std::numeric_limits<std::uint64_t>::max()
        : 2 * particle_count;
    const std::uint64_t active_plane_workers = std::min({
        available_threads,
        plane_tasks,
        maximum_nonempty_plane_tasks});

    plan.binning_thread_count = std::max<std::uint64_t>(binning_threads, 1);
    plan.requested_plane_worker_count = std::min(
        available_threads, plane_tasks);
    plan.active_plane_worker_count = std::max<std::uint64_t>(
        active_plane_workers, 1);
    plan.stable_index_bytes = cic_stable_index_bytes(particle_count);
    plan.bin_offset_bytes = checked_multiply(
        checked_add(mesh_size, 1, "bin offset count"),
        sizeof(std::size_t),
        "bin offsets");
    plan.bin_thread_metadata_bytes = checked_multiply(
        checked_multiply(
            checked_multiply(
                plan.binning_thread_count,
                mesh_size,
                "thread-bin count"),
            2,
            "count and offset arrays"),
        sizeof(std::size_t),
        "thread metadata");
    plan.retained_bin_bytes = checked_add(
        checked_add(
            plan.stable_index_bytes,
            plan.bin_offset_bytes,
            "stable indices and bin offsets"),
        plan.bin_thread_metadata_bytes,
        "complete retained CIC bin workspace");

    // Call-local metadata stores one visit count per plane task and one boundary
    // per active worker plus the terminal boundary.
    plan.task_particle_visit_bytes = checked_multiply(
        plane_tasks,
        sizeof(std::uint64_t),
        "plane-task particle visits");
    plan.worker_boundary_bytes = checked_multiply(
        checked_add(
            plan.active_plane_worker_count, 1, "worker boundary count"),
        sizeof(std::size_t),
        "worker boundaries");
    plan.call_local_bytes = checked_add(
        plan.task_particle_visit_bytes,
        plan.worker_boundary_bytes,
        "complete call-local CIC metadata");
    plan.peak_bytes = checked_add(
        plan.retained_bin_bytes,
        plan.call_local_bytes,
        "complete CIC deposit peak");
    return plan;
}

} // namespace cosmo_nbody::mesh
