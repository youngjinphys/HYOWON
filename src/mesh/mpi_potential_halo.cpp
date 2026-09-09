#include "cosmo_nbody/mesh/mpi_potential_halo.hpp"

#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::mesh {
namespace {

#ifdef COSMO_NBODY_HAS_MPI
std::size_t checked_mul_size(
    std::size_t lhs,
    std::size_t rhs,
    const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return lhs * rhs;
}

int checked_int(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(label) + " exceeds MPI int range");
    }
    return static_cast<int>(value);
}

std::vector<int> displacements(const std::vector<int>& counts) {
    std::vector<int> result(counts.size(), 0);
    long long total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] < 0
            || total > std::numeric_limits<int>::max()
            || total + counts[i] > std::numeric_limits<int>::max()) {
            throw std::overflow_error("MPI potential-halo displacement overflow");
        }
        result[i] = static_cast<int>(total);
        total += counts[i];
    }
    return result;
}

std::size_t total_count(
    const std::vector<int>& counts,
    const std::vector<int>& offsets) {
    if (counts.empty()) return 0;
    const std::size_t last = counts.size() - 1;
    if (counts[last] < 0 || offsets[last] < 0) {
        throw std::logic_error("Negative MPI potential-halo count");
    }
    return static_cast<std::size_t>(offsets[last])
        + static_cast<std::size_t>(counts[last]);
}

template <typename T>
T* nullable_data(std::vector<T>& values) noexcept {
    return values.empty() ? nullptr : values.data();
}

template <typename T>
const T* nullable_data(const std::vector<T>& values) noexcept {
    return values.empty() ? nullptr : values.data();
}
#endif

} // namespace

std::vector<PotentialHaloPlane> potential_halo_plan(
    const MpiFFTCommunicatorLayout& communicator,
    int rank) {
    const auto& local = communicator.rank_layout(rank);
    const std::size_t n = communicator.grid_size;
    if (n == 0 || local.local_n0 == 0) {
        throw std::invalid_argument(
            "Potential halo requires a non-empty periodic slab layout");
    }

    const std::size_t left =
        (local.local_0_start + n - 1) % n;
    const std::size_t right =
        (local.local_0_start + local.local_n0) % n;
    const std::size_t right_second = (right + 1) % n;
    const std::size_t requested[3] = {left, right, right_second};

    std::vector<PotentialHaloPlane> result;
    result.reserve(3);
    for (const std::size_t global_plane : requested) {
        const auto duplicate = std::find_if(
            result.begin(),
            result.end(),
            [global_plane](const PotentialHaloPlane& plane) {
                return plane.global_plane == global_plane;
            });
        if (duplicate != result.end()) continue;

        const int owner = communicator.owner_rank(global_plane);
        const auto& owner_layout = communicator.rank_layout(owner);
        result.push_back({
            global_plane,
            owner,
            global_plane - owner_layout.local_0_start});
    }
    return result;
}

std::vector<core::Real> exchange_potential_halo_planes(
    const MpiFFTCommunicatorLayout& communicator,
    int rank,
    std::span<const core::Real> local_field) {
#ifndef COSMO_NBODY_HAS_MPI
    (void)communicator;
    (void)rank;
    (void)local_field;
    throw std::runtime_error(
        "Potential halo exchange requires an MPI-enabled build");
#else
    static_assert(std::is_same_v<core::Real, double>,
                  "MPI potential halo requires double-precision core::Real");

    runtime::require_active_mpi_main_thread("Potential halo exchange");

    int world_rank = -1;
    int world_size = 0;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS
        || MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to query MPI topology for potential halo exchange");
    }

    const MpiFFTAllocationLayout* local_layout = nullptr;
    std::size_t plane_cells = 0;
    std::vector<PotentialHaloPlane> plan;
    std::vector<core::Real> result;
    std::vector<std::vector<std::size_t>> request_positions;
    std::vector<int> send_request_counts;
    std::exception_ptr stage_exception;
    try {
        if (rank != world_rank
            || communicator.ranks.size()
                != static_cast<std::size_t>(world_size)) {
            throw std::invalid_argument(
                "Potential halo communicator differs from MPI_COMM_WORLD");
        }
        local_layout = &communicator.rank_layout(rank);
        plane_cells = checked_mul_size(
            communicator.grid_size,
            communicator.grid_size,
            "potential halo plane cells");
        const std::size_t expected_local = checked_mul_size(
            local_layout->local_n0,
            plane_cells,
            "potential halo local field extent");
        if (local_field.size() != expected_local) {
            throw std::invalid_argument(
                "Potential halo local field size differs from its slab layout");
        }

        plan = potential_halo_plan(communicator, rank);
        result.resize(checked_mul_size(
            plan.size(), plane_cells, "potential halo result extent"));
        request_positions.resize(static_cast<std::size_t>(world_size));
        for (std::size_t plan_index = 0;
             plan_index < plan.size();
             ++plan_index) {
            const auto& request = plan[plan_index];
            if (request.owner_rank == rank) {
                const std::size_t source = checked_mul_size(
                    request.owner_local_plane,
                    plane_cells,
                    "local potential halo plane offset");
                std::copy_n(
                    local_field.data() + source,
                    plane_cells,
                    result.data() + plan_index * plane_cells);
            } else {
                request_positions[
                    static_cast<std::size_t>(request.owner_rank)]
                    .push_back(plan_index);
            }
        }

        send_request_counts.assign(
            static_cast<std::size_t>(world_size), 0);
        for (int owner = 0; owner < world_size; ++owner) {
            send_request_counts[static_cast<std::size_t>(owner)] =
                checked_int(
                    request_positions[static_cast<std::size_t>(owner)].size(),
                    "potential halo request count");
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception,
        world_size,
        "Potential halo request preparation");

    std::vector<int> recv_request_counts(
        static_cast<std::size_t>(world_size), 0);
    if (MPI_Alltoall(
            send_request_counts.data(),
            1,
            MPI_INT,
            recv_request_counts.data(),
            1,
            MPI_INT,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Alltoall failed for potential halo request counts");
    }

    std::vector<int> send_request_offsets;
    std::vector<int> recv_request_offsets;
    std::vector<std::uint64_t> send_requests;
    std::vector<std::uint64_t> recv_requests;
    stage_exception = nullptr;
    try {
        send_request_offsets = displacements(send_request_counts);
        recv_request_offsets = displacements(recv_request_counts);
        send_requests.resize(
            total_count(send_request_counts, send_request_offsets));
        for (int owner = 0; owner < world_size; ++owner) {
            const auto& positions =
                request_positions[static_cast<std::size_t>(owner)];
            const std::size_t offset = static_cast<std::size_t>(
                send_request_offsets[static_cast<std::size_t>(owner)]);
            for (std::size_t i = 0; i < positions.size(); ++i) {
                send_requests[offset + i] = static_cast<std::uint64_t>(
                    plan[positions[i]].global_plane);
            }
        }
        recv_requests.resize(
            total_count(recv_request_counts, recv_request_offsets));
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception,
        world_size,
        "Potential halo request-buffer preparation");

    if (MPI_Alltoallv(
            nullable_data(send_requests),
            send_request_counts.data(),
            send_request_offsets.data(),
            MPI_UINT64_T,
            nullable_data(recv_requests),
            recv_request_counts.data(),
            recv_request_offsets.data(),
            MPI_UINT64_T,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Alltoallv failed for potential halo plane requests");
    }

    std::vector<int> send_value_counts;
    std::vector<int> recv_value_counts;
    std::vector<int> send_value_offsets;
    std::vector<int> recv_value_offsets;
    std::vector<core::Real> send_values;
    std::vector<core::Real> recv_values;
    stage_exception = nullptr;
    try {
        send_value_counts.assign(
            static_cast<std::size_t>(world_size), 0);
        recv_value_counts.assign(
            static_cast<std::size_t>(world_size), 0);
        for (int peer = 0; peer < world_size; ++peer) {
            send_value_counts[static_cast<std::size_t>(peer)] = checked_int(
                checked_mul_size(
                    static_cast<std::size_t>(
                        recv_request_counts[static_cast<std::size_t>(peer)]),
                    plane_cells,
                    "potential halo response cells"),
                "potential halo response count");
            recv_value_counts[static_cast<std::size_t>(peer)] = checked_int(
                checked_mul_size(
                    static_cast<std::size_t>(
                        send_request_counts[static_cast<std::size_t>(peer)]),
                    plane_cells,
                    "potential halo receive cells"),
                "potential halo receive count");
        }
        send_value_offsets = displacements(send_value_counts);
        recv_value_offsets = displacements(recv_value_counts);
        send_values.resize(total_count(
            send_value_counts, send_value_offsets));
        recv_values.resize(total_count(
            recv_value_counts, recv_value_offsets));

        for (int destination = 0;
             destination < world_size;
             ++destination) {
            const std::size_t request_offset = static_cast<std::size_t>(
                recv_request_offsets[
                    static_cast<std::size_t>(destination)]);
            const std::size_t value_offset = static_cast<std::size_t>(
                send_value_offsets[
                    static_cast<std::size_t>(destination)]);
            const int count = recv_request_counts[
                static_cast<std::size_t>(destination)];
            for (int request_index = 0;
                 request_index < count;
                 ++request_index) {
                const std::uint64_t raw_global = recv_requests[
                    request_offset
                    + static_cast<std::size_t>(request_index)];
                if (raw_global
                    >= static_cast<std::uint64_t>(communicator.grid_size)) {
                    throw std::out_of_range(
                        "Potential halo global plane is outside the grid");
                }
                const std::size_t global_plane =
                    static_cast<std::size_t>(raw_global);
                if (communicator.owner_rank(global_plane) != rank) {
                    throw std::runtime_error(
                        "Potential halo request was sent to the wrong owner");
                }
                const std::size_t local_plane =
                    global_plane - local_layout->local_0_start;
                const std::size_t source = checked_mul_size(
                    local_plane,
                    plane_cells,
                    "potential halo response source");
                std::copy_n(
                    local_field.data() + source,
                    plane_cells,
                    send_values.data()
                        + value_offset
                        + static_cast<std::size_t>(request_index)
                            * plane_cells);
            }
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception,
        world_size,
        "Potential halo response preparation");

    if (MPI_Alltoallv(
            nullable_data(send_values),
            send_value_counts.data(),
            send_value_offsets.data(),
            MPI_DOUBLE,
            nullable_data(recv_values),
            recv_value_counts.data(),
            recv_value_offsets.data(),
            MPI_DOUBLE,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Alltoallv failed for potential halo plane values");
    }

    stage_exception = nullptr;
    try {
        for (int owner = 0; owner < world_size; ++owner) {
            const auto& positions =
                request_positions[static_cast<std::size_t>(owner)];
            const std::size_t value_offset = static_cast<std::size_t>(
                recv_value_offsets[static_cast<std::size_t>(owner)]);
            for (std::size_t i = 0; i < positions.size(); ++i) {
                std::copy_n(
                    recv_values.data() + value_offset + i * plane_cells,
                    plane_cells,
                    result.data() + positions[i] * plane_cells);
            }
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception,
        world_size,
        "Potential halo response unpack");
    return result;
#endif
}

std::span<const core::Real> potential_halo_plane(
    std::span<const core::Real> exchanged,
    std::size_t plane_cells,
    std::size_t plan_index) {
    if (plane_cells == 0
        || plan_index > std::numeric_limits<std::size_t>::max() / plane_cells) {
        throw std::invalid_argument("Potential halo plane extent is invalid");
    }
    const std::size_t offset = plan_index * plane_cells;
    if (offset > exchanged.size()
        || plane_cells > exchanged.size() - offset) {
        throw std::out_of_range(
            "Potential halo plane index exceeds exchanged storage");
    }
    return exchanged.subspan(offset, plane_cells);
}

} // namespace cosmo_nbody::mesh
