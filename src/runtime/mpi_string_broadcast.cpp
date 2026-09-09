#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"

#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::runtime {

std::string broadcast_string_collective(
    std::string_view value,
    int source_rank,
    int rank,
    int size,
    const char* context) {
    if (context == nullptr || *context == '\0') {
        throw std::invalid_argument(
            "MPI string broadcast context must not be empty");
    }
    if (size < 1 || rank < 0 || rank >= size
        || source_rank < 0 || source_rank >= size) {
        throw std::invalid_argument(
            std::string(context) + " has an invalid MPI topology");
    }
    if (size == 1) return std::string(value);

#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        std::string(context)
        + " requested a multi-rank string broadcast in a non-MPI build");
#else
    require_active_mpi_main_thread(context);

    std::uint64_t length = rank == source_rank
        ? static_cast<std::uint64_t>(value.size())
        : 0ULL;
    const int length_status = MPI_Bcast(
            &length,
            1,
            MPI_UINT64_T,
            source_rank,
            MPI_COMM_WORLD);
    if (length_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, length_status);
        std::abort();
    }
    if (length > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        // Every rank received the same length, so all participants leave before
        // the payload stage without requiring a second failure exchange.
        throw std::overflow_error(
            std::string(context) + " length does not fit size_t");
    }

    std::string result;
    std::exception_ptr local_allocation_exception;
    try {
        if (rank == source_rank) {
            if (!value.empty()) result.assign(value.data(), value.size());
        } else {
            result.resize(static_cast<std::size_t>(length));
        }
    } catch (...) {
        local_allocation_exception = std::current_exception();
    }

    const int local_allocation_failed =
        local_allocation_exception ? 1 : 0;
    int any_allocation_failed = 0;
    const int allocation_status = MPI_Allreduce(
            &local_allocation_failed,
            &any_allocation_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
    if (allocation_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, allocation_status);
        std::abort();
    }
    if (any_allocation_failed != 0) {
        if (local_allocation_exception) {
            std::rethrow_exception(local_allocation_exception);
        }
        throw std::runtime_error(
            std::string(context)
            + " receive-buffer allocation failed on another MPI rank");
    }

    std::size_t offset = 0;
    while (offset < result.size()) {
        const int count = static_cast<int>(std::min<std::size_t>(
            result.size() - offset,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        const int payload_status = MPI_Bcast(
                result.data() + offset,
                count,
                MPI_CHAR,
                source_rank,
                MPI_COMM_WORLD);
        if (payload_status != MPI_SUCCESS) {
            (void)MPI_Abort(MPI_COMM_WORLD, payload_status);
            std::abort();
        }
        offset += static_cast<std::size_t>(count);
    }
    return result;
#endif
}

} // namespace cosmo_nbody::runtime
