#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::runtime {
namespace {

void require_context(const char* context) {
    if (context == nullptr || *context == '\0') {
        throw std::invalid_argument(
            "MPI collective stage context must not be empty");
    }
}

#ifdef COSMO_NBODY_HAS_MPI
void require_live_mpi(const char* context) {
    int initialized = 0;
    const int initialized_status = MPI_Initialized(&initialized);
    if (initialized_status != MPI_SUCCESS) {
        std::abort();
    }
    if (!initialized) {
        throw std::runtime_error(
            std::string(context) + " requires initialized MPI");
    }
    int finalized = 0;
    const int finalized_status = MPI_Finalized(&finalized);
    if (finalized_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, finalized_status);
        std::abort();
    }
    if (finalized) {
        throw std::runtime_error(
            std::string(context) + " is unavailable after MPI_Finalize");
    }
    int is_main_thread = 0;
    const int main_thread_status = MPI_Is_thread_main(&is_main_thread);
    if (main_thread_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, main_thread_status);
        std::abort();
    }
    if (!is_main_thread) {
        throw std::runtime_error(
            std::string(context) + " requires the MPI main thread");
    }
}
#endif

} // namespace

void require_active_mpi_main_thread(const char* context) {
    require_context(context);
#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        std::string(context) + " requires an MPI-enabled build");
#else
    require_live_mpi(context);
#endif
}

void synchronize_mpi_failure(
    int local_failed,
    int size,
    const char* context) {
    require_context(context);
    if (size < 1) {
        throw std::invalid_argument(
            std::string(context) + " has an invalid MPI size");
    }
    if (size == 1) {
        if (local_failed != 0) throw std::runtime_error(context);
        return;
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_failed;
    throw std::runtime_error(
        std::string(context)
        + " requested multi-rank failure synchronization in a non-MPI build");
#else
    require_live_mpi(context);
    int any_failed = 0;
    const int status = MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
    if (any_failed != 0) throw std::runtime_error(context);
#endif
}

void synchronize_mpi_exception(
    std::exception_ptr local_exception,
    int size,
    const char* context) {
    require_context(context);
    if (size < 1) {
        throw std::invalid_argument(
            std::string(context) + " has an invalid MPI size");
    }
    if (size == 1) {
        if (local_exception) std::rethrow_exception(local_exception);
        return;
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_exception;
    throw std::runtime_error(
        std::string(context)
        + " requested multi-rank exception synchronization in a non-MPI build");
#else
    require_live_mpi(context);
    const int local_failed = local_exception ? 1 : 0;
    int any_failed = 0;
    const int status = MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
    if (any_failed != 0) {
        if (local_exception) std::rethrow_exception(local_exception);
        throw std::runtime_error(
            std::string(context) + " failed on another MPI rank");
    }
#endif
}

} // namespace cosmo_nbody::runtime
