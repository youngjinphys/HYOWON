#include "cosmo_nbody/mesh/mpi_fftw_runtime.hpp"

#include "cosmo_nbody/mesh/fftw_runtime.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(COSMO_NBODY_HAS_FFTW_MPI)
#include <mpi.h>
#endif

namespace cosmo_nbody::mesh {
namespace {

#if defined(COSMO_NBODY_HAS_FFTW_MPI)
int requested_rank_threads() noexcept {
#if defined(COSMO_NBODY_HAS_FFTW_THREADS)
    return requested_fftw_host_threads();
#else
    return 1;
#endif
}

int agreed_rank_threads() {
    const int local = requested_rank_threads();
    int minimum = 0;
    int maximum = 0;
    const int minimum_status = MPI_Allreduce(
        &local, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const int maximum_status = MPI_Allreduce(
        &local, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (minimum_status != MPI_SUCCESS || maximum_status != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to synchronize the FFTW-MPI rank-local thread count");
    }
    if (minimum < 1 || minimum != maximum) {
        throw std::invalid_argument(
            "FFTW-MPI ranks disagree on the rank-local OpenMP thread count");
    }
    return minimum;
}

void require_thread_support(int threads) {
    int provided = MPI_THREAD_SINGLE;
    const int query_failed = MPI_Query_thread(&provided) != MPI_SUCCESS ? 1 : 0;
    int any_query_failed = 0;
    if (MPI_Allreduce(
            &query_failed,
            &any_query_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to synchronize the MPI thread-support query");
    }
    if (any_query_failed != 0) {
        throw std::runtime_error("MPI_Query_thread failed on at least one rank");
    }

    const int local_insufficient =
        threads > 1 && provided < MPI_THREAD_FUNNELED ? 1 : 0;
    int any_insufficient = 0;
    if (MPI_Allreduce(
            &local_insufficient,
            &any_insufficient,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to synchronize FFTW-MPI thread-support admission");
    }
    if (any_insufficient != 0) {
        throw std::runtime_error(
            "Threaded FFTW-MPI requires MPI_THREAD_FUNNELED on every rank");
    }
}

void synchronize_initialization_exception(
    std::exception_ptr local_exception,
    const char* context) {
    const int local_failed = local_exception ? 1 : 0;
    int any_failed = 0;
    if (MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("Failed to synchronize ") + context);
    }
    if (any_failed != 0) {
        if (local_exception) std::rethrow_exception(local_exception);
        throw std::runtime_error(
            std::string(context) + " failed on another rank");
    }
}
#endif

} // namespace

MpiFftwPlannerSession::MpiFftwPlannerSession(
    std::unique_lock<std::mutex> planner_lock,
    int configured_thread_count) noexcept
    : planner_lock_(std::move(planner_lock)),
      configured_thread_count_(configured_thread_count) {}

void require_mpi_main_thread(const char* operation) {
#if !defined(COSMO_NBODY_HAS_FFTW_MPI)
    (void)operation;
    throw std::runtime_error(
        "MPI main-thread admission requires FFTW-MPI support");
#else
    runtime::require_active_mpi_main_thread(
        operation ? operation : "MPI operation");
#endif
}

MpiFftwPlannerSession configure_mpi_fftw_planner() {
#if !defined(COSMO_NBODY_HAS_FFTW_MPI)
    throw std::runtime_error(
        "FFTW-MPI planner configuration requires FFTW-MPI support");
#else
    require_mpi_main_thread("FFTW-MPI planner configuration");

    const int threads = agreed_rank_threads();
    require_thread_support(threads);

    std::optional<std::unique_lock<std::mutex>> planner_lock;
    std::exception_ptr planner_lock_exception;
    try {
        // Constructing the process-global mutex and acquiring it can both fail
        // rank-locally. Agree on that stage before any rank advances to the
        // collective FFTW initialization checks below.
        planner_lock.emplace(shared_fftw_planner_mutex());
    } catch (...) {
        planner_lock_exception = std::current_exception();
    }
    synchronize_initialization_exception(
        planner_lock_exception, "FFTW planner lock acquisition");

    std::exception_ptr thread_exception;
    try {
        require_fftw_threads_initialized();
    } catch (...) {
        thread_exception = std::current_exception();
    }
    synchronize_initialization_exception(
        thread_exception, "FFTW thread initialization");

    std::exception_ptr mpi_exception;
    try {
        initialize_fftw_mpi_once();
    } catch (...) {
        mpi_exception = std::current_exception();
    }
    synchronize_initialization_exception(
        mpi_exception, "FFTW-MPI initialization");

    configure_fftw_plan_threads(threads);
    return MpiFftwPlannerSession(std::move(*planner_lock), threads);
#endif
}

} // namespace cosmo_nbody::mesh
