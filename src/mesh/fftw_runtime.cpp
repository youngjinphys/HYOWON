#include "cosmo_nbody/mesh/fftw_runtime.hpp"

#include <mutex>
#include <stdexcept>

#if defined(COSMO_NBODY_HAS_FFTW_THREADS) \
    || defined(COSMO_NBODY_HAS_FFTW_MPI)
#include <fftw3.h>
#endif

#if defined(COSMO_NBODY_HAS_FFTW_MPI)
#include <fftw3-mpi.h>
#endif

#if defined(COSMO_NBODY_HAS_OPENMP)
#include <omp.h>
#endif

namespace cosmo_nbody::mesh {
namespace {

std::once_flag& fftw_threads_once_flag() {
    static auto* flag = new std::once_flag();
    return *flag;
}

int& fftw_threads_status() {
    // 0 = not attempted, 1 = initialized or unavailable by build, -1 = failed.
    static int status = 0;
    return status;
}

#if defined(COSMO_NBODY_HAS_FFTW_MPI)
std::once_flag& fftw_mpi_once_flag() {
    static auto* flag = new std::once_flag();
    return *flag;
}
#endif

} // namespace

std::mutex& shared_fftw_planner_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

int requested_fftw_host_threads() noexcept {
#if defined(COSMO_NBODY_HAS_OPENMP)
    const int requested = omp_get_max_threads();
    return requested > 0 ? requested : 1;
#else
    return 1;
#endif
}

void require_fftw_threads_initialized() {
    std::call_once(fftw_threads_once_flag(), [] {
#if defined(COSMO_NBODY_HAS_FFTW_THREADS)
        fftw_threads_status() = fftw_init_threads() != 0 ? 1 : -1;
#else
        fftw_threads_status() = 1;
#endif
    });
    if (fftw_threads_status() != 1) {
        throw std::runtime_error("Failed to initialize FFTW thread support");
    }
}

void configure_fftw_plan_threads(int thread_count) {
    if (thread_count < 1) {
        throw std::invalid_argument(
            "FFTW planner thread count must be positive");
    }
#if defined(COSMO_NBODY_HAS_FFTW_THREADS)
    fftw_plan_with_nthreads(thread_count);
#else
    (void)thread_count;
#endif
}

void initialize_fftw_mpi_once() {
#if !defined(COSMO_NBODY_HAS_FFTW_MPI)
    throw std::runtime_error(
        "FFTW-MPI initialization requires FFTW-MPI support");
#else
    require_fftw_threads_initialized();
    std::call_once(fftw_mpi_once_flag(), [] { fftw_mpi_init(); });
#endif
}

} // namespace cosmo_nbody::mesh
