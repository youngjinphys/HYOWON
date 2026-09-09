#pragma once

#include <exception>

namespace cosmo_nbody::runtime {

// Require initialized, non-finalized MPI on the main thread under
// MPI_THREAD_FUNNELED.
void require_active_mpi_main_thread(const char* context);

// Synchronize a local exception before the next collective; the origin rethrows
// while peers fail closed instead of advancing alone.
void synchronize_mpi_exception(
    std::exception_ptr local_exception,
    int size,
    const char* context);

void synchronize_mpi_failure(
    int local_failed,
    int size,
    const char* context);

} // namespace cosmo_nbody::runtime
