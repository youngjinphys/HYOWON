#pragma once

#include <mutex>

namespace cosmo_nbody::mesh {

// FFTW guarantees plan execution, but not planning or destruction, to be
// thread-safe. Serial and MPI backends must use this same process-global lock.
std::mutex& shared_fftw_planner_mutex();

// OpenMP policy visible to FFTW plan construction. Returns one without OpenMP.
int requested_fftw_host_threads() noexcept;

// Initializes FFTW thread support at most once for the process and fails closed
// when the linked runtime reports an initialization error. In a build without
// the FFTW threads library this is a successful no-op.
void require_fftw_threads_initialized();

// Applies to plans created after the call. It is a no-op when FFTW thread
// support is not linked.
void configure_fftw_plan_threads(int thread_count);

// Initializes the MPI extension at most once after shared FFTW thread initialization.
void initialize_fftw_mpi_once();

} // namespace cosmo_nbody::mesh
