#pragma once

#include <mutex>

namespace cosmo_nbody::mesh {

class MpiFftwPlannerSession {
public:
    explicit MpiFftwPlannerSession(
        std::unique_lock<std::mutex> planner_lock,
        int configured_thread_count) noexcept;

    MpiFftwPlannerSession(MpiFftwPlannerSession&&) noexcept = default;
    MpiFftwPlannerSession& operator=(MpiFftwPlannerSession&&) noexcept = default;
    MpiFftwPlannerSession(const MpiFftwPlannerSession&) = delete;
    MpiFftwPlannerSession& operator=(const MpiFftwPlannerSession&) = delete;

    int configured_thread_count() const noexcept {
        return configured_thread_count_;
    }

private:
    std::unique_lock<std::mutex> planner_lock_;
    int configured_thread_count_{1};
};

// Fail closed unless MPI is live and the caller is the World Model main thread
// (the thread that called MPI_Init or MPI_Init_thread). MPI_Query_thread and
// MPI_Is_thread_main are defined to be callable from threads irrespective of
// the provided thread level, so this guard can safely protect FUNNELED entry
// points before they issue communicator or collective calls.
void require_mpi_main_thread(const char* operation);

// Collective over MPI_COMM_WORLD. The caller must be the MPI main thread on
// every rank. The function agrees on one rank-local OpenMP thread count,
// initializes FFTW threads before FFTW-MPI, configures subsequent plans with the
// agreed count, and returns a session retaining the process-global planner lock.
// Keep the session alive through every FFTW planner call that depends on it.
[[nodiscard]] MpiFftwPlannerSession configure_mpi_fftw_planner();

} // namespace cosmo_nbody::mesh
