#include "cosmo_nbody/mesh/mpi_fft_allocation_layout.hpp"

#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <atomic>
#include <exception>
#include <map>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef COSMO_NBODY_HAS_FFTW_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::mesh {
namespace {

struct RegistryEntry {
    int communicator_size{0};
    int local_rank{0};
    MpiFFTCommunicatorLayout layout;
};

class NoThrowAtomicLock final {
public:
    explicit NoThrowAtomicLock(std::atomic_flag& flag) noexcept
        : flag_(flag) {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            flag_.wait(true, std::memory_order_relaxed);
        }
    }

    ~NoThrowAtomicLock() {
        flag_.clear(std::memory_order_release);
        flag_.notify_one();
    }

    NoThrowAtomicLock(const NoThrowAtomicLock&) = delete;
    NoThrowAtomicLock& operator=(const NoThrowAtomicLock&) = delete;

private:
    std::atomic_flag& flag_;
};

void require_live_world() {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    throw std::runtime_error(
        "Shared FFT layout registry requires FFTW-MPI support");
#else
    int initialized = 0;
    int finalized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized) {
        throw std::runtime_error(
            "Shared FFT layout registry requires initialized MPI");
    }
    if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized) {
        throw std::runtime_error(
            "Shared FFT layout registry is unavailable after MPI_Finalize");
    }
#endif
}

void require_mpi_main_thread() {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    throw std::runtime_error(
        "Shared FFT layout registry requires FFTW-MPI support");
#else
    int is_main_thread = 0;
    if (MPI_Is_thread_main(&is_main_thread) != MPI_SUCCESS || !is_main_thread) {
        throw std::runtime_error(
            "A new FFTW-MPI layout was requested outside the MPI main thread");
    }
#endif
}

int world_size() {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    return 0;
#else
    int size = 0;
    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size < 1) {
        throw std::runtime_error(
            "Failed to query MPI_COMM_WORLD for FFT layout registry");
    }
    return size;
#endif
}

int world_rank() {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    return 0;
#else
    int rank = 0;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS || rank < 0) {
        throw std::runtime_error(
            "Failed to query MPI rank for FFT layout registry");
    }
    return rank;
#endif
}

class MpiFFTLayoutRegistry final {
public:
    const MpiFFTCommunicatorLayout& communicator_layout(std::size_t grid_size) {
        return entry(grid_size).layout;
    }

    const MpiFFTAllocationLayout& allocation_layout(std::size_t grid_size) {
        const auto& record = entry(grid_size);
        return record.layout.rank_layout(record.local_rank);
    }

private:
    const RegistryEntry& entry(std::size_t grid_size) {
        // A cache miss immediately precedes communicator collectives. Use a
        // non-throwing process-local lock so one rank cannot peel off with a
        // std::mutex::lock exception while peers enter the layout query.
        NoThrowAtomicLock lock(lock_);

        // MPI_Initialized and MPI_Finalized are required by MPI to remain
        // thread-safe and callable before initialization and after finalization.
        // Probe lifecycle before serving cached data so an immutable layout cannot
        // outlive the MPI world that established its communicator identity.
        require_live_world();

        const auto found = entries_.find(grid_size);
        if (found != entries_.end()) {
            return found->second;
        }

        // A collective cache miss may only be materialized by MPI's actual main
        // thread. Do not infer ownership from whichever application thread happens
        // to request the first layout.
        require_mpi_main_thread();
        RegistryEntry record;
        record.communicator_size = world_size();
        std::exception_ptr prequery_exception;
        try {
            const std::thread::id current = std::this_thread::get_id();
            if (main_thread_bound_ && current != mpi_main_thread_) {
                throw std::runtime_error(
                    "A new FFTW-MPI layout was requested outside the registry MPI main thread");
            }
            if (!main_thread_bound_) {
                mpi_main_thread_ = current;
                main_thread_bound_ = true;
            }
            record.local_rank = world_rank();
        } catch (...) {
            prequery_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            prequery_exception,
            record.communicator_size,
            "FFTW-MPI layout registry prequery");
        record.layout = query_mpi_fft_communicator_layout(grid_size);
        const int communicator_size = record.communicator_size;
        auto published = entries_.end();
        std::exception_ptr publication_exception;
        try {
            if (record.layout.ranks.size()
                != static_cast<std::size_t>(record.communicator_size)) {
                throw std::logic_error(
                    "Exact FFT communicator layout has the wrong rank count");
            }

            const auto [inserted, ok] = entries_.emplace(
                grid_size, std::move(record));
            if (!ok) {
                throw std::logic_error(
                    "Failed to publish FFT layout registry entry");
            }
            published = inserted;
        } catch (...) {
            publication_exception = std::current_exception();
        }

        try {
            runtime::synchronize_mpi_exception(
                publication_exception,
                communicator_size,
                "FFTW-MPI layout registry publication");
        } catch (...) {
            // A successful rank must not retain a cache hit when a peer failed
            // to publish. Otherwise a retry would skip the collective layout
            // query here while the failed peer enters it alone.
            if (published != entries_.end()) entries_.erase(published);
            throw;
        }
        return published->second;
    }

    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    std::map<std::size_t, RegistryEntry> entries_;
    std::thread::id mpi_main_thread_{};
    bool main_thread_bound_{false};
};

MpiFFTLayoutRegistry& layout_registry() {
    static MpiFFTLayoutRegistry value;
    return value;
}

} // namespace

const MpiFFTCommunicatorLayout& shared_mpi_fft_communicator_layout(
    std::size_t grid_size) {
    return layout_registry().communicator_layout(grid_size);
}

const MpiFFTAllocationLayout& shared_mpi_fft_allocation_layout(
    std::size_t grid_size) {
    return layout_registry().allocation_layout(grid_size);
}

} // namespace cosmo_nbody::mesh
