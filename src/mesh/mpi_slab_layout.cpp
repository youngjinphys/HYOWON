#include "cosmo_nbody/mesh/mpi_slab_layout.hpp"

#include "cosmo_nbody/mesh/mpi_fft_allocation_layout.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef COSMO_NBODY_HAS_FFTW_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace mesh {
namespace {

MpiSlabLayout deterministic_fallback_layout(
    std::size_t n0,
    int rank,
    int size) {
    if (n0 == 0) {
        throw std::invalid_argument("MPI slab global extent must be positive");
    }
    if (size < 1 || rank < 0 || rank >= size) {
        throw std::invalid_argument("MPI slab rank topology is invalid");
    }

    const std::size_t size_u = static_cast<std::size_t>(size);
    const std::size_t rank_u = static_cast<std::size_t>(rank);
    const std::size_t base = n0 / size_u;
    const std::size_t remainder = n0 % size_u;
    const std::size_t count = base + (rank_u < remainder ? 1U : 0U);
    const std::size_t start = rank_u * base + std::min(rank_u, remainder);
    return {start, count};
}

#ifdef COSMO_NBODY_HAS_FFTW_MPI
bool live_fftw_mpi_world(int& world_size) {
    int initialized = 0;
    int finalized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized) {
        return false;
    }
    if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized) {
        return false;
    }
    runtime::require_active_mpi_main_thread(
        "FFTW slab ownership query");
    if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to query MPI_COMM_WORLD for slab ownership");
    }
    return true;
}
#endif

} // namespace

MpiSlabLayout mpi_slab_layout(
    std::size_t n0,
    int rank,
    int size) {
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    int live_size = 0;
    if (live_fftw_mpi_world(live_size)) {
        if (size != live_size) {
            throw std::invalid_argument(
                "Requested slab topology differs from active MPI_COMM_WORLD");
        }
        return shared_mpi_fft_communicator_layout(n0)
            .rank_layout(rank)
            .slab();
    }
#endif
    return deterministic_fallback_layout(n0, rank, size);
}

bool mpi_slab_layout_has_empty_rank(
    std::size_t n0,
    int size) {
    if (size < 1) {
        throw std::invalid_argument("MPI slab size must be positive");
    }
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    int live_size = 0;
    if (live_fftw_mpi_world(live_size)) {
        if (size != live_size) {
            throw std::invalid_argument(
                "Requested slab topology differs from active MPI_COMM_WORLD");
        }
        return shared_mpi_fft_communicator_layout(n0)
            .minimum_local_n0() == 0;
    }
#endif
    return deterministic_fallback_layout(n0, size - 1, size).count == 0;
}

} // namespace mesh
} // namespace cosmo_nbody
