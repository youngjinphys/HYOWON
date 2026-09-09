#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mpi_slab_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cosmo_nbody::mesh {

struct MpiFFTAllocationLayout {
    std::size_t grid_size{0};
    int rank{0};
    int communicator_size{1};
    std::size_t local_n0{0};
    std::size_t local_0_start{0};
    std::size_t alloc_local_complex_elements{0};
    std::size_t padded_last_dim{0};

    MpiSlabLayout slab() const noexcept {
        return {local_0_start, local_n0};
    }

    std::size_t compact_real_elements() const;
    std::size_t required_complex_elements() const;
    std::size_t padded_real_elements() const;
    std::uint64_t compact_real_bytes() const;
    std::uint64_t padded_real_bytes() const;
    std::uint64_t complex_bytes() const;
    // Padded real workspace plus caller-owned alloc_local complex storage.
    std::uint64_t transform_workspace_bytes() const;
};

MpiFFTAllocationLayout make_mpi_fft_allocation_layout(
    std::size_t grid_size,
    int rank,
    int communicator_size,
    std::size_t local_n0,
    std::size_t local_0_start,
    std::size_t alloc_local_complex_elements);

struct MpiFFTCommunicatorLayout {
    std::size_t grid_size{0};
    std::vector<MpiFFTAllocationLayout> ranks;

    const MpiFFTAllocationLayout& rank_layout(int rank) const;
    int owner_rank(std::size_t global_plane) const;
    std::size_t total_planes() const;
    std::size_t minimum_local_n0() const;
};

MpiFFTCommunicatorLayout make_mpi_fft_communicator_layout(
    std::size_t grid_size,
    std::vector<MpiFFTAllocationLayout> ranks);

// Collective MPI_COMM_WORLD queries available only with active FFTW-MPI.
MpiFFTAllocationLayout query_mpi_fft_allocation_layout(
    std::size_t grid_size);
MpiFFTCommunicatorLayout query_mpi_fft_communicator_layout(
    std::size_t grid_size);

// Process-local immutable registry shared by planner, ownership, and FFT users.
// Cache misses require the live MPI main thread; hits issue no communicator or
// collective calls and remain valid only while the MPI world is live.
const MpiFFTCommunicatorLayout& shared_mpi_fft_communicator_layout(
    std::size_t grid_size);
const MpiFFTAllocationLayout& shared_mpi_fft_allocation_layout(
    std::size_t grid_size);

} // namespace cosmo_nbody::mesh
