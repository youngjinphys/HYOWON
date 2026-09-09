#pragma once

#include <cstddef>

namespace cosmo_nbody {
namespace mesh {

struct MpiSlabLayout {
    std::size_t start{0};
    std::size_t count{0};
};

// Use live FFTW-MPI layout when available; otherwise deterministic quotient/remainder.
MpiSlabLayout mpi_slab_layout(
    std::size_t n0,
    int rank,
    int size);

bool mpi_slab_layout_has_empty_rank(
    std::size_t n0,
    int size);

} // namespace mesh
} // namespace cosmo_nbody
