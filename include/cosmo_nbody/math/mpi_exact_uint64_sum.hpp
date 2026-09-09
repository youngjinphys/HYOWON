#pragma once

#include <cstdint>

namespace cosmo_nbody::math {

// Sum one uint64 value per rank without allowing MPI's arithmetic to wrap.
// A multi-rank result outside uint64 is rejected identically on every rank.
// The implementation uses fixed stack limbs and never materializes one value
// per rank. Non-MPI builds and one-rank communicators return local_value.
std::uint64_t mpi_exact_uint64_sum(std::uint64_t local_value);

} // namespace cosmo_nbody::math
