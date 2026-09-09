#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <optional>
#include <span>

namespace cosmo_nbody::math {

// local_count is the number of represented local terms. In uniform mode the
// explicit span must be empty and the scalar is repeated local_count times. In
// explicit mode the span length must equal local_count. Every represented term
// must be finite and non-negative; a collective containing only zero terms, or
// no represented terms, is valid and returns positive zero. The result is
// independent of insertion order, MPI rank partition, and MPI reduction-tree
// choice; the exact integer sum is rounded to binary64 once after collective
// combination.
core::Real mpi_exact_nonnegative_sum(
    std::size_t local_count,
    std::span<const core::Real> explicit_values,
    std::optional<core::Real> uniform_value);

// The same exact collective operation for multiple disjoint local spans without
// concatenating or issuing one collective per span. Segment order and rank
// partition do not affect the result. Used by distributed CIC to represent the
// pre-exchange state as local mesh cells plus both outgoing overflow planes.
core::Real mpi_exact_nonnegative_sum_segments(
    std::span<const std::span<const core::Real>> local_segments);

} // namespace cosmo_nbody::math
