#include "cosmo_nbody/math/mpi_exact_uint64_sum.hpp"

#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::math {

std::uint64_t mpi_exact_uint64_sum(std::uint64_t local_value) {
#ifndef COSMO_NBODY_HAS_MPI
    return local_value;
#else
    runtime::require_active_mpi_main_thread("Exact MPI uint64 sum");

    int size = 0;
    const int size_status = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size_status != MPI_SUCCESS) {
        // Ordinary MPI offers no recovery contract after communicator failure;
        // a rank-local throw could strand peers in the next collective.
        (void)MPI_Abort(MPI_COMM_WORLD, size_status);
        throw std::runtime_error(
            "Exact MPI uint64 sum failed to query MPI_COMM_WORLD");
    }
    if (size < 1) {
        throw std::runtime_error(
            "Exact MPI uint64 sum failed to query MPI_COMM_WORLD");
    }
    if (size == 1) return local_value;

    constexpr std::uint64_t limb_mask = 0xffffffffULL;
    // Prove the actual communicator can sum a maximum 32-bit limb in uint64
    // before asking MPI_SUM to do so. Mainstream MPI uses a 32-bit int size,
    // but this check keeps the arithmetic valid without relying on that ABI.
    const auto rank_count = static_cast<std::uintmax_t>(size);
    if (rank_count
        > std::numeric_limits<std::uint64_t>::max() / limb_mask) {
        throw std::overflow_error(
            "Exact MPI uint64 sum communicator is too large for 32-bit limbs");
    }
    const std::array<std::uint64_t, 2> local_limbs{
        local_value >> 32U,
        local_value & limb_mask};
    std::array<std::uint64_t, 2> global_limbs{};
    const int reduction_status = MPI_Allreduce(
            local_limbs.data(),
            global_limbs.data(),
            static_cast<int>(global_limbs.size()),
            MPI_UINT64_T,
            MPI_SUM,
            MPI_COMM_WORLD);
    if (reduction_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, reduction_status);
        throw std::runtime_error("MPI_Allreduce failed for exact uint64 sum");
    }

    const std::uint64_t carry = global_limbs[1] >> 32U;
    if (global_limbs[0] > std::numeric_limits<std::uint64_t>::max() - carry) {
        throw std::overflow_error("Exact MPI uint64 sum overflows uint64_t");
    }
    const std::uint64_t high = global_limbs[0] + carry;
    if (high > limb_mask) {
        throw std::overflow_error("Exact MPI uint64 sum overflows uint64_t");
    }
    return (high << 32U) | (global_limbs[1] & limb_mask);
#endif
}

} // namespace cosmo_nbody::math
