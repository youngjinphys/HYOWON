#include "cosmo_nbody/math/mpi_exact_nonnegative_sum.hpp"

#include "cosmo_nbody/math/exact_positive_sum.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::math {
namespace {

ExactPositiveDoubleSum exact_nonnegative_double_accumulator(
    std::size_t term_count,
    std::span<const core::Real> explicit_values,
    std::optional<core::Real> uniform_value) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (term_count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Exact non-negative sum term count exceeds uint64_t");
        }
    }
    const auto terms = static_cast<std::uint64_t>(term_count);

    ExactPositiveDoubleSum result;
    if (uniform_value.has_value()) {
        if (!explicit_values.empty()) {
            throw std::invalid_argument(
                "Exact non-negative sum uniform mode requires an empty explicit span");
        }
        if (!std::isfinite(*uniform_value) || *uniform_value < 0.0) {
            throw std::invalid_argument(
                "Exact non-negative sum uniform value must be finite and non-negative");
        }
        result.add_repeated(*uniform_value, terms);
        return result;
    }

    if (explicit_values.size() != term_count) {
        throw std::invalid_argument(
            "Exact non-negative sum explicit value count mismatch");
    }
    for (const core::Real value : explicit_values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(
                "Exact non-negative sum explicit values must be finite and non-negative");
        }
        result.add(value);
    }
    return result;
}

ExactPositiveDoubleSum exact_nonnegative_segment_accumulator(
    std::span<const std::span<const core::Real>> segments) {
    ExactPositiveDoubleSum result;
    for (const auto segment : segments) {
        for (const core::Real value : segment) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "Exact non-negative segment sum requires finite non-negative values");
            }
            result.add(value);
        }
    }
    return result;
}

#ifdef COSMO_NBODY_HAS_MPI
void synchronize_exception(
    std::exception_ptr local_exception,
    const char* context) {
    const int local_failed = local_exception ? 1 : 0;
    int any_failed = 0;
    if (MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context);
    }
    if (any_failed != 0) {
        if (local_exception) std::rethrow_exception(local_exception);
        throw std::runtime_error(
            std::string(context) + " failed on another MPI rank");
    }
}

void require_active_mpi(int& rank, int& size) {
    int initialized = 0;
    int finalized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
        throw std::runtime_error(
            "Exact MPI non-negative sum requires initialized MPI");
    }
    if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
        throw std::runtime_error(
            "Exact MPI non-negative sum is unavailable after MPI_Finalize");
    }
    int is_main_thread = 0;
    if (MPI_Is_thread_main(&is_main_thread) != MPI_SUCCESS
        || is_main_thread == 0) {
        throw std::runtime_error(
            "Exact MPI non-negative sum requires the MPI main thread");
    }
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS
        || MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS
        || rank < 0 || size < 1 || rank >= size) {
        throw std::runtime_error(
            "Exact MPI non-negative sum failed to query MPI_COMM_WORLD");
    }
}

core::Real combine_exact_accumulator_collective(
    const ExactPositiveDoubleSum& local,
    int rank,
    int size,
    const char* context) {
    if (size == 1) return local.value();

    std::vector<std::uint64_t> gathered_limbs;
    std::vector<std::uint64_t> gathered_counts;
    std::exception_ptr stage_exception;
    try {
        if (rank == 0) {
            const std::size_t rank_count = static_cast<std::size_t>(size);
            if (rank_count
                > std::numeric_limits<std::size_t>::max()
                    / ExactPositiveDoubleSum::limb_count) {
                throw std::overflow_error(
                    "Exact MPI non-negative sum gathered limb count overflows size_t");
            }
            gathered_limbs.resize(
                rank_count * ExactPositiveDoubleSum::limb_count);
            gathered_counts.resize(rank_count);
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_exception(
        stage_exception, "exact MPI non-negative gather allocation");

    constexpr int limb_count = static_cast<int>(
        ExactPositiveDoubleSum::limb_count);
    const auto local_limbs = local.limbs();
    if (MPI_Gather(
            local_limbs.data(),
            limb_count,
            MPI_UINT64_T,
            rank == 0 ? gathered_limbs.data() : nullptr,
            limb_count,
            MPI_UINT64_T,
            0,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Gather failed for ") + context + " limbs");
    }

    const std::uint64_t local_terms = local.term_count();
    if (MPI_Gather(
            &local_terms,
            1,
            MPI_UINT64_T,
            rank == 0 ? gathered_counts.data() : nullptr,
            1,
            MPI_UINT64_T,
            0,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Gather failed for ") + context + " term counts");
    }

    core::Real total = 0.0;
    stage_exception = nullptr;
    try {
        if (rank == 0) {
            ExactPositiveDoubleSum global;
            for (int source = 0; source < size; ++source) {
                const std::size_t offset =
                    static_cast<std::size_t>(source)
                    * ExactPositiveDoubleSum::limb_count;
                global.combine_serialized(
                    std::span<const std::uint64_t>(
                        gathered_limbs.data() + offset,
                        ExactPositiveDoubleSum::limb_count),
                    gathered_counts[static_cast<std::size_t>(source)]);
            }
            total = global.value();
            if (!std::isfinite(total)) {
                throw std::overflow_error(
                    "Exact MPI non-negative sum is not finite");
            }
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_exception(
        stage_exception, "exact MPI non-negative global combination");

    if (MPI_Bcast(
            &total,
            1,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Bcast failed for ") + context + " result");
    }
    return total;
}
#endif

} // namespace

core::Real mpi_exact_nonnegative_sum(
    std::size_t local_count,
    std::span<const core::Real> explicit_values,
    std::optional<core::Real> uniform_value) {
#ifndef COSMO_NBODY_HAS_MPI
    return exact_nonnegative_double_accumulator(
        local_count, explicit_values, uniform_value).value();
#else
    int rank = 0;
    int size = 1;
    require_active_mpi(rank, size);

    ExactPositiveDoubleSum local;
    std::exception_ptr stage_exception;
    try {
        local = exact_nonnegative_double_accumulator(
            local_count, explicit_values, uniform_value);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_exception(
        stage_exception, "exact MPI non-negative local accumulation");
    return combine_exact_accumulator_collective(
        local, rank, size, "exact MPI non-negative sum");
#endif
}

core::Real mpi_exact_nonnegative_sum_segments(
    std::span<const std::span<const core::Real>> local_segments) {
#ifndef COSMO_NBODY_HAS_MPI
    return exact_nonnegative_segment_accumulator(local_segments).value();
#else
    int rank = 0;
    int size = 1;
    require_active_mpi(rank, size);

    ExactPositiveDoubleSum local;
    std::exception_ptr stage_exception;
    try {
        local = exact_nonnegative_segment_accumulator(local_segments);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_exception(
        stage_exception, "exact MPI non-negative segment accumulation");
    return combine_exact_accumulator_collective(
        local, rank, size, "exact MPI non-negative segment sum");
#endif
}

} // namespace cosmo_nbody::math
