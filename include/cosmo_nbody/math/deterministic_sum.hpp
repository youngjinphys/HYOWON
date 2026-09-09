// Fixed block partition and reduction order make these sums independent of
// OpenMP team size; block and final sums use Neumaier compensation.
#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/exact_signed_sum.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody::math {

namespace detail {

inline void neumaier_add(
    core::Accum value,
    core::Accum& sum,
    core::Accum& compensation) noexcept {
    const core::Accum updated = sum + value;
    if (std::abs(sum) >= std::abs(value)) {
        compensation += (sum - updated) + value;
    } else {
        compensation += (value - updated) + sum;
    }
    sum = updated;
}

template <typename Term>
core::Accum exact_signed_fallback(
    std::size_t count,
    const Term& term) {
    // Exceptional serial fallback distinguishes a non-finite intermediate
    // reduction from an unrepresentable final result without hot-path cost.
    ExactSignedDoubleSum exact;
    for (std::size_t index = 0; index < count; ++index) {
        exact.add(static_cast<core::Accum>(term(index)));
    }
    return exact.value();
}

} // namespace detail

template <typename Term>
core::Accum deterministic_blocked_sum(
    std::size_t count,
    const Term& term,
    std::size_t block_size = 4096) {
    if (block_size == 0) {
        throw std::invalid_argument(
            "deterministic_blocked_sum block_size must be positive");
    }
    if (count == 0) return core::Accum{0.0};

    const std::size_t block_count = 1 + (count - 1) / block_size;
    // Reuse O(block_count) partial storage on the caller thread; workers share it.
    static thread_local std::vector<core::Accum> reusable_partials;
    auto& partials = reusable_partials;
    partials.assign(block_count, core::Accum{0.0});
    std::vector<std::exception_ptr> block_exceptions(block_count);

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(block_count > 1)
#endif
    for (std::size_t block = 0; block < block_count; ++block) {
        try {
            const std::size_t begin = block * block_size;
            const std::size_t end = begin + std::min(block_size, count - begin);
            core::Accum sum = 0.0;
            core::Accum compensation = 0.0;
            for (std::size_t index = begin; index < end; ++index) {
                detail::neumaier_add(
                    static_cast<core::Accum>(term(index)),
                    sum,
                    compensation);
            }
            partials[block] = sum + compensation;
        } catch (...) {
            block_exceptions[block] = std::current_exception();
        }
    }

    // Exceptions must not escape an OpenMP region. Re-throw in fixed block
    // order before any partial result can become caller-visible.
    for (const auto& exception : block_exceptions) {
        if (exception) std::rethrow_exception(exception);
    }

    core::Accum total = 0.0;
    core::Accum compensation = 0.0;
    for (const core::Accum partial : partials) {
        detail::neumaier_add(partial, total, compensation);
    }
    const core::Accum result = total + compensation;
    if (std::isfinite(result)) return result;
    return detail::exact_signed_fallback(count, term);
}

} // namespace cosmo_nbody::math
