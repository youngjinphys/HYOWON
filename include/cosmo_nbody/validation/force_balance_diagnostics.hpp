// Numerically stable force-balance diagnostic for validation campaigns.
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody::validation {

// Dimensionless internal-force residual
//
//   R_F = ||sum_i m_i a_i||_2 / sum_i m_i ||a_i||_2.
//
// Exact symmetric pair forces give R_F = 0. A one-sided approximate tree may
// produce a non-zero value, which must be measured rather than hidden. Every
// large scan uses fixed logical blocks: blocks execute independently in
// parallel, then their compensated partials are merged in ascending block
// order. The arithmetic is therefore independent of the OpenMP team size.
struct ForceBalanceDiagnostics {
    core::Real relative_residual{0.0};
};

// Additive representation used by a distributed diagnostic. Every partial in
// one reduction must use the same common_exponent. The shared power-of-two
// factor cancels exactly from R_F, so ranks combine numerator components and the
// denominator before the ratio is evaluated. Rank-local ratios are not
// composable and must never be reduced.
struct ForceBalanceScaledSums {
    int common_exponent{std::numeric_limits<int>::min()};
    long double scaled_force_x{0.0L};
    long double scaled_force_y{0.0L};
    long double scaled_force_z{0.0L};
    long double scaled_denominator{0.0L};
    std::uint64_t particle_count{0};
};

// Requested dynamic payload for the deterministic blocked force-balance
// reduction. The exponent and scaled block arrays are separate thread-local
// vectors: after the first sample both remain retained on the calling thread
// until process teardown. MPI receive/partial arrays are call-local and overlap
// both retained vectors during every distributed sample. Vector capacity beyond
// the requested element count and allocator bookkeeping remain runtime
// high-water obligations.
struct ForceBalanceMemoryPlan {
    std::uint64_t particle_count{0};
    std::uint64_t block_count{0};
    std::uint64_t retained_exponent_block_bytes{0};
    std::uint64_t retained_scaled_block_bytes{0};
    std::uint64_t retained_block_bytes{0};
    std::uint64_t mpi_rank_count{0};
    std::uint64_t mpi_gathered_value_count{0};
    std::uint64_t mpi_gathered_value_bytes{0};
    std::uint64_t mpi_gathered_count_bytes{0};
    std::uint64_t mpi_partial_bytes{0};
    std::uint64_t mpi_gather_bytes{0};
    std::uint64_t sample_peak_bytes{0};
};

namespace detail {

// Fixed logical blocks keep the reduction order independent of OpenMP team size.
constexpr std::size_t force_balance_block_size = 4096;

class NeumaierAccumulator {
public:
    void add(long double value) noexcept {
        const long double updated = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            compensation_ += (sum_ - updated) + value;
        } else {
            compensation_ += (value - updated) + sum_;
        }
        sum_ = updated;
    }

    long double value() const noexcept { return sum_ + compensation_; }

private:
    long double sum_{0.0L};
    long double compensation_{0.0L};
};

inline int product_binary_exponent(core::Real mass, core::Real component) {
    if (component == 0.0) return std::numeric_limits<int>::min();

    int mass_exponent = 0;
    int component_exponent = 0;
    (void)std::frexp(static_cast<long double>(mass), &mass_exponent);
    (void)std::frexp(
        std::abs(static_cast<long double>(component)),
        &component_exponent);
    return mass_exponent + component_exponent;
}

inline long double scaled_force_component(
    core::Real mass,
    core::Real component,
    int common_exponent) {
    if (component == 0.0) return 0.0L;

    int mass_exponent = 0;
    int component_exponent = 0;
    const long double mass_mantissa = std::frexp(
        static_cast<long double>(mass), &mass_exponent);
    const long double component_mantissa = std::frexp(
        std::abs(static_cast<long double>(component)),
        &component_exponent);
    const int relative_exponent =
        mass_exponent + component_exponent - common_exponent;
    const long double magnitude = std::scalbn(
        mass_mantissa * component_mantissa,
        relative_exponent);
    return std::copysign(magnitude, static_cast<long double>(component));
}

inline void require_matching_force_arrays(
    std::size_t count,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    if (acceleration_x.size() != count
        || acceleration_y.size() != count
        || acceleration_z.size() != count) {
        throw std::invalid_argument(
            "Force-balance mass and acceleration arrays must have equal length");
    }
}

inline std::uint64_t checked_particle_count(std::size_t count) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Force-balance particle count exceeds uint64 range");
        }
    }
    return static_cast<std::uint64_t>(count);
}

inline std::size_t fixed_block_count(std::size_t count) noexcept {
    return count == 0
        ? 0
        : 1 + (count - 1) / force_balance_block_size;
}

inline std::uint64_t checked_memory_multiply(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* label) {
    if (lhs != 0
        && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(
            std::string("Force-balance memory plan overflow: ") + label);
    }
    return lhs * rhs;
}

inline std::uint64_t checked_memory_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string("Force-balance memory plan overflow: ") + label);
    }
    return lhs + rhs;
}

enum class ForceBalanceInputError : std::uint8_t {
    none = 0,
    invalid_mass = 1,
    invalid_acceleration = 2,
};

struct ForceBalanceExponentBlock {
    int maximum_exponent{std::numeric_limits<int>::min()};
    ForceBalanceInputError input_error{ForceBalanceInputError::none};
};

struct ForceBalanceScaledBlock {
    long double force_x{0.0L};
    long double force_y{0.0L};
    long double force_z{0.0L};
    long double denominator{0.0L};
    ForceBalanceInputError input_error{ForceBalanceInputError::none};
    bool underscaled{false};
    bool non_finite{false};
};

inline void throw_input_error(ForceBalanceInputError error) {
    if (error == ForceBalanceInputError::invalid_mass) {
        throw std::invalid_argument(
            "Force-balance particle masses must be finite and positive");
    }
    if (error == ForceBalanceInputError::invalid_acceleration) {
        throw std::invalid_argument(
            "Force-balance accelerations must be finite");
    }
}

template <typename MassAt>
inline int force_balance_common_exponent_impl(
    std::size_t count,
    MassAt&& mass_at,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    require_matching_force_arrays(
        count, acceleration_x, acceleration_y, acceleration_z);

    const std::size_t block_count = fixed_block_count(count);
    // The OpenMP team shares the caller thread's retained block vector.
    static thread_local std::vector<ForceBalanceExponentBlock>
        reusable_exponent_blocks;
    auto& blocks = reusable_exponent_blocks;
    blocks.assign(block_count, ForceBalanceExponentBlock{});

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(block_count > 1)
#endif
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t begin = block * force_balance_block_size;
        const std::size_t end = begin + std::min(
            force_balance_block_size, count - begin);
        ForceBalanceExponentBlock result;
        for (std::size_t index = begin; index < end; ++index) {
            const core::Real mass = mass_at(index);
            if (!std::isfinite(mass) || mass <= 0.0) {
                if (result.input_error == ForceBalanceInputError::none) {
                    result.input_error = ForceBalanceInputError::invalid_mass;
                }
                continue;
            }

            const core::Real ax = acceleration_x[index];
            const core::Real ay = acceleration_y[index];
            const core::Real az = acceleration_z[index];
            if (!std::isfinite(ax) || !std::isfinite(ay)
                || !std::isfinite(az)) {
                if (result.input_error == ForceBalanceInputError::none) {
                    result.input_error =
                        ForceBalanceInputError::invalid_acceleration;
                }
                continue;
            }

            result.maximum_exponent = std::max({
                result.maximum_exponent,
                product_binary_exponent(mass, ax),
                product_binary_exponent(mass, ay),
                product_binary_exponent(mass, az)});
        }
        blocks[block] = result;
    }

    int common_exponent = std::numeric_limits<int>::min();
    for (const auto& block : blocks) {
        throw_input_error(block.input_error);
        common_exponent = std::max(
            common_exponent, block.maximum_exponent);
    }
    return common_exponent;
}

template <typename MassAt>
inline ForceBalanceScaledSums compute_force_balance_scaled_sums_impl(
    std::size_t count,
    MassAt&& mass_at,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z,
    int common_exponent) {
    require_matching_force_arrays(
        count, acceleration_x, acceleration_y, acceleration_z);

    const std::size_t block_count = fixed_block_count(count);
    static thread_local std::vector<ForceBalanceScaledBlock>
        reusable_scaled_blocks;
    auto& blocks = reusable_scaled_blocks;
    blocks.assign(block_count, ForceBalanceScaledBlock{});

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(block_count > 1)
#endif
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t begin = block * force_balance_block_size;
        const std::size_t end = begin + std::min(
            force_balance_block_size, count - begin);
        NeumaierAccumulator net_x;
        NeumaierAccumulator net_y;
        NeumaierAccumulator net_z;
        NeumaierAccumulator force_sum;
        ForceBalanceScaledBlock result;

        for (std::size_t index = begin; index < end; ++index) {
            const core::Real mass = mass_at(index);
            if (!std::isfinite(mass) || mass <= 0.0) {
                if (result.input_error == ForceBalanceInputError::none) {
                    result.input_error = ForceBalanceInputError::invalid_mass;
                }
                continue;
            }

            const core::Real ax = acceleration_x[index];
            const core::Real ay = acceleration_y[index];
            const core::Real az = acceleration_z[index];
            if (!std::isfinite(ax) || !std::isfinite(ay)
                || !std::isfinite(az)) {
                if (result.input_error == ForceBalanceInputError::none) {
                    result.input_error =
                        ForceBalanceInputError::invalid_acceleration;
                }
                continue;
            }

            // Validate the caller-supplied exponent while computing block partials.
            const int local_exponent = std::max({
                product_binary_exponent(mass, ax),
                product_binary_exponent(mass, ay),
                product_binary_exponent(mass, az)});
            if (local_exponent > common_exponent) {
                result.underscaled = true;
                continue;
            }

            const long double fx = scaled_force_component(
                mass, ax, common_exponent);
            const long double fy = scaled_force_component(
                mass, ay, common_exponent);
            const long double fz = scaled_force_component(
                mass, az, common_exponent);
            const long double norm = core::scale_safe_norm3(fx, fy, fz);
            if (!std::isfinite(fx) || !std::isfinite(fy)
                || !std::isfinite(fz) || !std::isfinite(norm)
                || norm < 0.0L) {
                result.non_finite = true;
                continue;
            }

            net_x.add(fx);
            net_y.add(fy);
            net_z.add(fz);
            force_sum.add(norm);
        }

        result.force_x = net_x.value();
        result.force_y = net_y.value();
        result.force_z = net_z.value();
        result.denominator = force_sum.value();
        result.non_finite = result.non_finite
            || !std::isfinite(result.force_x)
            || !std::isfinite(result.force_y)
            || !std::isfinite(result.force_z)
            || !std::isfinite(result.denominator)
            || result.denominator < 0.0L;
        blocks[block] = result;
    }

    // Preserve error precedence and inspect blocks in logical order for
    // thread-independent diagnostics.
    for (const auto& block : blocks) {
        throw_input_error(block.input_error);
    }
    for (const auto& block : blocks) {
        if (block.underscaled) {
            throw std::invalid_argument(
                "Force-balance common exponent does not cover the local force range");
        }
    }
    for (const auto& block : blocks) {
        if (block.non_finite) {
            throw std::overflow_error(
                "Force-balance scaled accumulation is non-finite");
        }
    }

    NeumaierAccumulator net_x;
    NeumaierAccumulator net_y;
    NeumaierAccumulator net_z;
    NeumaierAccumulator force_sum;
    for (const auto& block : blocks) {
        net_x.add(block.force_x);
        net_y.add(block.force_y);
        net_z.add(block.force_z);
        force_sum.add(block.denominator);
    }

    ForceBalanceScaledSums sums;
    sums.common_exponent = common_exponent;
    sums.scaled_force_x = net_x.value();
    sums.scaled_force_y = net_y.value();
    sums.scaled_force_z = net_z.value();
    sums.scaled_denominator = force_sum.value();
    sums.particle_count = checked_particle_count(count);
    if (!std::isfinite(sums.scaled_force_x)
        || !std::isfinite(sums.scaled_force_y)
        || !std::isfinite(sums.scaled_force_z)
        || !std::isfinite(sums.scaled_denominator)
        || sums.scaled_denominator < 0.0L) {
        throw std::overflow_error(
            "Force-balance scaled accumulation is non-finite");
    }
    return sums;
}

} // namespace detail

inline ForceBalanceMemoryPlan force_balance_memory_plan(
    std::uint64_t particle_count,
    bool mpi_active,
    std::uint64_t mpi_ranks) {
    if (mpi_ranks == 0) {
        throw std::invalid_argument(
            "Force-balance memory planning requires at least one rank");
    }
    if (!mpi_active && mpi_ranks != 1) {
        throw std::invalid_argument(
            "Serial force-balance memory planning requires one rank");
    }
    if (mpi_ranks
        > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Force-balance MPI rank count exceeds the MPI int range");
    }
    if (particle_count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Force-balance particle count exceeds the host size_t range");
    }

    const std::uint64_t block_count = static_cast<std::uint64_t>(
        detail::fixed_block_count(
            static_cast<std::size_t>(particle_count)));
    const std::uint64_t exponent_bytes =
        detail::checked_memory_multiply(
            block_count,
            sizeof(detail::ForceBalanceExponentBlock),
            "retained exponent blocks");
    const std::uint64_t scaled_bytes =
        detail::checked_memory_multiply(
            block_count,
            sizeof(detail::ForceBalanceScaledBlock),
            "retained scaled blocks");
    const std::uint64_t retained_bytes = detail::checked_memory_add(
        exponent_bytes,
        scaled_bytes,
        "complete retained blocks");

    const std::uint64_t rank_count = mpi_active ? mpi_ranks : 0;
    const std::uint64_t gathered_value_count =
        detail::checked_memory_multiply(
            rank_count, 4, "MPI gathered value count");
    const std::uint64_t gathered_value_bytes =
        detail::checked_memory_multiply(
            gathered_value_count,
            sizeof(long double),
            "MPI gathered values");
    const std::uint64_t gathered_count_bytes =
        detail::checked_memory_multiply(
            rank_count,
            sizeof(std::uint64_t),
            "MPI gathered particle counts");
    const std::uint64_t partial_bytes =
        detail::checked_memory_multiply(
            rank_count,
            sizeof(ForceBalanceScaledSums),
            "MPI force-balance partials");
    const std::uint64_t gather_bytes = detail::checked_memory_add(
        detail::checked_memory_add(
            gathered_value_bytes,
            gathered_count_bytes,
            "MPI gathered values and counts"),
        partial_bytes,
        "complete MPI force-balance gather");

    return ForceBalanceMemoryPlan{
        particle_count,
        block_count,
        exponent_bytes,
        scaled_bytes,
        retained_bytes,
        rank_count,
        gathered_value_count,
        gathered_value_bytes,
        gathered_count_bytes,
        partial_bytes,
        gather_bytes,
        detail::checked_memory_add(
            retained_bytes,
            gather_bytes,
            "force-balance sample peak")};
}

inline int force_balance_common_exponent(
    std::span<const core::Real> masses,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    return detail::force_balance_common_exponent_impl(
        masses.size(),
        [&](std::size_t index) { return masses[index]; },
        acceleration_x,
        acceleration_y,
        acceleration_z);
}

inline int force_balance_common_exponent(
    core::Real uniform_mass,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    if (!std::isfinite(uniform_mass) || uniform_mass <= 0.0) {
        throw std::invalid_argument(
            "Force-balance uniform particle mass must be finite and positive");
    }
    return detail::force_balance_common_exponent_impl(
        acceleration_x.size(),
        [uniform_mass](std::size_t) { return uniform_mass; },
        acceleration_x,
        acceleration_y,
        acceleration_z);
}

inline ForceBalanceScaledSums compute_force_balance_scaled_sums(
    std::span<const core::Real> masses,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z,
    int common_exponent) {
    return detail::compute_force_balance_scaled_sums_impl(
        masses.size(),
        [&](std::size_t index) { return masses[index]; },
        acceleration_x,
        acceleration_y,
        acceleration_z,
        common_exponent);
}

inline ForceBalanceScaledSums compute_force_balance_scaled_sums(
    core::Real uniform_mass,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z,
    int common_exponent) {
    if (!std::isfinite(uniform_mass) || uniform_mass <= 0.0) {
        throw std::invalid_argument(
            "Force-balance uniform particle mass must be finite and positive");
    }
    return detail::compute_force_balance_scaled_sums_impl(
        acceleration_x.size(),
        [uniform_mass](std::size_t) { return uniform_mass; },
        acceleration_x,
        acceleration_y,
        acceleration_z,
        common_exponent);
}

inline ForceBalanceScaledSums combine_force_balance_scaled_sums(
    std::span<const ForceBalanceScaledSums> partials) {
    if (partials.empty()) {
        throw std::invalid_argument(
            "Force-balance distributed reduction requires at least one partial");
    }

    const int common_exponent = partials.front().common_exponent;
    detail::NeumaierAccumulator net_x;
    detail::NeumaierAccumulator net_y;
    detail::NeumaierAccumulator net_z;
    detail::NeumaierAccumulator force_sum;
    std::uint64_t particle_count = 0;
    for (const auto& partial : partials) {
        if (partial.common_exponent != common_exponent
            || !std::isfinite(partial.scaled_force_x)
            || !std::isfinite(partial.scaled_force_y)
            || !std::isfinite(partial.scaled_force_z)
            || !std::isfinite(partial.scaled_denominator)
            || partial.scaled_denominator < 0.0L) {
            throw std::invalid_argument(
                "Force-balance distributed partial is invalid or uses a different exponent");
        }
        if (partial.particle_count
            > std::numeric_limits<std::uint64_t>::max() - particle_count) {
            throw std::overflow_error(
                "Force-balance distributed particle count exceeds uint64 range");
        }
        particle_count += partial.particle_count;
        net_x.add(partial.scaled_force_x);
        net_y.add(partial.scaled_force_y);
        net_z.add(partial.scaled_force_z);
        force_sum.add(partial.scaled_denominator);
    }

    return ForceBalanceScaledSums{
        common_exponent,
        net_x.value(),
        net_y.value(),
        net_z.value(),
        force_sum.value(),
        particle_count};
}

inline ForceBalanceDiagnostics finalize_force_balance_diagnostics(
    const ForceBalanceScaledSums& sums) {
    if (!std::isfinite(sums.scaled_force_x)
        || !std::isfinite(sums.scaled_force_y)
        || !std::isfinite(sums.scaled_force_z)
        || !std::isfinite(sums.scaled_denominator)
        || sums.scaled_denominator < 0.0L
        || (sums.particle_count == 0
            && (sums.scaled_force_x != 0.0L
                || sums.scaled_force_y != 0.0L
                || sums.scaled_force_z != 0.0L
                || sums.scaled_denominator != 0.0L))) {
        throw std::invalid_argument(
            "Force-balance scaled sums are not a valid physical aggregate");
    }

    ForceBalanceDiagnostics diagnostics;
    if (sums.scaled_denominator == 0.0L) {
        if (sums.scaled_force_x != 0.0L
            || sums.scaled_force_y != 0.0L
            || sums.scaled_force_z != 0.0L) {
            throw std::logic_error(
                "Force-balance zero denominator has a nonzero numerator");
        }
        return diagnostics;
    }
    if (sums.common_exponent == std::numeric_limits<int>::min()) {
        throw std::logic_error(
            "Force-balance nonzero aggregate is missing a common exponent");
    }

    const long double net_norm = core::scale_safe_norm3(
        sums.scaled_force_x, sums.scaled_force_y, sums.scaled_force_z);
    if (!std::isfinite(net_norm) || net_norm < 0.0L) {
        throw std::overflow_error(
            "Force-balance net-force norm is non-finite");
    }

    long double ratio = net_norm / sums.scaled_denominator;
    if (!std::isfinite(ratio) || ratio < 0.0L) {
        throw std::overflow_error(
            "Force-balance relative residual is non-finite");
    }
    const long double upper_bound_tolerance = 64.0L
        * std::numeric_limits<long double>::epsilon()
        * std::max<long double>(
            1.0L, static_cast<long double>(sums.particle_count));
    if (ratio > 1.0L + upper_bound_tolerance) {
        throw std::logic_error(
            "Force-balance relative residual violates the triangle inequality");
    }
    ratio = std::min(ratio, 1.0L);
    diagnostics.relative_residual = static_cast<core::Real>(ratio);
    if (!std::isfinite(diagnostics.relative_residual)) {
        throw std::overflow_error(
            "Force-balance relative residual is not representable");
    }
    return diagnostics;
}

inline ForceBalanceDiagnostics force_balance_diagnostics_from_scaled_sums(
    const ForceBalanceScaledSums& sums) {
    return finalize_force_balance_diagnostics(sums);
}

inline ForceBalanceDiagnostics compute_force_balance_diagnostics(
    std::span<const core::Real> masses,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    const int common_exponent = force_balance_common_exponent(
        masses, acceleration_x, acceleration_y, acceleration_z);
    return finalize_force_balance_diagnostics(
        compute_force_balance_scaled_sums(
            masses,
            acceleration_x,
            acceleration_y,
            acceleration_z,
            common_exponent));
}

// Uniform-mass simulations do not need an N-element temporary mass array. The
// common mass factor cancels analytically from R_F, but it remains an admitted
// physical input and is therefore validated as finite and positive.
inline ForceBalanceDiagnostics compute_force_balance_diagnostics(
    core::Real uniform_mass,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    const int common_exponent = force_balance_common_exponent(
        uniform_mass, acceleration_x, acceleration_y, acceleration_z);
    return finalize_force_balance_diagnostics(
        compute_force_balance_scaled_sums(
            uniform_mass,
            acceleration_x,
            acceleration_y,
            acceleration_z,
            common_exponent));
}

} // namespace cosmo_nbody::validation
