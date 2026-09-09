#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::math {
namespace detail {

class ExactWeightedBinary64Mean final {
public:
    static constexpr std::size_t limb_count = 68;
    using Limbs = std::array<std::uint64_t, limb_count>;

    void add(core::Real value, core::Real weight) {
        if (!std::isfinite(value)
            || !std::isfinite(weight) || weight <= 0.0) {
            throw std::invalid_argument(
                "Exact weighted mean requires finite values and positive weights");
        }
        denominator_.add(weight);
        if (value == 0.0) return;
        const Decoded value_parts = decode_positive(std::abs(value));
        const Decoded weight_parts = decode_positive(weight);
        const ProductWords product = multiply_u64_exact(
            value_parts.mantissa, weight_parts.mantissa);
        Limbs& target = std::signbit(value) ? negative_ : positive_;
        const std::size_t shift = value_parts.shift + weight_parts.shift;
        add_shifted_word(target, product.low, shift);
        add_shifted_word(target, product.high, shift + 64U);
    }

    core::Real value(core::Real rounded_total_weight, std::string_view role) const {
        if (!std::isfinite(rounded_total_weight) || rounded_total_weight <= 0.0) {
            throw std::invalid_argument(
                std::string(role) + " requires a finite positive total weight");
        }
        const core::Real canonical_total = denominator_.value();
        if (canonical_total != rounded_total_weight) {
            throw std::invalid_argument(
                std::string(role)
                + " total weight disagrees with the exact source-value sum");
        }

        const auto denominator_source = denominator_.limbs();
        Limbs denominator{};
        std::copy(
            denominator_source.begin(), denominator_source.end(),
            denominator.begin());
        const int ordering = compare(positive_, negative_);
        if (ordering == 0) return 0.0;
        const bool negative_result = ordering < 0;
        const Limbs numerator = ordering > 0
            ? subtract(positive_, negative_)
            : subtract(negative_, positive_);
        core::Real rounded = rounded_ratio(numerator, denominator, role);
        return negative_result ? -rounded : rounded;
    }

private:
    struct Decoded {
        std::uint64_t mantissa{0};
        std::size_t shift{0};
    };
    struct ProductWords {
        std::uint64_t high{0};
        std::uint64_t low{0};
    };

    static Decoded decode_positive(core::Real value) {
        static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
        static_assert(std::numeric_limits<core::Real>::is_iec559);
        static_assert(std::numeric_limits<core::Real>::radix == 2);
        static_assert(std::numeric_limits<core::Real>::digits == 53);
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(
                "Exact weighted mean requires a positive binary64 magnitude");
        }
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint64_t exponent = (bits >> 52U) & 0x7ffU;
        const std::uint64_t fraction =
            bits & ((std::uint64_t{1} << 52U) - 1U);
        if (exponent == 0) return {fraction, 0};
        return {
            (std::uint64_t{1} << 52U) | fraction,
            static_cast<std::size_t>(exponent - 1U)};
    }

    static ProductWords multiply_u64_exact(
        std::uint64_t lhs,
        std::uint64_t rhs) noexcept {
        const std::uint64_t lhs_low = static_cast<std::uint32_t>(lhs);
        const std::uint64_t lhs_high = lhs >> 32U;
        const std::uint64_t rhs_low = static_cast<std::uint32_t>(rhs);
        const std::uint64_t rhs_high = rhs >> 32U;
        const std::uint64_t low_low = lhs_low * rhs_low;
        const std::uint64_t low_high = lhs_low * rhs_high;
        const std::uint64_t high_low = lhs_high * rhs_low;
        const std::uint64_t high_high = lhs_high * rhs_high;

        std::uint64_t low = low_low;
        std::uint64_t carry = 0;
        std::uint64_t previous = low;
        low += low_high << 32U;
        if (low < previous) ++carry;
        previous = low;
        low += high_low << 32U;
        if (low < previous) ++carry;
        return {
            high_high + (low_high >> 32U) + (high_low >> 32U) + carry,
            low};
    }

    static void add_word(Limbs& target, std::size_t index, std::uint64_t word) {
        while (word != 0) {
            if (index >= target.size()) {
                throw std::overflow_error(
                    "Exact weighted mean fixed accumulator overflow");
            }
            const std::uint64_t previous = target[index];
            target[index] += word;
            word = target[index] < previous ? 1U : 0U;
            ++index;
        }
    }

    static void add_shifted_word(
        Limbs& target,
        std::uint64_t word,
        std::size_t shift) {
        if (word == 0) return;
        const std::size_t index = shift / 64U;
        const unsigned offset = static_cast<unsigned>(shift % 64U);
        add_word(target, index, word << offset);
        if (offset != 0) {
            add_word(target, index + 1U, word >> (64U - offset));
        }
    }

    static int compare(const Limbs& lhs, const Limbs& rhs) noexcept {
        for (std::size_t reverse = limb_count; reverse > 0; --reverse) {
            const std::size_t index = reverse - 1;
            if (lhs[index] < rhs[index]) return -1;
            if (lhs[index] > rhs[index]) return 1;
        }
        return 0;
    }

    static Limbs subtract(const Limbs& lhs, const Limbs& rhs) {
        Limbs result{};
        bool borrow = false;
        for (std::size_t index = 0; index < limb_count; ++index) {
            const std::uint64_t first = lhs[index] - rhs[index];
            const bool first_borrow = lhs[index] < rhs[index];
            const std::uint64_t second =
                first - static_cast<std::uint64_t>(borrow);
            const bool second_borrow =
                first < static_cast<std::uint64_t>(borrow);
            result[index] = second;
            borrow = first_borrow || second_borrow;
        }
        if (borrow) {
            throw std::logic_error(
                "Exact weighted mean magnitude subtraction underflowed");
        }
        return result;
    }

    static std::size_t highest_set_bit(const Limbs& value) noexcept {
        for (std::size_t reverse = limb_count; reverse > 0; --reverse) {
            const std::size_t index = reverse - 1;
            const std::uint64_t word = value[index];
            if (word != 0) {
                return index * 64U
                    + static_cast<std::size_t>(63U - std::countl_zero(word));
            }
        }
        return no_bit;
    }

    static Limbs shift_left(const Limbs& value, std::size_t shift) {
        Limbs result{};
        const std::size_t word_shift = shift / 64U;
        const unsigned bit_shift = static_cast<unsigned>(shift % 64U);
        for (std::size_t source = 0; source < limb_count; ++source) {
            const std::uint64_t word = value[source];
            if (word == 0) continue;
            const std::size_t target = source + word_shift;
            if (target >= limb_count) {
                throw std::overflow_error(
                    "Exact weighted mean comparison shift exceeded its envelope");
            }
            add_word(result, target, word << bit_shift);
            if (bit_shift != 0) {
                add_word(result, target + 1U, word >> (64U - bit_shift));
            }
        }
        return result;
    }

    static Limbs multiply_small(const Limbs& value, std::uint64_t factor) {
        Limbs result{};
        if (factor == 0) return result;
        for (std::size_t index = 0; index < limb_count; ++index) {
            if (value[index] == 0) continue;
            const ProductWords product = multiply_u64_exact(value[index], factor);
            add_word(result, index, product.low);
            if (product.high != 0) add_word(result, index + 1U, product.high);
        }
        return result;
    }

    static int compare_numerator_to_denominator_multiple(
        const Limbs& numerator,
        const Limbs& denominator,
        std::size_t denominator_shift,
        std::uint64_t multiplier) {
        return compare(
            numerator,
            shift_left(multiply_small(denominator, multiplier), denominator_shift));
    }

    static int compare_double_numerator_to_odd_half_step(
        const Limbs& numerator,
        const Limbs& denominator,
        std::size_t denominator_shift,
        std::uint64_t odd_multiplier) {
        return compare(
            shift_left(numerator, 1U),
            shift_left(
                multiply_small(denominator, odd_multiplier),
                denominator_shift));
    }

    static int floor_log2_ratio(
        const Limbs& numerator,
        const Limbs& denominator) {
        const std::size_t numerator_high = highest_set_bit(numerator);
        const std::size_t denominator_high = highest_set_bit(denominator);
        if (numerator_high == no_bit || denominator_high == no_bit) {
            throw std::logic_error(
                "Exact weighted mean ratio requires non-zero integers");
        }
        const long long delta = static_cast<long long>(numerator_high)
            - static_cast<long long>(denominator_high);
        if (delta >= 0) {
            const int comparison = compare(
                numerator,
                shift_left(denominator, static_cast<std::size_t>(delta)));
            return comparison >= 0
                ? static_cast<int>(delta)
                : static_cast<int>(delta - 1);
        }
        const int comparison = compare(
            shift_left(numerator, static_cast<std::size_t>(-delta)),
            denominator);
        return comparison >= 0
            ? static_cast<int>(delta)
            : static_cast<int>(delta - 1);
    }

    static std::uint64_t extract_u64(
        const Limbs& value,
        std::size_t shift) noexcept {
        const std::size_t index = shift / 64U;
        const unsigned offset = static_cast<unsigned>(shift % 64U);
        if (index >= limb_count) return 0;
        std::uint64_t result = value[index] >> offset;
        if (offset != 0 && index + 1U < limb_count) {
            result |= value[index + 1U] << (64U - offset);
        }
        return result;
    }

    static long double leading_mantissa(
        const Limbs& value,
        std::size_t highest) noexcept {
        if (highest < 63U) {
            return std::ldexp(
                static_cast<long double>(value[0]),
                -static_cast<int>(highest));
        }
        const std::size_t shift = highest - 63U;
        const std::uint64_t leading = extract_u64(value, shift);
        return std::ldexp(static_cast<long double>(leading), -63);
    }

    static std::uint64_t floor_scaled_quotient_binary_search(
        const Limbs& numerator,
        const Limbs& denominator,
        std::size_t denominator_shift) {
        constexpr std::uint64_t upper_bound = std::uint64_t{1} << 53U;
        std::uint64_t low = 0;
        std::uint64_t high = upper_bound;
        while (low < high) {
            const std::uint64_t middle = low + (high - low + 1U) / 2U;
            if (compare_numerator_to_denominator_multiple(
                    numerator, denominator, denominator_shift, middle) >= 0) {
                low = middle;
            } else {
                high = middle - 1U;
            }
        }
        return low;
    }

    static std::uint64_t floor_scaled_quotient(
        const Limbs& numerator,
        const Limbs& denominator,
        std::size_t denominator_shift) {
        constexpr std::uint64_t upper_bound = std::uint64_t{1} << 53U;
        const std::size_t numerator_high = highest_set_bit(numerator);
        const std::size_t denominator_high = highest_set_bit(denominator);
        const long long exponent_difference =
            static_cast<long long>(numerator_high)
            - static_cast<long long>(denominator_high)
            - static_cast<long long>(denominator_shift);
        const long double approximate = std::ldexp(
            leading_mantissa(numerator, numerator_high)
                / leading_mantissa(denominator, denominator_high),
            static_cast<int>(exponent_difference));
        std::uint64_t quotient = 0;
        if (std::isfinite(approximate) && approximate > 0.0L) {
            quotient = approximate >= static_cast<long double>(upper_bound)
                ? upper_bound
                : static_cast<std::uint64_t>(std::floor(approximate));
        }

        // Top-bit division is only a seed. Every adjustment is authorized by
        // exact integer comparison, so long-double width cannot change the
        // returned quotient. Eight ulps is deliberately generous for a seed
        // formed from 64 leading bits; a bounded binary-search fallback keeps
        // the result exact on unusual ABIs or adversarial bit patterns.
        for (int correction = 0; correction < 8; ++correction) {
            if (compare_numerator_to_denominator_multiple(
                    numerator, denominator, denominator_shift, quotient) < 0) {
                if (quotient == 0) break;
                --quotient;
                continue;
            }
            if (quotient < upper_bound
                && compare_numerator_to_denominator_multiple(
                    numerator, denominator, denominator_shift, quotient + 1U) >= 0) {
                ++quotient;
                continue;
            }
            return quotient;
        }
        return floor_scaled_quotient_binary_search(
            numerator, denominator, denominator_shift);
    }

    static core::Real rounded_ratio(
        const Limbs& numerator,
        const Limbs& denominator,
        std::string_view role) {
        const int ratio_log2 = floor_log2_ratio(numerator, denominator);
        const int value_exponent = ratio_log2 - 1074;
        if (value_exponent > std::numeric_limits<core::Real>::max_exponent - 1) {
            throw std::overflow_error(
                std::string(role) + " exact weighted mean exceeds binary64");
        }
        const int output_exponent = std::max(value_exponent, -1022);
        const std::size_t denominator_shift = static_cast<std::size_t>(
            output_exponent + 1022);
        std::uint64_t significand = floor_scaled_quotient(
            numerator, denominator, denominator_shift);
        const std::uint64_t odd_half_step = 2U * significand + 1U;
        const int half_comparison = compare_double_numerator_to_odd_half_step(
            numerator, denominator, denominator_shift, odd_half_step);
        if (half_comparison > 0
            || (half_comparison == 0 && (significand & 1U) != 0)) {
            ++significand;
        }

        if (output_exponent == -1022) {
            if (significand > (std::uint64_t{1} << 53U)) {
                throw std::logic_error(
                    "Exact weighted mean minimum-binade rounding exceeded its envelope");
            }
            return std::ldexp(static_cast<core::Real>(significand), -1074);
        }

        int normalized_exponent = output_exponent;
        if (significand == (std::uint64_t{1} << 53U)) {
            significand >>= 1U;
            ++normalized_exponent;
        }
        if (normalized_exponent > 1023) {
            throw std::overflow_error(
                std::string(role) + " exact weighted mean rounds outside binary64");
        }
        if (significand < (std::uint64_t{1} << 52U)
            || significand >= (std::uint64_t{1} << 53U)) {
            throw std::logic_error(
                "Exact weighted mean produced a non-normalized significand");
        }
        return std::ldexp(
            static_cast<core::Real>(significand),
            normalized_exponent - 52);
    }

    static constexpr std::size_t no_bit =
        std::numeric_limits<std::size_t>::max();

    ExactPositiveDoubleSum denominator_;
    Limbs positive_{};
    Limbs negative_{};
};

} // namespace detail

// Correctly rounded positive-weight binary64 mean. The source values and weights
// are interpreted as exact IEEE-754 numbers; no weight*value product and no
// reciprocal of the rounded total weight is ever materialized. A fixed integer
// accumulator carries the exact signed first moment and exact positive mass sum,
// and the final rational quotient is rounded once, directly to binary64 using
// round-to-nearest, ties-to-even. The total_weight argument is retained as a
// caller-binding check only; it must equal the canonical correctly rounded sum
// of the supplied source weights.
template <typename ValueAt, typename WeightAt>
core::Real range_safe_weighted_mean(
    std::size_t count,
    ValueAt&& value_at,
    WeightAt&& weight_at,
    core::Real total_weight,
    std::string_view role) {
    if (count == 0) {
        throw std::invalid_argument(
            std::string(role) + " requires at least one value");
    }
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                std::string(role) + " value count exceeds the exact accumulator envelope");
        }
    }

    detail::ExactWeightedBinary64Mean accumulator;
    core::Real minimum = value_at(0);
    core::Real maximum = minimum;
    for (std::size_t index = 0; index < count; ++index) {
        const core::Real value = value_at(index);
        const core::Real weight = weight_at(index);
        accumulator.add(value, weight);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const core::Real result = accumulator.value(total_weight, role);
    if (!std::isfinite(result) || result < minimum || result > maximum) {
        throw std::logic_error(
            std::string(role) + " exact weighted mean escaped its convex hull");
    }
    return result;
}

} // namespace cosmo_nbody::math
