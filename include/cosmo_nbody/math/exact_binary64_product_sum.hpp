// Exact fallback for cancellation-sensitive sums of a few binary64 products.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

namespace cosmo_nbody::math {

struct ExactBinary64ProductTerm {
    std::array<double, 5> factors{};
    std::size_t factor_count{0};
};

namespace exact_binary64_product_sum_detail {

static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::radix == 2);
static_assert(std::numeric_limits<double>::digits == 53);
static_assert(std::numeric_limits<double>::max_exponent == 1024);
static_assert(std::numeric_limits<double>::min_exponent == -1021);
static_assert(sizeof(double) == sizeof(std::uint64_t));

// Five finite binary64 factors have a minimum dyadic exponent of -5*1074.
// With that common unit, their largest exact product plus a small fixed number
// of summed terms fits below bit 10492. 166 64-bit limbs provide 10624 bits.
inline constexpr int base_exponent = -5370;
inline constexpr std::size_t accumulator_limbs = 166;
inline constexpr std::size_t product_limbs = 6;

struct UInt128Words {
    std::uint64_t high{0};
    std::uint64_t low{0};
};

inline UInt128Words multiply_u64(
    std::uint64_t lhs,
    std::uint64_t rhs) noexcept {
    const std::uint64_t lhs_low = static_cast<std::uint32_t>(lhs);
    const std::uint64_t lhs_high = lhs >> 32U;
    const std::uint64_t rhs_low = static_cast<std::uint32_t>(rhs);
    const std::uint64_t rhs_high = rhs >> 32U;

    const std::uint64_t product_low_low = lhs_low * rhs_low;
    const std::uint64_t product_low_high = lhs_low * rhs_high;
    const std::uint64_t product_high_low = lhs_high * rhs_low;
    const std::uint64_t product_high_high = lhs_high * rhs_high;

    std::uint64_t low = product_low_low;
    std::uint64_t carry = 0;
    std::uint64_t previous = low;
    low += product_low_high << 32U;
    if (low < previous) ++carry;
    previous = low;
    low += product_high_low << 32U;
    if (low < previous) ++carry;

    return {
        product_high_high
            + (product_low_high >> 32U)
            + (product_high_low >> 32U)
            + carry,
        low,
    };
}

struct DecodedBinary64 {
    bool negative{false};
    std::uint64_t significand{0};
    int exponent{0};
};

inline DecodedBinary64 decode_binary64(double value) {
    // Inspect the IEEE-754 object representation directly. This validation must
    // remain correct even when a caller supplies aggressive floating-point
    // compiler flags that allow ordinary floating predicates to assume finite
    // operands.
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const bool negative = (bits >> 63U) != 0U;
    const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
    if (exponent_bits == 0x7ffU) {
        throw std::invalid_argument(
            "Exact binary64 product sum requires finite factors");
    }
    const std::uint64_t fraction =
        bits & ((std::uint64_t{1} << 52U) - 1U);
    if (exponent_bits == 0U) {
        return {negative, fraction, -1074};
    }
    return {
        negative,
        (std::uint64_t{1} << 52U) | fraction,
        static_cast<int>(exponent_bits) - 1075,
    };
}

inline void multiply_product(
    std::array<std::uint64_t, product_limbs>& product,
    std::uint64_t factor) {
    std::uint64_t carry = 0;
    for (std::size_t index = 0; index < product.size(); ++index) {
        const UInt128Words wide = multiply_u64(product[index], factor);
        const std::uint64_t low = wide.low + carry;
        const std::uint64_t carry_from_low = low < wide.low ? 1U : 0U;
        product[index] = low;
        carry = wide.high + carry_from_low;
    }
    if (carry != 0U) {
        throw std::overflow_error(
            "Exact binary64 factor product exceeded its proven limb envelope");
    }
}

using Accumulator = std::array<std::uint64_t, accumulator_limbs>;

inline void add_word(
    Accumulator& accumulator,
    std::size_t index,
    std::uint64_t word) {
    while (word != 0U) {
        if (index >= accumulator.size()) {
            throw std::overflow_error(
                "Exact binary64 product sum exceeded its proven limb envelope");
        }
        const std::uint64_t previous = accumulator[index];
        accumulator[index] = previous + word;
        word = accumulator[index] < previous ? 1U : 0U;
        ++index;
    }
}

inline void add_shifted_product(
    Accumulator& accumulator,
    const std::array<std::uint64_t, product_limbs>& product,
    std::size_t shift) {
    const std::size_t word_offset = shift / 64U;
    const unsigned bit_offset = static_cast<unsigned>(shift % 64U);
    for (std::size_t index = 0; index < product.size(); ++index) {
        const std::uint64_t word = product[index];
        if (word == 0U) continue;
        add_word(
            accumulator,
            word_offset + index,
            word << bit_offset);
        if (bit_offset != 0U) {
            add_word(
                accumulator,
                word_offset + index + 1U,
                word >> (64U - bit_offset));
        }
    }
}

inline int compare(
    const Accumulator& lhs,
    const Accumulator& rhs) noexcept {
    for (std::size_t reverse = lhs.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1U;
        if (lhs[index] < rhs[index]) return -1;
        if (lhs[index] > rhs[index]) return 1;
    }
    return 0;
}

inline Accumulator subtract(
    const Accumulator& lhs,
    const Accumulator& rhs) {
    Accumulator difference{};
    std::uint64_t borrow = 0;
    for (std::size_t index = 0; index < difference.size(); ++index) {
        const std::uint64_t rhs_word = rhs[index];
        const std::uint64_t subtrahend = rhs_word + borrow;
        const bool carry_into_subtrahend = subtrahend < rhs_word;
        const std::uint64_t lhs_word = lhs[index];
        difference[index] = lhs_word - subtrahend;
        borrow = (carry_into_subtrahend || lhs_word < subtrahend)
            ? 1U : 0U;
    }
    if (borrow != 0U) {
        throw std::logic_error(
            "Exact binary64 product subtraction violated ordering");
    }
    return difference;
}

inline std::size_t highest_set_bit(const Accumulator& value) noexcept {
    for (std::size_t reverse = value.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1U;
        const std::uint64_t word = value[index];
        if (word != 0U) {
            return index * 64U
                + static_cast<std::size_t>(63U - std::countl_zero(word));
        }
    }
    return std::numeric_limits<std::size_t>::max();
}

inline bool bit(const Accumulator& value, std::size_t position) noexcept {
    return ((value[position / 64U] >> (position % 64U)) & 1U) != 0U;
}

inline bool any_bits_below(
    const Accumulator& value,
    std::size_t exclusive_end) noexcept {
    const std::size_t complete_words = exclusive_end / 64U;
    for (std::size_t index = 0; index < complete_words; ++index) {
        if (value[index] != 0U) return true;
    }
    const unsigned remainder = static_cast<unsigned>(exclusive_end % 64U);
    if (remainder != 0U) {
        const std::uint64_t mask =
            (std::uint64_t{1} << remainder) - 1U;
        return (value[complete_words] & mask) != 0U;
    }
    return false;
}

inline std::uint64_t extract_u64(
    const Accumulator& value,
    std::size_t shift) noexcept {
    const std::size_t index = shift / 64U;
    const unsigned offset = static_cast<unsigned>(shift % 64U);
    std::uint64_t result = index < value.size()
        ? value[index] >> offset : 0U;
    if (offset != 0U && index + 1U < value.size()) {
        result |= value[index + 1U] << (64U - offset);
    }
    return result;
}

inline std::uint64_t rounded_shifted_integer(
    const Accumulator& magnitude,
    std::size_t shift) noexcept {
    std::uint64_t significand = extract_u64(magnitude, shift);
    if (shift == 0U) return significand;
    const bool guard = bit(magnitude, shift - 1U);
    const bool sticky = any_bits_below(magnitude, shift - 1U);
    if (guard && (sticky || (significand & 1U) != 0U)) {
        ++significand;
    }
    return significand;
}

inline double rounded_binary64(
    const Accumulator& magnitude,
    bool negative) {
    const std::size_t highest = highest_set_bit(magnitude);
    if (highest == std::numeric_limits<std::size_t>::max()) return 0.0;

    const long long value_exponent =
        static_cast<long long>(highest) + base_exponent;
    // Finish in the same integer domain as the exact accumulator. ldexp() and
    // floating predicates depend on rounding/FTZ modes and compiler finite-math
    // assumptions: they can flush a valid subnormal or hide an overflow after
    // the exact integer calculation has already succeeded.
    std::uint64_t bits = negative ? (std::uint64_t{1} << 63U) : 0U;
    if (value_exponent < -1022) {
        constexpr std::size_t subnormal_shift =
            static_cast<std::size_t>(-1074 - base_exponent);
        const std::uint64_t units = rounded_shifted_integer(
            magnitude, subnormal_shift);
        if (units == 0U) {
            throw std::underflow_error(
                "Exact binary64 product sum is nonzero but not representable");
        }
        // units <= 2^52; the upper endpoint is the smallest normal value.
        bits |= units;
    } else {
        std::size_t shift = highest > 52U ? highest - 52U : 0U;
        std::uint64_t significand = rounded_shifted_integer(
            magnitude, shift);
        if (significand == (std::uint64_t{1} << 53U)) {
            significand >>= 1U;
            ++shift;
        }
        const long long rounded_exponent =
            base_exponent + static_cast<long long>(shift) + 52;
        if (rounded_exponent > 1023) {
            throw std::overflow_error(
                "Exact binary64 product sum is not representable");
        }
        bits |= static_cast<std::uint64_t>(rounded_exponent + 1023) << 52U;
        bits |= significand & ((std::uint64_t{1} << 52U) - 1U);
    }
    return std::bit_cast<double>(bits);
}

} // namespace exact_binary64_product_sum_detail

// Sum a small fixed set of products as the exact mathematical values of their
// binary64 factors, then round once to nearest, ties-to-even binary64,
// independently of the floating rounding and denormal modes. This is for rare
// cancellation fallbacks, not as a general high-throughput reduction.
inline double exact_binary64_product_sum(
    std::span<const ExactBinary64ProductTerm> terms) {
    using namespace exact_binary64_product_sum_detail;
    Accumulator positive{};
    Accumulator negative{};

    for (const ExactBinary64ProductTerm& term : terms) {
        if (term.factor_count == 0U
            || term.factor_count > term.factors.size()) {
            throw std::invalid_argument(
                "Exact binary64 product term factor count must lie in [1,5]");
        }

        bool product_negative = false;
        bool product_zero = false;
        int exponent = 0;
        std::array<std::uint64_t, product_limbs> product{};
        product[0] = 1U;
        for (std::size_t index = 0; index < term.factor_count; ++index) {
            const DecodedBinary64 decoded = decode_binary64(term.factors[index]);
            product_negative ^= decoded.negative;
            if (decoded.significand == 0U) {
                product_zero = true;
                continue;
            }
            // Once any factor is exactly zero the mathematical product is zero,
            // but every remaining factor must still be decoded so NaN/Inf can
            // never be hidden by factor ordering.
            if (product_zero) continue;
            exponent += decoded.exponent;
            multiply_product(product, decoded.significand);
        }
        if (product_zero) continue;

        const long long shift =
            static_cast<long long>(exponent) - base_exponent;
        if (shift < 0) {
            throw std::overflow_error(
                "Exact binary64 product fell below its proven dyadic base");
        }
        add_shifted_product(
            product_negative ? negative : positive,
            product,
            static_cast<std::size_t>(shift));
    }

    const int ordering = compare(positive, negative);
    if (ordering == 0) return 0.0;
    if (ordering > 0) {
        return rounded_binary64(subtract(positive, negative), false);
    }
    return rounded_binary64(subtract(negative, positive), true);
}

} // namespace cosmo_nbody::math
