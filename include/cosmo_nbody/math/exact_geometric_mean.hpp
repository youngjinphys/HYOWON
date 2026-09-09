// Correctly rounded geometric mean of two positive binary64 values.
//
// sqrt(lhs*rhs) is formed from the exact integer product of normalized
// IEEE-754 significands. The only binary64 rounding is the final ties-to-even
// decision.
#pragma once

#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::math {
namespace {

static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::digits == 53);
static_assert(sizeof(double) == sizeof(std::uint64_t));

struct UInt128 {
    std::uint64_t high{0};
    std::uint64_t low{0};
};

inline UInt128 multiply_u64(std::uint64_t lhs, std::uint64_t rhs) noexcept {
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
        low};
}

inline int compare_u128(UInt128 lhs, UInt128 rhs) noexcept {
    if (lhs.high < rhs.high) return -1;
    if (lhs.high > rhs.high) return 1;
    if (lhs.low < rhs.low) return -1;
    if (lhs.low > rhs.low) return 1;
    return 0;
}

inline bool square_fits(std::uint64_t root, UInt128 limit) noexcept {
    return compare_u128(multiply_u64(root, root), limit) <= 0;
}

// Normalized input significands are 53 bits. The geometric-mean significand
// product is therefore at most 107 bits after the odd-exponent adjustment, so
// its integer square root fits in 54 bits. Searching bit 53 downward never
// squares outside this two-limb product.
inline std::uint64_t isqrt_u128(UInt128 value) noexcept {
    std::uint64_t root = 0;
    for (int bit = 53; bit >= 0; --bit) {
        const std::uint64_t candidate = root | (std::uint64_t{1} << bit);
        if (square_fits(candidate, value)) root = candidate;
    }
    return root;
}

struct Decoded {
    std::uint64_t mantissa{0};
    int exponent{0};
};

inline Decoded decode_positive(double value) {
    const std::uint64_t bits =
        core::portable_bit_cast<std::uint64_t>(value);
    const int biased = static_cast<int>((bits >> 52U) & 0x7ffU);
    const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52U) - 1U);
    if (biased == 0) {
        if (fraction == 0) return {0, -1074};
        // Normalize subnormals before multiplication. Leaving a short raw
        // fraction here would make the later integer square root carry fewer
        // than 53 significant bits, so normal results from mixed
        // subnormal/normal inputs could be rounded at the wrong scale.
        const int msb = static_cast<int>(std::bit_width(fraction)) - 1;
        const int shift = 52 - msb;
        return {
            fraction << static_cast<unsigned>(shift),
            -1074 - shift};
    }
    return {
        (std::uint64_t{1} << 52U) | fraction,
        biased - 1023 - 52};
}

// Encode an already-rounded binary64 significand/exponent pair. If the
// result is subnormal, any remaining shift is exact: rounding onto the
// fixed 2^-1074 grid has already happened against the exact radicand.
inline double encode_positive_binary64(
    std::uint64_t significand,
    int result_exponent) {
    if (significand == 0) {
        throw std::overflow_error(
            "exact geometric mean underflowed binary64");
    }
    const int msb = static_cast<int>(std::bit_width(significand)) - 1;
    const int ieee_unbiased = result_exponent + msb;
    if (ieee_unbiased > 1023) {
        throw std::overflow_error(
            "exact geometric mean exceeds finite binary64");
    }
    if (ieee_unbiased >= -1022) {
        const unsigned shift = static_cast<unsigned>(52 - msb);
        const std::uint64_t mantissa = significand << shift;
        const int biased = ieee_unbiased + 1023;
        const std::uint64_t bits =
            (static_cast<std::uint64_t>(biased) << 52U)
            | (mantissa & ((std::uint64_t{1} << 52U) - 1U));
        return core::portable_bit_cast<double>(bits);
    }

    const int power = result_exponent + 1074;
    if (power < 0) {
        throw std::logic_error(
            "exact geometric mean was not rounded on the subnormal grid");
    }

    std::uint64_t integer = significand;
    if (power > 0) {
        const unsigned shift = static_cast<unsigned>(power);
        if (shift >= 64U
            || (integer << shift) >> shift != integer) {
            throw std::overflow_error(
                "exact geometric mean exceeds finite binary64");
        }
        integer <<= shift;
    }
    if (integer == 0) {
        throw std::overflow_error(
            "exact geometric mean underflowed binary64");
    }
    if (integer == (std::uint64_t{1} << 52U)) {
        return core::portable_bit_cast<double>(
            std::uint64_t{1} << 52U);
    }
    if (integer > (std::uint64_t{1} << 52U)) {
        throw std::overflow_error(
            "exact geometric mean exceeds finite binary64");
    }
    return core::portable_bit_cast<double>(integer);
}

} // namespace

inline core::Real exact_geometric_mean(core::Real lhs, core::Real rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)
        || lhs <= 0.0 || rhs <= 0.0) {
        throw std::invalid_argument(
            "exact geometric mean requires finite positive operands");
    }

    const Decoded left = decode_positive(lhs);
    const Decoded right = decode_positive(rhs);
    if (left.mantissa == 0 || right.mantissa == 0) {
        throw std::overflow_error(
            "exact geometric mean underflowed binary64");
    }

    UInt128 product = multiply_u64(left.mantissa, right.mantissa);
    int exponent = left.exponent + right.exponent;
    if ((exponent & 1) != 0) {
        // sqrt(m * 2^(2k+1)) = sqrt(2m) * 2^k. 2m has at most 107 bits.
        product.high = (product.high << 1U) | (product.low >> 63U);
        product.low <<= 1U;
        --exponent;
    }
    const int power = exponent / 2;
    const std::uint64_t root = isqrt_u128(product);

    // If power < -1074, first rounding sqrt(product) to an integer and then
    // shifting that integer onto the subnormal grid would round the same
    // mathematical value twice. Round directly from the exact radicand to
    // integer multiples of 2^-1074 instead.
    const int subnormal_grid_shift = -1074 - power;
    if (subnormal_grid_shift > 0) {
        // sqrt(product) is less than 2^54 for normalized binary64 inputs.
        // A grid spacing above 2^54 therefore rounds the positive result to
        // zero, which this fail-closed primitive reports as underflow.
        if (subnormal_grid_shift > 54) {
            throw std::overflow_error(
                "exact geometric mean underflowed binary64");
        }

        const unsigned shift =
            static_cast<unsigned>(subnormal_grid_shift);
        std::uint64_t rounded = root >> shift;
        // The midpoint between rounded and rounded+1, expressed on the
        // integer sqrt(product) scale, is exactly
        // (2*rounded+1)*2^(shift-1). Compare its square with the exact
        // radicand to implement round-to-nearest, ties-to-even once.
        const std::uint64_t midpoint =
            (2U * rounded + 1U) << (shift - 1U);
        const int relation = compare_u128(
            product, multiply_u64(midpoint, midpoint));
        if (relation > 0
            || (relation == 0 && (rounded & 1U) != 0)) {
            ++rounded;
        }
        if (rounded == 0) {
            throw std::overflow_error(
                "exact geometric mean underflowed binary64");
        }
        return encode_positive_binary64(rounded, -1074);
    }

    const UInt128 square = multiply_u64(root, root);
    const bool exact_square = square.high == product.high
        && square.low == product.low;

    std::uint64_t significand = root;
    int result_exponent = power;
    if (root >= (std::uint64_t{1} << 53U)) {
        // A 54-bit floor(root(sqrt(P))) must be rounded to a multiple of two
        // before the exponent is raised. Compare the exact radicand with the
        // square of the odd midpoint between adjacent even integer candidates.
        // This preserves round-to-nearest, ties-to-even for both exact and
        // non-square radicands; treating every non-square remainder as a sticky
        // bit is incorrect when the floor root is even.
        const std::uint64_t lower = root & ~std::uint64_t{1};
        significand = lower >> 1U;
        const std::uint64_t midpoint = lower + 1U;
        const int relation = compare_u128(
            product, multiply_u64(midpoint, midpoint));
        if (relation > 0
            || (relation == 0 && (significand & 1U) != 0)) {
            ++significand;
        }
        ++result_exponent;
        if (significand >= (std::uint64_t{1} << 53U)) {
            significand >>= 1U;
            ++result_exponent;
        }
    } else if (!exact_square) {
        // sqrt(P) > root+1/2 iff 4P > (2*root+1)^2. 4P has at most 108 bits
        // in this branch because root < 2^53.
        const UInt128 four_product{
            (product.high << 2U) | (product.low >> 62U),
            product.low << 2U};
        const UInt128 midpoint_square = multiply_u64(
            2U * root + 1U, 2U * root + 1U);
        const int relation = compare_u128(four_product, midpoint_square);
        if (relation > 0 || (relation == 0 && (root & 1U) != 0)) {
            ++significand;
            if (significand >= (std::uint64_t{1} << 53U)) {
                significand >>= 1U;
                ++result_exponent;
            }
        }
    }

    return encode_positive_binary64(significand, result_exponent);
}

} // namespace cosmo_nbody::math
