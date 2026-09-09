#pragma once

#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace cosmo_nbody::math {
namespace detail {

struct UInt128 {
    std::uint64_t high{0};
    std::uint64_t low{0};
};

inline bool uint128_is_zero(UInt128 value) noexcept {
    return value.high == 0 && value.low == 0;
}

inline int compare_uint128(UInt128 lhs, UInt128 rhs) noexcept {
    if (lhs.high < rhs.high) return -1;
    if (lhs.high > rhs.high) return 1;
    if (lhs.low < rhs.low) return -1;
    if (lhs.low > rhs.low) return 1;
    return 0;
}

inline unsigned uint128_bit_width(UInt128 value) noexcept {
    return value.high != 0
        ? 64U + std::bit_width(value.high)
        : std::bit_width(value.low);
}

inline UInt128 shift_left_uint128(UInt128 value, unsigned shift) noexcept {
    if (shift == 0) return value;
    if (shift >= 128) return {};
    if (shift >= 64) return {value.low << (shift - 64), 0};
    return {
        (value.high << shift) | (value.low >> (64 - shift)),
        value.low << shift};
}

inline UInt128 multiply_u64_exact(
    std::uint64_t lhs,
    std::uint64_t rhs) noexcept {
    const std::uint64_t lhs_low = static_cast<std::uint32_t>(lhs);
    const std::uint64_t lhs_high = lhs >> 32;
    const std::uint64_t rhs_low = static_cast<std::uint32_t>(rhs);
    const std::uint64_t rhs_high = rhs >> 32;

    const std::uint64_t product_low_low = lhs_low * rhs_low;
    const std::uint64_t product_low_high = lhs_low * rhs_high;
    const std::uint64_t product_high_low = lhs_high * rhs_low;
    const std::uint64_t product_high_high = lhs_high * rhs_high;

    std::uint64_t low = product_low_low;
    std::uint64_t carry = 0;

    const std::uint64_t middle_low_high = product_low_high << 32;
    std::uint64_t previous = low;
    low += middle_low_high;
    if (low < previous) ++carry;

    const std::uint64_t middle_high_low = product_high_low << 32;
    previous = low;
    low += middle_high_low;
    if (low < previous) ++carry;

    const std::uint64_t high = product_high_high
        + (product_low_high >> 32)
        + (product_high_low >> 32)
        + carry;
    return {high, low};
}

struct Dyadic {
    bool negative{false};
    UInt128 magnitude{};
    int exponent{0};
};

inline Dyadic dyadic_from_u64(
    bool negative,
    std::uint64_t significand,
    int exponent) noexcept {
    if (significand == 0) return {};
    return {negative, {0, significand}, exponent};
}

inline int compare_dyadic(Dyadic lhs, Dyadic rhs) noexcept {
    const bool lhs_zero = uint128_is_zero(lhs.magnitude);
    const bool rhs_zero = uint128_is_zero(rhs.magnitude);
    if (lhs_zero && rhs_zero) return 0;
    if (lhs_zero) return rhs.negative ? 1 : -1;
    if (rhs_zero) return lhs.negative ? -1 : 1;
    if (lhs.negative != rhs.negative) return lhs.negative ? -1 : 1;

    const unsigned lhs_width = uint128_bit_width(lhs.magnitude);
    const unsigned rhs_width = uint128_bit_width(rhs.magnitude);
    const long long lhs_top =
        static_cast<long long>(lhs.exponent) + lhs_width;
    const long long rhs_top =
        static_cast<long long>(rhs.exponent) + rhs_width;

    int magnitude_comparison = 0;
    if (lhs_top < rhs_top) {
        magnitude_comparison = -1;
    } else if (lhs_top > rhs_top) {
        magnitude_comparison = 1;
    } else if (lhs.exponent == rhs.exponent) {
        magnitude_comparison = compare_uint128(
            lhs.magnitude, rhs.magnitude);
    } else if (lhs.exponent > rhs.exponent) {
        magnitude_comparison = compare_uint128(
            shift_left_uint128(
                lhs.magnitude,
                static_cast<unsigned>(lhs.exponent - rhs.exponent)),
            rhs.magnitude);
    } else {
        magnitude_comparison = compare_uint128(
            lhs.magnitude,
            shift_left_uint128(
                rhs.magnitude,
                static_cast<unsigned>(rhs.exponent - lhs.exponent)));
    }
    return lhs.negative ? -magnitude_comparison : magnitude_comparison;
}

inline Dyadic decode_binary64(core::Real value) noexcept {
    const std::uint64_t bits = core::portable_bit_cast<std::uint64_t>(value);
    const bool negative = (bits >> 63) != 0;
    const std::uint64_t exponent_bits = (bits >> 52) & 0x7ffU;
    const std::uint64_t fraction_bits =
        bits & ((std::uint64_t{1} << 52) - 1);
    if (exponent_bits == 0) {
        return dyadic_from_u64(negative, fraction_bits, -1074);
    }
    return dyadic_from_u64(
        negative,
        (std::uint64_t{1} << 52) | fraction_bits,
        static_cast<int>(exponent_bits) - 1075);
}

inline bool binary64_significand_is_even(core::Real value) noexcept {
    return (core::portable_bit_cast<std::uint64_t>(value) & 1U) == 0;
}

inline Dyadic adjacent_midpoint(
    core::Real lhs,
    core::Real rhs) noexcept {
    Dyadic left = decode_binary64(lhs);
    Dyadic right = decode_binary64(rhs);
    if (uint128_is_zero(left.magnitude)) {
        --right.exponent;
        return right;
    }
    if (uint128_is_zero(right.magnitude)) {
        --left.exponent;
        return left;
    }

    const int common_exponent = std::min(left.exponent, right.exponent);
    const unsigned left_shift =
        static_cast<unsigned>(left.exponent - common_exponent);
    const unsigned right_shift =
        static_cast<unsigned>(right.exponent - common_exponent);
    const std::uint64_t left_magnitude =
        left.magnitude.low << left_shift;
    const std::uint64_t right_magnitude =
        right.magnitude.low << right_shift;

    if (left.negative == right.negative) {
        return dyadic_from_u64(
            left.negative,
            left_magnitude + right_magnitude,
            common_exponent - 1);
    }
    if (left_magnitude == right_magnitude) return {};
    if (left_magnitude > right_magnitude) {
        return dyadic_from_u64(
            left.negative,
            left_magnitude - right_magnitude,
            common_exponent - 1);
    }
    return dyadic_from_u64(
        right.negative,
        right_magnitude - left_magnitude,
        common_exponent - 1);
}

inline Dyadic binary64_overflow_boundary(bool negative) noexcept {
    // max_finite + 0.5 ulp(max_finite)
    //   = (2^54 - 1) * 2^970.
    return dyadic_from_u64(
        negative,
        (std::uint64_t{1} << 54) - 1,
        970);
}

struct DyadicInterval {
    Dyadic lower{};
    Dyadic upper{};
    bool lower_closed{false};
    bool upper_closed{false};
};

inline std::optional<DyadicInterval> binary64_rounding_preimage(
    core::Real value) noexcept {
    if (!std::isfinite(value)) return std::nullopt;
    if (value == 0.0) {
        Dyadic half_minimum = dyadic_from_u64(false, 1, -1075);
        Dyadic negative_half = half_minimum;
        negative_half.negative = true;
        return DyadicInterval{
            negative_half, half_minimum, true, true};
    }

    const core::Real previous = std::nextafter(
        value, -std::numeric_limits<core::Real>::infinity());
    const core::Real next = std::nextafter(
        value, std::numeric_limits<core::Real>::infinity());
    const Dyadic lower = std::isfinite(previous)
        ? adjacent_midpoint(previous, value)
        : binary64_overflow_boundary(true);
    const Dyadic upper = std::isfinite(next)
        ? adjacent_midpoint(value, next)
        : binary64_overflow_boundary(false);
    const bool midpoint_belongs_to_value =
        binary64_significand_is_even(value);
    return DyadicInterval{
        lower,
        upper,
        midpoint_belongs_to_value,
        midpoint_belongs_to_value};
}

inline Dyadic multiply_dyadic(Dyadic lhs, Dyadic rhs) noexcept {
    if (uint128_is_zero(lhs.magnitude)
        || uint128_is_zero(rhs.magnitude)) {
        return {};
    }
    // A binary64 rounding-preimage endpoint has at most 54 significant bits,
    // so an endpoint product has at most 108 and fits this two-limb integer.
    return {
        lhs.negative != rhs.negative,
        multiply_u64_exact(lhs.magnitude.low, rhs.magnitude.low),
        lhs.exponent + rhs.exponent};
}

inline std::optional<DyadicInterval> multiply_positive_interval(
    const DyadicInterval& positive,
    const DyadicInterval& other) noexcept {
    const Dyadic zero{};
    if (compare_dyadic(positive.lower, zero) <= 0) {
        return std::nullopt;
    }

    const int lower_sign = compare_dyadic(other.lower, zero);
    const int upper_sign = compare_dyadic(other.upper, zero);
    if (lower_sign >= 0) {
        return DyadicInterval{
            multiply_dyadic(positive.lower, other.lower),
            multiply_dyadic(positive.upper, other.upper),
            positive.lower_closed && other.lower_closed,
            positive.upper_closed && other.upper_closed};
    }
    if (upper_sign <= 0) {
        return DyadicInterval{
            multiply_dyadic(positive.upper, other.lower),
            multiply_dyadic(positive.lower, other.upper),
            positive.upper_closed && other.lower_closed,
            positive.lower_closed && other.upper_closed};
    }
    return DyadicInterval{
        multiply_dyadic(positive.upper, other.lower),
        multiply_dyadic(positive.upper, other.upper),
        positive.upper_closed && other.lower_closed,
        positive.upper_closed && other.upper_closed};
}

inline bool intervals_overlap(
    const DyadicInterval& lhs,
    const DyadicInterval& rhs) noexcept {
    const int lhs_upper_vs_rhs_lower =
        compare_dyadic(lhs.upper, rhs.lower);
    if (lhs_upper_vs_rhs_lower < 0) return false;
    if (lhs_upper_vs_rhs_lower == 0
        && !(lhs.upper_closed && rhs.lower_closed)) {
        return false;
    }

    const int rhs_upper_vs_lhs_lower =
        compare_dyadic(rhs.upper, lhs.lower);
    if (rhs_upper_vs_lhs_lower < 0) return false;
    if (rhs_upper_vs_lhs_lower == 0
        && !(rhs.upper_closed && lhs.lower_closed)) {
        return false;
    }
    return true;
}

} // namespace detail

inline bool matches_independently_rounded_quotient(
    core::Real rounded_numerator,
    core::Real rounded_denominator,
    core::Real rounded_quotient) noexcept {
    static_assert(
        sizeof(core::Real) == sizeof(std::uint64_t)
        && std::numeric_limits<core::Real>::radix == 2
        && std::numeric_limits<core::Real>::digits == 53
        && std::numeric_limits<core::Real>::is_iec559,
        "Independent-rounding relation currently requires IEEE-754 binary64 Real");
    if (!std::isfinite(rounded_numerator)
        || !std::isfinite(rounded_denominator)
        || rounded_denominator <= 0.0
        || !std::isfinite(rounded_quotient)) {
        return false;
    }
    const auto numerator = detail::binary64_rounding_preimage(
        rounded_numerator);
    const auto denominator = detail::binary64_rounding_preimage(
        rounded_denominator);
    const auto quotient = detail::binary64_rounding_preimage(
        rounded_quotient);
    if (!numerator.has_value()
        || !denominator.has_value()
        || !quotient.has_value()) {
        return false;
    }
    const auto product = detail::multiply_positive_interval(
        *denominator, *quotient);
    return product.has_value()
        && detail::intervals_overlap(*numerator, *product);
}

inline bool matches_independently_rounded_quotient_with_exact_denominator(
    core::Real rounded_numerator,
    core::Real exact_denominator,
    core::Real rounded_quotient) noexcept {
    static_assert(
        sizeof(core::Real) == sizeof(std::uint64_t)
        && std::numeric_limits<core::Real>::radix == 2
        && std::numeric_limits<core::Real>::digits == 53
        && std::numeric_limits<core::Real>::is_iec559,
        "Independent-rounding relation currently requires IEEE-754 binary64 Real");
    if (!std::isfinite(rounded_numerator)
        || !std::isfinite(exact_denominator)
        || exact_denominator <= 0.0
        || !std::isfinite(rounded_quotient)) {
        return false;
    }
    const auto numerator = detail::binary64_rounding_preimage(
        rounded_numerator);
    const auto quotient = detail::binary64_rounding_preimage(
        rounded_quotient);
    if (!numerator.has_value() || !quotient.has_value()) return false;

    const detail::Dyadic exact = detail::decode_binary64(exact_denominator);
    const detail::DyadicInterval denominator{
        exact, exact, true, true};
    const auto product = detail::multiply_positive_interval(
        denominator, *quotient);
    return product.has_value()
        && detail::intervals_overlap(*numerator, *product);
}

} // namespace cosmo_nbody::math
