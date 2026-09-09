#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/exact_binary64_product_sum.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace cosmo_nbody::analysis::detail {

inline core::Real spectral_norm3(
    core::Real x,
    core::Real y,
    core::Real z) noexcept {
    return core::scale_safe_norm3(x, y, z);
}

// Compute log(upper/lower) without subtracting two separately rounded logs.
// This remains non-zero for narrow positive ranges on ABIs where long double
// has binary64 precision, while the fallback covers ratios too wide to form.
template <typename Floating>
inline Floating stable_positive_log_ratio(
    Floating upper,
    Floating lower) {
    static_assert(std::is_floating_point_v<Floating>);
    if (!std::isfinite(upper) || !std::isfinite(lower)
        || lower <= Floating{0} || upper < lower) {
        throw std::invalid_argument(
            "Stable logarithmic ratio requires finite 0 < lower <= upper");
    }
    if (upper == lower) return Floating{0};

    const Floating relative_difference = (upper - lower) / lower;
    const Floating value = std::isfinite(relative_difference)
        ? std::log1p(relative_difference)
        : std::log(upper) - std::log(lower);
    if (!std::isfinite(value) || value <= Floating{0}) {
        throw std::overflow_error(
            "Stable logarithmic ratio is not representable");
    }
    return value;
}

inline core::Real stable_positive_log_interpolate(
    core::Real lower,
    core::Real upper,
    core::Real fraction) {
    if (!std::isfinite(lower) || !std::isfinite(upper)
        || lower <= 0.0 || upper < lower
        || !std::isfinite(fraction)
        || fraction < 0.0 || fraction > 1.0) {
        throw std::invalid_argument(
            "Stable logarithmic interpolation requires finite 0 < lower <= upper and fraction in [0,1]");
    }
    if (fraction == 0.0 || lower == upper) return lower;
    if (fraction == 1.0) return upper;

    const core::Real log_span = stable_positive_log_ratio(upper, lower);
    core::Real result = 0.0;
    if (log_span <= 1.0) {
        result = lower * std::exp(fraction * log_span);
    } else {
        const long double log_value =
            std::log(static_cast<long double>(lower))
            + static_cast<long double>(fraction)
                * static_cast<long double>(log_span);
        const long double wide = std::exp(log_value);
        const long double maximum = static_cast<long double>(
            std::numeric_limits<core::Real>::max());
        if (!std::isfinite(wide) || wide <= 0.0L || wide > maximum) {
            throw std::overflow_error(
                "Stable logarithmic interpolation is not representable");
        }
        result = static_cast<core::Real>(wide);
    }
    if (!std::isfinite(result) || result <= 0.0
        || result < lower || result > upper) {
        throw std::overflow_error(
            "Stable logarithmic interpolation escaped its admitted endpoints");
    }
    return result;
}

// One representable edge grid owns both shell assignment and publication.
// This avoids disagreeing at rounded interior edges when classification is
// independently reconstructed from logarithms at wider precision.
class PositiveLogBinGrid {
public:
    PositiveLogBinGrid(
        core::Real minimum,
        core::Real maximum,
        int bin_count) {
        if (!std::isfinite(minimum) || !std::isfinite(maximum)
            || minimum <= 0.0 || maximum <= minimum || bin_count <= 0) {
            throw std::invalid_argument(
                "Positive logarithmic bin grid requires finite 0 < minimum < maximum and positive bin_count");
        }
        const std::size_t count = static_cast<std::size_t>(bin_count);
        edges_.resize(count + 1U);
        for (std::size_t index = 0; index <= count; ++index) {
            edges_[index] = stable_positive_log_interpolate(
                minimum,
                maximum,
                static_cast<core::Real>(index)
                    / static_cast<core::Real>(count));
            if (index != 0 && !(edges_[index] > edges_[index - 1])) {
                throw std::overflow_error(
                    "Positive logarithmic bin edges are not strictly representable");
            }
        }
    }

    std::size_t bin_count() const noexcept { return edges_.size() - 1U; }

    core::Real edge(std::size_t index) const { return edges_.at(index); }

    // Bins are [edge_i, edge_(i+1)), except that the final bin includes the
    // final edge. Values outside represented support return -1.
    int inclusive_bin_index(core::Real value) const {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Positive logarithmic bin lookup requires a finite value");
        }
        if (value < edges_.front() || value > edges_.back()) return -1;
        if (value == edges_.back()) {
            return static_cast<int>(bin_count() - 1U);
        }
        const auto upper = std::upper_bound(
            edges_.begin(), edges_.end(), value);
        if (upper == edges_.begin() || upper == edges_.end()) {
            throw std::logic_error(
                "Positive logarithmic bin lookup escaped its admitted grid");
        }
        return static_cast<int>(
            static_cast<std::size_t>(upper - edges_.begin()) - 1U);
    }

private:
    std::vector<core::Real> edges_;
};

inline core::Real checked_scaled_result(
    long double coefficient,
    int exponent,
    std::string_view label) {
    const long double value = std::scalbn(coefficient, exponent);
    const long double maximum = static_cast<long double>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || (coefficient != 0.0L && value == 0.0L)
        || value < -maximum || value > maximum) {
        throw std::overflow_error(
            std::string(label) + " is not representable in core::Real");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result) || (value != 0.0L && result == 0.0)) {
        throw std::overflow_error(
            std::string(label) + " rounded outside core::Real");
    }
    return result;
}

inline long double checked_scaled_wide_result(
    long double coefficient,
    int exponent,
    std::string_view label) {
    const long double value = std::scalbn(coefficient, exponent);
    if (!std::isfinite(value) || (coefficient != 0.0L && value == 0.0L)) {
        throw std::overflow_error(
            std::string(label) + " is not representable in long double");
    }
    return value;
}

inline long double scaled_box_power_times_wide(
    core::Real box_size,
    int box_power,
    long double value,
    std::string_view label) {
    if (!std::isfinite(box_size) || box_size <= 0.0
        || box_power < 0 || box_power > 6 || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(label)
            + " requires finite positive box_size, box power in [0,6], and finite value");
    }
    if (value == 0.0L) return 0.0L;

    int box_exponent = 0;
    int value_exponent = 0;
    const long double box_fraction = std::frexp(
        static_cast<long double>(box_size), &box_exponent);
    const long double value_fraction = std::frexp(value, &value_exponent);
    long double coefficient = value_fraction;
    for (int factor = 0; factor < box_power; ++factor) {
        coefficient *= box_fraction;
    }
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    const int exponent = box_power * box_exponent
        + value_exponent + coefficient_exponent;
    return checked_scaled_wide_result(coefficient, exponent, label);
}

inline core::Real checked_real_result(
    long double value,
    std::string_view label) {
    const long double maximum = static_cast<long double>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || std::abs(value) > maximum) {
        throw std::overflow_error(
            std::string(label) + " is not representable in core::Real");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result) || (value != 0.0L && result == 0.0)) {
        throw std::overflow_error(
            std::string(label) + " rounded outside core::Real");
    }
    return result;
}

inline long double scaled_box_volume_times_wide(
    core::Real box_size,
    long double value,
    std::string_view label) {
    return scaled_box_power_times_wide(box_size, 3, value, label);
}

inline core::Real scaled_box_volume_times(
    core::Real box_size,
    long double value,
    std::string_view label) {
    return checked_real_result(
        scaled_box_volume_times_wide(box_size, value, label), label);
}

inline long double scaled_box_volume_squared_times_wide(
    core::Real box_size,
    long double value,
    std::string_view label) {
    return scaled_box_power_times_wide(box_size, 6, value, label);
}

inline core::Real scaled_box_volume_squared_times(
    core::Real box_size,
    long double value,
    std::string_view label) {
    return checked_real_result(
        scaled_box_volume_squared_times_wide(box_size, value, label), label);
}

inline core::Real scaled_box_volume_ratio(
    core::Real box_size,
    core::Real numerator,
    core::Real denominator,
    std::string_view label) {
    if (!std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(numerator) || numerator < 0.0
        || !std::isfinite(denominator) || denominator <= 0.0) {
        throw std::invalid_argument(
            std::string(label) + " requires finite positive box_size/denominator and non-negative numerator");
    }
    if (numerator == 0.0) return 0.0;

    int box_exponent = 0;
    int numerator_exponent = 0;
    int denominator_exponent = 0;
    const long double box_fraction = std::frexp(
        static_cast<long double>(box_size), &box_exponent);
    const long double numerator_fraction = std::frexp(
        static_cast<long double>(numerator), &numerator_exponent);
    const long double denominator_fraction = std::frexp(
        static_cast<long double>(denominator), &denominator_exponent);
    long double coefficient = box_fraction * box_fraction * box_fraction
        * numerator_fraction
        / (denominator_fraction * denominator_fraction);
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    return checked_scaled_result(
        coefficient,
        3 * box_exponent + numerator_exponent
            - 2 * denominator_exponent + coefficient_exponent,
        label);
}

inline core::Real checked_complex_scaled_box_volume_product(
    core::Real box_size,
    long double shape,
    core::Real lhs_scale,
    core::Real rhs_scale,
    std::string_view label) {
    if (!std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(shape)
        || !std::isfinite(lhs_scale) || lhs_scale < 0.0
        || !std::isfinite(rhs_scale) || rhs_scale < 0.0) {
        throw std::invalid_argument(
            std::string(label) + " received invalid scaled complex product inputs");
    }
    if (shape == 0.0L || lhs_scale == 0.0 || rhs_scale == 0.0) return 0.0;

    int box_exponent = 0;
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const long double box_fraction = std::frexp(
        static_cast<long double>(box_size), &box_exponent);
    const long double lhs_fraction = std::frexp(
        static_cast<long double>(lhs_scale), &lhs_exponent);
    const long double rhs_fraction = std::frexp(
        static_cast<long double>(rhs_scale), &rhs_exponent);
    long double coefficient = shape * lhs_fraction * rhs_fraction
        * box_fraction * box_fraction * box_fraction;
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    return checked_scaled_result(
        coefficient,
        lhs_exponent + rhs_exponent + 3 * box_exponent + coefficient_exponent,
        label);
}

inline bool cancellation_needs_exact_product_sum(
    long double result,
    long double absolute_term_sum) noexcept {
    if (!(absolute_term_sum > 0.0L) || !std::isfinite(absolute_term_sum)) {
        return false;
    }
    // A small fixed number of long-double products/scalings precede each
    // decision below. If the retained result is inside a conservative forward
    // rounding envelope, recompute the original binary64 polynomial exactly.
    const long double arithmetic_envelope =
        32.0L * std::numeric_limits<long double>::epsilon()
        * absolute_term_sum;
    return std::abs(result) <= arithmetic_envelope;
}

// Form |value|^2 * box_size^3 without materializing either component square.
inline core::Real scaled_box_volume_times_complex_norm_squared(
    core::Real box_size,
    const std::complex<core::Real>& value,
    std::string_view label) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
        throw std::invalid_argument(
            std::string(label) + " requires a finite complex value");
    }
    const core::Real scale = std::max(
        std::abs(value.real()), std::abs(value.imag()));
    if (scale == 0.0) return 0.0;

    const long double real_scaled =
        static_cast<long double>(value.real()) / static_cast<long double>(scale);
    const long double imag_scaled =
        static_cast<long double>(value.imag()) / static_cast<long double>(scale);
    const long double shape = real_scaled * real_scaled + imag_scaled * imag_scaled;
    return checked_complex_scaled_box_volume_product(
        box_size, shape, scale, scale, label);
}

// Form |value|^2 * box_size^3 - subtraction in a shared exponent domain. This
// preserves a representable difference even when the positive raw power itself
// lies outside binary64. Severe cancellation is recomputed from the original
// binary64 factors before any product or subtraction is rounded away.
inline core::Real scaled_box_volume_times_complex_norm_squared_minus(
    core::Real box_size,
    const std::complex<core::Real>& value,
    core::Real subtraction,
    std::string_view label) {
    if (!std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(value.real()) || !std::isfinite(value.imag())
        || !std::isfinite(subtraction) || subtraction < 0.0) {
        throw std::invalid_argument(
            std::string(label)
            + " requires finite positive box_size, a finite complex value, and finite non-negative subtraction");
    }

    const core::Real scale = std::max(
        std::abs(value.real()), std::abs(value.imag()));
    if (scale == 0.0) return subtraction == 0.0 ? 0.0 : -subtraction;

    const long double real_scaled =
        static_cast<long double>(value.real()) / static_cast<long double>(scale);
    const long double imag_scaled =
        static_cast<long double>(value.imag()) / static_cast<long double>(scale);
    const long double shape = real_scaled * real_scaled + imag_scaled * imag_scaled;

    int box_exponent = 0;
    int scale_exponent = 0;
    const long double box_fraction = std::frexp(
        static_cast<long double>(box_size), &box_exponent);
    const long double scale_fraction = std::frexp(
        static_cast<long double>(scale), &scale_exponent);
    long double raw_coefficient = shape
        * scale_fraction * scale_fraction
        * box_fraction * box_fraction * box_fraction;
    int raw_coefficient_exponent = 0;
    raw_coefficient = std::frexp(
        raw_coefficient, &raw_coefficient_exponent);
    const int raw_exponent = 2 * scale_exponent + 3 * box_exponent
        + raw_coefficient_exponent;

    if (subtraction == 0.0) {
        return checked_scaled_result(raw_coefficient, raw_exponent, label);
    }

    int subtraction_exponent = 0;
    const long double subtraction_coefficient = std::frexp(
        static_cast<long double>(subtraction), &subtraction_exponent);
    const int common_exponent = std::max(raw_exponent, subtraction_exponent);
    const long double aligned_raw = std::scalbn(
        raw_coefficient, raw_exponent - common_exponent);
    const long double aligned_subtraction = std::scalbn(
        subtraction_coefficient, subtraction_exponent - common_exponent);
    const long double difference_coefficient =
        aligned_raw - aligned_subtraction;
    if (cancellation_needs_exact_product_sum(
            difference_coefficient,
            std::abs(aligned_raw) + std::abs(aligned_subtraction))) {
        const std::array<math::ExactBinary64ProductTerm, 3> terms{{
            {{{value.real(), value.real(), box_size, box_size, box_size}}, 5U},
            {{{value.imag(), value.imag(), box_size, box_size, box_size}}, 5U},
            {{{-subtraction, 0.0, 0.0, 0.0, 0.0}}, 1U},
        }};
        return math::exact_binary64_product_sum(terms);
    }
    return checked_scaled_result(difference_coefficient, common_exponent, label);
}

// Form |value|^2 * box_size^3 / denominator^2 without materializing either
// the component square or the large numerator before the denominator acts.
inline long double scaled_box_volume_times_complex_norm_squared_ratio_wide(
    core::Real box_size,
    const std::complex<core::Real>& value,
    long double denominator,
    std::string_view label) {
    if (!std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(value.real()) || !std::isfinite(value.imag())
        || !std::isfinite(denominator) || denominator <= 0.0L) {
        throw std::invalid_argument(
            std::string(label) + " requires finite positive box_size/denominator and a finite complex value");
    }
    const core::Real scale = std::max(
        std::abs(value.real()), std::abs(value.imag()));
    if (scale == 0.0) return 0.0L;

    const long double real_scaled =
        static_cast<long double>(value.real()) / static_cast<long double>(scale);
    const long double imag_scaled =
        static_cast<long double>(value.imag()) / static_cast<long double>(scale);
    const long double shape = real_scaled * real_scaled + imag_scaled * imag_scaled;

    int box_exponent = 0;
    int scale_exponent = 0;
    int denominator_exponent = 0;
    const long double box_fraction = std::frexp(
        static_cast<long double>(box_size), &box_exponent);
    const long double scale_fraction = std::frexp(
        static_cast<long double>(scale), &scale_exponent);
    const long double denominator_fraction = std::frexp(
        denominator, &denominator_exponent);
    long double coefficient = shape
        * scale_fraction * scale_fraction
        * box_fraction * box_fraction * box_fraction
        / (denominator_fraction * denominator_fraction);
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    return checked_scaled_wide_result(
        coefficient,
        2 * scale_exponent + 3 * box_exponent
            - 2 * denominator_exponent + coefficient_exponent,
        label);
}

// Form Re(lhs * conj(rhs)) * box_size^3 without materializing the complex
// product. Severe cancellation is recomputed from the original binary64
// factors so a representable cross power is not rounded to zero upstream.
inline core::Real scaled_box_volume_times_complex_cross_real(
    core::Real box_size,
    const std::complex<core::Real>& lhs,
    const std::complex<core::Real>& rhs,
    std::string_view label) {
    if (!std::isfinite(lhs.real()) || !std::isfinite(lhs.imag())
        || !std::isfinite(rhs.real()) || !std::isfinite(rhs.imag())) {
        throw std::invalid_argument(
            std::string(label) + " requires finite complex values");
    }
    const core::Real lhs_scale = std::max(
        std::abs(lhs.real()), std::abs(lhs.imag()));
    const core::Real rhs_scale = std::max(
        std::abs(rhs.real()), std::abs(rhs.imag()));
    if (lhs_scale == 0.0 || rhs_scale == 0.0) return 0.0;

    const long double lhs_real =
        static_cast<long double>(lhs.real()) / static_cast<long double>(lhs_scale);
    const long double lhs_imag =
        static_cast<long double>(lhs.imag()) / static_cast<long double>(lhs_scale);
    const long double rhs_real =
        static_cast<long double>(rhs.real()) / static_cast<long double>(rhs_scale);
    const long double rhs_imag =
        static_cast<long double>(rhs.imag()) / static_cast<long double>(rhs_scale);
    const long double real_term = lhs_real * rhs_real;
    const long double imag_term = lhs_imag * rhs_imag;
    const long double shape = real_term + imag_term;
    if (cancellation_needs_exact_product_sum(
            shape, std::abs(real_term) + std::abs(imag_term))) {
        const std::array<math::ExactBinary64ProductTerm, 2> terms{{
            {{{lhs.real(), rhs.real(), box_size, box_size, box_size}}, 5U},
            {{{lhs.imag(), rhs.imag(), box_size, box_size, box_size}}, 5U},
        }};
        return math::exact_binary64_product_sum(terms);
    }
    return checked_complex_scaled_box_volume_product(
        box_size, shape, lhs_scale, rhs_scale, label);
}

} // namespace cosmo_nbody::analysis::detail