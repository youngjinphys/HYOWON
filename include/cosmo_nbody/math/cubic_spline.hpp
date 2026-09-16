#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/exact_binary64_product_sum.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace math {

class CubicSpline {
public:
    CubicSpline() = default;

    CubicSpline(const std::vector<core::Real>& x, const std::vector<core::Real>& y) {
        build(x, y);
    }

    void build(const std::vector<core::Real>& x, const std::vector<core::Real>& y) {
        if (x.size() != y.size() || x.size() < 2) {
            throw std::invalid_argument("CubicSpline requires at least 2 points of equal length.");
        }
        for (size_t i = 0; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
                throw std::invalid_argument(
                    "CubicSpline knots and ordinates must be finite.");
            }
        }
        for (size_t i = 1; i < x.size(); ++i) {
            if (!(x[i] > x[i - 1])) {
                throw std::invalid_argument("CubicSpline x values must be strictly increasing.");
            }
        }

        std::vector<core::Real> new_x(x);
        std::vector<core::Real> new_y(y);
        std::vector<core::Real> new_y2;
        // A two-knot natural spline is a line: retain only the original knots.
        if (new_x.size() == 2 || try_build_native(new_x, new_y, new_y2)) {
            x_.swap(new_x);
            y_.swap(new_y);
            y2_.swap(new_y2);
            normalized_x_.clear();
            normalized_y_.clear();
            normalized_y2_.clear();
            normalized_y_scale_ = 1.0L;
            normalized_fallback_ = false;
            return;
        }

        std::vector<long double> new_normalized_x;
        std::vector<long double> new_normalized_y;
        std::vector<long double> new_normalized_y2;
        long double new_y_scale = 1.0L;
        build_normalized(
            new_x,
            new_y,
            new_normalized_x,
            new_normalized_y,
            new_normalized_y2,
            new_y_scale);

        x_.swap(new_x);
        y_.swap(new_y);
        y2_.clear();
        normalized_x_.swap(new_normalized_x);
        normalized_y_.swap(new_normalized_y);
        normalized_y2_.swap(new_normalized_y2);
        normalized_y_scale_ = new_y_scale;
        normalized_fallback_ = true;
    }

    core::Real interpolate(core::Real x) const {
        if (x_.empty()) {
            throw std::runtime_error("CubicSpline not built.");
        }
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "CubicSpline interpolation query must be finite.");
        }

        const auto exact = std::lower_bound(x_.begin(), x_.end(), x);
        if (exact != x_.end() && *exact == x) {
            return y_[static_cast<std::size_t>(exact - x_.begin())];
        }

        if (x_.size() == 2) return interpolate_two_knots(x);

        if (normalized_fallback_) {
            return interpolate_normalized(x);
        }

        // Extrapolate with linear if out of bounds (using end derivatives)
        if (x < x_.front()) {
            const core::Real dx = x_[1] - x_[0];
            const core::Real dy = y_[1] - y_[0];
            const core::Real y1_prime =
                dy / dx - dx * (2.0 * y2_[0] + y2_[1]) / 6.0;
            return checked_result(
                y_[0] + y1_prime * (x - x_[0]));
        }
        if (x > x_.back()) {
            const size_t n = x_.size() - 1;
            const core::Real dx = x_[n] - x_[n - 1];
            const core::Real dy = y_[n] - y_[n - 1];
            const core::Real yn_prime =
                dy / dx + dx * (y2_[n - 1] + 2.0 * y2_[n]) / 6.0;
            return checked_result(
                y_[n] + yn_prime * (x - x_[n]));
        }

        auto it = std::lower_bound(x_.begin(), x_.end(), x);
        size_t khi = std::distance(x_.begin(), it);
        size_t klo = khi - 1;

        const core::Real h = x_[khi] - x_[klo];
        const core::Real a = (x_[khi] - x) / h;
        const core::Real b = (x - x_[klo]) / h;

        return checked_result(
            a * y_[klo] + b * y_[khi]
            + ((a * a * a - a) * y2_[klo]
               + (b * b * b - b) * y2_[khi]) * (h * h) / 6.0);
    }

private:
    core::Real interpolate_two_knots(core::Real x) const {
        // Evaluate ((x1-x)*y0 + (x-x0)*y1)/(x1-x0) without rounding
        // either a normalized coordinate or a cancelling numerator first.
        // Two finite binary64 factors per term fit the existing exact dyadic
        // accumulator; even the final subnormal scaling (at most 1074 bits)
        // stays below bit 8494, within its 10624-bit storage. No wider floating
        // type, dimensional product, or heap scratch is required.
        using namespace exact_binary64_product_sum_detail;
        const auto sum_products = [](auto terms) {
            Accumulator positive{};
            Accumulator negative{};
            for (const auto& [a, b] : terms) {
                const auto da = decode_binary64(a);
                const auto db = decode_binary64(b);
                const auto product = multiply_u64(da.significand, db.significand);
                std::array<std::uint64_t, product_limbs> words{};
                words[0] = product.low;
                words[1] = product.high;
                add_shifted_product(
                    da.negative != db.negative ? negative : positive,
                    words,
                    static_cast<std::size_t>(
                        da.exponent + db.exponent - base_exponent));
            }
            const bool is_negative = compare(positive, negative) < 0;
            return std::pair{
                is_negative ? subtract(negative, positive)
                            : subtract(positive, negative),
                is_negative};
        };
        const auto [numerator, negative] = sum_products(
            std::array<std::pair<core::Real, core::Real>, 4>{{
                {x_[1], y_[0]}, {-x, y_[0]},
                {x, y_[1]}, {-x_[0], y_[1]}}});
        const auto denominator = sum_products(
            std::array<std::pair<core::Real, core::Real>, 2>{{
                {x_[1], 1.0}, {-x_[0], 1.0}}}).first;
        // build() has established x1>x0, so this denominator is positive.
        const auto highest = highest_set_bit(numerator);
        if (highest == std::numeric_limits<std::size_t>::max()) return 0.0;

        const auto shifted = [](const Accumulator& value, unsigned count) {
            Accumulator result{};
            const std::size_t offset = count / 64U;
            const unsigned bits = count % 64U;
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (value[i] == 0U) continue;
                add_word(result, i + offset, value[i] << bits);
                if (bits != 0U) {
                    add_word(result, i + offset + 1U, value[i] >> (64U - bits));
                }
            }
            return result;
        };
        int exponent = static_cast<int>(highest)
            - static_cast<int>(highest_set_bit(denominator));
        const int ordering = exponent >= 0
            ? compare(numerator, shifted(denominator, static_cast<unsigned>(exponent)))
            : compare(shifted(numerator, static_cast<unsigned>(-exponent)), denominator);
        if (ordering < 0) --exponent;
        std::uint64_t result_bits = negative ? (std::uint64_t{1} << 63U) : 0U;
        if (exponent > 1023) {
            throw std::overflow_error("CubicSpline result is not representable.");
        }
        if (exponent < -1075) {
            return core::portable_bit_cast<core::Real>(result_bits);
        }

        // Integer long division at the final binary64 quantum. The quotient
        // needs at most 53 bits; 2*remainder versus divisor implements exact
        // nearest-even rounding, including the half-min-subnormal boundary.
        int unit_exponent = std::max(exponent - 52, -1074);
        Accumulator remainder = unit_exponent < 0
            ? shifted(numerator, static_cast<unsigned>(-unit_exponent))
            : numerator;
        const Accumulator divisor = unit_exponent > 0
            ? shifted(denominator, static_cast<unsigned>(unit_exponent))
            : denominator;
        std::uint64_t significand = 0;
        const int first_bit = std::min(52,
            static_cast<int>(highest_set_bit(remainder))
                - static_cast<int>(highest_set_bit(divisor)));
        for (int bit_index = first_bit; bit_index >= 0; --bit_index) {
            const auto multiple = shifted(divisor, static_cast<unsigned>(bit_index));
            if (compare(remainder, multiple) >= 0) {
                remainder = subtract(remainder, multiple);
                significand |= std::uint64_t{1} << bit_index;
            }
        }
        const int halfway = compare(shifted(remainder, 1U), divisor);
        if (halfway > 0 || (halfway == 0 && (significand & 1U) != 0U)) {
            ++significand;
        }
        if (unit_exponent == -1074 && significand <= (std::uint64_t{1} << 52U)) {
            // Subnormals and the smallest normal share this integer encoding.
            result_bits |= significand;
        } else {
            if (significand == (std::uint64_t{1} << 53U)) {
                significand >>= 1U;
                ++unit_exponent;
            }
            const int rounded_exponent = unit_exponent + 52;
            if (rounded_exponent > 1023) {
                throw std::overflow_error("CubicSpline result is not representable.");
            }
            result_bits |= static_cast<std::uint64_t>(rounded_exponent + 1023) << 52U;
            result_bits |= significand & ((std::uint64_t{1} << 52U) - 1U);
        }
        return core::portable_bit_cast<core::Real>(result_bits);
    }

    static bool try_build_native(
        const std::vector<core::Real>& x,
        const std::vector<core::Real>& y,
        std::vector<core::Real>& result_y2) {
#if LDBL_MAX_EXP <= DBL_MAX_EXP || LDBL_MIN_EXP >= DBL_MIN_EXP
        // Apple ARM64 defines long double as binary64. In that ABI, the native
        // path cannot observe a coefficient that underflows before conversion,
        // so its conversion-loss guard is ineffective. The normalized path is
        // scale-safe without requiring a wider exponent range.
        return false;
#endif
        const size_t n = x.size() - 1;
        std::vector<core::Real> y2(n + 1, 0.0);
        std::vector<core::Real> u(n, 0.0);

        y2[0] = 0.0;
        u[0] = 0.0;

        // Native interpolation later multiplies each second derivative by h^2.
        // If that square is not finite or loses normal precision, the coefficient
        // product can manufacture NaN or large scale-dependent curvature error.
        // Route the complete spline to the scale-normalized path first.
        for (size_t i = 1; i < x.size(); ++i) {
            const core::Real interval = x[i] - x[i - 1];
            const core::Real interval_squared = interval * interval;
            if (!std::isfinite(interval)
                || !std::isfinite(interval_squared)
                || (interval != 0.0
                    && (interval_squared == 0.0
                        || std::fpclassify(interval_squared) == FP_SUBNORMAL))) {
                return false;
            }
        }

        for (size_t i = 1; i < n; ++i) {
            const core::Real span = x[i + 1] - x[i - 1];
            const core::Real right_dy = y[i + 1] - y[i];
            const core::Real left_dy = y[i] - y[i - 1];
            if (!std::isfinite(span)
                || !std::isfinite(right_dy)
                || !std::isfinite(left_dy)) {
                return false;
            }
            const core::Real sig = (x[i] - x[i - 1]) / span;
            const core::Real p = sig * y2[i - 1] + 2.0;
            const core::Real right_slope =
                right_dy / (x[i + 1] - x[i]);
            const core::Real left_slope =
                left_dy / (x[i] - x[i - 1]);
            const core::Real slope_difference = right_slope - left_slope;
            if (!std::isfinite(sig) || !std::isfinite(p) || p == 0.0
                || !std::isfinite(right_slope)
                || !std::isfinite(left_slope)
                || !std::isfinite(slope_difference)
                || (right_dy != 0.0
                    && (right_slope == 0.0
                        || std::fpclassify(right_slope) == FP_SUBNORMAL))
                || (left_dy != 0.0
                    && (left_slope == 0.0
                        || std::fpclassify(left_slope) == FP_SUBNORMAL))) {
                return false;
            }

            const long double next_y2 =
                (static_cast<long double>(sig) - 1.0L)
                / static_cast<long double>(p);
            const long double next_u =
                ((static_cast<long double>(slope_difference)
                  / static_cast<long double>(span))
                    * 6.0L
                 - static_cast<long double>(sig)
                    * static_cast<long double>(u[i - 1]))
                / static_cast<long double>(p);
            if (!representable_real(next_y2) || !representable_real(next_u)) {
                return false;
            }
            const core::Real converted_y2 = static_cast<core::Real>(next_y2);
            const core::Real converted_u = static_cast<core::Real>(next_u);
            if ((next_y2 != 0.0L && converted_y2 == 0.0)
                || (next_u != 0.0L && converted_u == 0.0)
                || std::fpclassify(converted_y2) == FP_SUBNORMAL
                || std::fpclassify(converted_u) == FP_SUBNORMAL) {
                return false;
            }
            y2[i] = converted_y2;
            u[i] = converted_u;
        }

        y2[n] = 0.0;
        for (size_t i = n; i-- > 0;) {
            y2[i] = y2[i] * y2[i + 1] + u[i];
            if (!std::isfinite(y2[i])) {
                return false;
            }
        }
        result_y2.swap(y2);
        return true;
    }

    static void build_normalized(
        const std::vector<core::Real>& x,
        const std::vector<core::Real>& y,
        std::vector<long double>& normalized_x,
        std::vector<long double>& normalized_y,
        std::vector<long double>& normalized_y2,
        long double & y_scale) {
        const size_t n = x.size() - 1;
        const core::Real x_scale =
            std::max(std::abs(x.front()), std::abs(x.back()));
        int x_exponent = 0;
        if (x_scale > 0.0) {
            (void)std::frexp(x_scale, &x_exponent);
        }

        normalized_x.resize(x.size());
        const long double first_scaled =
            std::scalbn(static_cast<long double>(x.front()), -x_exponent);
        const long double last_scaled =
            std::scalbn(static_cast<long double>(x.back()), -x_exponent);
        const long double full_span = last_scaled - first_scaled;
        if (!std::isfinite(full_span) || full_span <= 0.0L) {
            throw std::overflow_error(
                "CubicSpline normalized knot span is not representable.");
        }
        for (size_t i = 0; i < x.size(); ++i) {
            const long double scaled =
                std::scalbn(static_cast<long double>(x[i]), -x_exponent);
            normalized_x[i] = (scaled - first_scaled) / full_span;
            if (!std::isfinite(normalized_x[i])
                || (i > 0 && !(normalized_x[i] > normalized_x[i - 1]))) {
                throw std::overflow_error(
                    "CubicSpline normalized knot spacing is not representable.");
            }
        }

        y_scale = 0.0L;
        for (const core::Real value : y) {
            y_scale = std::max(y_scale, std::abs(static_cast<long double>(value)));
        }
        if (y_scale == 0.0L) {
            y_scale = 1.0L;
        }

        normalized_y.resize(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            normalized_y[i] = static_cast<long double>(y[i]) / y_scale;
        }

        normalized_y2.assign(n + 1, 0.0L);
        std::vector<long double> u(n, 0.0L);
        for (size_t i = 1; i < n; ++i) {
            const long double left =
                normalized_x[i] - normalized_x[i - 1];
            const long double right =
                normalized_x[i + 1] - normalized_x[i];
            const long double span = left + right;
            const long double sig = left / span;
            const long double p = sig * normalized_y2[i - 1] + 2.0L;
            const long double slope_difference =
                (normalized_y[i + 1] - normalized_y[i]) / right
                - (normalized_y[i] - normalized_y[i - 1]) / left;
            const long double next_y2 = (sig - 1.0L) / p;
            const long double next_u =
                (6.0L * slope_difference / span - sig * u[i - 1]) / p;
            if (!std::isfinite(next_y2) || !std::isfinite(next_u)) {
                throw std::overflow_error(
                    "CubicSpline normalized coefficient construction is not representable.");
            }
            normalized_y2[i] = next_y2;
            u[i] = next_u;
        }

        for (size_t i = n; i-- > 0;) {
            normalized_y2[i] =
                normalized_y2[i] * normalized_y2[i + 1] + u[i];
            if (!std::isfinite(normalized_y2[i])) {
                throw std::overflow_error(
                    "CubicSpline normalized back substitution is not representable.");
            }
        }
    }

    core::Real interpolate_normalized(core::Real x) const {
        const long double query = normalized_coordinate(x);

        if (x < x_.front()) {
            const long double dt = normalized_x_[1] - normalized_x_[0];
            const long double derivative =
                (normalized_y_[1] - normalized_y_[0]) / dt
                - dt * (2.0L * normalized_y2_[0] + normalized_y2_[1])
                    / 6.0L;
            return checked_result(
                (normalized_y_[0]
                 + derivative * (query - normalized_x_[0]))
                * normalized_y_scale_);
        }
        if (x > x_.back()) {
            const size_t n = x_.size() - 1;
            const long double dt =
                normalized_x_[n] - normalized_x_[n - 1];
            const long double derivative =
                (normalized_y_[n] - normalized_y_[n - 1]) / dt
                + dt
                    * (normalized_y2_[n - 1]
                       + 2.0L * normalized_y2_[n])
                    / 6.0L;
            return checked_result(
                (normalized_y_[n]
                 + derivative * (query - normalized_x_[n]))
                * normalized_y_scale_);
        }

        const auto it = std::lower_bound(x_.begin(), x_.end(), x);
        const size_t khi = static_cast<size_t>(it - x_.begin());
        const size_t klo = khi - 1;
        const long double h = normalized_x_[khi] - normalized_x_[klo];
        const long double a = (normalized_x_[khi] - query) / h;
        const long double b = (query - normalized_x_[klo]) / h;
        return checked_result(
            (a * normalized_y_[klo] + b * normalized_y_[khi]
             + ((a * a * a - a) * normalized_y2_[klo]
                + (b * b * b - b) * normalized_y2_[khi])
                    * h * h / 6.0L)
            * normalized_y_scale_);
    }

    long double normalized_coordinate(core::Real x) const {
        const core::Real scale = std::max({
            std::abs(x),
            std::abs(x_.front()),
            std::abs(x_.back())});
        int exponent = 0;
        if (scale > 0.0) {
            (void)std::frexp(scale, &exponent);
        }
        const long double query_scaled =
            std::scalbn(static_cast<long double>(x), -exponent);
        const long double first_scaled =
            std::scalbn(static_cast<long double>(x_.front()), -exponent);
        const long double last_scaled =
            std::scalbn(static_cast<long double>(x_.back()), -exponent);
        const long double span = last_scaled - first_scaled;
        const long double result = (query_scaled - first_scaled) / span;
        if (!std::isfinite(result)) {
            throw std::overflow_error(
                "CubicSpline normalized interpolation coordinate is not representable.");
        }
        return result;
    }

    static bool representable_real(long double value) {
        constexpr long double maximum =
            static_cast<long double>(std::numeric_limits<core::Real>::max());
        return std::isfinite(value) && value <= maximum && value >= -maximum;
    }

    static core::Real checked_result(long double value) {
        if (!representable_real(value)) {
            throw std::overflow_error(
                "CubicSpline result is not representable.");
        }
        const core::Real converted = static_cast<core::Real>(value);
        if (!std::isfinite(converted)) {
            throw std::overflow_error(
                "CubicSpline result is not representable.");
        }
        return converted;
    }

    std::vector<core::Real> x_;
    std::vector<core::Real> y_;
    std::vector<core::Real> y2_;
    std::vector<long double> normalized_x_;
    std::vector<long double> normalized_y_;
    std::vector<long double> normalized_y2_;
    long double normalized_y_scale_{1.0L};
    bool normalized_fallback_{false};
};

} // namespace math
} // namespace cosmo_nbody
