#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/floating_environment.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::math {

// Composite Gauss-Legendre integration resolved to the downstream binary64
// representation. The 16-point production estimate is accompanied by an
// independent 8-point estimate on the same cells. Refinement stops only when
// (1) the 8/16 order gap and (2) the 16-point change under cell doubling are
// both no larger than one spacing of the returned binary64 value. The spacing
// is a representation target, not a scientific accuracy threshold. The order
// gap is an a-posteriori truncation indicator, not an analytic error bound.
namespace detail {

inline constexpr std::array<core::Real, 8> gauss_legendre_16_x{
    0.095012509837637440,
    0.28160355077925891,
    0.45801677765722739,
    0.61787624440264375,
    0.75540440835500303,
    0.86563120238783174,
    0.94457502307323258,
    0.98940093499164993};
inline constexpr std::array<core::Real, 8> gauss_legendre_16_w{
    0.18945061045506850,
    0.18260341504492359,
    0.16915651939500254,
    0.14959598881657673,
    0.12462897125553387,
    0.095158511682492785,
    0.062253523938647893,
    0.027152459411754095};
inline constexpr std::array<core::Real, 4> gauss_legendre_8_x{
    0.18343464249564980,
    0.52553240991632899,
    0.79666647741362674,
    0.96028985649753623};
inline constexpr std::array<core::Real, 4> gauss_legendre_8_w{
    0.36268378337836198,
    0.31370664587788729,
    0.22238103445337447,
    0.10122853629037626};

inline void compensated_add(
    long double term,
    long double& sum,
    long double& compensation) noexcept {
    const long double updated = sum + term;
    if (std::abs(sum) >= std::abs(term)) {
        compensation += (sum - updated) + term;
    } else {
        compensation += (term - updated) + sum;
    }
    sum = updated;
}

inline long double binary64_spacing(core::Real value, std::string_view role) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            std::string(role) + " quadrature spacing received a non-finite value");
    }
    const core::Real upward = std::nextafter(
        value, std::numeric_limits<core::Real>::infinity());
    const core::Real downward = std::nextafter(
        value, -std::numeric_limits<core::Real>::infinity());
    long double spacing = 0.0L;
    if (std::isfinite(upward)) {
        spacing = std::max(
            spacing,
            std::abs(static_cast<long double>(upward)
                     - static_cast<long double>(value)));
    }
    if (std::isfinite(downward)) {
        spacing = std::max(
            spacing,
            std::abs(static_cast<long double>(value)
                     - static_cast<long double>(downward)));
    }
    if (!(spacing > 0.0L) || !std::isfinite(spacing)) {
        throw std::underflow_error(
            std::string(role) + " quadrature could not resolve binary64 spacing");
    }
    return spacing;
}

struct CompositeGaussLegendreEstimate {
    long double order8{0.0L};
    long double order16{0.0L};
};

template <typename Integrand>
CompositeGaussLegendreEstimate composite_gauss_legendre_8_16(
    const Integrand& integrand,
    core::Real lower,
    core::Real upper,
    std::size_t subdivisions,
    std::string_view role) {
    if (subdivisions == 0U) {
        throw std::invalid_argument(
            std::string(role) + " quadrature requires positive subdivisions");
    }
    const core::Real width = upper - lower;
    const core::Real cell_width = width / static_cast<core::Real>(subdivisions);
    const core::Real half_width = core::Real{0.5} * cell_width;
    if (!std::isfinite(width) || !(width > 0.0)
        || !std::isfinite(cell_width) || !(cell_width > 0.0)
        || !std::isfinite(half_width) || !(half_width > 0.0)) {
        throw std::underflow_error(
            std::string(role) + " quadrature cells are not representable");
    }

    long double total8 = 0.0L;
    long double compensation8 = 0.0L;
    long double total16 = 0.0L;
    long double compensation16 = 0.0L;
    for (std::size_t cell = 0; cell < subdivisions; ++cell) {
        const core::Real center = lower
            + (static_cast<core::Real>(cell) + core::Real{0.5}) * cell_width;
        if (!std::isfinite(center)) {
            throw std::overflow_error(
                std::string(role) + " quadrature cell center is invalid");
        }

        long double local8 = 0.0L;
        long double local8_compensation = 0.0L;
        for (std::size_t index = 0; index < gauss_legendre_8_x.size(); ++index) {
            const core::Real offset = half_width * gauss_legendre_8_x[index];
            const core::Real left = center - offset;
            const core::Real right = center + offset;
            if (!(lower < left && left < right && right < upper)) {
                throw std::underflow_error(
                    std::string(role)
                    + " 8-point quadrature nodes collapsed at binary64 resolution");
            }
            const core::Real left_value = integrand(left);
            const core::Real right_value = integrand(right);
            if (!std::isfinite(left_value) || !std::isfinite(right_value)) {
                throw std::overflow_error(
                    std::string(role) + " quadrature integrand is non-finite");
            }
            compensated_add(
                static_cast<long double>(gauss_legendre_8_w[index])
                    * (static_cast<long double>(left_value)
                       + static_cast<long double>(right_value)),
                local8,
                local8_compensation);
        }

        long double local16 = 0.0L;
        long double local16_compensation = 0.0L;
        for (std::size_t index = 0; index < gauss_legendre_16_x.size(); ++index) {
            const core::Real offset = half_width * gauss_legendre_16_x[index];
            const core::Real left = center - offset;
            const core::Real right = center + offset;
            if (!(lower < left && left < right && right < upper)) {
                throw std::underflow_error(
                    std::string(role)
                    + " 16-point quadrature nodes collapsed at binary64 resolution");
            }
            const core::Real left_value = integrand(left);
            const core::Real right_value = integrand(right);
            if (!std::isfinite(left_value) || !std::isfinite(right_value)) {
                throw std::overflow_error(
                    std::string(role) + " quadrature integrand is non-finite");
            }
            compensated_add(
                static_cast<long double>(gauss_legendre_16_w[index])
                    * (static_cast<long double>(left_value)
                       + static_cast<long double>(right_value)),
                local16,
                local16_compensation);
        }

        compensated_add(
            static_cast<long double>(half_width)
                * (local8 + local8_compensation),
            total8,
            compensation8);
        compensated_add(
            static_cast<long double>(half_width)
                * (local16 + local16_compensation),
            total16,
            compensation16);
    }

    const long double estimate8 = total8 + compensation8;
    const long double estimate16 = total16 + compensation16;
    if (!std::isfinite(estimate8) || !std::isfinite(estimate16)
        || estimate16 > static_cast<long double>(
            std::numeric_limits<core::Real>::max())
        || estimate16 < -static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(
            std::string(role) + " quadrature estimate is not representable");
    }
    return {estimate8, estimate16};
}

} // namespace detail

template <typename Integrand>
core::Real roundoff_resolved_gauss_legendre_16(
    const Integrand& integrand,
    core::Real lower,
    core::Real upper,
    std::string_view role) {
    require_strict_floating_environment(role);
    if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper)) {
        throw std::invalid_argument(
            std::string(role) + " quadrature requires finite lower < upper");
    }

    auto previous = detail::composite_gauss_legendre_8_16(
        integrand, lower, upper, 1U, role);
    std::size_t subdivisions = 2U;
    while (true) {
        const auto current = detail::composite_gauss_legendre_8_16(
            integrand, lower, upper, subdivisions, role);
        const core::Real represented = static_cast<core::Real>(current.order16);
        const long double spacing = detail::binary64_spacing(represented, role);
        const long double order_gap = std::abs(current.order16 - current.order8);
        const long double refinement_change =
            std::abs(current.order16 - previous.order16);
        if (order_gap <= spacing && refinement_change <= spacing) {
            return represented;
        }

        previous = current;
        if (subdivisions > std::numeric_limits<std::size_t>::max() / 2U) {
            throw std::overflow_error(
                std::string(role)
                + " quadrature subdivision count overflowed before roundoff resolution");
        }
        subdivisions *= 2U;
    }
}

} // namespace cosmo_nbody::math
