#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <span>

namespace cosmo_nbody::halo::detail {

inline std::optional<math::ScaledPositiveProduct> so_scaled_density_at_radius(
    core::Real enclosed_mass,
    core::Real radius) noexcept {
    if (!std::isfinite(enclosed_mass) || enclosed_mass <= 0.0
        || !std::isfinite(radius) || radius <= 0.0) {
        return std::nullopt;
    }
    const core::Real pi = std::acos(core::Real{-1.0});
    const std::array<core::Real, 2> numerators{3.0, enclosed_mass};
    const std::array<core::Real, 5> denominators{
        4.0, pi, radius, radius, radius};
    try {
        return math::ScaledPositiveProduct::from_quotient(
            numerators, denominators, "SO enclosed density");
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// An ordering not certifiable at working precision is unresolved, not equality.
inline std::optional<bool> so_density_at_or_above_target(
    const math::ScaledPositiveProduct& density,
    const math::ScaledPositiveProduct& target) noexcept {
    try {
        return density.compare_to(target) >= 0;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

inline std::optional<core::Real> so_crossing_radius(
    core::Real enclosed_mass,
    core::Real effective_overdensity,
    core::Real reference_density) noexcept {
    if (!std::isfinite(enclosed_mass) || enclosed_mass <= 0.0
        || !std::isfinite(effective_overdensity)
        || effective_overdensity <= 0.0
        || !std::isfinite(reference_density)
        || reference_density <= 0.0) {
        return std::nullopt;
    }
    const core::Real pi = std::acos(core::Real{-1.0});
    const std::array<core::Real, 2> numerators{3.0, enclosed_mass};
    const std::array<core::Real, 4> denominators{
        4.0, pi, effective_overdensity, reference_density};
    try {
        return math::ScaledPositiveProduct::from_quotient(
            numerators, denominators, "SO crossing-radius ratio")
            .cube_root_value("SO crossing radius");
    } catch (const std::overflow_error&) {
        // A finite mathematical crossing above binary64 range is above every
        // finite representable aperture; preserve that ordering as +infinity.
        return std::numeric_limits<core::Real>::infinity();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

inline math::ScaledPositiveProduct so_target_density(
    core::Real effective_overdensity,
    core::Real reference_density) {
    const std::array<core::Real, 2> numerators{
        effective_overdensity, reference_density};
    return math::ScaledPositiveProduct::from_quotient(
        numerators,
        std::span<const core::Real>{},
        "SO target density");
}

} // namespace cosmo_nbody::halo::detail
