#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::math {

// Lower bound of max(|delta|-half, 0) and upper bound of |delta|+half.
// Nearest-even subtraction can return |delta| when the exact remainder is
// smaller, which would prune a cube that still intersects the open cutoff
// ball. Directed rounding keeps the lower edge from rounding up and the
// upper edge from rounding down.
inline std::pair<core::Real, core::Real> conservative_axis_extent(
    core::Real delta,
    core::Real half) {
    if (!std::isfinite(delta) || !std::isfinite(half) || half < 0.0) {
        throw std::invalid_argument(
            "conservative axis extent requires a finite displacement and a "
            "non-negative finite half-width");
    }

    const core::Real absolute = std::abs(delta);
    core::Real minimum = std::max<core::Real>(absolute - half, 0.0);
    if (minimum > 0.0) {
        minimum = std::nextafter(minimum, 0.0);
    }

    if (half > std::numeric_limits<core::Real>::max() - absolute) {
        return {minimum, std::numeric_limits<core::Real>::infinity()};
    }
    const core::Real maximum = std::nextafter(
        absolute + half,
        std::numeric_limits<core::Real>::infinity());
    return {minimum, maximum};
}

// Exact geometric class of an axis-aligned cube versus an open cutoff ball.
// -1: every represented point is at least `radius` away (safe to prune).
//  0: the cube straddles the sphere.
// +1: every represented point is strictly inside the ball.
inline int aabb_cutoff_relation(
    core::Real dx,
    core::Real dy,
    core::Real dz,
    core::Real half,
    core::Real radius) {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)
        || !std::isfinite(half) || half < 0.0
        || !std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument(
            "AABB cutoff relation requires finite displacements, a "
            "non-negative half-width, and a non-negative radius");
    }
    if constexpr (core::detail::has_binary64_semantics<core::Real>) {
        return core::detail::exact_binary64_aabb_cutoff_relation(
            static_cast<double>(dx),
            static_cast<double>(dy),
            static_cast<double>(dz),
            static_cast<double>(half),
            static_cast<double>(radius));
    }
    const auto [min_x, max_x] = conservative_axis_extent(dx, half);
    const auto [min_y, max_y] = conservative_axis_extent(dy, half);
    const auto [min_z, max_z] = conservative_axis_extent(dz, half);
    if (!core::scale_safe_norm3_less(min_x, min_y, min_z, radius)) {
        return -1;
    }
    if (core::scale_safe_norm3_less(max_x, max_y, max_z, radius)) {
        return 1;
    }
    return 0;
}

} // namespace cosmo_nbody::math
