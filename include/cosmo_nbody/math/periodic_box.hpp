// Periodic geometry shared by force, halo, domain, and analysis code. wrap maps
// to [0,L); minimum_image maps to [-L/2,L/2), with the exact half-box cut locus
// explicitly non-directional.
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <limits>

namespace cosmo_nbody {
namespace math {

enum class BoundaryMode {
    Open,
    PeriodicMinimumImage,
    PeriodicMesh // Used for force grids where minimum-image is implicit via FFT
};

// Map finite x to [0,L) without the overflow-prone x-L*floor(x/L) form.
inline core::Real wrap(core::Real x, core::Real L) {
    if (!std::isfinite(x) || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (x >= 0.0 && x < L) return x;

    core::Real wrapped = std::fmod(x, L);
    if (!std::isfinite(wrapped)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (wrapped < 0.0) wrapped += L;
    // Preserve the canonical half-open interval if reduction rounds to L.
    if (wrapped >= L) wrapped = 0.0;
    return wrapped == 0.0 ? core::Real{0.0} : wrapped;
}

// Map finite dx to [-L/2,L/2); +L/2 ties map to -L/2.
inline core::Real minimum_image(core::Real dx, core::Real L) {
    if (!std::isfinite(dx) || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    const core::Real half_L = 0.5 * L;
    if (!std::isfinite(half_L) || half_L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (dx >= -half_L && dx < half_L) return dx;

    core::Real wrapped = std::remainder(dx, L);
    if (!std::isfinite(wrapped)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (wrapped >= half_L) wrapped -= L;
    else if (wrapped < -half_L) wrapped += L;
    return wrapped == 0.0 ? core::Real{0.0} : wrapped;
}

// Preserve an exact representable endpoint difference; detect overflow before
// subtraction and fall back to independently wrapped endpoints when necessary.
inline core::Real minimum_image_displacement(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }

    const core::Real maximum = std::numeric_limits<core::Real>::max();
    const bool direct_overflows =
        (from < 0.0 && to > maximum + from)
        || (from > 0.0 && to < -maximum + from);
    if (!direct_overflows) {
        return minimum_image(to - from, L);
    }

    const core::Real wrapped_from = wrap(from, L);
    const core::Real wrapped_to = wrap(to, L);
    if (!std::isfinite(wrapped_from) || !std::isfinite(wrapped_to)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    return minimum_image(wrapped_to - wrapped_from, L);
}

// Exact half-box separation has two equally short directions and is therefore
// not directionally unique; invalid geometry also returns false.
inline bool minimum_image_displacement_is_directionally_unique(
    core::Real from,
    core::Real to,
    core::Real L) {
    const core::Real delta = minimum_image_displacement(from, to, L);
    const core::Real half_L = 0.5 * L;
    return std::isfinite(delta) && std::isfinite(half_L) && half_L > 0.0
        && std::abs(delta) != half_L;
}

inline core::Vec3 wrap(const core::Vec3& x, core::Real L) {
    return { wrap(x.x, L), wrap(x.y, L), wrap(x.z, L) };
}

inline core::Vec3 minimum_image(const core::Vec3& dx, core::Real L) {
    return { minimum_image(dx.x, L), minimum_image(dx.y, L), minimum_image(dx.z, L) };
}

inline core::Vec3 minimum_image_displacement(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L) {
    return {
        minimum_image_displacement(from.x, to.x, L),
        minimum_image_displacement(from.y, to.y, L),
        minimum_image_displacement(from.z, to.z, L)};
}

inline bool minimum_image_displacement_is_directionally_unique(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L) {
    return minimum_image_displacement_is_directionally_unique(from.x, to.x, L)
        && minimum_image_displacement_is_directionally_unique(from.y, to.y, L)
        && minimum_image_displacement_is_directionally_unique(from.z, to.z, L);
}

} // namespace math
} // namespace cosmo_nbody
