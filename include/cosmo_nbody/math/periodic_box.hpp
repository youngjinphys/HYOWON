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

// Unsigned distance for two canonical endpoints. Distance-only consumers must
// not first round a near-L subtraction or inherit the directional cut-locus
// convention. The shorter nonnegative route is selected before any cancellation.
// When the across-face route can be shorter, L-high is exact by Sterbenz; its
// addition to low and the direct subtraction each have one final rounding.
// This is a component distance, not an exact three-dimensional norm predicate.
inline core::Real minimum_image_distance_wrapped(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0
        || from < 0.0 || from >= L || to < 0.0 || to >= L) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    const core::Real high = from > to ? from : to;
    const core::Real low = from > to ? to : from;
    const core::Real distance = std::fmin(high - low, (L - high) + low);
    return distance == 0.0 ? core::Real{0.0} : distance;
}

// Compute candidate-reference from two already wrapped endpoints without first
// forming a near-L subtraction. At opposite box faces, subtracting the endpoints
// can round away a small exact route before minimum_image() sees it. The side of
// the half-box cut is classified from endpoint comparisons; high-half_L is exact
// by Sterbenz for high in [L/2,L). If a unique exact route rounds onto the cut
// locus, move it one representable step inward so its sign and uniqueness are not
// replaced by the exact-tie convention.
inline core::Real minimum_image_displacement_wrapped(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0
        || from < 0.0 || from >= L || to < 0.0 || to >= L) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    const core::Real half_L = 0.5 * L;
    if (!std::isfinite(half_L) || half_L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }

    const bool forward = to >= from;
    const core::Real high = forward ? to : from;
    const core::Real low = forward ? from : to;

    bool direct_shorter = false;
    bool across_shorter = false;
    if (high < half_L) {
        direct_shorter = true;
    } else {
        const core::Real threshold = high - half_L;
        if (low > threshold) direct_shorter = true;
        else if (low < threshold) across_shorter = true;
        else return -half_L; // Exact represented cut-locus tie.
    }

    const core::Real inward_half = std::nextafter(half_L, core::Real{0.0});
    core::Real magnitude = 0.0;
    core::Real result = 0.0;
    if (direct_shorter) {
        magnitude = high - low;
        if (!std::isfinite(magnitude) || magnitude < 0.0) {
            return std::numeric_limits<core::Real>::quiet_NaN();
        }
        if (magnitude >= half_L) magnitude = inward_half;
        result = forward ? magnitude : -magnitude;
    } else if (across_shorter) {
        // L-high is exact for high in [L/2,L), and the sum contains only
        // non-negative terms. This route retains tiny cross-boundary distances.
        magnitude = (L - high) + low;
        if (!std::isfinite(magnitude) || magnitude < 0.0) {
            return std::numeric_limits<core::Real>::quiet_NaN();
        }
        if (magnitude >= half_L) magnitude = inward_half;
        result = forward ? -magnitude : magnitude;
    } else {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    return result == 0.0 ? core::Real{0.0} : result;
}

// Preserve a directly representable small difference for arbitrary finite input.
// Production particle/node coordinates are canonical [0,L) values and therefore
// take the endpoint-aware path above. Noncanonical callers retain the historical
// small-difference fast path; otherwise each endpoint is wrapped independently.
inline core::Real minimum_image_displacement(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    const core::Real half_L = 0.5 * L;
    if (!std::isfinite(half_L) || half_L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }

    if (from >= 0.0 && from < L && to >= 0.0 && to < L) {
        return minimum_image_displacement_wrapped(from, to, L);
    }

    const core::Real direct = to - from;
    if (std::isfinite(direct) && direct >= -half_L && direct < half_L) {
        return direct == 0.0 ? core::Real{0.0} : direct;
    }

    const core::Real wrapped_from = wrap(from, L);
    const core::Real wrapped_to = wrap(to, L);
    if (!std::isfinite(wrapped_from) || !std::isfinite(wrapped_to)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    return minimum_image_displacement_wrapped(wrapped_from, wrapped_to, L);
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
