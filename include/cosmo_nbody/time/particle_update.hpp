#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <cmath>
#include <limits>

namespace cosmo_nbody::time {

struct KickDriftCandidate {
    core::Real position_x{0.0};
    core::Real position_y{0.0};
    core::Real position_z{0.0};
    core::Real momentum_x{0.0};
    core::Real momentum_y{0.0};
    core::Real momentum_z{0.0};
};

inline bool finite_kick_candidate(
    core::Real momentum_x,
    core::Real momentum_y,
    core::Real momentum_z,
    core::Real acceleration_x,
    core::Real acceleration_y,
    core::Real acceleration_z,
    core::Real kick_factor,
    core::Real& next_x,
    core::Real& next_y,
    core::Real& next_z) noexcept {
    if (!std::isfinite(momentum_x) || !std::isfinite(momentum_y)
        || !std::isfinite(momentum_z) || !std::isfinite(acceleration_x)
        || !std::isfinite(acceleration_y) || !std::isfinite(acceleration_z)
        || !std::isfinite(kick_factor)) {
        return false;
    }
    next_x = momentum_x + acceleration_x * kick_factor;
    next_y = momentum_y + acceleration_y * kick_factor;
    next_z = momentum_z + acceleration_z * kick_factor;
    return std::isfinite(next_x)
        && std::isfinite(next_y)
        && std::isfinite(next_z);
}

inline core::Real add_periodic_displacement(
    core::Real position,
    core::Real displacement,
    core::Real box_size) noexcept {
    if (!std::isfinite(position) || !std::isfinite(displacement)
        || !std::isfinite(box_size) || box_size <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    // Full-box translations carry no information on the torus. Reduce the
    // displacement to its signed minimum-image representative before adding it
    // to the stored coordinate. A signed remainder preserves tiny negative
    // drifts near zero while still preventing a huge whole-box translation from
    // rounding away the sub-box coordinate (for example 0.25 + 2^53 in a unit
    // box). The final addition is bounded to O(L), avoiding otherwise needless
    // overflow and catastrophic loss of the stored position.
    const core::Real wrapped_position = math::wrap(position, box_size);
    const core::Real reduced_displacement = math::minimum_image(
        displacement, box_size);
    if (!std::isfinite(wrapped_position)
        || !std::isfinite(reduced_displacement)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    return math::wrap(wrapped_position + reduced_displacement, box_size);
}

inline bool finite_kick_drift_candidate(
    core::Real position_x,
    core::Real position_y,
    core::Real position_z,
    core::Real momentum_x,
    core::Real momentum_y,
    core::Real momentum_z,
    core::Real acceleration_x,
    core::Real acceleration_y,
    core::Real acceleration_z,
    core::Real kick_factor,
    core::Real drift_factor,
    core::Real box_size,
    KickDriftCandidate& candidate) noexcept {
    if (!std::isfinite(position_x) || !std::isfinite(position_y)
        || !std::isfinite(position_z) || !std::isfinite(drift_factor)
        || !std::isfinite(box_size) || box_size <= 0.0) {
        return false;
    }
    if (!finite_kick_candidate(
            momentum_x, momentum_y, momentum_z,
            acceleration_x, acceleration_y, acceleration_z,
            kick_factor,
            candidate.momentum_x,
            candidate.momentum_y,
            candidate.momentum_z)) {
        return false;
    }

    const core::Real drift_x = candidate.momentum_x * drift_factor;
    const core::Real drift_y = candidate.momentum_y * drift_factor;
    const core::Real drift_z = candidate.momentum_z * drift_factor;
    if (!std::isfinite(drift_x) || !std::isfinite(drift_y)
        || !std::isfinite(drift_z)) {
        return false;
    }

    candidate.position_x = add_periodic_displacement(
        position_x, drift_x, box_size);
    candidate.position_y = add_periodic_displacement(
        position_y, drift_y, box_size);
    candidate.position_z = add_periodic_displacement(
        position_z, drift_z, box_size);
    return std::isfinite(candidate.position_x)
        && std::isfinite(candidate.position_y)
        && std::isfinite(candidate.position_z);
}

} // namespace cosmo_nbody::time
