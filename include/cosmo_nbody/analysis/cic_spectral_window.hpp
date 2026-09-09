#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cosmo_nbody::analysis::detail {

// CIC contributes sinc^2(k_i dx/2) per axis. Evaluate in long double and clamp
// only roundoff excursions to the exact bound |sinc|<=1, without ULP heuristics.
inline long double cic_window_axis_wide(
    core::Real k_component,
    core::Real cell_size) {
    if (!std::isfinite(k_component)
        || !std::isfinite(cell_size) || cell_size <= 0.0) {
        throw std::invalid_argument(
            "CIC spectral window requires finite k and positive cell size");
    }

    const long double argument = 0.5L
        * static_cast<long double>(k_component)
        * static_cast<long double>(cell_size);
    if (!std::isfinite(argument)) {
        throw std::overflow_error(
            "CIC spectral-window argument is not representable");
    }
    if (argument == 0.0L) return 1.0L;

    const long double raw_sinc = std::sin(argument) / argument;
    if (!std::isfinite(raw_sinc)) {
        throw std::overflow_error(
            "CIC spectral sinc is not representable");
    }
    const long double bounded_magnitude = std::min(
        1.0L, std::abs(raw_sinc));
    return bounded_magnitude * bounded_magnitude;
}

inline long double cic_window_3d_wide(
    core::Real kx,
    core::Real ky,
    core::Real kz,
    core::Real cell_size) {
    const long double wx = cic_window_axis_wide(kx, cell_size);
    const long double wy = cic_window_axis_wide(ky, cell_size);
    const long double wz = cic_window_axis_wide(kz, cell_size);
    const long double window = wx * wy * wz;
    if (!std::isfinite(window) || window <= 0.0L || window > 1.0L) {
        throw std::overflow_error(
            "CIC spectral window escaped its exact range (0,1]");
    }
    return window;
}

} // namespace cosmo_nbody::analysis::detail
