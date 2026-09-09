// Scale-invariant normalization for the in-run Layzer-Irvine measurement.
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace cosmo_nbody::validation {

inline std::optional<core::Real> layzer_irvine_ratio(
    core::Real residual,
    core::Real kinetic_energy,
    core::Real potential_energy) {
    if (!std::isfinite(residual)
        || !std::isfinite(kinetic_energy)
        || !std::isfinite(potential_energy)) {
        return std::nullopt;
    }

    const core::Real abs_residual = std::abs(residual);
    const core::Real denominator = std::max(
        std::abs(potential_energy), std::abs(kinetic_energy));
    const core::Real scale = std::max(abs_residual, denominator);
    const core::Real tolerance = core::Real{64.0}
        * std::numeric_limits<core::Real>::epsilon() * scale;
    if (denominator <= tolerance) {
        if (abs_residual <= tolerance) return core::Real{0.0};
        return std::nullopt;
    }

    const core::Real ratio = abs_residual / denominator;
    return std::isfinite(ratio)
        ? std::optional<core::Real>{ratio}
        : std::nullopt;
}

} // namespace cosmo_nbody::validation
