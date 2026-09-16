// Scale-invariant normalization for the in-run Layzer-Irvine measurement.
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cmath>
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

    const core::Real denominator = std::max(
        std::abs(potential_energy), std::abs(kinetic_energy));
    if (!(denominator > 0.0)) {
        // The normalized ratio is mathematically undefined when both energy
        // scales vanish, including the 0/0 case. Do not manufacture a zero or
        // use an epsilon threshold that can hide a large finite ratio.
        return std::nullopt;
    }

    const core::Real ratio = std::abs(residual) / denominator;
    return std::isfinite(ratio)
        ? std::optional<core::Real>{ratio}
        : std::nullopt;
}

} // namespace cosmo_nbody::validation
