#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"

#include <cmath>

namespace cosmo_nbody::cosmology {

// Accept flat closure when the exact dyadic sum of the stored binary64 density
// parameters rounds to 1.0 in binary64. This preserves ordinary decimal input
// pairs such as 0.3 + 0.7 while making the rounding point explicit and
// independent of intermediate floating-point evaluation details.
inline bool has_flat_matter_lambda_closure(
    core::Real omega_m,
    core::Real omega_lambda) {
    if (!std::isfinite(omega_m) || !std::isfinite(omega_lambda)
        || omega_m < 0.0 || omega_lambda < 0.0
        || omega_m > 1.0 || omega_lambda > 1.0) {
        return false;
    }

    math::ExactPositiveDoubleSum density_sum;
    density_sum.add(omega_m);
    density_sum.add(omega_lambda);
    return density_sum.value() == core::Real{1.0};
}

} // namespace cosmo_nbody::cosmology
