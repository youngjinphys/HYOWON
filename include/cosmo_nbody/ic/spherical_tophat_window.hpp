#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <stdexcept>

namespace cosmo_nbody::ic {

// Fourier transform of a unit-normalized spherical top-hat:
// W(x) = 3 [sin(x) - x cos(x)] / x^3.
//
// For large finite |x|, materializing x^3 can overflow even though the true
// window, which decays as O(x^-2), remains representable. Evaluate the same
// expression through reciprocal factors instead. Near the removable
// singularity, direct subtraction loses significant bits. The even Maclaurin
// expansion through x^18 is evaluated in Horner form for |x| <= 1.6. Its first
// omitted term is x^20 / 391697223316439040000, about 3.1e-17 at the branch,
// below half a binary64 ULP for the represented window value there.
inline core::Real spherical_tophat_fourier_window(core::Real x) {
    if (!std::isfinite(x)) {
        throw std::invalid_argument(
            "spherical top-hat window argument must be finite");
    }

    const core::Real magnitude = std::abs(x);
    if (magnitude <= 1.6) {
        const core::Real x2 = x * x;
        return 1.0 + x2 * (
            -1.0 / 10.0 + x2 * (
                1.0 / 280.0 + x2 * (
                    -1.0 / 15120.0 + x2 * (
                        1.0 / 1330560.0 + x2 * (
                            -1.0 / 172972800.0 + x2 * (
                                1.0 / 31135104000.0 + x2 * (
                                    -1.0 / 7410154752000.0 + x2 * (
                                        1.0 / 2252687044608000.0 + x2 * (
                                            -1.0 / 851515702861824000.0)))))))));
    }

    const core::Real inverse_x = 1.0 / x;
    const core::Real result = 3.0
        * (std::sin(x) * inverse_x - std::cos(x))
        * inverse_x * inverse_x;
    if (!std::isfinite(result)) {
        throw std::overflow_error(
            "spherical top-hat window result is not finite");
    }
    return result;
}

} // namespace cosmo_nbody::ic
