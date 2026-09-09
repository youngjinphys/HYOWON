#pragma once

#include "cosmo_nbody/core/types.hpp"

namespace cosmo_nbody::mesh {

// Hessian numerator for tensor-product real trigonometric interpolation at
// mesh nodes. An even-grid Nyquist cosine has zero first derivative at every
// node, so a mixed derivative vanishes if either differentiated axis is
// Nyquist. Diagonal second derivatives retain k_i^2, preserving the Laplacian
// and the Poisson-Hessian trace. Hermitian symmetry alone would also allow
// double-Nyquist mixed terms, but those violate individual-axis reflections.
// Components may be physical wave numbers or uniformly rescaled wave numbers.
inline core::Real real_fourier_hessian_numerator(
    int axis_a,
    int axis_b,
    const core::Real components[3],
    const bool nyquist[3]) noexcept {
    if (axis_a != axis_b && (nyquist[axis_a] || nyquist[axis_b])) {
        return 0.0;
    }
    return components[axis_a] * components[axis_b];
}

} // namespace cosmo_nbody::mesh
