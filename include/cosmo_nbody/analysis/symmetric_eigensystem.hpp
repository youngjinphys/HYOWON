#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <optional>

namespace cosmo_nbody::analysis {

struct SymmetricEigenvalues3 {
    core::Real largest{0.0};
    core::Real middle{0.0};
    core::Real smallest{0.0};
};

// Scale-invariant eigensystem for a real symmetric 3x3 matrix. The matrix is
// normalized before Jacobi rotations. Rotations continue while a nonzero
// coupling has a representable rotation; no heuristic epsilon sets convergence.
// Exact zero invariants are restored only when proven by binary64 arithmetic.
SymmetricEigenvalues3 symmetric_eigenvalues_3x3(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22);

// Return the canonical unit eigenvector only when exactly one represented
// eigenvalue equals the supplied value. If the requested value is absent or is
// repeated at core::Real precision, its direction is not uniquely represented
// and std::nullopt is returned rather than using a heuristic degeneracy cutoff.
std::optional<core::Vec3> symmetric_eigenvector_3x3(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22,
    core::Real eigenvalue);

} // namespace cosmo_nbody::analysis
