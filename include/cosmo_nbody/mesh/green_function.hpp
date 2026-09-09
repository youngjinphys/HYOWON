// Periodic Poisson kernels selected by PMForceMethod.
// Pure PM uses the 7-point discrete Laplacian; TreePM long range uses continuum
// -k^2 with Gaussian splitting and W_CIC^-2 compensation.
// k=0 is zero; 4*pi*G normalization is applied here.
#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/pm_force_method.hpp"

#include <vector>

namespace cosmo_nbody {
namespace mesh {

class GreenFunction {
public:
    GreenFunction(const MeshGeometry& geom, PMForceMethod method);

    // In-place delta_rho(k) -> phi(k).
    void apply(ComplexField& field_k) const;

private:
    const MeshGeometry& geom_;
    PMForceMethod method_;
    std::vector<core::Real> laplacian_symbol_1d_;
    std::vector<core::Real> assignment_window_1d_;

    core::Real evaluate_kernel(std::size_t ix, std::size_t iy, std::size_t iz) const;
    core::Real W_k_1D(std::size_t i) const;
};

} // namespace mesh
} // namespace cosmo_nbody
