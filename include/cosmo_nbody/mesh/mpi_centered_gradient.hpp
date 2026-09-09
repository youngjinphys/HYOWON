#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/mpi_potential_halo.hpp"

#include <span>

namespace cosmo_nbody::mesh {

// Gather centered-gradient/CIC values from a compact potential slab and exact
// periodic halo planes, preserving full-mesh arithmetic grouping without a force slab.
// Validation recomputes without mutation; validate_only returns after that pass
// so MPI callers can agree collectively before publishing rank-local outputs.
void interpolate_centered_gradient3_add_slab(
    const MeshGeometry& geometry,
    const RealField& potential,
    std::span<const PotentialHaloPlane> halo_plan,
    std::span<const core::Real> halo_values,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> out_x,
    std::span<core::Real> out_y,
    std::span<core::Real> out_z,
    bool validate_only = false);

} // namespace cosmo_nbody::mesh
