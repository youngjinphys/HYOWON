// Particle-lattice Fourier support for internally generated initial conditions.
//
// The force mesh may be finer than the particle lattice.  A particle lattice
// with N cells per axis cannot represent arbitrary force-mesh Fourier modes:
// sampling a mode beyond its Brillouin zone aliases it onto a lower mode.
// These helpers define the conservative, non-aliased cubic support used by
// the generated-IC path.  For even N the Nyquist planes are excluded because
// their odd-derivative convention is not unique on the particle lattice.
#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace cosmo_nbody {
namespace ic {

// Use one represented arithmetic path for Fourier magnitudes at spectrum
// admission and at mode generation. The long-double integer square is only an
// intermediate range guard; the radius is narrowed before multiplying by the
// binary64 fundamental wavenumber, matching the generated-mode convention.
inline core::Real represented_mode_wavenumber(
    std::int64_t mode_x,
    std::int64_t mode_y,
    std::int64_t mode_z,
    core::Real fundamental_wavenumber) noexcept {
    const long double mode_squared =
        static_cast<long double>(mode_x) * static_cast<long double>(mode_x)
        + static_cast<long double>(mode_y) * static_cast<long double>(mode_y)
        + static_cast<long double>(mode_z) * static_cast<long double>(mode_z);
    return static_cast<core::Real>(std::sqrt(mode_squared))
        * fundamental_wavenumber;
}

// Inclusive Cartesian coordinate bound, defined even for INT64_MIN.
bool mode_within_axis_limit(std::int64_t mode, std::size_t max_mode) noexcept;

// Return whether an integer Fourier mode is safely representable on the
// particle lattice.  The convention is a Cartesian (not spherical) cutoff so
// it exactly matches separable lattice sampling in every coordinate.
bool particle_lattice_mode_representable(
    std::int64_t mode,
    std::size_t particles_per_dimension) noexcept;

// Set all non-representable r2c Fourier coefficients to zero.  The field must
// use a mesh dimension that is an integer multiple of the particle dimension,
// which is also the LPT sampling rule.
void apply_particle_lattice_bandlimit(
    const mesh::MeshGeometry& geometry,
    std::size_t particles_per_dimension,
    mesh::ComplexField& field,
    std::optional<std::size_t> max_mode_per_axis = std::nullopt);

} // namespace ic
} // namespace cosmo_nbody
