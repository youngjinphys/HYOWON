#include "cosmo_nbody/ic/particle_lattice_bandlimit.hpp"

#include <cstdint>
#include <stdexcept>

namespace cosmo_nbody {
namespace ic {

bool mode_within_axis_limit(std::int64_t mode, std::size_t max_mode) noexcept {
    const std::uintmax_t magnitude = mode < 0
        ? static_cast<std::uintmax_t>(-(mode + 1)) + 1U
        : static_cast<std::uintmax_t>(mode);
    return magnitude <= static_cast<std::uintmax_t>(max_mode);
}

bool particle_lattice_mode_representable(
    std::int64_t mode,
    std::size_t particles_per_dimension) noexcept {
    if (particles_per_dimension == 0) return false;

    // Form the magnitude without negating INT64_MIN.  This helper is public and
    // noexcept, so its full int64_t input domain must be defined even though
    // production MeshGeometry mode numbers are much smaller.
    const std::uintmax_t magnitude = mode < 0
        ? static_cast<std::uintmax_t>(-(mode + 1)) + 1U
        : static_cast<std::uintmax_t>(mode);
    const std::uintmax_t half = static_cast<std::uintmax_t>(
        particles_per_dimension / 2);

    // For an even lattice, +/- N/2 are the same sampled Nyquist plane.  The
    // production LPT rule zeros odd derivatives there, so exclude the
    // plane rather than retain an orientation-dependent partial mode.
    if (particles_per_dimension % 2 == 0) {
        return magnitude < half;
    }
    return magnitude <= half;
}

void apply_particle_lattice_bandlimit(
    const mesh::MeshGeometry& geometry,
    std::size_t particles_per_dimension,
    mesh::ComplexField& field,
    std::optional<std::size_t> max_mode_per_axis) {
    if (particles_per_dimension == 0) {
        throw std::invalid_argument(
            "Particle-lattice bandlimit requires a positive particle dimension");
    }
    if (geometry.grid_size() % particles_per_dimension != 0) {
        throw std::invalid_argument(
            "IC mesh dimension must be an integer multiple of the particle dimension");
    }
    if (field.size() != geometry.complex_size()) {
        throw std::invalid_argument(
            "Particle-lattice bandlimit field size does not match mesh geometry");
    }
    const std::size_t limit = max_mode_per_axis.value_or(
        (particles_per_dimension - 1) / 2);
    if (limit > (particles_per_dimension - 1) / 2) {
        throw std::invalid_argument("IC Fourier support exceeds the safe particle-lattice support");
    }

    const std::size_t mesh_n = geometry.grid_size();
    const std::size_t nz_complex = mesh_n / 2 + 1;
    for (std::size_t ix = 0; ix < mesh_n; ++ix) {
        const bool keep_x = mode_within_axis_limit(
            geometry.mode_number(ix), limit);
        for (std::size_t iy = 0; iy < mesh_n; ++iy) {
            const bool keep_xy = keep_x && mode_within_axis_limit(
                geometry.mode_number(iy), limit);
            for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                if (keep_xy && mode_within_axis_limit(
                        geometry.mode_number(iz), limit)) {
                    continue;
                }
                field[geometry.complex_index(ix, iy, iz)] = {0.0, 0.0};
            }
        }
    }
}

} // namespace ic
} // namespace cosmo_nbody
