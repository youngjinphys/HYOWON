#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/cosmology/cosmology_model.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"

#include <span>

namespace cosmo_nbody {
namespace ic {

class LPTDisplacement {
public:
    // All six arrays contain N^3 elements in the same corner-lattice ordering.
    // The first-order density spectrum follows the configured IC Fourier
    // support, as produced by RandomField. Hessians use the same reflection-
    // compatible real trigonometric convention as the tidal-web analysis;
    // the generated bandlimit excludes the affected Nyquist planes.
    static void apply_to_arrays(
        const config::SimulationParameters& config,
        const cosmology::CosmologyModel& cosmo,
        mesh::FFTBackend& fft,
        const mesh::ComplexField& density_k_field,
        std::span<core::Real> pos_x,
        std::span<core::Real> pos_y,
        std::span<core::Real> pos_z,
        std::span<core::Real> mom_x,
        std::span<core::Real> mom_y,
        std::span<core::Real> mom_z);

    // Generated-IC path that consumes density_k_field and reuses its storage for
    // the 2LPT source after the last first-order use.
    static void apply_to_arrays_reusing_density(
        const config::SimulationParameters& config,
        const cosmology::CosmologyModel& cosmo,
        mesh::FFTBackend& fft,
        mesh::ComplexField& density_k_field,
        std::span<core::Real> pos_x,
        std::span<core::Real> pos_y,
        std::span<core::Real> pos_z,
        std::span<core::Real> mom_x,
        std::span<core::Real> mom_y,
        std::span<core::Real> mom_z);

    // Same in-place reuse when the input complex storage is already file-backed.
    static void apply_to_arrays_reusing_file_backed_density(
        const config::SimulationParameters& config,
        const cosmology::CosmologyModel& cosmo,
        mesh::FFTBackend& fft,
        mesh::ComplexField& density_k_field,
        std::span<core::Real> pos_x,
        std::span<core::Real> pos_y,
        std::span<core::Real> pos_z,
        std::span<core::Real> mom_x,
        std::span<core::Real> mom_y,
        std::span<core::Real> mom_z);
};

} // namespace ic
} // namespace cosmo_nbody
