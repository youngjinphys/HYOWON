#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/ic/linear_power_spectrum.hpp"
#include "cosmo_nbody/ic/realised_ic_checks.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cosmo_nbody::ic {

struct RealisedICPowerBin {
    std::size_t shell_index{0};
    core::Real k_low{0.0};
    core::Real k_high{0.0};
    core::Real k_mean{0.0};
    core::Real realised_power{0.0};
    core::Real target_power{0.0};
    core::Real realised_to_target_ratio{0.0};
    std::size_t mode_count{0};
};

struct ExactFourierEvidence {
    std::size_t resolved_shell_count{0};
    core::Real k_min{0.0};
    core::Real k_max{0.0};
    std::string status{"unavailable"};
    LinearPowerNormalizationDiagnostics power_normalization;
    std::vector<RealisedICPowerBin> bins;
};

struct RealisedICEvidence {
    std::uint64_t seed{0};
    std::uint64_t particles_per_dimension{0};
    std::uint64_t ic_mesh_per_dimension{0};
    std::optional<std::uint64_t> ic_max_mode_per_axis;
    std::uint64_t ic_effective_max_mode_per_axis{0};
    std::uint64_t evolution_mesh_per_dimension{0};
    int lpt_order{0};
    core::Real box_size_Mpc_h{0.0};
    core::Real start_redshift{0.0};
    std::string amplitude_mode;
    std::string phase_pairing;
    std::string power_spectrum_file;
    std::string power_spectrum_sha256;

    RealisedICSummary structural;
    LinearPowerNormalizationDiagnostics power_normalization;
    std::string exact_fourier_estimator{"generated_fourier_modes"};
    std::string exact_fourier_status{"unavailable"};
    core::Real evaluated_k_max{0.0};
    std::vector<RealisedICPowerBin> exact_bins;

    std::string to_json() const;
};

// Particle-lattice k_f shells over 0<k<k_Ny,particle. Explicit Cartesian IC
// support restricts the measured set; intentionally removed modes are excluded.
ExactFourierEvidence summarize_generated_fourier_realisation(
    const config::SimulationParameters& config,
    const LinearPowerSpectrum& power_spectrum,
    const mesh::ComplexField& density_k,
    std::size_t ic_mesh_per_dimension);

// Particle-spectrum estimation lives in nbody_analyze where its estimator
// coordinates are explicit; this evidence keeps generated modes and structure.
RealisedICEvidence build_realised_ic_evidence(
    const config::SimulationParameters& config,
    const RealisedICSummary& structural,
    ExactFourierEvidence exact,
    const core::ParticleStore& particles,
    bool include_particle_estimator);

} // namespace cosmo_nbody::ic
