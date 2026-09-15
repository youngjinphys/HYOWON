#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/cosmology/flat_matter_lambda.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"
#include "cosmo_nbody/time/time_stepper.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace cosmo_nbody {
namespace config {

namespace {

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

std::string_view scratch_mode_name(ScratchMode mode) noexcept {
    switch (mode) {
        case ScratchMode::Memory: return "memory";
        case ScratchMode::Disk: return "disk";
    }
    return "unknown";
}

ScratchMode parse_scratch_mode(std::string value) {
    value = lower_ascii(std::move(value));
    if (value == "memory") return ScratchMode::Memory;
    if (value == "disk") return ScratchMode::Disk;
    throw std::invalid_argument(
        "scratch mode must be memory or disk");
}

SimulationParameters::SimulationParameters(
    CosmologyParams c,
    BoxParams b,
    GravityParams g,
    TimeParams t,
    ICParams i,
    OutputParams o,
    RuntimeParams r,
    ValidationParams v,
    MemoryPolicyParams m)
    : cosmo(std::move(c)), box(std::move(b)), gravity(std::move(g)),
      time(std::move(t)), ic(std::move(i)), output(std::move(o)),
      validation(std::move(v)), runtime(std::move(r)),
      memory_policy(std::move(m)) {
    derive_ic_input_hashes();
    validate_and_derive();
}

std::string SimulationParameters::ascii_lower(std::string value) {
    return lower_ascii(std::move(value));
}

void SimulationParameters::require_finite_positive(
    core::Real value,
    const char* label) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::overflow_error(
            std::string("Derived simulation quantity is not finite and positive: ")
            + label);
    }
}

void SimulationParameters::derive_ic_input_hashes() {
    ic.power_spectrum_sha256.clear();
    ic.snapshot_sha256.clear();
    if (ic.mode != "generate" || ic.power_spectrum_file.empty()) return;

    std::error_code ec;
    const std::filesystem::path path(ic.power_spectrum_file);
    const bool regular = std::filesystem::is_regular_file(path, ec);
    if (ec || !regular) {
        return;
    }
    ic.power_spectrum_sha256 = io::sha256_file(path);
}

void SimulationParameters::validate_cosmology() {
    if (!std::isfinite(cosmo.h) || !std::isfinite(cosmo.omega_m)
        || !std::isfinite(cosmo.omega_lambda) || !std::isfinite(cosmo.omega_b)
        || !std::isfinite(cosmo.sigma8) || !std::isfinite(cosmo.n_s)) {
        throw std::invalid_argument("Cosmology parameters must be finite");
    }
    if (cosmo.h <= 0.0 || cosmo.omega_m <= 0.0 || cosmo.omega_lambda < 0.0) {
        throw std::invalid_argument(
            "h and omega_m must be positive; omega_lambda must be non-negative");
    }
    if (cosmo.omega_b < 0.0 || cosmo.omega_b > cosmo.omega_m) {
        throw std::invalid_argument("omega_b must lie in [0, omega_m]");
    }
    if (cosmo.sigma8 < 0.0) {
        throw std::invalid_argument("sigma8 must be non-negative");
    }
    if (!cosmology::has_flat_matter_lambda_closure(
            cosmo.omega_m,
            cosmo.omega_lambda)) {
        throw std::invalid_argument(
            "Flat LambdaCDM is required: omega_m + omega_lambda must equal 1.0");
    }
}

void SimulationParameters::resolve_and_validate_resolution() {
    if (!std::isfinite(box.L) || box.L <= 0.0) {
        throw std::invalid_argument("Box size must be finite and positive");
    }
    if (box.N == 0 || box.N_mesh == 0) {
        throw std::invalid_argument("Particle and mesh dimensions must be positive");
    }
    const auto max_uint64 = std::numeric_limits<std::uint64_t>::max();
    if (box.N > max_uint64 / box.N) {
        throw std::overflow_error(
            "Configured particle population N^3 overflows uint64_t");
    }
    const std::uint64_t particle_plane = box.N * box.N;
    if (particle_plane > max_uint64 / box.N) {
        throw std::overflow_error(
            "Configured particle population N^3 overflows uint64_t");
    }
    derived_num_particles = particle_plane * box.N;
    if (derived_num_particles > static_cast<std::uint64_t>(
            std::numeric_limits<unsigned int>::max())) {
        throw std::overflow_error(
            "Configured particle population N^3 exceeds the uint32 count range of the mandatory native snapshot format");
    }
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (derived_num_particles > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "Configured particle population N^3 does not fit size_t");
        }
    }

    if (ic.mode == "generate") {
        if (!ic.mesh_per_dimension.specified()) {
            throw std::invalid_argument(
                "Generated IC configuration requires an explicit mesh_per_dimension coordinate; zero explicitly selects the automatic mesh policy");
        }
        const std::uint64_t requested_ic_mesh = ic.mesh_per_dimension.value();
        if (requested_ic_mesh != 0) {
            derived_ic_mesh = requested_ic_mesh;
        } else if (ic.lpt_order == 2) {
            if (box.N > std::numeric_limits<std::uint64_t>::max() / 2) {
                throw std::overflow_error(
                    "Automatic de-aliased 2LPT IC mesh dimension overflows uint64_t");
            }
            derived_ic_mesh = 2 * box.N;
        } else {
            derived_ic_mesh = box.N;
        }
    } else if (ic.mode == "snapshot") {
        derived_ic_mesh = 0;
    } else {
        throw std::invalid_argument("ic.mode must be 'generate' or 'snapshot'");
    }

    if (ic.mode == "generate" && derived_ic_mesh == 0) {
        throw std::invalid_argument("Generated IC mesh dimension must be positive");
    }
    const auto max_int_dimension = static_cast<std::uint64_t>(
        std::numeric_limits<int>::max());
    if (box.N > max_int_dimension || box.N_mesh > max_int_dimension
        || derived_ic_mesh > max_int_dimension) {
        throw std::overflow_error(
            "Particle, evolution mesh, and IC mesh dimensions must fit the int-based FFT and mesh APIs");
    }
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        const auto max_size = static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max());
        if (box.N > max_size || box.N_mesh > max_size
            || derived_ic_mesh > max_size) {
            throw std::overflow_error(
                "Particle, evolution mesh, or IC mesh dimension does not fit size_t");
        }
    }
}

void SimulationParameters::validate_time() const {
    if (!std::isfinite(time.z_start) || !std::isfinite(time.z_final)
        || !std::isfinite(time.delta_ln_a)) {
        throw std::invalid_argument("Time parameters must be finite");
    }
    if (time.z_start <= -1.0 || time.z_final <= -1.0) {
        throw std::invalid_argument(
            "Redshifts must satisfy z > -1 so scale factor remains positive");
    }
    if (time.z_final < 0.0) {
        throw std::invalid_argument(
            "Future-time evolution (z_final < 0, a_final > 1) is not supported by the current growth table");
    }
    if (time.z_start <= time.z_final) {
        throw std::invalid_argument("z_start must be greater than z_final");
    }
    if (time.delta_ln_a <= 0.0) {
        throw std::invalid_argument("delta_ln_a must be positive");
    }
    if (time.step_policy != "global") {
        throw std::invalid_argument(
            "Only 'global' step_policy is supported by the maintained integrator");
    }
}

void SimulationParameters::validate_gravity() {
    if (gravity.solver == "PM") {
        if (!gravity.deconvolve_cic.specified()) {
            throw std::invalid_argument(
                "Pure PM configuration requires an explicit deconvolve_cic force-method choice");
        }
        derived_solver_kind = SolverKind::PM;
        gravity.eps = 0.0;
        gravity.theta = 0.0;
        gravity.split_scale_cells = 0.0;
        gravity.cutoff_multiplier = 0.0;
    } else if (gravity.solver == "TreePM") {
        derived_solver_kind = SolverKind::TreePM;
        if (!std::isfinite(gravity.eps) || gravity.eps <= 0.0) {
            throw std::invalid_argument(
                "TreePM softening_comoving_Mpc_h must be finite and positive");
        }
        gravity.deconvolve_cic = true;
        if (!std::isfinite(gravity.theta) || gravity.theta <= 0.0) {
            throw std::invalid_argument("theta must be finite and positive");
        }
        if (!std::isfinite(gravity.split_scale_cells)
            || gravity.split_scale_cells <= 0.0) {
            throw std::invalid_argument("split_scale_cells must be finite and positive");
        }
        if (!std::isfinite(gravity.cutoff_multiplier)
            || gravity.cutoff_multiplier <= 0.0) {
            throw std::invalid_argument("cutoff_multiplier must be finite and positive");
        }
    } else {
        throw std::invalid_argument(
            "gravity.solver must be one of: PM, TreePM");
    }
}

void SimulationParameters::validate_initial_conditions() {
    if (ic.mode != "generate" && ic.mode != "snapshot") {
        throw std::invalid_argument("ic.mode must be 'generate' or 'snapshot'");
    }

    if (ic.mode == "generate") {
        if (!ic.seed.specified()) {
            throw std::invalid_argument(
                "Generated IC configuration requires an explicit seed; seed zero is valid and must be selected explicitly");
        }
        if (!ic.mesh_per_dimension.specified()) {
            throw std::invalid_argument(
                "Generated IC configuration requires explicit mesh_per_dimension");
        }
        derived_ic_seed = ic.seed.value();
        if (ic.max_mode_per_axis.has_value()
            && (*ic.max_mode_per_axis == 0
                || *ic.max_mode_per_axis > (box.N - 1) / 2)) {
            throw std::invalid_argument(
                "ic.max_mode_per_axis must be positive and satisfy 2*K < particles_per_dimension");
        }
        if (ic.lpt_order != 1 && ic.lpt_order != 2) {
            throw std::invalid_argument("ic.lpt_order must be 1 or 2");
        }
        if (ic.power_spectrum_file.empty()) {
            throw std::invalid_argument(
                "ic.power_spectrum_file is required when ic.mode = 'generate'");
        }
        if (!std::isfinite(ic.power_spectrum_redshift)
            || ic.power_spectrum_redshift <= -1.0) {
            throw std::invalid_argument("power_spectrum_redshift must be finite and > -1");
        }
        if (ic.power_spectrum_fidelity != "precision_boltzmann") {
            throw std::invalid_argument(
                "Generated cosmological ICs require ic.power_spectrum_fidelity="
                "precision_boltzmann and an externally produced linear spectrum; "
                "changing a fidelity label does not validate the input data");
        }
        if (derived_ic_mesh % box.N != 0) {
            throw std::invalid_argument(
                "Generated IC mesh must be an integer multiple of particles_per_dimension");
        }
        if (ic.lpt_order == 2 && derived_ic_mesh / 2 < box.N) {
            throw std::invalid_argument(
                "Generated 2LPT ICs require ic.mesh_per_dimension >= 2 * particles_per_dimension");
        }
        if (ic.amplitude_mode != "gaussian" && ic.amplitude_mode != "fixed") {
            throw std::invalid_argument("ic.amplitude_mode must be 'gaussian' or 'fixed'");
        }
        if (ic.phase_pairing != "independent"
            && ic.phase_pairing != "pair_a"
            && ic.phase_pairing != "pair_b") {
            throw std::invalid_argument(
                "ic.phase_pairing must be 'independent', 'pair_a', or 'pair_b'");
        }
        if (!ic.snapshot_file.empty() || !ic.expected_snapshot_sha256.empty()) {
            throw std::invalid_argument(
                "Generated IC configuration cannot carry external snapshot input coordinates");
        }
    } else {
        derived_ic_seed = 0;
        if (ic.snapshot_file.empty()) {
            throw std::invalid_argument(
                "ic.snapshot_file is required when ic.mode = 'snapshot'");
        }
        if (ic.seed.specified() || ic.mesh_per_dimension.specified()
            || ic.max_mode_per_axis.has_value()
            || ic.lpt_order != 0
            || !ic.power_spectrum_file.empty()
            || std::isfinite(ic.power_spectrum_redshift)
            || !ic.power_spectrum_fidelity.empty()
            || !ic.amplitude_mode.empty()
            || !ic.phase_pairing.empty()) {
            throw std::invalid_argument(
                "Snapshot IC configuration cannot carry generated-IC scientific coordinates");
        }
        // Canonicalize inactive generated-IC fields only after snapshot-mode
        // admission rejects every generated-IC coordinate.
        ic.seed = std::uint64_t{0};
        ic.mesh_per_dimension = std::uint64_t{0};
    }
}

void SimulationParameters::normalize_and_validate_output() {
    const core::Real a_start = 1.0 / (1.0 + time.z_start);
    const core::Real a_final = 1.0 / (1.0 + time.z_final);
    require_finite_positive(a_start, "a_start");
    require_finite_positive(a_final, "a_final");
    core::Real previous = a_start;
    for (const core::Real target : output.snapshot_scale_factors) {
        if (!std::isfinite(target) || target <= a_start || target > a_final) {
            throw std::invalid_argument(
                "snapshot_scale_factors must lie strictly after a_start and at or before a_final");
        }
        if (!(target > previous)) {
            throw std::invalid_argument("snapshot_scale_factors must be strictly increasing");
        }
        previous = target;
    }
    if (output.formats.size() != 1) {
        throw std::invalid_argument(
            "Exactly one output format is supported by the maintained writer");
    }
    output.formats.front() = ascii_lower(output.formats.front());
    if (output.formats.front() != "hdf5") {
        throw std::invalid_argument("Only output format 'hdf5' is implemented");
    }
    if (output.root_directory.empty()) {
        throw std::invalid_argument("output.root_directory must not be empty");
    }
    if (output.snapshot_batch_particles == 0) {
        throw std::invalid_argument("output.snapshot_batch_particles must be >= 1");
    }
    try {
        (void)time::TimeStepper::planned_boundary_storage_bytes(
            a_start,
            a_final,
            time.delta_ln_a,
            output.snapshot_scale_factors);
    } catch (const std::exception& error) {
        throw std::invalid_argument(
            std::string(
                "TimeStepper boundary table is not representable at the "
                "requested delta_ln_a: ")
            + error.what());
    }
}

void SimulationParameters::normalize_runtime() {
}

void SimulationParameters::derive_physical_scales() {
    derived_d_mean = box.L / static_cast<core::Real>(box.N);
    require_finite_positive(derived_d_mean, "d_mean");
    const std::array<core::Real, 5> particle_mass_factors{
        cosmology::units::rho_crit0,
        cosmo.omega_m,
        derived_d_mean,
        derived_d_mean,
        derived_d_mean};
    derived_m_p = math::scaled_positive_product_quotient(
        particle_mass_factors,
        std::span<const core::Real>{},
        "particle mass");
    require_finite_positive(derived_m_p, "particle_mass");

    if (derived_solver_kind == SolverKind::TreePM) {
        derived_eps = gravity.eps;
        require_finite_positive(derived_eps, "softening");
        const core::Real dx = box.L / static_cast<core::Real>(box.N_mesh);
        require_finite_positive(dx, "mesh_cell_size");
        if (gravity.split_scale_cells
            > std::numeric_limits<core::Real>::max() / dx) {
            throw std::overflow_error(
                "TreePM split scale (split_scale_cells * dx) overflows");
        }
        derived_r_s = gravity.split_scale_cells * dx;
        require_finite_positive(derived_r_s, "r_s");
        if (!(derived_r_s < box.L)) {
            throw std::overflow_error(
                "TreePM split scale r_s must be smaller than the box size");
        }
        if (derived_r_s
            > std::numeric_limits<core::Real>::max()
                / gravity.cutoff_multiplier) {
            throw std::overflow_error(
                "TreePM short-range cutoff (cutoff_multiplier * r_s) overflows");
        }
        derived_r_cut = gravity.cutoff_multiplier * derived_r_s;
        require_finite_positive(derived_r_cut, "r_cut");
        if (!(derived_r_cut < 0.5 * box.L)) {
            throw std::invalid_argument(
                "TreePM requires r_cut < L/2 so the minimum-image short-range neighborhood is unambiguous");
        }
    } else {
        derived_eps = 0.0;
        derived_r_s = 0.0;
        derived_r_cut = 0.0;
    }

    derived_k_Nyq = std::numbers::pi * static_cast<core::Real>(box.N) / box.L;
    require_finite_positive(derived_k_Nyq, "k_Nyq");
}

void SimulationParameters::validate_and_derive() {
    validate_cosmology();
    resolve_and_validate_resolution();
    validate_time();
    validate_gravity();
    validate_initial_conditions();
    normalize_and_validate_output();
    normalize_runtime();
    derive_physical_scales();
}

} // namespace config
} // namespace cosmo_nbody
