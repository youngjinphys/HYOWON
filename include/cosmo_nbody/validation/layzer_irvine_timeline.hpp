#pragma once

#include "cosmo_nbody/validation/force_energy_work.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

namespace cosmo_nbody::validation {

struct ForceEnergyWorkIntervalRecord {
    ForceEnergyWorkPoint start{};
    ForceEnergyWorkPoint end{};
    double integrated_work{0.0};
    double integrated_work_compensation{0.0};
    double closure_residual{0.0};
};

struct LayzerIrvineSampleRecord {
    std::uint64_t step{0};
    double scale_factor_start{0.0};
    double scale_factor_end{0.0};
    double delta_ln_a{0.0};
    double kinetic_energy{0.0};
    double potential_energy_raw{0.0};
    double cic_self_energy{0.0};
    double potential_energy_pair{0.0};
    double initial_energy{0.0};
    double source_start{0.0};
    double source{0.0};
    double integrated_source{0.0};
    double integrated_source_compensation{0.0};
    double residual{0.0};
    std::optional<double> ratio;
    bool reused_force_potential{false};
    std::optional<ForceEnergyWorkIntervalRecord> force_energy_work;
};

std::string layzer_irvine_timeline_to_json(
    std::span<const LayzerIrvineSampleRecord> samples);
void write_layzer_irvine_timeline_atomic(
    std::span<const LayzerIrvineSampleRecord> samples,
    const std::filesystem::path& path);

} // namespace cosmo_nbody::validation
