#pragma once

#include "cosmo_nbody/analysis/analysis_resource_policy.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cosmo_nbody::analysis {

enum class AnalysisInputRole {
    Snapshot,
    ProgenitorSnapshot,
    CrossSnapshot,
};

enum class FieldShotNoiseTreatment {
    Raw,
    SubtractCICAliasedPoisson,
};

struct AnalysisInputArgument {
    AnalysisInputRole role{AnalysisInputRole::Snapshot};
    std::size_t argument_index{0};
};

// Normalized analyzer invocation; requested scientific estimator coordinates are
// explicit, while resource settings affect execution/storage only.
struct AnalysisRequest {
    std::string snapshot_path;
    std::string output_directory;
    std::size_t output_directory_argument_index{0};
    bool output_directory_explicit{false};
    std::vector<AnalysisInputArgument> input_arguments;
    AnalysisResourcePolicy resource_policy{};

    std::optional<int> field_mesh;
    int bins{0};
    bool bins_explicit{false};
    std::optional<bool> field_interlaced;
    std::optional<FieldShotNoiseTreatment> field_shot_noise_treatment;
    core::Real field_max_k_fraction_nyquist{0.0};
    bool field_max_k_fraction_nyquist_explicit{false};

    bool skip_halos{false};
    core::Real fof_linking_length_b{0.0};
    bool fof_linking_length_explicit{false};
    std::size_t fof_min_particles{0};
    bool fof_min_particles_explicit{false};

    std::size_t peak_density_k_neighbors{0};
    bool peak_density_k_neighbors_explicit{false};
    std::size_t deblended_min_particles{0};
    bool deblended_min_particles_explicit{false};
    core::Real peak_saddle_merge_ratio{0.0};
    bool peak_saddle_merge_ratio_explicit{false};
    bool write_standard_so{false};

    std::optional<core::Real> hmf_min_mass;
    std::optional<core::Real> hmf_max_mass;
    int hmf_bins{0};
    bool hmf_bins_explicit{false};

    std::optional<core::Real> xi_min_radius;
    std::optional<core::Real> xi_max_radius;
    int xi_bins{0};
    bool xi_bins_explicit{false};
    std::string xi_binning;

    bool write_bispectrum{false};
    int bispectrum_bins{0};
    bool bispectrum_bins_explicit{false};
    core::Real bispectrum_max_k_fraction_nyquist{0.0};
    bool bispectrum_max_k_fraction_nyquist_explicit{false};

    std::string progenitor_snapshot;
    bool write_shapes{false};
    std::optional<bool> shape_use_reduced_tensor;
    int shape_max_iterations{0};
    core::Real shape_convergence_tolerance{0.0};

    bool write_tidal_web{false};
    core::Real tidal_web_gaussian_smoothing_radius_Mpc_h{0.0};
    bool tidal_web_gaussian_smoothing_radius_explicit{false};
    core::Real tidal_web_lambda_threshold{0.0};
    bool tidal_web_lambda_threshold_explicit{false};

    std::string cross_snapshot;
    int cross_mesh{0};
    bool cross_mesh_explicit{false};
    bool write_cross_phase_space_summary{false};
};

std::string analysis_request_usage();

AnalysisRequest parse_analysis_request(
    std::span<const std::string> arguments);

std::string& analysis_input_path(
    AnalysisRequest& request,
    AnalysisInputRole role);

const std::string& analysis_input_path(
    const AnalysisRequest& request,
    AnalysisInputRole role);

std::string_view analysis_input_role_name(AnalysisInputRole role) noexcept;
std::string_view analysis_input_label(AnalysisInputRole role) noexcept;

} // namespace cosmo_nbody::analysis
