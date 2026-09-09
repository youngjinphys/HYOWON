#include "cosmo_nbody/analysis/analysis_request.hpp"

#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace cosmo_nbody::analysis {
namespace {

const std::string& next_value(
    std::size_t& index,
    std::span<const std::string> arguments,
    std::string_view option) {
    if (index + 1 >= arguments.size()) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return arguments[++index];
}

void require_numeric_token(const std::string& value, std::string_view option) {
    if (value.empty()) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    for (const unsigned char c : value) {
        if (std::isspace(c)) {
            throw std::invalid_argument(
                std::string(option) + " numeric value contains whitespace");
        }
    }
}

long long parse_integer(const std::string& value, std::string_view option) {
    require_numeric_token(value, option);
    try {
        std::size_t consumed = 0;
        const long long result = std::stoll(value, &consumed, 10);
        if (consumed != value.size()) throw std::invalid_argument("trailing");
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string(option) + " requires a canonical base-10 integer");
    }
}

int positive_int(const std::string& value, std::string_view option) {
    const long long result = parse_integer(value, option);
    if (result <= 0
        || result > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string(option) + " must lie in [1,INT_MAX]");
    }
    return static_cast<int>(result);
}

std::size_t positive_size(const std::string& value, std::string_view option) {
    const long long result = parse_integer(value, option);
    if (result <= 0) {
        throw std::invalid_argument(std::string(option) + " must be positive");
    }
    const auto unsigned_result = static_cast<unsigned long long>(result);
    if (unsigned_result > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string(option) + " exceeds SIZE_MAX");
    }
    return static_cast<std::size_t>(unsigned_result);
}

core::Real finite_real(const std::string& value, std::string_view option) {
    require_numeric_token(value, option);
    try {
        std::size_t consumed = 0;
        const core::Real result = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(result)) {
            throw std::invalid_argument("invalid");
        }
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string(option) + " requires a finite floating-point value");
    }
}

void require_fraction(core::Real value, std::string_view option) {
    if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(option) + " must be finite and lie in (0,1]");
    }
}

void validate(const AnalysisRequest& r) {
    if (!r.output_directory_explicit || r.output_directory.empty()) {
        throw std::invalid_argument(
            "Native analysis requires an explicit non-empty --output-directory naming a new execution directory");
    }
    if (!r.field_mesh.has_value() || *r.field_mesh <= 0) {
        throw std::invalid_argument(
            "Native field analysis requires explicit positive --field-mesh");
    }
    if (!r.bins_explicit || r.bins <= 0) {
        throw std::invalid_argument(
            "Native field analysis requires explicit positive --bins");
    }
    if (!r.field_interlaced.has_value()) {
        throw std::invalid_argument(
            "Native field analysis requires explicit --field-interlacing interlaced|single-grid");
    }
    if (!r.field_shot_noise_treatment.has_value()) {
        throw std::invalid_argument(
            "Native field analysis requires explicit --field-shot-noise raw|subtract-cic-aliased-poisson");
    }
    if (!r.field_max_k_fraction_nyquist_explicit) {
        throw std::invalid_argument(
            "Native field analysis requires explicit --field-max-k-fraction-nyquist; this declares estimator support, not a trust threshold");
    }
    require_fraction(
        r.field_max_k_fraction_nyquist,
        "--field-max-k-fraction-nyquist");
    if (*r.field_interlaced
        && *r.field_shot_noise_treatment
            == FieldShotNoiseTreatment::SubtractCICAliasedPoisson) {
        throw std::invalid_argument(
            "--field-shot-noise subtract-cic-aliased-poisson requires --field-interlacing single-grid because the implemented discreteness correction is not interlacing-matched");
    }
    if (!std::isfinite(r.resource_policy.memory_budget_gib)
        || r.resource_policy.memory_budget_gib < 0.0) {
        throw std::invalid_argument(
            "Analysis memory budget must be finite and non-negative");
    }

    const bool any_deblend = r.peak_density_k_neighbors_explicit
        || r.deblended_min_particles_explicit
        || r.peak_saddle_merge_ratio_explicit;
    if (!r.skip_halos) {
        if (!r.fof_linking_length_explicit
            || !std::isfinite(r.fof_linking_length_b)
            || r.fof_linking_length_b <= 0.0) {
            throw std::invalid_argument(
                "Halo analysis requires explicit positive --fof-linking-length-b");
        }
        if (!r.fof_min_particles_explicit || r.fof_min_particles == 0) {
            throw std::invalid_argument(
                "Halo analysis requires explicit positive --fof-min-particles");
        }
    } else if (r.fof_linking_length_explicit
               || r.fof_min_particles_explicit || any_deblend) {
        throw std::invalid_argument(
            "FoF/deblending parameters cannot be combined with --skip-halos");
    }

    if (r.write_standard_so) {
        if (r.skip_halos) {
            throw std::invalid_argument(
                "--standard-so cannot be combined with --skip-halos");
        }
        if (r.fof_min_particles < 2) {
            throw std::invalid_argument(
                "--standard-so requires --fof-min-particles >= 2");
        }
        if (!r.peak_density_k_neighbors_explicit
            || !r.deblended_min_particles_explicit
            || !r.peak_saddle_merge_ratio_explicit) {
            throw std::invalid_argument(
                "--standard-so requires explicit density-peak neighbor, minimum-particle, and saddle-merge parameters");
        }
        if (r.peak_density_k_neighbors == 0
            || r.deblended_min_particles == 0
            || !std::isfinite(r.peak_saddle_merge_ratio)
            || r.peak_saddle_merge_ratio < 0.0
            || r.peak_saddle_merge_ratio > 1.0) {
            throw std::invalid_argument(
                "Standard-SO deblending parameters are outside their domains");
        }
    } else if (any_deblend) {
        throw std::invalid_argument(
            "Density-peak/deblending parameters require --standard-so");
    }

    if (r.hmf_min_mass.has_value() != r.hmf_max_mass.has_value()) {
        throw std::invalid_argument(
            "--hmf-min-mass and --hmf-max-mass must be specified together");
    }
    if (r.hmf_min_mass.has_value()) {
        if (!r.hmf_bins_explicit || r.hmf_bins <= 0) {
            throw std::invalid_argument(
                "HMF output requires explicit positive --hmf-bins");
        }
        if (*r.hmf_min_mass <= 0.0
            || *r.hmf_max_mass <= *r.hmf_min_mass || r.skip_halos) {
            throw std::invalid_argument(
                "HMF requires 0 < min_mass < max_mass and halo analysis enabled");
        }
    } else if (r.hmf_bins_explicit) {
        throw std::invalid_argument(
            "--hmf-bins requires the complete HMF mass range");
    }

    if (r.xi_min_radius.has_value() != r.xi_max_radius.has_value()) {
        throw std::invalid_argument(
            "--xi-min-radius and --xi-max-radius must be specified together");
    }
    if (r.xi_min_radius.has_value()) {
        if (!r.xi_bins_explicit || r.xi_bins <= 0
            || (r.xi_binning != "linear" && r.xi_binning != "logarithmic")) {
            throw std::invalid_argument(
                "2PCF requires explicit positive --xi-bins and --xi-binning linear|logarithmic");
        }
        if (*r.xi_min_radius < 0.0
            || *r.xi_max_radius <= *r.xi_min_radius || r.skip_halos) {
            throw std::invalid_argument(
                "2PCF requires 0 <= min_radius < max_radius and halo analysis enabled");
        }
        if (r.xi_binning == "logarithmic" && *r.xi_min_radius <= 0.0) {
            throw std::invalid_argument(
                "Logarithmic 2PCF binning requires min_radius > 0");
        }
    } else if (r.xi_bins_explicit || !r.xi_binning.empty()) {
        throw std::invalid_argument(
            "2PCF bin settings require the complete radius range");
    }

    if (r.write_bispectrum) {
        if (!r.bispectrum_bins_explicit || r.bispectrum_bins <= 0
            || !r.bispectrum_max_k_fraction_nyquist_explicit) {
            throw std::invalid_argument(
                "--bispectrum requires explicit bin count and Nyquist support fraction");
        }
        // An unpadded cubic FFT contraction closes wavevectors modulo the
        // mesh. Strictly sub-2/3-Nyquist support excludes wrapped triads.
        if (!std::isfinite(r.bispectrum_max_k_fraction_nyquist)
            || r.bispectrum_max_k_fraction_nyquist <= 0.0
            || r.bispectrum_max_k_fraction_nyquist >= 2.0 / 3.0) {
            throw std::invalid_argument(
                "--bispectrum-max-k-fraction-nyquist must be in (0,2/3) to exclude wrapped FFT triads");
        }
    } else if (r.bispectrum_bins_explicit
               || r.bispectrum_max_k_fraction_nyquist_explicit) {
        throw std::invalid_argument(
            "Bispectrum settings require --bispectrum");
    }

    const bool shape_requested = r.write_shapes || r.write_standard_so;
    const bool any_shape = r.shape_use_reduced_tensor.has_value()
        || r.shape_max_iterations != 0
        || r.shape_convergence_tolerance != 0.0;
    if (shape_requested) {
        if (!r.shape_use_reduced_tensor.has_value()
            || r.shape_max_iterations <= 0
            || !std::isfinite(r.shape_convergence_tolerance)
            || r.shape_convergence_tolerance <= 0.0) {
            throw std::invalid_argument(
                "Shape-producing analysis requires explicit --shape-tensor, --shape-max-iterations, and positive --shape-convergence-tolerance");
        }
    } else if (any_shape) {
        throw std::invalid_argument(
            "Shape estimator parameters require --write-shapes or --standard-so");
    }
    if (r.write_shapes && r.skip_halos) {
        throw std::invalid_argument(
            "--write-shapes cannot be combined with --skip-halos");
    }

    if (r.write_tidal_web) {
        if (!r.tidal_web_gaussian_smoothing_radius_explicit
            || !std::isfinite(r.tidal_web_gaussian_smoothing_radius_Mpc_h)
            || r.tidal_web_gaussian_smoothing_radius_Mpc_h < 0.0) {
            throw std::invalid_argument(
                "--tidal-web requires explicit finite non-negative --tidal-web-smoothing-radius-mpc-h; zero explicitly selects the unsmoothed mesh diagnostic");
        }
        if (!r.tidal_web_lambda_threshold_explicit
            || !std::isfinite(r.tidal_web_lambda_threshold)) {
            throw std::invalid_argument(
                "--tidal-web requires explicit finite --tidal-web-lambda-threshold");
        }
    } else if (r.tidal_web_gaussian_smoothing_radius_explicit
               || r.tidal_web_lambda_threshold_explicit) {
        throw std::invalid_argument(
            "Tidal-web parameters require --tidal-web");
    }

    if (r.cross_snapshot.empty()) {
        if (r.cross_mesh_explicit) {
            throw std::invalid_argument(
                "--cross-mesh requires --cross-snapshot");
        }
        if (r.write_cross_phase_space_summary) {
            throw std::invalid_argument(
                "--cross-phase-space-summary requires --cross-snapshot");
        }
    } else if (!r.cross_mesh_explicit || r.cross_mesh <= 0) {
        throw std::invalid_argument(
            "--cross-snapshot requires explicit positive --cross-mesh; comparison discretization must not inherit --field-mesh implicitly");
    }
    if (!r.progenitor_snapshot.empty() && r.skip_halos) {
        throw std::invalid_argument(
            "--progenitor-snapshot cannot be combined with --skip-halos");
    }
}

} // namespace

std::string analysis_request_usage() {
    return "Usage: hyowon_analyze <snapshot.hdf5> "
        "[--analysis-threads N] [--analysis-memory-budget-gib G] "
        "--output-directory DIR --field-mesh N --bins N "
        "--field-interlacing interlaced|single-grid "
        "--field-shot-noise raw|subtract-cic-aliased-poisson "
        "--field-max-k-fraction-nyquist F "
        "[--skip-halos | --fof-linking-length-b B --fof-min-particles N] "
        "[--standard-so --peak-density-k-neighbors N --deblended-min-particles N "
        "--peak-saddle-merge-ratio X] "
        "[--hmf-min-mass M --hmf-max-mass M --hmf-bins N] "
        "[--xi-min-radius R --xi-max-radius R --xi-bins N "
        "--xi-binning linear|logarithmic] "
        "[--bispectrum --bispectrum-bins N --bispectrum-max-k-fraction-nyquist F] "
        "[--progenitor-snapshot PATH] "
        "[--write-shapes] [--shape-tensor standard|reduced "
        "--shape-max-iterations N --shape-convergence-tolerance X] "
        "[--tidal-web --tidal-web-smoothing-radius-mpc-h R "
        "--tidal-web-lambda-threshold X] "
        "[--cross-snapshot PATH --cross-mesh N [--cross-phase-space-summary]]\n"
        "For cross comparisons: positional snapshot = candidate B; "
        "--cross-snapshot = reference A.";
}

AnalysisRequest parse_analysis_request(
    std::span<const std::string> arguments) {
    if (arguments.size() < 2) {
        throw std::invalid_argument(analysis_request_usage());
    }

    AnalysisRequest r;
    r.snapshot_path = arguments[1];
    r.input_arguments = {{AnalysisInputRole::Snapshot, 1}};
    std::unordered_set<std::string> seen;

    for (std::size_t i = 2; i < arguments.size(); ++i) {
        const std::string& option = arguments[i];
        if (!seen.insert(option).second) {
            throw std::invalid_argument("Duplicate option: " + option);
        }
        if (option == "--analysis-threads") {
            r.resource_policy.requested_threads =
                positive_size(next_value(i, arguments, option), option);
        } else if (option == "--analysis-memory-budget-gib") {
            r.resource_policy.memory_budget_gib =
                finite_real(next_value(i, arguments, option), option);
            if (r.resource_policy.memory_budget_gib <= 0.0) {
                throw std::invalid_argument(
                    "--analysis-memory-budget-gib must be positive");
            }
        } else if (option == "--output-directory") {
            r.output_directory = next_value(i, arguments, option);
            r.output_directory_argument_index = i;
            r.output_directory_explicit = true;
        } else if (option == "--field-mesh") {
            r.field_mesh = positive_int(next_value(i, arguments, option), option);
        } else if (option == "--bins") {
            r.bins = positive_int(next_value(i, arguments, option), option);
            r.bins_explicit = true;
        } else if (option == "--field-interlacing") {
            const std::string& value = next_value(i, arguments, option);
            if (value == "interlaced") r.field_interlaced = true;
            else if (value == "single-grid") r.field_interlaced = false;
            else throw std::invalid_argument(
                "--field-interlacing must be interlaced or single-grid");
        } else if (option == "--field-shot-noise") {
            const std::string& value = next_value(i, arguments, option);
            if (value == "raw") {
                r.field_shot_noise_treatment = FieldShotNoiseTreatment::Raw;
            } else if (value == "subtract-cic-aliased-poisson") {
                r.field_shot_noise_treatment =
                    FieldShotNoiseTreatment::SubtractCICAliasedPoisson;
            } else {
                throw std::invalid_argument(
                    "--field-shot-noise must be raw or subtract-cic-aliased-poisson");
            }
        } else if (option == "--field-max-k-fraction-nyquist") {
            r.field_max_k_fraction_nyquist =
                finite_real(next_value(i, arguments, option), option);
            r.field_max_k_fraction_nyquist_explicit = true;
        } else if (option == "--skip-halos") {
            r.skip_halos = true;
        } else if (option == "--fof-linking-length-b") {
            r.fof_linking_length_b =
                finite_real(next_value(i, arguments, option), option);
            r.fof_linking_length_explicit = true;
        } else if (option == "--fof-min-particles") {
            r.fof_min_particles =
                positive_size(next_value(i, arguments, option), option);
            r.fof_min_particles_explicit = true;
        } else if (option == "--standard-so") {
            r.write_standard_so = true;
        } else if (option == "--peak-density-k-neighbors") {
            r.peak_density_k_neighbors =
                positive_size(next_value(i, arguments, option), option);
            r.peak_density_k_neighbors_explicit = true;
        } else if (option == "--deblended-min-particles") {
            r.deblended_min_particles =
                positive_size(next_value(i, arguments, option), option);
            r.deblended_min_particles_explicit = true;
        } else if (option == "--peak-saddle-merge-ratio") {
            r.peak_saddle_merge_ratio =
                finite_real(next_value(i, arguments, option), option);
            r.peak_saddle_merge_ratio_explicit = true;
        } else if (option == "--hmf-min-mass") {
            r.hmf_min_mass = finite_real(next_value(i, arguments, option), option);
        } else if (option == "--hmf-max-mass") {
            r.hmf_max_mass = finite_real(next_value(i, arguments, option), option);
        } else if (option == "--hmf-bins") {
            r.hmf_bins = positive_int(next_value(i, arguments, option), option);
            r.hmf_bins_explicit = true;
        } else if (option == "--xi-min-radius") {
            r.xi_min_radius = finite_real(next_value(i, arguments, option), option);
        } else if (option == "--xi-max-radius") {
            r.xi_max_radius = finite_real(next_value(i, arguments, option), option);
        } else if (option == "--xi-bins") {
            r.xi_bins = positive_int(next_value(i, arguments, option), option);
            r.xi_bins_explicit = true;
        } else if (option == "--xi-binning") {
            r.xi_binning = next_value(i, arguments, option);
        } else if (option == "--bispectrum") {
            r.write_bispectrum = true;
        } else if (option == "--bispectrum-bins") {
            r.bispectrum_bins = positive_int(next_value(i, arguments, option), option);
            r.bispectrum_bins_explicit = true;
        } else if (option == "--bispectrum-max-k-fraction-nyquist") {
            r.bispectrum_max_k_fraction_nyquist =
                finite_real(next_value(i, arguments, option), option);
            r.bispectrum_max_k_fraction_nyquist_explicit = true;
        } else if (option == "--progenitor-snapshot") {
            r.progenitor_snapshot = next_value(i, arguments, option);
            r.input_arguments.push_back(
                {AnalysisInputRole::ProgenitorSnapshot, i});
        } else if (option == "--write-shapes") {
            r.write_shapes = true;
        } else if (option == "--shape-tensor") {
            const std::string& value = next_value(i, arguments, option);
            if (value == "reduced") r.shape_use_reduced_tensor = true;
            else if (value == "standard") r.shape_use_reduced_tensor = false;
            else throw std::invalid_argument(
                "--shape-tensor must be standard or reduced");
        } else if (option == "--shape-max-iterations") {
            r.shape_max_iterations =
                positive_int(next_value(i, arguments, option), option);
        } else if (option == "--shape-convergence-tolerance") {
            r.shape_convergence_tolerance =
                finite_real(next_value(i, arguments, option), option);
        } else if (option == "--tidal-web") {
            r.write_tidal_web = true;
        } else if (option == "--tidal-web-smoothing-radius-mpc-h") {
            r.tidal_web_gaussian_smoothing_radius_Mpc_h =
                finite_real(next_value(i, arguments, option), option);
            r.tidal_web_gaussian_smoothing_radius_explicit = true;
        } else if (option == "--tidal-web-lambda-threshold") {
            r.tidal_web_lambda_threshold =
                finite_real(next_value(i, arguments, option), option);
            r.tidal_web_lambda_threshold_explicit = true;
        } else if (option == "--cross-snapshot") {
            r.cross_snapshot = next_value(i, arguments, option);
            r.input_arguments.push_back({AnalysisInputRole::CrossSnapshot, i});
        } else if (option == "--cross-mesh") {
            r.cross_mesh = positive_int(next_value(i, arguments, option), option);
            r.cross_mesh_explicit = true;
        } else if (option == "--cross-phase-space-summary") {
            r.write_cross_phase_space_summary = true;
        } else {
            throw std::invalid_argument("Unknown hyowon_analyze option: " + option);
        }
    }

    validate(r);
    return r;
}

std::string& analysis_input_path(AnalysisRequest& request, AnalysisInputRole role) {
    switch (role) {
        case AnalysisInputRole::Snapshot: return request.snapshot_path;
        case AnalysisInputRole::ProgenitorSnapshot:
            return request.progenitor_snapshot;
        case AnalysisInputRole::CrossSnapshot: return request.cross_snapshot;
    }
    throw std::logic_error("Unknown analysis input role");
}

const std::string& analysis_input_path(
    const AnalysisRequest& request,
    AnalysisInputRole role) {
    return analysis_input_path(const_cast<AnalysisRequest&>(request), role);
}

std::string_view analysis_input_role_name(AnalysisInputRole role) noexcept {
    switch (role) {
        case AnalysisInputRole::Snapshot: return "snapshot";
        case AnalysisInputRole::ProgenitorSnapshot: return "progenitor_snapshot";
        case AnalysisInputRole::CrossSnapshot: return "cross_snapshot";
    }
    return "unknown";
}

std::string_view analysis_input_label(AnalysisInputRole role) noexcept {
    switch (role) {
        case AnalysisInputRole::Snapshot: return "analysis snapshot";
        case AnalysisInputRole::ProgenitorSnapshot:
            return "analysis progenitor snapshot";
        case AnalysisInputRole::CrossSnapshot: return "analysis cross snapshot";
    }
    return "analysis input";
}

} // namespace cosmo_nbody::analysis
