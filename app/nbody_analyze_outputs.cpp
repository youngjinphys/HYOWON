#include "nbody_analyze_outputs.hpp"

#include "cosmo_nbody/io/checked_output_file.hpp"
#include "cosmo_nbody/io/output_schema.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <stdexcept>

namespace cosmo_nbody::app::nbody_analyze {
namespace {

io::CheckedOutputFile open_output(const std::filesystem::path& path) {
    return io::open_checked_output_file(path, "native analysis product");
}

void close_output(
    io::CheckedOutputFile& output,
    const std::filesystem::path& path) {
    io::close_checked_output_file(output, path, "native analysis product");
}

void require_source_provenance(
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a,
    const char* role) {
    const bool canonical_sha256 =
        source_native_snapshot_object_sha256.size() == 64U
        && std::all_of(
            source_native_snapshot_object_sha256.begin(),
            source_native_snapshot_object_sha256.end(),
            [](char value) {
                return (value >= '0' && value <= '9')
                    || (value >= 'a' && value <= 'f');
            });
    if (source_snapshot.empty()
        || !canonical_sha256
        || !std::isfinite(snapshot_a)
        || snapshot_a <= 0.0) {
        throw std::invalid_argument(
            std::string(role) + " source provenance is invalid");
    }
}

} // namespace

void write_particle_tracer_csv_metadata(std::ostream& output) {
    output << "# tracer_species=" << io::schema::VAL_PARTICLE_SPECIES << '\n'
           << "# tracer_interpretation="
           << io::schema::VAL_PARTICLE_INTERPRETATION << '\n'
           << "# mass_normalization="
           << io::schema::VAL_PARTICLE_MASS_NORMALIZATION << '\n'
           << "# baryons_separately_evolved="
           << io::schema::VAL_BARYONS_SEPARATELY_EVOLVED << '\n';
}

void write_single_snapshot_csv_metadata(
    std::ostream& output,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a) {
    require_source_provenance(
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a,
        "Native analysis output");
    output << "# source_snapshot=" << source_snapshot << '\n'
           << "# source_native_snapshot_object_sha256="
           << source_native_snapshot_object_sha256 << '\n'
           << "# scale_factor=" << snapshot_a << '\n';
    write_particle_tracer_csv_metadata(output);
}

void write_particle_tracer_json_metadata(std::ostream& output) {
    output << "  \"tracer_species\": \""
           << io::schema::VAL_PARTICLE_SPECIES << "\",\n"
           << "  \"tracer_interpretation\": \""
           << io::schema::VAL_PARTICLE_INTERPRETATION << "\",\n"
           << "  \"mass_normalization\": \""
           << io::schema::VAL_PARTICLE_MASS_NORMALIZATION << "\",\n"
           << "  \"baryons_separately_evolved\": "
           << io::schema::VAL_BARYONS_SEPARATELY_EVOLVED << ",\n";
}

void write_pk_csv(
    const std::filesystem::path& path,
    const std::vector<analysis::PowerSpectrumBin>& bins,
    int mesh_per_dimension,
    core::Real box_size_Mpc_h,
    const analysis::PowerSpectrumOptions& options,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a) {
    require_source_provenance(
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a,
        "Power-spectrum output");
    if (mesh_per_dimension < 2
        || !std::isfinite(box_size_Mpc_h)
        || box_size_Mpc_h <= 0.0) {
        throw std::invalid_argument(
            "Power-spectrum output requires positive mesh and box size");
    }
    if (!std::isfinite(options.max_k_fraction_nyquist)
        || options.max_k_fraction_nyquist <= 0.0
        || options.max_k_fraction_nyquist > 1.0) {
        throw std::invalid_argument(
            "Power-spectrum output requires the exact admitted max_k_fraction_nyquist in (0,1]");
    }
    if (options.num_bins <= 0
        || bins.size() != static_cast<std::size_t>(options.num_bins)) {
        throw std::invalid_argument(
            "Power-spectrum output shell count disagrees with the estimator request");
    }

    const analysis::PowerSpectrumBin* first_nonempty = nullptr;
    std::size_t represented_shell_count = 0;
    for (const auto& bin : bins) {
        if (bin.mode_count != 0) {
            ++represented_shell_count;
            if (first_nonempty == nullptr) first_nonempty = &bin;
        }
    }
    if (first_nonempty == nullptr) {
        throw std::invalid_argument(
            "Power-spectrum output requires at least one nonempty shell");
    }
    const core::Real shot_noise_reference =
        first_nonempty->shot_noise_reference;
    if (!std::isfinite(shot_noise_reference)
        || shot_noise_reference < 0.0) {
        throw std::invalid_argument(
            "Power-spectrum shot-noise reference is outside its finite domain");
    }

    core::Real previous_high = 0.0;
    bool have_previous = false;
    for (const auto& bin : bins) {
        if (bin.mode_count == 0) continue;
        if (!std::isfinite(bin.k_low)
            || !std::isfinite(bin.k_mean)
            || !std::isfinite(bin.k_high)
            || bin.k_low <= 0.0
            || bin.k_mean < bin.k_low
            || bin.k_mean > bin.k_high
            || bin.k_high <= bin.k_low
            || !std::isfinite(bin.p_raw)
            || !std::isfinite(bin.p_corrected)
            || !std::isfinite(
                bin.mean_cic_power_deconvolution_factor)
            || bin.mean_cic_power_deconvolution_factor < 1.0
            || bin.shot_noise_reference != shot_noise_reference) {
            throw std::invalid_argument(
                "Power-spectrum bin metadata is internally inconsistent");
        }
        if (have_previous && bin.k_low < previous_high) {
            throw std::invalid_argument(
                "Power-spectrum nonempty bins overlap or are out of order");
        }
        previous_high = bin.k_high;
        have_previous = true;
    }

    auto output = open_output(path);
    output.precision(17);
    const core::Real k_nyquist_h_Mpc = std::numbers::pi
        * static_cast<core::Real>(mesh_per_dimension) / box_size_Mpc_h;
    const core::Real evaluated_k_max_h_Mpc =
        options.max_k_fraction_nyquist * k_nyquist_h_Mpc;
    const core::Real k_fundamental_h_Mpc =
        2.0 * std::numbers::pi / box_size_Mpc_h;
    output << "# product_kind=analysis_pk\n";
    write_single_snapshot_csv_metadata(
        output,
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a);
    output << "# binning=logarithmic\n"
           << "# bin_convention=left_closed_right_open_last_closed\n"
           << "# field=particle_mass_weighted_density_contrast\n"
           << "# k_unit=h_per_Mpc\n"
           << "# power_unit=(Mpc_per_h)^3\n"
           << "# fourier_mode_normalization=forward_fft_divided_by_mesh_cell_count\n"
           << "# power_normalization=box_volume_times_mode_norm_squared\n"
           << "# shell_count=" << options.num_bins << '\n'
           << "# represented_shell_count=" << represented_shell_count << '\n'
           << "# empty_shell_count="
           << bins.size() - represented_shell_count << '\n'
           << "# mesh_per_dimension=" << mesh_per_dimension << '\n'
           << "# k_fundamental_h_Mpc=" << k_fundamental_h_Mpc << '\n'
           << "# k_nyquist_h_Mpc=" << k_nyquist_h_Mpc << '\n'
           << "# evaluated_k_max_h_Mpc=" << evaluated_k_max_h_Mpc
           << '\n'
           << "# max_k_fraction_nyquist="
           << options.max_k_fraction_nyquist << '\n'
           << "# mass_assignment=CIC\n"
           << "# interlaced="
           << (options.interlaced ? "true" : "false") << '\n'
           << "# shot_noise_subtracted="
           << (options.subtract_shot_noise ? "true" : "false") << '\n'
           << "# shot_noise_reference=" << shot_noise_reference << '\n'
           << "# shot_noise_reference_kind="
              "box_volume_times_sum_mass_squared_over_sum_mass_squared\n"
           << "# shot_noise_subtraction_model="
           << (options.subtract_shot_noise
                   ? "cic_alias_factor_noninterlaced" : "none")
           << '\n'
           << "# cic_window_deconvolved_for_p_corrected=true\n"
           << "# signal_aliasing_removed=false\n"
           << "# range_interpretation="
              "explicit_estimator_support_measurement_only_accuracy_requires_independent_convergence\n"
           << "shell_index,k_low,k_mean,k_high,p_raw,p_corrected,mode_count,"
              "mean_cic_power_deconvolution_factor\n";
    for (std::size_t shell_index = 0; shell_index < bins.size(); ++shell_index) {
        const auto& bin = bins[shell_index];
        if (bin.mode_count == 0) continue;
        output << shell_index << ',' << bin.k_low << ',' << bin.k_mean << ','
               << bin.k_high
               << ',' << bin.p_raw << ',' << bin.p_corrected << ','
               << bin.mode_count << ','
               << bin.mean_cic_power_deconvolution_factor << '\n';
    }
    close_output(output, path);
}

void write_hmf_csv(
    const std::filesystem::path& path,
    const analysis::HaloMassFunctionResult& hmf,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles) {
    require_source_provenance(
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a,
        "FoF-candidate HMF");
    if (!std::isfinite(fof_linking_length_b) || fof_linking_length_b <= 0.0
        || fof_min_particles == 0U) {
        throw std::invalid_argument("FoF-candidate HMF provenance is invalid");
    }
    auto output = open_output(path);
    output.precision(17);
    output << "# product_kind=analysis_fof_candidate_hmf\n";
    write_single_snapshot_csv_metadata(
        output,
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a);
    output << "# object_kind=fof_candidate\n"
           << "# selection=periodic_friends_of_friends\n"
           << "# fof_linking_length_b=" << fof_linking_length_b << '\n'
           << "# fof_min_particles=" << fof_min_particles << '\n'
           << "# mass_definition=" << hmf.mass_definition << '\n'
           << "# mass_unit=1e10_Msun_per_h\n"
           << "# volume=" << hmf.volume << '\n'
           << "# volume_unit=(Mpc/h)^3\n"
           << "# binning=logarithmic\n"
           << "# bin_convention=left_closed_right_open_last_closed\n"
           << "# bin_count=" << hmf.bins.size() << '\n'
           << "# delta_ln_mass=" << hmf.delta_ln_mass << '\n'
           << "# input_halo_count=" << hmf.input_halo_count << '\n'
           << "# underflow_count=" << hmf.underflow_count << '\n'
           << "# overflow_count=" << hmf.overflow_count << '\n'
           << "# poisson_error_definition=sqrt_count_over_volume_delta_ln_mass\n"
           << "# sample_variance_modeled=false\n"
           << "# poisson_error_is_confidence_interval=false\n"
           << "mass_low,mass_high,mass_geometric_mean,count,"
              "number_density_per_ln_mass,poisson_error_per_ln_mass\n";
    for (const auto& bin : hmf.bins) {
        output << bin.mass_low << ',' << bin.mass_high << ','
               << bin.mass_geometric_mean << ',' << bin.count << ','
               << bin.number_density_per_ln_mass << ','
               << bin.poisson_error_per_ln_mass << '\n';
    }
    close_output(output, path);
}

void write_xi_csv(
    const std::filesystem::path& path,
    const analysis::TwoPointResult& xi,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles) {
    require_source_provenance(
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a,
        "FoF-candidate 2PCF");
    if (!std::isfinite(fof_linking_length_b) || fof_linking_length_b <= 0.0
        || fof_min_particles == 0U) {
        throw std::invalid_argument("FoF-candidate 2PCF provenance is invalid");
    }
    auto output = open_output(path);
    output.precision(17);
    output << "# product_kind=analysis_fof_candidate_xi\n";
    write_single_snapshot_csv_metadata(
        output,
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a);
    output << "# object_kind=fof_candidate\n"
           << "# selection=periodic_friends_of_friends\n"
           << "# fof_linking_length_b=" << fof_linking_length_b << '\n'
           << "# fof_min_particles=" << fof_min_particles << '\n'
           << "# center=fof_candidate_circular_mass_weighted_com\n"
           << "# radius_unit=Mpc_per_h\n"
           << "# box_size_Mpc_h=" << xi.box_size << '\n'
           << "# estimator=" << xi.estimator << '\n'
           << "# input_point_count=" << xi.input_point_count << '\n'
           << "# used_point_count=" << xi.used_point_count << '\n'
           << "# dd_pair_count_in_range="
           << xi.data_data_pair_count_in_range << '\n'
           << "# random_point_count=" << xi.random_point_count << '\n'
           << "# dr_pair_count_in_range="
           << xi.data_random_pair_count_in_range << '\n'
           << "# rr_pair_count_in_range="
           << xi.random_random_pair_count_in_range << '\n'
           << "# binning="
           << (xi.binning == analysis::TwoPointBinning::Linear
                   ? "linear" : "logarithmic") << '\n'
           << "# bin_convention=" << xi.bin_convention << '\n'
           << "# random_catalog=" << xi.random_catalog << '\n'
           << "# reference_measure=" << xi.reference_measure << '\n'
           << "# tracer=" << xi.tracer_label << '\n'
           << "radius_low,radius_high,radius_midpoint,"
              "dd_raw,dd_normalized,expected_uniform_pair_probability,xi,"
              "dr_raw,rr_raw,dr_normalized,rr_normalized\n";
    for (const auto& bin : xi.bins) {
        output << bin.radius_low << ',' << bin.radius_high << ','
               << bin.radius_midpoint << ',' << bin.data_data_pair_count << ','
               << bin.data_data_normalized << ','
               << bin.expected_pair_probability << ',' << bin.xi << ','
               << bin.data_random_pair_count << ','
               << bin.random_random_pair_count << ','
               << bin.data_random_normalized << ','
               << bin.random_random_normalized << '\n';
    }
    close_output(output, path);
}

void write_bispectrum_csv(
    const std::filesystem::path& path,
    const std::vector<analysis::BispectrumBin>& bins,
    int analysis_mesh,
    core::Real box_size_Mpc_h,
    const analysis::BispectrumOptions& options,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a) {
    require_source_provenance(
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a,
        "Bispectrum output");
    if (analysis_mesh < 4
        || !std::isfinite(box_size_Mpc_h)
        || box_size_Mpc_h <= 0.0) {
        throw std::invalid_argument(
            "Bispectrum output requires an admitted mesh and positive box size");
    }
    if (!std::isfinite(options.max_k_fraction_nyquist)
        || options.max_k_fraction_nyquist <= 0.0
        || options.max_k_fraction_nyquist > 1.0) {
        throw std::invalid_argument(
            "Bispectrum output requires max_k_fraction_nyquist in (0,1]");
    }
    if (options.num_k_bins < 1
        || bins.size()
            != static_cast<std::size_t>(options.num_k_bins)) {
        throw std::invalid_argument(
            "Bispectrum output shell count does not match the requested bins");
    }
    const auto shell_is_represented = [](const auto& bin) {
        return std::round(bin.closed_triad_count_estimate) > 0.0;
    };
    const std::size_t represented_shell_count = static_cast<std::size_t>(
        std::count_if(
            bins.begin(), bins.end(), shell_is_represented));
    if (represented_shell_count == 0) {
        throw std::invalid_argument(
            "Bispectrum output requires at least one represented shell");
    }
    core::Real previous_k_mean = 0.0;
    bool have_previous_k_mean = false;
    for (const auto& bin : bins) {
        if (!std::isfinite(bin.closed_triad_count_estimate)
            || !std::isfinite(bin.closed_triad_integer_residual)
            || bin.closed_triad_integer_residual < 0.0
            || bin.closed_triad_integer_residual > 0.5) {
            throw std::invalid_argument(
                "Bispectrum closed-triad metadata is outside its finite domain");
        }
        if (!shell_is_represented(bin)) {
            if (bin.k_mean != 0.0 || bin.B_raw != 0.0
                || bin.B_corrected != 0.0
                || bin.closed_triad_count_estimate != 0.0
                || bin.closed_triad_integer_residual != 0.0) {
                throw std::invalid_argument(
                    "Bispectrum omitted shell carries noncanonical values");
            }
            continue;
        }
        if (!std::isfinite(bin.k_mean) || bin.k_mean <= 0.0
            || !std::isfinite(bin.B_raw)
            || !std::isfinite(bin.B_corrected)
            || (have_previous_k_mean && bin.k_mean <= previous_k_mean)) {
            throw std::invalid_argument(
                "Bispectrum represented shells are invalid or out of order");
        }
        previous_k_mean = bin.k_mean;
        have_previous_k_mean = true;
    }
    auto output = open_output(path);
    output.precision(17);
    const core::Real k_nyquist_h_Mpc = std::numbers::pi
        * static_cast<core::Real>(analysis_mesh) / box_size_Mpc_h;
    const core::Real configured_k_max_h_Mpc =
        options.max_k_fraction_nyquist * k_nyquist_h_Mpc;
    output << "# product_kind=analysis_bispectrum\n";
    write_single_snapshot_csv_metadata(
        output,
        source_snapshot,
        source_native_snapshot_object_sha256,
        snapshot_a);
    output << "# table_schema_version=2\n"
           << "# shell_index_base=0\n"
           << "# analysis_mesh=" << analysis_mesh << '\n'
           << "# configured_mesh=" << analysis_mesh << '\n'
           << "# estimator=equilateral_log_shell_fft_indicator_contraction\n"
           << "# mass_assignment=CIC\n"
           << "# cic_window_deconvolution_for_B_corrected=true\n"
           << "# interlaced=false\n"
           << "# signal_aliasing_removed=false\n"
           << "# poisson_shot_noise_subtracted_for_B_corrected=true\n"
           << "# shot_noise_power_weighting=ordered_closed_triads\n"
           << "# closed_triad_count_kind=floating_fft_indicator_contraction\n"
           << "# closed_triad_integer_residual=absolute_distance_to_nearest_integer_no_acceptance_threshold\n"
           << "# requested_shell_count=" << bins.size() << '\n'
           << "# represented_shell_count=" << represented_shell_count << '\n'
           << "# omitted_shell_count="
           << bins.size() - represented_shell_count << '\n'
           << "# omitted_shell_policy=nearest_integer_closed_triad_count_nonpositive\n"
           << "# represented_shell_indices_zero_based=";
    bool first_index = true;
    for (std::size_t index = 0; index < bins.size(); ++index) {
        if (!shell_is_represented(bins[index])) continue;
        if (!first_index) output << ',';
        output << index;
        first_index = false;
    }
    output << '\n' << "# omitted_shell_indices_zero_based=";
    first_index = true;
    for (std::size_t index = 0; index < bins.size(); ++index) {
        if (shell_is_represented(bins[index])) continue;
        if (!first_index) output << ',';
        output << index;
        first_index = false;
    }
    output << '\n'
           << "# max_k_fraction_nyquist="
           << options.max_k_fraction_nyquist << '\n'
           << "# k_nyquist_h_Mpc=" << k_nyquist_h_Mpc << '\n'
           << "# configured_k_max_h_Mpc=" << configured_k_max_h_Mpc
           << '\n'
           << "k_mean,B_raw,B_corrected,closed_triad_count_estimate,"
              "closed_triad_integer_residual,shell_index\n";
    for (std::size_t index = 0; index < bins.size(); ++index) {
        const auto& bin = bins[index];
        if (shell_is_represented(bin)) {
            output << bin.k_mean << ',' << bin.B_raw << ','
                   << bin.B_corrected << ','
                   << bin.closed_triad_count_estimate << ','
                   << bin.closed_triad_integer_residual << ','
                   << index << '\n';
        }
    }
    close_output(output, path);
}

void write_pair_links_json(
    const std::filesystem::path& path,
    const halo::PairLinkSet& pair_links,
    const std::filesystem::path& earlier_snapshot,
    std::string_view earlier_native_snapshot_object_sha256,
    core::Real earlier_scale_factor,
    const std::filesystem::path& later_snapshot,
    std::string_view later_native_snapshot_object_sha256,
    core::Real later_scale_factor,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles) {
    require_source_provenance(
        earlier_snapshot,
        earlier_native_snapshot_object_sha256,
        earlier_scale_factor,
        "Pair-link earlier snapshot");
    require_source_provenance(
        later_snapshot,
        later_native_snapshot_object_sha256,
        later_scale_factor,
        "Pair-link later snapshot");
    if (!std::isfinite(earlier_scale_factor)
        || !std::isfinite(later_scale_factor)
        || earlier_scale_factor <= 0.0
        || later_scale_factor <= earlier_scale_factor
        || !std::isfinite(fof_linking_length_b)
        || fof_linking_length_b <= 0.0
        || fof_min_particles == 0U) {
        throw std::invalid_argument(
            "Pair-link publication requires finite ordered snapshots and explicit FoF selection");
    }
    auto output = open_output(path);
    output << std::setprecision(17)
           << "{\n"
           << "  \"product_kind\": \"analysis_pair_links\",\n";
    write_particle_tracer_json_metadata(output);
    output << "  \"object_kind\": \"fof_candidate\",\n"
           << "  \"candidate_selection\": \"periodic_friends_of_friends\",\n"
           << "  \"fof_linking_length_b\": " << fof_linking_length_b << ",\n"
           << "  \"fof_min_particles\": " << fof_min_particles << ",\n"
           << "  \"earlier_native_snapshot_object_sha256\": \""
           << earlier_native_snapshot_object_sha256 << "\",\n"
           << "  \"later_native_snapshot_object_sha256\": \""
           << later_native_snapshot_object_sha256 << "\",\n"
           << "  \"method\": \"stable_particle_id_overlap_two_snapshots\",\n"
           << "  \"earlier_scale_factor\": " << earlier_scale_factor << ",\n"
           << "  \"later_scale_factor\": " << later_scale_factor << ",\n"
           << "  \"links\": [\n";
    for (std::size_t index = 0; index < pair_links.links.size(); ++index) {
        const auto& link = pair_links.links[index];
        output << "    {\"earlier_candidate_id\": "
               << link.earlier_candidate_id
               << ", \"later_candidate_id\": "
               << link.later_candidate_id
               << ", \"shared_particle_count\": "
               << link.shared_particle_count
               << ", \"earlier_shared_fraction\": "
               << link.earlier_shared_fraction
               << ", \"later_shared_fraction\": "
               << link.later_shared_fraction
               << ", \"primary_earlier_candidate_for_later\": "
               << (link.is_primary_earlier_candidate_for_later
                    ? "true" : "false")
               << ", \"unambiguous_forward_match\": "
               << (link.is_unambiguous_forward_match ? "true" : "false")
               << ", \"ambiguous\": "
               << (link.is_ambiguous ? "true" : "false") << "}"
               << (index + 1 < pair_links.links.size() ? "," : "") << '\n';
    }
    output << "  ],\n  \"unmatched_earlier_candidate_ids\": [";
    for (std::size_t index = 0;
         index < pair_links.unmatched_earlier_candidate_ids.size();
         ++index) {
        if (index != 0) output << ", ";
        output << pair_links.unmatched_earlier_candidate_ids[index];
    }
    output << "],\n  \"new_later_candidate_ids\": [";
    for (std::size_t index = 0;
         index < pair_links.new_later_candidate_ids.size();
         ++index) {
        if (index != 0) output << ", ";
        output << pair_links.new_later_candidate_ids[index];
    }
    output << "],\n  \"link_count\": " << pair_links.links.size()
           << "\n}\n";
    close_output(output, path);
}

} // namespace cosmo_nbody::app::nbody_analyze
