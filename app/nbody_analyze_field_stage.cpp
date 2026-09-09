#include "nbody_analyze_field_stage.hpp"

#include "nbody_analyze_outputs.hpp"

#include "cosmo_nbody/analysis/bispectrum.hpp"
#include "cosmo_nbody/analysis/field_cross_correlation.hpp"
#include "cosmo_nbody/analysis/filament_statistics.hpp"
#include "cosmo_nbody/analysis/matter_power_spectrum.hpp"
#include "cosmo_nbody/analysis/periodic_domain.hpp"
#include "cosmo_nbody/analysis/phase_space_cross.hpp"
#include "cosmo_nbody/io/analysis_snapshot_reader.hpp"
#include "cosmo_nbody/io/checked_output_file.hpp"
#include "cosmo_nbody/io/output_schema.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace cosmo_nbody::app::nbody_analyze {

void run_field_statistics_stage(
    const analysis::AnalysisRequest& request,
    const io::SnapshotDescriptor& snapshot,
    const io::SnapshotDescriptor* cross_snapshot,
    const core::ParticleStore& particles,
    core::Real snapshot_a,
    int configured_mesh,
    const std::filesystem::path& output_directory) {
    const analysis::PeriodicDomain domain(snapshot.box_size_Mpc_h);
    if (!request.field_interlaced.has_value()) {
        throw std::logic_error(
            "Validated analysis request lost explicit field-interlacing policy");
    }
    if (!request.field_shot_noise_treatment.has_value()) {
        throw std::logic_error(
            "Validated analysis request lost explicit field shot-noise policy");
    }

    const bool field_interlaced = *request.field_interlaced;
    const core::Real field_k_fraction =
        request.field_max_k_fraction_nyquist;
    const core::Real field_k_nyquist = std::numbers::pi
        * static_cast<core::Real>(configured_mesh) / domain.box_size();
    const core::Real field_evaluated_k_max =
        field_k_fraction * field_k_nyquist;
    if (!std::isfinite(field_evaluated_k_max)
        || !(field_evaluated_k_max > 0.0)) {
        throw std::logic_error(
            "Validated field estimator produced an invalid physical k support");
    }

    {
        analysis::MatterPowerSpectrum pk(
            domain,
            configured_mesh,
            request.resource_policy.memory);
        analysis::PowerSpectrumOptions options;
        options.num_bins = request.bins;
        options.subtract_shot_noise =
            *request.field_shot_noise_treatment
            == analysis::FieldShotNoiseTreatment::SubtractCICAliasedPoisson;
        options.interlaced = field_interlaced;
        options.max_k_fraction_nyquist = field_k_fraction;
        const auto bins = pk.compute_pk(particles, options);
        write_pk_csv(
            output_directory / "analysis_pk.csv",
            bins,
            configured_mesh,
            domain.box_size(),
            options,
            request.snapshot_path,
            snapshot.native_snapshot_object_sha256,
            snapshot_a);
    }

    if (!request.cross_snapshot.empty()) {
        if (!request.cross_mesh_explicit || request.cross_mesh <= 0) {
            throw std::logic_error(
                "Validated cross-snapshot request lost its explicit estimator mesh");
        }
        if (cross_snapshot == nullptr) {
            throw std::logic_error(
                "Cross-snapshot descriptor was not checked before payload analysis");
        }
        const io::SnapshotDescriptor& other_descriptor = *cross_snapshot;
        const int mesh = request.cross_mesh;
        const core::Real cross_k_nyquist = std::numbers::pi
            * static_cast<core::Real>(mesh) / domain.box_size();
        if (!std::isfinite(cross_k_nyquist)
            || cross_k_nyquist < field_evaluated_k_max) {
            throw std::invalid_argument(
                "--cross-mesh cannot represent the primary field's physical k support; "
                "increase --cross-mesh or reduce --field-max-k-fraction-nyquist");
        }
        const core::Real cross_k_fraction =
            field_evaluated_k_max / cross_k_nyquist;
        if (!std::isfinite(cross_k_fraction) || cross_k_fraction <= 0.0) {
            throw std::logic_error(
                "Field-cross support fraction is not representable");
        }

        core::ParticleStore other;
        core::Real other_a = 0.0;
        io::AnalysisSnapshotReader::read_complete(
            request.cross_snapshot, other_descriptor, other, other_a);
        other.release_accelerations();

        if (request.write_cross_phase_space_summary) {
            if (other_a != snapshot_a) {
                throw std::invalid_argument(
                    "phase-space cross summary requires reference and candidate at the exact same epoch");
            }
            if (other_descriptor.particles_per_dimension
                != snapshot.particles_per_dimension) {
                throw std::invalid_argument(
                    "phase-space cross summary requires the same particles-per-dimension coordinate");
            }

            std::size_t matching_workspace_limit_bytes = 0;
            if (request.resource_policy.memory_budget_gib > 0.0) {
                const long double requested_bytes =
                    static_cast<long double>(
                        request.resource_policy.memory_budget_gib)
                    * 1024.0L * 1024.0L * 1024.0L;
                if (requested_bytes < 1.0L) {
                    matching_workspace_limit_bytes = 1;
                } else if (requested_bytes
                    >= static_cast<long double>(
                        std::numeric_limits<std::size_t>::max())) {
                    matching_workspace_limit_bytes =
                        std::numeric_limits<std::size_t>::max();
                } else {
                    matching_workspace_limit_bytes =
                        static_cast<std::size_t>(requested_bytes);
                }
            }

            const auto phase_space =
                analysis::compare_phase_space_by_stable_id(
                    other,
                    other_a,
                    particles,
                    snapshot_a,
                    domain.box_size(),
                    snapshot.particles_per_dimension,
                    matching_workspace_limit_bytes);

            const auto phase_path =
                output_directory / "analysis_phase_space_cross.csv";
            auto phase_output = io::open_checked_output_file(
                phase_path, "phase-space cross analysis product");
            phase_output << std::setprecision(17)
                         << "# product_kind=analysis_phase_space_cross\n"
                         << "# reference_snapshot=" << request.cross_snapshot << '\n'
                         << "# candidate_snapshot=" << request.snapshot_path << '\n'
                         << "# comparison_basis="
                            "same_epoch_exact_stable_particle_id_and_mass_correspondence_in_shared_periodic_volume\n"
                         << "# reference_native_snapshot_object_sha256="
                         << other_descriptor.native_snapshot_object_sha256 << '\n'
                         << "# candidate_native_snapshot_object_sha256="
                         << snapshot.native_snapshot_object_sha256 << '\n';
            write_particle_tracer_csv_metadata(phase_output);
            phase_output
                << "# reference_a=" << other_a << '\n'
                << "# candidate_a=" << snapshot_a << '\n'
                << "# particle_matching="
                   "reference_stable_id_sorted_index_binary_search_candidate_ids\n"
                << "# particle_mass_match=exact_binary64_per_stable_particle_id\n"
                << "# particles_per_dimension="
                << snapshot.particles_per_dimension << '\n'
                << "# particle_grid_coordinate_match=exact\n"
                << "# matching_workspace=single_size_t_index_vector\n"
                << "# matching_workspace_bytes="
                << phase_space.matching_workspace_bytes << '\n'
                << "# position_difference="
                   "candidate_minus_reference_periodic_minimum_image_per_component\n"
                << "# minimum_image_interval=[-L/2,L/2)\n"
                << "# minimum_image_tie="
                   "positive_half_box_maps_to_negative_half_box\n"
                << "# position_normalization=mean_particle_spacing_L_over_N\n"
                << "# position_interpretation="
                   "instantaneous_shortest_periodic_separation_not_unwrapped_trajectory_path_length\n"
                << "# canonical_momentum_difference="
                   "candidate_minus_reference_particle_store_p\n"
                << "# canonical_momentum_convention="
                << io::schema::VAL_MOMENTUM_CONVENTION << '\n'
                << "# canonical_momentum_unit=a_times_km_per_s\n"
                << "# velocity_difference="
                   "same_epoch_candidate_minus_reference_peculiar_velocity\n"
                << "# velocity_reconstruction="
                   "reported_momentum_rms_and_max_divided_by_common_scale_factor_a\n"
                << "# velocity_unit=km_per_s\n"
                << "# rms_accumulation="
                   "position_and_canonical_momentum_use_two_pass_max_scaled_exact_positive_binary64_sum_round_once_velocity_is_derived_from_momentum_at_common_epoch\n"
                << "# exact_binary64_phase_space_equal_scope="
                   "stable_id_matched_position_and_canonical_momentum_components\n"
                << "# convergence_assessment=false\n"
                << "# study_relationship_inferred=false\n"
                << "particle_count,position_rms_mean_spacing,"
                   "position_max_mean_spacing,canonical_momentum_rms_a_km_s,"
                   "canonical_momentum_max_a_km_s,velocity_rms_km_s,velocity_max_km_s,"
                   "exact_binary64_phase_space_equal\n"
                << phase_space.particle_count << ','
                << phase_space.position_rms_mean_spacing << ','
                << phase_space.position_max_mean_spacing << ','
                << phase_space.canonical_momentum_rms_a_km_s << ','
                << phase_space.canonical_momentum_max_a_km_s << ','
                << phase_space.velocity_rms_km_s << ','
                << phase_space.velocity_max_km_s << ','
                << (phase_space.exact_binary64_phase_space_equal ? "true" : "false")
                << '\n';
            io::close_checked_output_file(
                phase_output,
                phase_path,
                "phase-space cross analysis product");
        }

        analysis::FieldCrossCorrelation cross(
            domain,
            mesh,
            request.resource_policy.memory);
        analysis::FieldComparisonOptions options;
        options.num_bins = request.bins;
        options.interlaced = field_interlaced;
        // Preserve the primary field's absolute physical support directly.
        // The mesh-relative fraction is output metadata only; reconstructing the
        // absolute cutoff through fraction*Nyquist would add a rounding-dependent
        // support coordinate.
        options.max_k_h_Mpc = field_evaluated_k_max;
        const auto summary = cross.compare(other, particles, options);
        if (summary.bins.size()
            != static_cast<std::size_t>(options.num_bins)) {
            throw std::logic_error(
                "Field-cross estimator did not preserve its requested shell count");
        }
        if (summary.evaluated_k_max != field_evaluated_k_max) {
            throw std::logic_error(
                "Field-cross estimator did not preserve the primary absolute k support");
        }
        std::size_t represented_shell_count = 0;
        for (const auto& bin : summary.bins) {
            if (bin.mode_count != 0) ++represented_shell_count;
        }

        const auto path = output_directory / "analysis_field_cross.csv";
        auto output = io::open_checked_output_file(
            path, "field-cross analysis product");
        output << std::setprecision(17)
               << "# product_kind=analysis_field_cross\n"
               << "# reference_snapshot=" << request.cross_snapshot << '\n'
               << "# candidate_snapshot=" << request.snapshot_path << '\n'
               << "# comparison_basis=shared_periodic_volume_and_phase_space_semantics\n"
               << "# reference_native_snapshot_object_sha256="
               << other_descriptor.native_snapshot_object_sha256 << '\n'
               << "# candidate_native_snapshot_object_sha256="
               << snapshot.native_snapshot_object_sha256 << '\n';
        write_particle_tracer_csv_metadata(output);
        output << "# reference_producer_pm_mesh_per_dimension="
               << other_descriptor.pm_mesh_per_dimension << '\n'
               << "# candidate_producer_pm_mesh_per_dimension="
               << snapshot.pm_mesh_per_dimension << '\n'
               << "# comparison_support_basis=primary_field_absolute_k_max\n"
               << "# binning=logarithmic\n"
               << "# bin_convention=left_closed_right_open_last_closed\n"
               << "# shell_count=" << summary.bins.size() << '\n'
               << "# represented_shell_count=" << represented_shell_count << '\n'
               << "# empty_shell_count="
               << summary.bins.size() - represented_shell_count << '\n'
               << "# estimator_mesh_per_dimension=" << mesh << '\n'
               << "# k_fundamental_h_Mpc=" << summary.k_fundamental << '\n'
               << "# k_nyquist_h_Mpc=" << summary.k_nyquist << '\n'
               << "# primary_field_evaluated_k_max_h_Mpc="
               << field_evaluated_k_max << '\n'
               << "# reference_a=" << other_a << '\n'
               << "# candidate_a=" << snapshot_a << '\n'
               << "# mass_assignment=CIC\n"
               << "# field=particle_mass_weighted_density_contrast\n"
               << "# k_unit=h_per_Mpc\n"
               << "# power_unit=(Mpc_per_h)^3\n"
               << "# fourier_mode_normalization=forward_fft_divided_by_mesh_cell_count\n"
               << "# power_normalization=box_volume_times_mode_product\n"
               << "# cic_window_deconvolution="
                  "per_mode_amplitude_divided_by_full_3d_CIC_window_before_auto_cross_and_residual_power\n"
               << "# interlaced=" << (options.interlaced ? "true" : "false") << '\n'
               << "# shot_noise_subtracted=false\n"
               << "# signal_aliasing_removed=false\n"
               << "# max_k_fraction_nyquist="
               << cross_k_fraction << '\n'
               << "# evaluated_k_max=" << summary.evaluated_k_max << '\n'
               << "# p_delta_definition="
                  "direct_shell_mean_power_of_candidate_minus_reference_fourier_mode\n"
               << "# p_delta_numeric_policy="
                  "accumulated_from_mode_residual_not_reconstructed_from_rounded_auto_cross_columns\n"
               << "# delta_p_fraction_definition=p_candidate_over_p_reference_minus_one\n"
               << "# transfer_amplitude_ratio_definition=sqrt_p_candidate_over_p_reference\n"
               << "# e_delta_definition=sqrt_p_delta_over_p_reference\n"
               << "# relative_metric_policy="
                  "valid_only_when_relative_metrics_defined_is_true\n"
               << "# correlation_policy=raw_floating_ratio_not_clamped_to_unit_interval\n"
               << "# correlation_defined_policy="
                  "false_when_geometric_auto_power_normalization_is_zero\n"
               << "# correlation_validation_basis="
                  "producer_long_double_shell_accumulators_not_reconstructible_from_rounded_means\n"
               << "# absolute_cauchy_excess=max(0,abs(r_k)-1)_diagnostic_only_when_correlation_defined\n"
               << "# range_interpretation=explicit_estimator_support_measurement_only_alias_accuracy_requires_independent_convergence\n"
               << "# global_reference_mode_rms=" << summary.reference_rms << '\n'
               << "# global_residual_mode_rms=" << summary.residual_rms << '\n'
               << "# global_mode_rms_definition="
                  "hermitian_multiplicity_weighted_rms_of_cic_deconvolved_density_contrast_fourier_mode_amplitudes_over_evaluated_nonzero_k_support\n"
               << "# normalized_residual_definition="
                  "global_residual_mode_rms_over_global_reference_mode_rms\n"
               << "# normalized_residual_policy="
                  "valid_only_when_normalized_residual_defined_is_true\n"
               << "# normalized_residual=" << summary.normalized_residual
               << '\n'
               << "# normalized_residual_defined="
               << (summary.normalized_residual_defined ? "true" : "false")
               << '\n'
               << "shell_index,k_low,k_mean,k_high,p_reference,p_candidate,p_cross,p_delta,"
                  "delta_p_fraction,transfer_amplitude_ratio,e_delta,relative_metrics_defined,"
                  "r_k,correlation_defined,absolute_cauchy_excess,mode_count\n";
        for (const auto& bin : summary.bins) {
            if (bin.mode_count == 0) continue;
            output << bin.shell_index << ','
                   << bin.k_low << ',' << bin.k_mean << ',' << bin.k_high
                   << ',' << bin.p_a << ',' << bin.p_b
                   << ',' << bin.p_cross << ',' << bin.p_delta
                   << ',' << bin.delta_p_fraction
                   << ',' << bin.transfer_amplitude_ratio
                   << ',' << bin.e_delta
                   << ',' << (bin.relative_metrics_defined ? "true" : "false")
                   << ',' << bin.r
                   << ',' << (bin.correlation_defined ? "true" : "false")
                   << ',' << bin.absolute_cauchy_excess
                   << ',' << bin.mode_count << '\n';
        }
        io::close_checked_output_file(
            output, path, "field-cross analysis product");
    }

    if (request.write_tidal_web) {
        analysis::TidalWebOptions options;
        options.mesh_size = configured_mesh;
        options.gaussian_smoothing_radius =
            request.tidal_web_gaussian_smoothing_radius_Mpc_h;
        options.lambda_threshold = request.tidal_web_lambda_threshold;
        options.materialize_filament_directions = false;
        options.memory_policy = request.resource_policy.memory;
        const auto summary = analysis::FilamentStatistics::classify_tidal_web(
            domain, particles, options);
        const auto path = output_directory / "analysis_tidal_web.json";
        auto output = io::open_checked_output_file(
            path, "tidal-web analysis product");
        const bool smoothed = summary.gaussian_smoothing_radius > 0.0;
        output << std::setprecision(17)
               << "{\n"
               << "  \"product_kind\": \"analysis_tidal_web\",\n"
               << "  \"source_native_snapshot_object_sha256\": \""
               << snapshot.native_snapshot_object_sha256 << "\",\n"
               << "  \"scale_factor\": " << snapshot_a << ",\n";
        write_particle_tracer_json_metadata(output);
        output << "  \"mesh_per_dimension\": " << options.mesh_size << ",\n"
               << "  \"mesh_matches_simulation\": "
               << (static_cast<std::uint64_t>(configured_mesh)
                        == snapshot.pm_mesh_per_dimension
                    ? "true"
                    : "false")
               << ",\n"
               << "  \"spectral_smoothing\": "
               << (smoothed ? "true" : "false") << ",\n"
               << "  \"spectral_filter\": \"gaussian_exp_minus_k2R2_over_2\",\n"
               << "  \"gaussian_smoothing_radius_Mpc_h\": "
               << summary.gaussian_smoothing_radius << ",\n"
               << "  \"lambda_threshold\": " << summary.lambda_threshold << ",\n"
               << "  \"void_volume_fraction\": "
               << summary.void_volume_fraction << ",\n"
               << "  \"sheet_volume_fraction\": "
               << summary.sheet_volume_fraction << ",\n"
               << "  \"filament_volume_fraction\": "
               << summary.filament_volume_fraction << ",\n"
               << "  \"node_volume_fraction\": "
               << summary.node_volume_fraction << "\n}\n";
        io::close_checked_output_file(
            output, path, "tidal-web analysis product");
    }

    if (request.write_bispectrum) {
        analysis::BispectrumEstimator estimator(
            domain,
            configured_mesh,
            request.resource_policy.memory);
        analysis::BispectrumOptions options;
        options.num_k_bins = request.bispectrum_bins;
        options.max_k_fraction_nyquist =
            request.bispectrum_max_k_fraction_nyquist;
        const auto bins = estimator.compute_equilateral(particles, options);
        write_bispectrum_csv(
            output_directory / "analysis_bispectrum.csv",
            bins,
            configured_mesh,
            domain.box_size(),
            options,
            request.snapshot_path,
            snapshot.native_snapshot_object_sha256,
            snapshot_a);
    }
}

} // namespace cosmo_nbody::app::nbody_analyze
