#include "nbody_analyze_halo_stage.hpp"

#include "nbody_analyze_deblend_stage.hpp"
#include "nbody_analyze_outputs.hpp"
#include "nbody_analyze_pair_link_halos.hpp"
#include "nbody_analyze_parallel.hpp"

#include "cosmo_nbody/analysis/halo_derived_properties.hpp"
#include "cosmo_nbody/analysis/halo_mass_function.hpp"
#include "cosmo_nbody/analysis/halo_shape.hpp"
#include "cosmo_nbody/analysis/two_point_correlation.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"
#include "cosmo_nbody/io/checked_output_file.hpp"
#include "cosmo_nbody/io/fof_analysis_catalog.hpp"
#include "cosmo_nbody/io/output_schema.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {
namespace {

constexpr std::size_t halo_result_batch_target_bytes =
    std::size_t{4} * 1024 * 1024;

template <typename T>
std::size_t halo_result_batch_capacity(std::size_t row_count) noexcept {
    if (row_count == 0) return 0;
    const std::size_t byte_limited = std::max<std::size_t>(
        1,
        halo_result_batch_target_bytes / sizeof(T));
    return std::min(row_count, byte_limited);
}

void release_memberships(
    std::vector<halo::FoFMembership>& memberships) noexcept {
    for (auto& membership : memberships) {
        std::vector<std::size_t>{}.swap(membership.particle_indices);
    }
    std::vector<halo::FoFMembership>{}.swap(memberships);
}

io::CheckedOutputFile open_halo_output(const std::filesystem::path& path) {
    return io::open_checked_output_file(path, "halo analysis product");
}

void close_halo_output(
    io::CheckedOutputFile& output,
    const std::filesystem::path& path) {
    io::close_checked_output_file(output, path, "halo analysis product");
}

const char* shape_status(const analysis::HaloShapeResult& shape) noexcept {
    if (shape.periodic_cut_locus_ambiguous) {
        return "periodic_cut_locus_ambiguous";
    }
    if (shape.zero_spatial_extent) return "zero_spatial_extent";
    if (shape.solver_indeterminate) return "solver_indeterminate";
    if (shape.rank_deficient_reduced_metric) {
        return "rank_deficient_reduced_metric";
    }
    if (!shape.computed()) return "insufficient_particles";
    return shape.converged
        ? "computed_converged"
        : "computed_not_converged";
}

analysis::HaloShapeOptions shape_options_from_request(
    const analysis::AnalysisRequest& request) {
    if (!request.shape_use_reduced_tensor.has_value()) {
        throw std::logic_error(
            "Validated shape request lost explicit tensor method");
    }
    analysis::HaloShapeOptions options;
    options.use_reduced_tensor = *request.shape_use_reduced_tensor;
    options.max_iterations = request.shape_max_iterations;
    options.convergence_tolerance = request.shape_convergence_tolerance;
    return options;
}

core::Real hmf_mass_msun_h_to_native(core::Real mass_msun_h) {
    const core::Real native =
        mass_msun_h / cosmology::units::MassUnit_in_Msun_per_h;
    if (!std::isfinite(native) || native <= 0.0) {
        throw std::overflow_error(
            "HMF mass bound in Msun/h is not representable in HYOWON native mass units");
    }
    return native;
}

} // namespace

HaloStageResult run_halo_stage(
    const analysis::AnalysisRequest& request,
    const HaloStageContext& context,
    const core::ParticleStore& particles,
    core::Real snapshot_a,
    const std::filesystem::path& output_directory) {
    HaloStageResult result;
    if (request.skip_halos) return result;

    halo::FoFMembershipFinder finder(
        context.box_size,
        context.mean_spacing,
        request.fof_linking_length_b,
        request.fof_min_particles);
    auto memberships = finder.find_memberships(particles);
    std::vector<analysis::HaloDerivedProperties> properties(
        memberships.size());
    parallel_for_indices(
        memberships.size(),
        [&](std::size_t group_index) {
            properties[group_index] =
                analysis::HaloDerivedPropertyAnalyzer::compute(
                    particles,
                    memberships[group_index],
                    context.box_size,
                    snapshot_a);
        });

    io::FoFAnalysisCatalogIO fof_catalog_io(context.fof_catalog);
    const std::filesystem::path fof_catalog_path =
        output_directory / "analysis_fof_candidate_all.hdf5";
    fof_catalog_io.write_durable(
        fof_catalog_path.string(),
        memberships,
        properties,
        particles,
        request.fof_linking_length_b,
        request.fof_min_particles,
        snapshot_a,
        io::ProductLineage::DerivedAnalysis);

    if (request.write_standard_so) {
        run_deblended_host_stage(
            request,
            context.so,
            particles,
            memberships,
            properties,
            snapshot_a,
            context.source_native_snapshot_object_sha256,
            output_directory);
    }

    if (request.hmf_min_mass.has_value()) {
        std::vector<core::Real> masses;
        masses.reserve(properties.size());
        for (const auto& property : properties) {
            masses.push_back(property.mass);
        }
        analysis::HaloMassFunctionOptions options;
        options.min_mass = hmf_mass_msun_h_to_native(*request.hmf_min_mass);
        options.max_mass = hmf_mass_msun_h_to_native(*request.hmf_max_mass);
        options.num_bins = request.hmf_bins;
        options.mass_definition = io::schema::VAL_FOF_MASS_DEFINITION;
        const auto hmf = analysis::HaloMassFunction::compute(
            masses, context.box_size, options);
        write_hmf_csv(
            output_directory / "analysis_fof_candidate_hmf.csv",
            hmf,
            request.snapshot_path,
            context.source_native_snapshot_object_sha256,
            snapshot_a,
            request.fof_linking_length_b,
            request.fof_min_particles);
    }

    if (request.xi_min_radius.has_value()) {
        if (properties.size() < 2) {
            throw std::runtime_error(
                "Requested candidate-center 2PCF requires at least two FoF candidates");
        }
        std::vector<core::Vec3> centers;
        centers.reserve(properties.size());
        for (const auto& property : properties) {
            centers.push_back(property.center_of_mass);
        }
        analysis::TwoPointOptions options;
        options.min_radius = *request.xi_min_radius;
        options.max_radius = *request.xi_max_radius;
        options.num_bins = request.xi_bins;
        if (request.xi_binning == "linear") {
            options.binning = analysis::TwoPointBinning::Linear;
        } else if (request.xi_binning == "logarithmic") {
            options.binning = analysis::TwoPointBinning::Logarithmic;
        } else {
            throw std::logic_error(
                "Validated 2PCF request lost explicit binning method");
        }
        options.tracer_label = "fof_candidate_all_centers";
        const auto xi = analysis::TwoPointCorrelation::compute(
            centers,
            context.box_size,
            options);
        write_xi_csv(
            output_directory / "analysis_fof_candidate_xi.csv",
            xi,
            request.snapshot_path,
            context.source_native_snapshot_object_sha256,
            snapshot_a,
            request.fof_linking_length_b,
            request.fof_min_particles);
    }

    if (request.write_shapes) {
        const analysis::HaloShapeOptions options =
            shape_options_from_request(request);
        const std::size_t batch_capacity =
            halo_result_batch_capacity<analysis::HaloShapeResult>(
                memberships.size());
        std::unique_ptr<analysis::HaloShapeResult[]> shape_batch;
        if (batch_capacity != 0) {
            shape_batch =
                std::make_unique<analysis::HaloShapeResult[]>(batch_capacity);
        }
        const auto path =
            output_directory / "analysis_fof_candidate_shapes.csv";
        auto output = open_halo_output(path);
        output.precision(17);
        output << "# product_kind=analysis_fof_candidate_shapes\n";
        write_single_snapshot_csv_metadata(
            output,
            request.snapshot_path,
            context.source_native_snapshot_object_sha256,
            snapshot_a);
        output << "# object_kind=fof_candidate\n"
               << "# selection=periodic_friends_of_friends\n"
               << "# fof_linking_length_b="
               << request.fof_linking_length_b << '\n'
               << "# fof_min_particles=" << request.fof_min_particles << '\n'
               << "# estimator=fixed_fof_membership_shape_tensor\n"
               << "# center=fof_candidate_circular_mass_weighted_com\n"
               << "# aperture=fixed_fof_candidate_all_membership\n"
               << "# membership_reselection=false\n"
               << "# substructure_removal=false\n"
               << "# use_reduced_tensor="
               << (options.use_reduced_tensor ? "true" : "false") << '\n'
               << "# tensor_weighting="
               << (options.use_reduced_tensor
                       ? "mass_over_iterated_ellipsoidal_radius_squared"
                       : "mass")
               << '\n'
               << "# tensor_normalization=total_fixed_membership_mass\n"
               << "# reduced_zero_displacement_policy=exclude_undefined_directional_numerator_include_mass_in_normalization\n"
               << "# directional_particle_count=nonzero_directionally_unique_periodic_displacement_members\n"
               << "# reduced_iteration_initialization=unreduced_tensor_eigenbasis\n"
               << "# iteration_count=requested_tensor_solve_iterations_excludes_reduced_bootstrap\n"
               << "# projected_displacement_evaluation=binary64_epsilon_bounded_wide_real_with_exact_dyadic_fallback\n"
               << "# tensor_factorization=streaming_qr_one_sided_svd\n"
               << "# psd_preservation=square_root_factorization\n"
               << "# rank_determination=exact_binary64_displacement_span_predicate\n"
               << "# svd_termination=strictly_decreasing_normalized_offdiagonal_gram_energy\n"
               << "# eigensystem_certificate=outward_rounded_bauer_fike_residual_on_returned_axes_of_represented_qr_gram\n"
               << "# numerical_failure_status=solver_indeterminate\n"
               << "# geometric_unavailable_statuses=periodic_cut_locus_ambiguous,zero_spatial_extent,rank_deficient_reduced_metric\n"
               << "# convergence_metric=member_natural_log_relative_tensor_weight_pairwise_range\n"
               << "# max_iterations=" << options.max_iterations << '\n'
               << "# convergence_tolerance="
               << options.convergence_tolerance << '\n'
               << "# shape_availability=computed_flag\n"
               << "# triaxiality_definition=(lambda_a-lambda_b)/(lambda_a-lambda_c)\n"
               << "# triaxiality_availability=triaxiality_defined_flag\n"
               << "# unavailable_numeric_fields=zero_placeholders\n"
               << "id,num_particles,directional_particle_count,shape_status,computed,b_over_a,c_over_a,"
                  "triaxiality,triaxiality_defined,converged,iterations,"
                  "use_reduced_tensor\n";
        for (std::size_t batch_begin = 0;
             batch_begin < memberships.size();
             batch_begin += batch_capacity) {
            const std::size_t batch_count = std::min(
                batch_capacity, memberships.size() - batch_begin);
            parallel_for_indices(
                batch_count,
                [&](std::size_t local_index) {
                    const std::size_t group_index =
                        batch_begin + local_index;
                    shape_batch[local_index] =
                        analysis::HaloShapeAnalyzer::compute(
                            particles,
                            properties[group_index].center_of_mass,
                            context.box_size,
                            memberships[group_index].particle_indices,
                            options);
                });
            for (std::size_t local_index = 0;
                 local_index < batch_count;
                 ++local_index) {
                const std::size_t group_index = batch_begin + local_index;
                const auto& membership = memberships[group_index];
                const auto& shape = shape_batch[local_index];
                output << membership.id << ',' << shape.particle_count << ','
                       << shape.directional_particle_count << ','
                       << shape_status(shape) << ','
                       << (shape.computed() ? 1 : 0) << ','
                       << shape.axis_ratio_b_over_a << ','
                       << shape.axis_ratio_c_over_a << ',' << shape.triaxiality
                       << ',' << (shape.triaxiality_defined() ? 1 : 0)
                       << ',' << (shape.converged ? 1 : 0) << ','
                       << shape.iterations << ','
                       << (options.use_reduced_tensor ? 1 : 0) << '\n';
            }
        }
        close_halo_output(output, path);
    }

    if (!request.progenitor_snapshot.empty()) {
        result.later_pair_link_candidates = materialize_pair_link_halos(
            memberships, properties, particles);
    }

    release_memberships(memberships);
    std::vector<analysis::HaloDerivedProperties>{}.swap(properties);
    return result;
}

} // namespace cosmo_nbody::app::nbody_analyze
