#include "nbody_analyze_named_so_products.hpp"
#include "nbody_analyze_outputs.hpp"

#include "cosmo_nbody/analysis/halo_membership_catalog.hpp"
#include "cosmo_nbody/analysis/halo_shape.hpp"
#include "cosmo_nbody/analysis/halo_spin.hpp"
#include "cosmo_nbody/analysis/halo_vmax.hpp"
#include "cosmo_nbody/io/checked_output_file.hpp"
#include "cosmo_nbody/io/output_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {
namespace {

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

} // namespace

void write_named_so_derived_products(
    const std::filesystem::path& path,
    const core::ParticleStore& particles,
    const halo::ExactPeriodicApertureQuery* aperture_query,
    analysis::HaloMembershipCatalogWriter& membership_output,
    const std::vector<halo::StandardSOSeedEvaluation>& evaluations,
    core::Real box_size,
    core::Real scale_factor,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles,
    std::size_t peak_density_k_neighbors,
    std::size_t deblended_min_particles,
    core::Real peak_saddle_merge_ratio,
    const analysis::HaloShapeOptions& shape_options) {
    const bool canonical_source_sha256 =
        source_native_snapshot_object_sha256.size() == 64U
        && std::all_of(
            source_native_snapshot_object_sha256.begin(),
            source_native_snapshot_object_sha256.end(),
            [](char value) {
                return (value >= '0' && value <= '9')
                    || (value >= 'a' && value <= 'f');
            });
    if (!std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(scale_factor) || scale_factor <= 0.0
        || source_snapshot.empty()
        || !canonical_source_sha256
        || !std::isfinite(fof_linking_length_b)
        || fof_linking_length_b <= 0.0
        || fof_min_particles == 0U
        || peak_density_k_neighbors == 0U
        || deblended_min_particles == 0U
        || !std::isfinite(peak_saddle_merge_ratio)
        || peak_saddle_merge_ratio < 0.0
        || peak_saddle_merge_ratio > 1.0) {
        throw std::invalid_argument(
            "Named SO derived products require complete source and seed-selection provenance");
    }
    if (shape_options.max_iterations <= 0
        || !std::isfinite(shape_options.convergence_tolerance)
        || shape_options.convergence_tolerance <= 0.0) {
        throw std::invalid_argument(
            "Named SO shape products require explicit positive iteration count and convergence tolerance");
    }

    bool requires_aperture_query = false;
    for (const auto& evaluation : evaluations) {
        for (const auto& measurement : evaluation.measurements) {
            requires_aperture_query = requires_aperture_query
                || measurement.crossing_resolved;
        }
    }
    if (requires_aperture_query && aperture_query == nullptr) {
        throw std::invalid_argument(
            "Resolved named SO measurements require a validated exact periodic aperture query");
    }
    if (aperture_query != nullptr
        && !aperture_query->describes(particles, box_size)) {
        throw std::invalid_argument(
            "Named SO derived products require an aperture query for the supplied particle store and box");
    }

    auto output = io::open_checked_output_file(
        path, "named SO derived analysis product");
    output.precision(17);
    output << "# product_kind=analysis_standard_so_derived\n";
    write_single_snapshot_csv_metadata(
        output,
        source_snapshot,
        source_native_snapshot_object_sha256,
        scale_factor);
    output
        << "# object_kind=density_peak_centered_standard_spherical_overdensity\n"
        << "# seed_selection=fof_candidates_then_fixed_k_density_peak_deblend_retained_host_minimum_then_named_so\n"
        << "# fof_linking_length_b=" << fof_linking_length_b << '\n'
        << "# fof_min_particles=" << fof_min_particles << '\n'
        << "# peak_density_k_neighbors=" << peak_density_k_neighbors << '\n'
        << "# deblended_min_particles=" << deblended_min_particles << '\n'
        << "# peak_saddle_merge_ratio=" << peak_saddle_merge_ratio << '\n'
        << "# radius_unit=comoving_Mpc_per_h\n"
        << "# mass_unit=1e10_Msun_per_h\n"
        << "# reference_density_unit=1e10_Msun_per_h_per_(Mpc_per_h)^3_comoving\n"
        << "# center=density_peak_particle\n"
        << "# membership=geometric_so_all\n"
        << "# membership_product=analysis_halo_membership_views.hdf5\n"
        << "# membership_materialization=same_exact_aperture_query_as_derived_properties\n"
        << "# aperture=named_so_radius\n"
        << "# aperture_query=exact_periodic_index_squared_distance_filter\n"
        << "# vmax_radius=physical_a_times_comoving_radius\n"
        << "# spin_definition=bullock_2001_exact_geometric_so_membership\n"
        << "# shape_tensor=fixed_spherical_membership_shape_tensor\n"
        << "# shape_use_reduced_tensor="
        << (shape_options.use_reduced_tensor ? "true" : "false") << '\n'
        << "# shape_tensor_weighting="
        << (shape_options.use_reduced_tensor
                ? "mass_over_iterated_ellipsoidal_radius_squared"
                : "mass")
        << '\n'
        << "# shape_tensor_normalization=total_fixed_membership_mass\n"
        << "# shape_reduced_zero_displacement_policy=exclude_undefined_directional_numerator_include_mass_in_normalization\n"
        << "# shape_directional_particle_count=nonzero_directionally_unique_periodic_displacement_members\n"
        << "# shape_reduced_iteration_initialization=unreduced_tensor_eigenbasis\n"
        << "# shape_iteration_count=requested_tensor_solve_iterations_excludes_reduced_bootstrap\n"
        << "# shape_projected_displacement_evaluation=binary64_epsilon_bounded_wide_real_with_exact_dyadic_fallback\n"
        << "# shape_tensor_factorization=streaming_qr_one_sided_svd\n"
        << "# shape_psd_preservation=square_root_factorization\n"
        << "# shape_rank_determination=exact_binary64_displacement_span_predicate\n"
        << "# shape_svd_termination=strictly_decreasing_normalized_offdiagonal_gram_energy\n"
        << "# shape_eigensystem_certificate=outward_rounded_bauer_fike_residual_on_returned_axes_of_represented_qr_gram\n"
        << "# shape_numerical_failure_status=solver_indeterminate\n"
        << "# shape_geometric_unavailable_statuses=periodic_cut_locus_ambiguous,zero_spatial_extent,rank_deficient_reduced_metric\n"
        << "# shape_convergence_metric=member_natural_log_relative_tensor_weight_pairwise_range\n"
        << "# shape_max_iterations=" << shape_options.max_iterations << '\n'
        << "# shape_convergence_tolerance="
        << shape_options.convergence_tolerance << '\n'
        << "# shape_ellipsoidal_reselection=false\n"
        << "# shape_availability=shape_computed_flag\n"
        << "# triaxiality_definition=(lambda_a-lambda_b)/(lambda_a-lambda_c)\n"
        << "# triaxiality_availability=triaxiality_defined_flag\n"
        << "# shape_unavailable_numeric_fields=zero_placeholders\n"
        << "candidate_id,deblended_seed_id,peak_particle_id,mass_definition,"
           "center_kind,membership_view,aperture_radius_comoving,"
           "aperture_mass,particle_count,crossing_resolved,spin_status,"
           "spin_computed,bullock_lambda,vmax_computed,vmax_km_s,"
           "r_vmax_comoving_Mpc_h,shape_status,shape_directional_particle_count,"
           "shape_computed,shape_converged,"
           "shape_iterations,b_over_a,c_over_a,triaxiality,"
           "triaxiality_defined\n";

    std::vector<halo::PeriodicNeighbor> members;
    std::vector<std::size_t> member_indices;
    std::vector<core::ParticleId> persisted_member_ids;
    for (const auto& evaluation : evaluations) {
        for (const auto& measurement : evaluation.measurements) {
            const std::string definition(
                halo::standard_so_mass_definition_name(measurement.definition));
            output << measurement.candidate_id << ','
                   << measurement.deblended_seed_id << ','
                   << measurement.peak_particle_id << ',' << definition << ','
                   << halo::halo_center_kind_name(
                          measurement.center_selection.selected_kind)
                   << ",geometric_so_all," << measurement.radius << ','
                   << measurement.mass << ','
                   << measurement.geometric_particle_count << ','
                   << (measurement.crossing_resolved ? 1 : 0) << ',';

            if (!measurement.crossing_resolved) {
                membership_output.append_geometric_so(
                    measurement.candidate_id,
                    measurement.deblended_seed_id,
                    measurement.peak_particle_id,
                    measurement.definition,
                    false,
                    std::span<const core::ParticleId>{});
                output
                    << "crossing_unresolved,0,0,0,0,0,"
                    << "crossing_unresolved,0,0,0,0,0,0,0,0\n";
                continue;
            }
            if (aperture_query == nullptr) {
                throw std::logic_error(
                    "Resolved named SO measurement is missing its validated exact periodic query");
            }

            aperture_query->collect(
                evaluation.seed.center,
                measurement.radius,
                members);
            if (members.size() != measurement.geometric_particle_count) {
                throw std::logic_error(
                    "Named SO derived membership count disagrees with the mass measurement");
            }

            // One exact geometric membership feeds every downstream consumer and
            // the persisted membership catalog. Vmax establishes radius order;
            // HaloSpin then canonicalizes the same vector by stable ParticleID.
            const analysis::VmaxResult vmax = analysis::compute_vmax(
                particles, members, scale_factor);
            const analysis::SpinResult spin = analysis::HaloSpin::compute_members(
                particles,
                evaluation.seed.center,
                measurement.radius,
                measurement.mass,
                box_size,
                scale_factor,
                members);

            member_indices.clear();
            persisted_member_ids.clear();
            if (member_indices.capacity() < members.size()) {
                member_indices.reserve(members.size());
            }
            if (persisted_member_ids.capacity() < members.size()) {
                persisted_member_ids.reserve(members.size());
            }
            const auto particle_ids = particles.get_ids();
            for (const auto& member : members) {
                member_indices.push_back(member.particle_index);
                persisted_member_ids.push_back(
                    particle_ids[member.particle_index]);
            }
            membership_output.append_geometric_so(
                measurement.candidate_id,
                measurement.deblended_seed_id,
                measurement.peak_particle_id,
                measurement.definition,
                true,
                std::span<const core::ParticleId>(persisted_member_ids));

            const analysis::HaloShapeResult shape =
                analysis::HaloShapeAnalyzer::compute(
                    particles,
                    evaluation.seed.center,
                    box_size,
                    member_indices,
                    shape_options);

            output << spin.status << ',' << (spin.computed() ? 1 : 0) << ','
                   << spin.bullock_lambda << ','
                   << (vmax.computed ? 1 : 0) << ',' << vmax.vmax_km_s << ','
                   << vmax.r_vmax_comoving_Mpc_h << ','
                   << shape_status(shape) << ','
                   << shape.directional_particle_count << ','
                   << (shape.computed() ? 1 : 0) << ','
                   << (shape.converged ? 1 : 0) << ','
                   << shape.iterations << ','
                   << shape.axis_ratio_b_over_a << ','
                   << shape.axis_ratio_c_over_a << ',' << shape.triaxiality
                   << ',' << (shape.triaxiality_defined() ? 1 : 0) << '\n';
        }
    }
    io::close_checked_output_file(
        output, path, "named SO derived analysis product");
}

} // namespace cosmo_nbody::app::nbody_analyze
