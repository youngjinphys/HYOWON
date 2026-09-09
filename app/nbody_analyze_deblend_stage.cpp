#include "nbody_analyze_deblend_stage.hpp"
#include "nbody_analyze_named_so_products.hpp"
#include "nbody_analyze_outputs.hpp"

#include "cosmo_nbody/analysis/halo_membership_catalog.hpp"
#include "cosmo_nbody/halo/density_peak_deblender.hpp"
#include "cosmo_nbody/halo/exact_periodic_aperture.hpp"
#include "cosmo_nbody/halo/halo_definition.hpp"
#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"
#include "cosmo_nbody/halo/spherical_overdensity.hpp"
#include "cosmo_nbody/halo/standard_so_evaluator.hpp"
#include "cosmo_nbody/io/checked_output_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {
namespace {

struct SeedContext {
    halo::NamedSOSeed seed;
    std::vector<halo::HaloCenterCandidate> center_candidates;
};

io::CheckedOutputFile open_output(const std::filesystem::path& path) {
    return io::open_checked_output_file(
        path, "deblended host analysis product");
}

void close_output(
    io::CheckedOutputFile& output,
    const std::filesystem::path& path) {
    io::close_checked_output_file(
        output, path, "deblended host analysis product");
}

std::vector<core::ParticleId> sorted_member_ids(
    const core::ParticleStore& particles,
    const std::vector<std::size_t>& indices) {
    const auto ids = particles.get_ids();
    std::vector<core::ParticleId> result;
    result.reserve(indices.size());
    for (const std::size_t index : indices) {
        if (index >= particles.num_owned_particles()) {
            throw std::out_of_range(
                "Deblended product membership contains a non-owned particle index");
        }
        result.push_back(ids[index]);
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument(
            "Deblended product membership contains duplicate stable particle IDs");
    }
    return result;
}

std::vector<halo::HaloCenterCandidate> center_candidates_for_host(
    const analysis::HaloDerivedProperties& candidate_property,
    const halo::DeblendedHostSeed& host) {
    return {
        {
            halo::HaloCenterKind::PeriodicMassWeightedCircularCOM,
            candidate_property.center_of_mass,
            true,
        },
        {
            halo::HaloCenterKind::DensityPeakParticle,
            host.peak_position,
            true,
        },
    };
}

} // namespace

void run_deblended_host_stage(
    const analysis::AnalysisRequest& request,
    const halo::SOContext& so_context,
    const core::ParticleStore& particles,
    const std::vector<halo::FoFMembership>& candidates,
    const std::vector<analysis::HaloDerivedProperties>& candidate_properties,
    core::Real snapshot_a,
    std::string_view source_native_snapshot_object_sha256,
    const std::filesystem::path& output_directory) {
    const bool canonical_source_sha256 =
        source_native_snapshot_object_sha256.size() == 64U
        && std::all_of(
            source_native_snapshot_object_sha256.begin(),
            source_native_snapshot_object_sha256.end(),
            [](char value) {
                return (value >= '0' && value <= '9')
                    || (value >= 'a' && value <= 'f');
            });
    if (candidates.size() != candidate_properties.size()) {
        throw std::invalid_argument(
            "Deblended host stage requires one derived-property row per FoF candidate");
    }
    if (!canonical_source_sha256
        || !std::isfinite(snapshot_a)
        || snapshot_a <= 0.0) {
        throw std::invalid_argument(
            "Deblended host stage requires canonical source identity and epoch");
    }

    halo::DensityPeakDeblendOptions deblend_options;
    deblend_options.k_neighbors = request.peak_density_k_neighbors;
    deblend_options.minimum_host_particles = request.deblended_min_particles;
    deblend_options.saddle_to_lower_peak_merge_ratio =
        request.peak_saddle_merge_ratio;
    const halo::DensityPeakDeblender deblender(deblend_options);
    const core::Real box_size = so_context.box_size;

    const auto peaks_path = output_directory / "analysis_density_peaks_all.csv";
    const auto hosts_path = output_directory / "analysis_deblended_hosts_all.csv";
    const auto membership_path =
        output_directory / "analysis_halo_membership_views.hdf5";
    auto peaks_output = open_output(peaks_path);
    auto hosts_output = open_output(hosts_path);
    analysis::HaloMembershipCatalogWriter membership_output(
        membership_path,
        request.snapshot_path,
        source_native_snapshot_object_sha256,
        snapshot_a,
        request.fof_linking_length_b,
        request.fof_min_particles,
        deblend_options.k_neighbors,
        deblend_options.minimum_host_particles,
        deblend_options.saddle_to_lower_peak_merge_ratio,
        "analysis_fof_candidate_all.hdf5:/FoFMembership");

    peaks_output << "# product_kind=analysis_density_peaks_all\n";
    write_single_snapshot_csv_metadata(
        peaks_output,
        request.snapshot_path,
        source_native_snapshot_object_sha256,
        snapshot_a);
    peaks_output
        << "# parent_object_kind=fof_candidate\n"
        << "# fof_linking_length_b=" << request.fof_linking_length_b << '\n'
        << "# fof_min_particles=" << request.fof_min_particles << '\n'
        << "# method=fixed_k_periodic_density_ascent_with_discrete_saddle_merging\n"
        << "# k_neighbors=" << deblend_options.k_neighbors << "\n"
        << "# saddle_to_lower_peak_merge_ratio="
        << deblend_options.saddle_to_lower_peak_merge_ratio << "\n"
        << "candidate_id,peak_particle_id,peak_particle_index,x,y,z,density,"
           "retained_as_host,merged_into_peak_particle_id,effective_k_neighbors\n";
    hosts_output << "# product_kind=analysis_deblended_hosts_all\n";
    write_single_snapshot_csv_metadata(
        hosts_output,
        request.snapshot_path,
        source_native_snapshot_object_sha256,
        snapshot_a);
    hosts_output
        << "# parent_object_kind=fof_candidate\n"
        << "# fof_linking_length_b=" << request.fof_linking_length_b << '\n'
        << "# fof_min_particles=" << request.fof_min_particles << '\n'
        << "# peak_density_k_neighbors=" << deblend_options.k_neighbors << '\n'
        << "# peak_saddle_merge_ratio="
        << deblend_options.saddle_to_lower_peak_merge_ratio << '\n'
        << "# center=density_peak_particle\n"
        << "# membership=deblended_density_basin_all\n"
        << "# membership_product=analysis_halo_membership_views.hdf5\n"
        << "# minimum_host_particles="
        << deblend_options.minimum_host_particles << "\n"
        << "candidate_id,seed_id,peak_particle_id,peak_particle_index,x,y,z,"
           "peak_density,particle_count,meets_minimum_particle_count,"
           "selected_for_named_so\n";

    std::vector<SeedContext> seed_contexts;
    for (std::size_t candidate_index = 0;
         candidate_index < candidates.size();
         ++candidate_index) {
        const auto& candidate = candidates[candidate_index];
        if (candidate.particle_indices.size() < 2) {
            throw std::invalid_argument(
                "FoF candidate is too small for the declared fixed-k density estimator");
        }

        const auto deblended = deblender.deblend(
            particles, candidate, box_size);
        for (const auto& peak : deblended.peaks) {
            peaks_output << peak.candidate_id << ',' << peak.peak_particle_id
                         << ',' << peak.peak_particle_index << ','
                         << peak.position.x << ',' << peak.position.y << ','
                         << peak.position.z << ',' << peak.density << ','
                         << (peak.retained_as_host ? 1 : 0) << ',';
            if (peak.merged_into_peak_particle_id.has_value()) {
                peaks_output << *peak.merged_into_peak_particle_id;
            }
            peaks_output << ',' << deblended.effective_k_neighbors << '\n';
        }

        for (const auto& host : deblended.hosts) {
            const bool selected_for_named_so = host.meets_minimum_particle_count;
            hosts_output << host.candidate_id << ',' << host.seed_id << ','
                         << host.peak_particle_id << ','
                         << host.peak_particle_index << ','
                         << host.peak_position.x << ',' << host.peak_position.y
                         << ',' << host.peak_position.z << ','
                         << host.peak_density << ','
                         << host.particle_indices.size() << ','
                         << (host.meets_minimum_particle_count ? 1 : 0) << ','
                         << (selected_for_named_so ? 1 : 0) << '\n';
            const auto deblended_member_ids =
                sorted_member_ids(particles, host.particle_indices);
            membership_output.append_deblended_density_basin(
                host.candidate_id,
                host.seed_id,
                host.peak_particle_id,
                std::span<const core::ParticleId>(deblended_member_ids));

            if (!selected_for_named_so) continue;

            SeedContext context;
            context.seed.candidate_id = candidate.id;
            context.seed.deblended_seed_id = host.seed_id;
            context.seed.peak_particle_id = host.peak_particle_id;
            context.seed.center = host.peak_position;
            context.seed.center_selection = {
                halo::HaloCenterKind::DensityPeakParticle,
            };
            context.center_candidates = center_candidates_for_host(
                candidate_properties[candidate_index], host);
            seed_contexts.push_back(std::move(context));
        }
    }
    close_output(peaks_output, peaks_path);
    close_output(hosts_output, hosts_path);

    std::vector<halo::NamedSOSeed> named_seeds;
    named_seeds.reserve(seed_contexts.size());
    for (const auto& context : seed_contexts) {
        named_seeds.push_back(context.seed);
    }

    const auto standard_evaluations =
        halo::StandardSOEvaluator(so_context).evaluate(
            particles, named_seeds, snapshot_a);
    if (standard_evaluations.size() != seed_contexts.size()) {
        throw std::logic_error(
            "Standard SO evaluator did not preserve seed cardinality");
    }

    std::optional<halo::PeriodicNeighborIndex> so_neighbor_index;
    std::optional<halo::ExactPeriodicApertureQuery> so_aperture_query;
    if (!named_seeds.empty()) {
        // The workspace plan is definition-independent. Bind it to one of the
        // named products that this stage actually computes rather than relying
        // on a generic SO constructor default.
        halo::SphericalOverdensityFinder planning_finder(
            so_context,
            core::Real{200.0},
            halo::SOReferenceDensity::MeanMatter);
        const auto so_plan = planning_finder.execution_plan(
            particles.num_owned_particles(), named_seeds.size());
        so_neighbor_index.emplace(
            particles, box_size, so_plan.spatial_index_cap_bytes);
        so_aperture_query.emplace(
            *so_neighbor_index, particles, box_size);
    }

    const auto standard_so_path =
        output_directory / "analysis_standard_so.csv";
    auto standard_output = open_output(standard_so_path);
    standard_output << "# product_kind=analysis_standard_so\n";
    write_single_snapshot_csv_metadata(
        standard_output,
        request.snapshot_path,
        source_native_snapshot_object_sha256,
        snapshot_a);
    standard_output
        << "# object_kind=density_peak_centered_standard_spherical_overdensity\n"
        << "# seed_selection=fof_candidates_then_fixed_k_density_peak_deblend_retained_host_minimum_then_named_so\n"
        << "# fof_linking_length_b=" << request.fof_linking_length_b << '\n'
        << "# fof_min_particles=" << request.fof_min_particles << '\n'
        << "# peak_density_k_neighbors="
        << deblend_options.k_neighbors << '\n'
        << "# deblended_min_particles="
        << deblend_options.minimum_host_particles << '\n'
        << "# peak_saddle_merge_ratio="
        << deblend_options.saddle_to_lower_peak_merge_ratio << '\n'
        << "# radius_unit=comoving_Mpc_per_h\n"
        << "# mass_unit=1e10_Msun_per_h\n"
        << "# reference_density_unit=1e10_Msun_per_h_per_(Mpc_per_h)^3_comoving\n"
        << "# definitions=M200m,M200c,Mvir_BN98\n"
        << "# membership=geometric_so_all_when_crossing_resolved\n"
        << "# membership_product=analysis_halo_membership_views.hdf5\n"
        << "# membership_query=exact_periodic_index_squared_distance_filter\n"
        << "candidate_id,deblended_seed_id,peak_particle_id,mass_definition,"
           "center_kind,scale_factor,reference_density_kind,"
           "reference_density,overdensity_threshold,radius,mass,"
           "geometric_particle_count,crossing_resolved\n";

    for (std::size_t seed_index = 0;
         seed_index < standard_evaluations.size();
         ++seed_index) {
        const auto& evaluated = standard_evaluations[seed_index];
        const auto& context = seed_contexts[seed_index];
        if (evaluated.seed.candidate_id != context.seed.candidate_id
            || evaluated.seed.deblended_seed_id
                != context.seed.deblended_seed_id) {
            throw std::logic_error(
                "Standard SO evaluator changed deterministic seed order");
        }

        for (const auto& measurement : evaluated.measurements) {
            const std::string definition(
                halo::standard_so_mass_definition_name(
                    measurement.definition));
            standard_output
                << measurement.candidate_id << ','
                << measurement.deblended_seed_id << ','
                << measurement.peak_particle_id << ',' << definition << ','
                << halo::halo_center_kind_name(
                       measurement.center_selection.selected_kind)
                << ',' << measurement.scale_factor << ','
                << halo::so_reference_density_name(
                       measurement.reference_density_kind)
                << ',' << measurement.reference_density << ','
                << measurement.overdensity_threshold << ','
                << measurement.radius << ',' << measurement.mass << ','
                << measurement.geometric_particle_count << ','
                << (measurement.crossing_resolved ? 1 : 0) << '\n';

            halo::HaloMembership membership{
                measurement.crossing_resolved
                    ? halo::HaloMembershipView::GeometricSOAll
                    : halo::HaloMembershipView::DeblendedDensityBasinAll,
                true,
            };
            halo::validate_halo_definition(
                context.center_candidates,
                measurement.center_selection,
                measurement,
                membership);
        }
    }
    close_output(standard_output, standard_so_path);

    if (!request.shape_use_reduced_tensor.has_value()) {
        throw std::logic_error(
            "Validated standard-SO request lost explicit shape tensor method");
    }
    analysis::HaloShapeOptions shape_options;
    shape_options.use_reduced_tensor = *request.shape_use_reduced_tensor;
    shape_options.max_iterations = request.shape_max_iterations;
    shape_options.convergence_tolerance = request.shape_convergence_tolerance;
    write_named_so_derived_products(
        output_directory / "analysis_standard_so_derived.csv",
        particles,
        so_aperture_query.has_value() ? &*so_aperture_query : nullptr,
        membership_output,
        standard_evaluations,
        box_size,
        snapshot_a,
        request.snapshot_path,
        source_native_snapshot_object_sha256,
        request.fof_linking_length_b,
        request.fof_min_particles,
        deblend_options.k_neighbors,
        deblend_options.minimum_host_particles,
        deblend_options.saddle_to_lower_peak_merge_ratio,
        shape_options);
    membership_output.finalize();
}

} // namespace cosmo_nbody::app::nbody_analyze
