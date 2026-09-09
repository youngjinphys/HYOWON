#include "nbody_analyze_pair_link_stage.hpp"

#include "nbody_analyze_outputs.hpp"
#include "nbody_analyze_pair_link_halos.hpp"
#include "nbody_analyze_parallel.hpp"

#include "cosmo_nbody/analysis/halo_derived_properties.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"
#include "cosmo_nbody/io/analysis_snapshot_reader.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {
namespace {

core::Real descriptor_mean_spacing(const io::SnapshotDescriptor& descriptor) {
    if (descriptor.particles_per_dimension == 0) {
        throw std::invalid_argument(
            "Pair-link snapshot descriptor has zero particles_per_dimension");
    }
    const core::Real spacing = descriptor.box_size_Mpc_h
        / static_cast<core::Real>(descriptor.particles_per_dimension);
    if (!std::isfinite(spacing) || spacing <= 0.0) {
        throw std::invalid_argument(
            "Pair-link snapshot descriptor implies invalid mean particle spacing");
    }
    return spacing;
}

} // namespace

void run_pair_link_stage(
    const analysis::AnalysisRequest& request,
    const io::SnapshotDescriptor& later_snapshot,
    core::Real later_scale_factor,
    const std::vector<halo::PairLinkHalo>& later_candidates,
    const std::filesystem::path& output_directory) {
    if (request.progenitor_snapshot.empty()) {
        throw std::logic_error(
            "Pair-link stage requires an explicit earlier snapshot");
    }

    const io::SnapshotDescriptor earlier_descriptor =
        io::read_snapshot_descriptor(request.progenitor_snapshot);
    io::require_pair_link_match(earlier_descriptor, later_snapshot);

    core::ParticleStore earlier_particles;
    core::Real earlier_scale_factor = 0.0;
    io::AnalysisSnapshotReader::read_complete(
        request.progenitor_snapshot,
        earlier_descriptor,
        earlier_particles,
        earlier_scale_factor);
    earlier_particles.release_accelerations();
    if (!(earlier_scale_factor < later_scale_factor)) {
        throw std::invalid_argument(
            "Pair-link earlier snapshot scale factor must be smaller than the later snapshot scale factor");
    }

    halo::FoFMembershipFinder finder(
        earlier_descriptor.box_size_Mpc_h,
        descriptor_mean_spacing(earlier_descriptor),
        request.fof_linking_length_b,
        request.fof_min_particles);
    auto memberships = finder.find_memberships(earlier_particles);
    std::vector<analysis::HaloDerivedProperties> properties(
        memberships.size());
    parallel_for_indices(
        memberships.size(),
        [&](std::size_t group_index) {
            properties[group_index] =
                analysis::HaloDerivedPropertyAnalyzer::compute(
                    earlier_particles,
                    memberships[group_index],
                    earlier_descriptor.box_size_Mpc_h,
                    earlier_scale_factor);
        });
    const auto earlier_candidates = materialize_pair_link_halos(
        memberships, properties, earlier_particles);

    const auto pair_links = halo::PairLinkBuilder().link_pair(
        earlier_candidates, later_candidates);
    write_pair_links_json(
        output_directory / "analysis_pair_links.json",
        pair_links,
        request.progenitor_snapshot,
        earlier_descriptor.native_snapshot_object_sha256,
        earlier_scale_factor,
        request.snapshot_path,
        later_snapshot.native_snapshot_object_sha256,
        later_scale_factor,
        request.fof_linking_length_b,
        request.fof_min_particles);
}

} // namespace cosmo_nbody::app::nbody_analyze
