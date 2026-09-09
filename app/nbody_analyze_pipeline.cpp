#include "nbody_analyze_pipeline.hpp"

#include "nbody_analyze_field_stage.hpp"
#include "nbody_analyze_halo_stage.hpp"
#include "nbody_analyze_pair_link_stage.hpp"

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/io/analysis_snapshot_reader.hpp"
#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace cosmo_nbody::app::nbody_analyze {
namespace {

int checked_mesh_dimension(std::uint64_t value, const char* source) {
    if (value == 0
        || value > static_cast<std::uint64_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string(source)
            + " mesh dimension must lie in [1, INT_MAX]");
    }
    return static_cast<int>(value);
}

std::uint64_t checked_thread_count(std::size_t value) {
    if (value == 0) {
        throw std::invalid_argument(
            "Analyzer effective thread count must be positive");
    }
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Analyzer effective thread count exceeds uint64 range");
        }
    }
    return static_cast<std::uint64_t>(value);
}

HaloStageContext make_halo_stage_context(
    const analysis::AnalysisRequest& request,
    const io::SnapshotDescriptor& snapshot,
    std::size_t effective_threads,
    std::size_t automatic_thread_capacity,
    bool thread_count_automatic,
    std::string_view thread_selection_reason) {
    if (snapshot.particles_per_dimension == 0) {
        throw std::invalid_argument(
            "Snapshot descriptor has zero particles_per_dimension");
    }
    const core::Real mean_spacing = snapshot.box_size_Mpc_h
        / static_cast<core::Real>(snapshot.particles_per_dimension);
    if (!std::isfinite(mean_spacing) || mean_spacing <= 0.0) {
        throw std::runtime_error(
            "Snapshot descriptor implies an invalid mean particle spacing");
    }

    io::RunMetadata metadata = io::RunMetadata::from_snapshot_analysis(
        snapshot,
        request.snapshot_path,
        request.resource_policy,
        effective_threads,
        automatic_thread_capacity,
        thread_count_automatic,
        thread_selection_reason);
    metadata.snapshot_sha256 = snapshot.native_snapshot_object_sha256;

    return HaloStageContext{
        snapshot.box_size_Mpc_h,
        mean_spacing,
        halo::SOContext{
            snapshot.box_size_Mpc_h,
            snapshot.omega_m,
            snapshot.omega_lambda,
            checked_thread_count(effective_threads),
            halo::SOResourcePolicy{
                request.resource_policy.memory_budget_gib}},
        io::FoFAnalysisCatalogContext{
            snapshot.box_size_Mpc_h,
            snapshot.physics_fingerprint,
            std::move(metadata),
            snapshot.scale_factor},
        snapshot.native_snapshot_object_sha256};
}

void require_private_output_directory(
    const std::filesystem::path& directory) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(directory, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(
            "Analyzer output directory must be an existing non-symlink directory: "
            + directory.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
}

} // namespace

void run_pipeline(
    const analysis::AnalysisRequest& request,
    std::size_t effective_threads,
    std::size_t automatic_thread_capacity,
    bool thread_count_automatic,
    std::string_view thread_selection_reason,
    const std::filesystem::path& output_directory) {
    require_private_output_directory(output_directory);

    const io::SnapshotDescriptor snapshot =
        io::read_snapshot_descriptor(request.snapshot_path);

    std::optional<io::SnapshotDescriptor> cross_snapshot;
    if (!request.cross_snapshot.empty()) {
        cross_snapshot.emplace(
            io::read_snapshot_descriptor(request.cross_snapshot));
        io::require_field_comparison_match(*cross_snapshot, snapshot);
    }

    // The native complete-periodic-cube estimator uses the ordinary spherical
    // shell measure, which is exact only while the shell is wholly inside the
    // minimum-image cube: r <= L/2. This is geometry, not an empirical cutoff.
    if (request.xi_max_radius.has_value()
        && *request.xi_max_radius > 0.5 * snapshot.box_size_Mpc_h) {
        throw std::invalid_argument(
            "--xi-max-radius exceeds L/2, the exact spherical-shell support of the native periodic analytic 2PCF estimator");
    }

    core::ParticleStore particles;
    core::Real snapshot_a = 0.0;
    io::AnalysisSnapshotReader::read_complete(
        request.snapshot_path,
        snapshot,
        particles,
        snapshot_a);
    particles.release_accelerations();

    if (!request.field_mesh.has_value()) {
        throw std::logic_error(
            "Validated analysis request lost its explicit field mesh");
    }
    const int field_mesh = checked_mesh_dimension(
        static_cast<std::uint64_t>(*request.field_mesh),
        "Explicit analysis field");
    run_field_statistics_stage(
        request,
        snapshot,
        cross_snapshot ? &*cross_snapshot : nullptr,
        particles,
        snapshot_a,
        field_mesh,
        output_directory);

    const HaloStageContext halo_context = make_halo_stage_context(
        request,
        snapshot,
        effective_threads,
        automatic_thread_capacity,
        thread_count_automatic,
        thread_selection_reason);
    HaloStageResult halo_result = run_halo_stage(
        request,
        halo_context,
        particles,
        snapshot_a,
        output_directory);

    if (halo_result.later_pair_link_candidates.has_value()) {
        particles = core::ParticleStore{};
        run_pair_link_stage(
            request,
            snapshot,
            snapshot_a,
            *halo_result.later_pair_link_candidates,
            output_directory);
        halo_result.later_pair_link_candidates.reset();
    } else {
        particles = core::ParticleStore{};
    }

    std::cout << "Analyzed snapshot '" << request.snapshot_path
              << "' at a=" << snapshot_a
              << "; field_mesh=" << field_mesh
              << " (explicit_analysis_method)\n";
}

} // namespace cosmo_nbody::app::nbody_analyze
