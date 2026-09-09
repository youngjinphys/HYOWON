#pragma once

#include "cosmo_nbody/analysis/analysis_request.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include <filesystem>

namespace cosmo_nbody::app::nbody_analyze {

// If present, cross_snapshot is prevalidated for estimator geometry and
// phase-space semantics before payload reads.
void run_field_statistics_stage(
    const analysis::AnalysisRequest& request,
    const io::SnapshotDescriptor& snapshot,
    const io::SnapshotDescriptor* cross_snapshot,
    const core::ParticleStore& particles,
    core::Real snapshot_a,
    int configured_mesh,
    const std::filesystem::path& output_directory);

} // namespace cosmo_nbody::app::nbody_analyze
