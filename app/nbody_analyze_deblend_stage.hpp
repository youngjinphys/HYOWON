#pragma once

#include "cosmo_nbody/analysis/analysis_request.hpp"
#include "cosmo_nbody/analysis/halo_derived_properties.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"
#include "cosmo_nbody/halo/spherical_overdensity.hpp"

#include <filesystem>
#include <string_view>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {

// Reuse FoF topology without recomputing FoF or mutating earlier pipeline products.
void run_deblended_host_stage(
    const analysis::AnalysisRequest& request,
    const halo::SOContext& so_context,
    const core::ParticleStore& particles,
    const std::vector<halo::FoFMembership>& candidates,
    const std::vector<analysis::HaloDerivedProperties>& candidate_properties,
    core::Real snapshot_a,
    std::string_view source_native_snapshot_object_sha256,
    const std::filesystem::path& output_directory);

} // namespace cosmo_nbody::app::nbody_analyze
