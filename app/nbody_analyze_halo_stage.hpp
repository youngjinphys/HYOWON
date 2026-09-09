#pragma once

#include "cosmo_nbody/analysis/analysis_request.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/pair_links.hpp"
#include "cosmo_nbody/halo/spherical_overdensity.hpp"
#include "cosmo_nbody/io/fof_analysis_catalog.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {

// Halo-analysis inputs only; trajectory solver, time, IC, and output state are excluded.
struct HaloStageContext {
    core::Real box_size{0.0};
    core::Real mean_spacing{0.0};
    halo::SOContext so;
    io::FoFAnalysisCatalogContext fof_catalog;
    std::string source_native_snapshot_object_sha256;
};

struct HaloStageResult {
    std::optional<std::vector<halo::PairLinkHalo>> later_pair_link_candidates;
};

HaloStageResult run_halo_stage(
    const analysis::AnalysisRequest& request,
    const HaloStageContext& context,
    const core::ParticleStore& particles,
    core::Real snapshot_a,
    const std::filesystem::path& output_directory);

} // namespace cosmo_nbody::app::nbody_analyze
