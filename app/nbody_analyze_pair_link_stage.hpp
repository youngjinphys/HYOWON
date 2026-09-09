#pragma once

#include "cosmo_nbody/analysis/analysis_request.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/pair_links.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include <filesystem>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {

void run_pair_link_stage(
    const analysis::AnalysisRequest& request,
    const io::SnapshotDescriptor& later_snapshot,
    core::Real later_scale_factor,
    const std::vector<halo::PairLinkHalo>& later_candidates,
    const std::filesystem::path& output_directory);

} // namespace cosmo_nbody::app::nbody_analyze
