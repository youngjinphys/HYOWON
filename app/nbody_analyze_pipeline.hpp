#pragma once

#include "cosmo_nbody/analysis/analysis_request.hpp"

#include <cstddef>
#include <filesystem>
#include <string_view>

namespace cosmo_nbody::app::nbody_analyze {

void run_pipeline(
    const analysis::AnalysisRequest& request,
    std::size_t effective_threads,
    std::size_t automatic_thread_capacity,
    bool thread_count_automatic,
    std::string_view thread_selection_reason,
    const std::filesystem::path& output_directory);

} // namespace cosmo_nbody::app::nbody_analyze