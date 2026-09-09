#include "cosmo_nbody/io/run_artifact_layout.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cosmo_nbody::io {
namespace {

std::filesystem::path absolute_normalized(
    const std::filesystem::path& path) {
    if (path.empty()) {
        throw std::invalid_argument(
            "Run artifact directory must not be empty");
    }
    std::error_code error;
    const auto absolute = std::filesystem::absolute(path, error);
    if (error) {
        throw std::runtime_error(
            "Could not resolve run artifact directory: " + error.message());
    }
    return absolute.lexically_normal();
}

void require_directory_no_symlink(
    const std::filesystem::path& path,
    const char* role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(
            std::string(role) + " must be an existing non-symlink directory: "
            + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
}

void create_directory_exclusive(
    const std::filesystem::path& path,
    const char* role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (!error && std::filesystem::exists(status)) {
        throw std::runtime_error(
            std::string(role) + " already exists: " + path.string());
    }
    if (error && error != std::errc::no_such_file_or_directory) {
        throw std::runtime_error(
            std::string("Could not inspect ") + role + ": "
            + error.message());
    }
    error.clear();
    if (!std::filesystem::create_directory(path, error) || error) {
        throw std::runtime_error(
            std::string("Could not create ") + role + ": " + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
    require_directory_no_symlink(path, role);
}

} // namespace

RunArtifactLayout::RunArtifactLayout(
    std::filesystem::path run_directory,
    bool organized)
    : run_directory_(absolute_normalized(run_directory)),
      organized_(organized) {}

RunArtifactLayout RunArtifactLayout::organized(
    const std::filesystem::path& run_directory) {
    return RunArtifactLayout(run_directory, true);
}

RunArtifactLayout RunArtifactLayout::flat(
    const std::filesystem::path& run_directory) {
    return RunArtifactLayout(run_directory, false);
}

std::filesystem::path RunArtifactLayout::snapshots_directory() const {
    return organized_ ? run_directory_ / "snapshots" : run_directory_;
}

std::filesystem::path RunArtifactLayout::checkpoints_directory() const {
    return organized_ ? run_directory_ / "checkpoints" : run_directory_;
}

std::filesystem::path RunArtifactLayout::diagnostics_directory() const {
    return organized_ ? run_directory_ / "diagnostics" : run_directory_;
}

std::filesystem::path RunArtifactLayout::snapshot_path(
    int snapshot_index) const {
    if (snapshot_index < 0) {
        throw std::invalid_argument(
            "Run artifact snapshot index must be non-negative");
    }
    return snapshots_directory()
        / ("snapshot_" + std::to_string(snapshot_index) + ".hdf5");
}

std::filesystem::path RunArtifactLayout::restart_base_path() const {
    return checkpoints_directory() / "restart";
}

std::filesystem::path RunArtifactLayout::layzer_irvine_timeline_path() const {
    return diagnostics_directory() / "layzer_irvine_timeline.json";
}

std::filesystem::path RunArtifactLayout::validation_report_path() const {
    return diagnostics_directory() / "runtime_diagnostics.json";
}

void RunArtifactLayout::prepare_directories_exclusive() const {
    require_directory_no_symlink(run_directory_, "Run artifact root");
    if (!organized_) return;

    const std::array<std::pair<std::filesystem::path, const char*>, 3>
        directories{{
            {snapshots_directory(), "Snapshot artifact directory"},
            {checkpoints_directory(), "Checkpoint artifact directory"},
            {diagnostics_directory(), "Diagnostic artifact directory"},
        }};
    for (const auto& [path, role] : directories) {
        create_directory_exclusive(path, role);
    }
}

} // namespace cosmo_nbody::io
