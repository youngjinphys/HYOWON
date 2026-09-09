#pragma once

#include <filesystem>

namespace cosmo_nbody::io {

class RunArtifactLayout {
public:
    static RunArtifactLayout organized(
        const std::filesystem::path& run_directory);
    static RunArtifactLayout flat(
        const std::filesystem::path& run_directory);

    const std::filesystem::path& run_directory() const noexcept {
        return run_directory_;
    }

    std::filesystem::path snapshots_directory() const;
    std::filesystem::path checkpoints_directory() const;
    std::filesystem::path diagnostics_directory() const;

    std::filesystem::path snapshot_path(int snapshot_index) const;
    std::filesystem::path restart_base_path() const;
    std::filesystem::path layzer_irvine_timeline_path() const;
    std::filesystem::path validation_report_path() const;

    void prepare_directories_exclusive() const;

private:
    RunArtifactLayout(
        std::filesystem::path run_directory,
        bool organized);

    std::filesystem::path run_directory_;
    bool organized_{false};
};

} // namespace cosmo_nbody::io
