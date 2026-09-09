#pragma once

#include "cosmo_nbody/io/hdf5_handle.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::io {

enum class DurableFilePublicationPolicy {
    ReplaceExisting,
    RequireAbsent,
};

// Atomic RequireAbsent destination collision, distinct from durability/write errors.
class DurableFilePublicationCollision : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Sync an existing non-empty regular file and its parent directory.
void sync_regular_file_and_parent_durable(
    const std::filesystem::path& requested_path,
    std::string_view role);

// Same-directory staged publication for externally written files. RequireAbsent
// uses atomic no-replace semantics and fails if the filesystem cannot provide them;
// observed HDF5 close failures reject publication.
class DurableFilePublication {
public:
    DurableFilePublication(
        const std::filesystem::path& requested_path,
        std::string role,
        DurableFilePublicationPolicy policy =
            DurableFilePublicationPolicy::ReplaceExisting);
    ~DurableFilePublication() noexcept;

    DurableFilePublication(const DurableFilePublication&) = delete;
    DurableFilePublication& operator=(const DurableFilePublication&) = delete;
    DurableFilePublication(DurableFilePublication&&) = delete;
    DurableFilePublication& operator=(DurableFilePublication&&) = delete;

    const std::filesystem::path& staging_path() const noexcept {
        return stage_path_;
    }
    bool canonical_visible() const noexcept {
        return published_.final_visible();
    }
    void publish_nonempty();
    void discard_unpublished();

private:
    class PublicationState {
    public:
        PublicationState() noexcept
            : hdf5_close_failure_epoch_(
                  current_hdf5_close_failure_epoch()) {}

        void require_hdf5_close_integrity() const {
            require_no_hdf5_close_failures_since(
                hdf5_close_failure_epoch_,
                "durable file publication");
        }

        bool final_visible() const noexcept { return final_visible_; }
        void mark_final_visible() noexcept { final_visible_ = true; }

        bool committed() const noexcept { return committed_; }
        void mark_committed() noexcept { committed_ = true; }

        bool discarded() const noexcept { return discarded_; }
        void mark_discarded() noexcept { discarded_ = true; }

    private:
        std::uint64_t hdf5_close_failure_epoch_{0};
        bool final_visible_{false};
        bool committed_{false};
        bool discarded_{false};
    };

    std::filesystem::path parent_;
    std::filesystem::path final_path_;
    std::filesystem::path stage_path_;
    std::string role_;
    DurableFilePublicationPolicy policy_{
        DurableFilePublicationPolicy::ReplaceExisting};
    PublicationState published_;
};

// Atomic no-replace publication of a non-empty flat output directory. Product
// writers sync their files; publication verifies/syncs the directory and rejects
// unsafe non-sticky group/world-writable parents as a namespace-integrity guard.
class DurableDirectoryPublication {
public:
    DurableDirectoryPublication(
        const std::filesystem::path& requested_path,
        std::string role);
    ~DurableDirectoryPublication() noexcept;

    DurableDirectoryPublication(const DurableDirectoryPublication&) = delete;
    DurableDirectoryPublication& operator=(
        const DurableDirectoryPublication&) = delete;
    DurableDirectoryPublication(DurableDirectoryPublication&&) = delete;
    DurableDirectoryPublication& operator=(
        DurableDirectoryPublication&&) = delete;

    const std::filesystem::path& staging_path() const noexcept {
        return stage_path_;
    }
    bool canonical_visible() const noexcept { return final_visible_; }
    void publish_nonempty();
    void discard_unpublished();

private:
    std::filesystem::path parent_;
    std::filesystem::path final_path_;
    std::filesystem::path stage_path_;
    std::filesystem::path stage_name_;
    std::string role_;
    int parent_descriptor_{-1};
    std::uintmax_t parent_device_{0};
    std::uintmax_t parent_inode_{0};
    std::uintmax_t stage_device_{0};
    std::uintmax_t stage_inode_{0};
    std::uint64_t hdf5_close_failure_epoch_{
        current_hdf5_close_failure_epoch()};
    bool final_visible_{false};
    bool committed_{false};
    bool discarded_{false};
};

} // namespace cosmo_nbody::io
