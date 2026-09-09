#pragma once

#include "cosmo_nbody/io/hdf5_handle.hpp"

#include <filesystem>
#include <string>

namespace cosmo_nbody::io {

class SnapshotIO;

// Own one immutable opened snapshot object for exact IC ingestion, remaining
// bound across pathname replacement; bytes are hashed before and after use.
class VerifiedSnapshotSource {
public:
    explicit VerifiedSnapshotSource(
        const std::filesystem::path& path,
        std::string expected_sha256 = {});
    ~VerifiedSnapshotSource() noexcept;

    VerifiedSnapshotSource(const VerifiedSnapshotSource&) = delete;
    VerifiedSnapshotSource& operator=(const VerifiedSnapshotSource&) = delete;

    VerifiedSnapshotSource(VerifiedSnapshotSource&& other) noexcept;
    VerifiedSnapshotSource& operator=(VerifiedSnapshotSource&& other) noexcept;

    const std::string& hdf5_read_path() const;
    const std::string& sha256() const noexcept { return sha256_; }

    // Re-hash the same opened object and reject in-place mutation during ingestion.
    void verify_unchanged() const;

private:
    friend class SnapshotIO;

    hid_t hdf5_file_id() const noexcept { return hdf5_file_.get(); }
    void close() noexcept;

    H5FileHandle hdf5_file_;
    int descriptor_{-1};
    std::string descriptor_path_;
    std::string sha256_;
};

} // namespace cosmo_nbody::io
