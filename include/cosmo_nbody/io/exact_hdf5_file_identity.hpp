#pragma once

#include "cosmo_nbody/io/hdf5_handle.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace cosmo_nbody::io {

// Move-only binding between one lexical path, one POSIX regular-file object,
// and the HDF5 SEC2 handle opened from that exact object. finish_identity()
// verifies that the bound descriptor, HDF5 descriptor, and pathname still name
// the admitted object without rereading its payload. finish_sha256() is the
// stronger raw-container identity operation for callers that actually need a
// byte-for-byte file digest.
class ExactHdf5FileIdentity {
public:
    static ExactHdf5FileIdentity open_readonly(
        const std::filesystem::path& path,
        std::string context);

    ExactHdf5FileIdentity(ExactHdf5FileIdentity&&) noexcept;
    ExactHdf5FileIdentity& operator=(ExactHdf5FileIdentity&&) noexcept;
    ~ExactHdf5FileIdentity();

    ExactHdf5FileIdentity(const ExactHdf5FileIdentity&) = delete;
    ExactHdf5FileIdentity& operator=(const ExactHdf5FileIdentity&) = delete;

    hid_t file_id() const noexcept;
    const std::filesystem::path& requested_path() const noexcept;
    bool exact_identity_available() const noexcept;
    std::uint64_t size_bytes() const;
    void read_raw_exact_at(
        std::uint64_t offset,
        std::span<std::byte> destination) const;
    void finish_identity() const;
    std::string finish_sha256() const;

private:
    struct Impl;
    explicit ExactHdf5FileIdentity(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace cosmo_nbody::io
