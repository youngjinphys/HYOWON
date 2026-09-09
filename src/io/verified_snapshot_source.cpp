#include "cosmo_nbody/io/verified_snapshot_source.hpp"

#include "cosmo_nbody/io/content_hash.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cosmo_nbody::io {
namespace {

#if !defined(_WIN32)
std::runtime_error system_error(
    const std::string& context,
    int error_number = errno) {
    return std::runtime_error(
        context + ": " + std::string(std::strerror(error_number)));
}

bool advisory_lock_unsupported(int error_number) noexcept {
    bool unsupported = error_number == ENOSYS;
#ifdef EOPNOTSUPP
    unsupported = unsupported || error_number == EOPNOTSUPP;
#endif
#ifdef ENOTSUP
    unsupported = unsupported || error_number == ENOTSUP;
#endif
    return unsupported;
}

void acquire_optional_shared_lock(int descriptor) {
    if (::flock(descriptor, LOCK_SH | LOCK_NB) == 0) return;
    const int error_number = errno;
    if (advisory_lock_unsupported(error_number)) {
        // Object identity plus before/after content digests remain decisive.
        // Some HPC/network filesystems do not implement flock; absence of that
        // cooperative optimization must not reject otherwise verifiable bytes.
        return;
    }
    if (error_number == EWOULDBLOCK || error_number == EAGAIN) {
        throw std::runtime_error(
            "Snapshot IC source is locked by a conflicting writer");
    }
    throw system_error(
        "Could not acquire a shared lock for exact snapshot IC ingestion",
        error_number);
}

void rewind_regular_file(int descriptor) {
    if (::lseek(descriptor, 0, SEEK_SET) < 0) {
        throw system_error(
            "Could not rewind opened snapshot IC source before descriptor-backed read");
    }
}

std::string descriptor_alias(int descriptor, const struct stat& admitted) {
    for (const char* root : {"/proc/self/fd", "/dev/fd"}) {
        std::error_code error;
        if (!std::filesystem::is_directory(root, error) || error) continue;
        const std::filesystem::path candidate =
            std::filesystem::path(root) / std::to_string(descriptor);

        // Do not infer the target object from stat(path). On some fdescfs
        // implementations (notably macOS), stat("/dev/fd/N") may describe the
        // descriptor namespace entry rather than the regular file reached when
        // the alias is opened. Open the alias and compare the resulting object
        // with the already-admitted descriptor using fstat on both descriptors.
        int alias_flags = O_RDONLY;
#ifdef O_CLOEXEC
        alias_flags |= O_CLOEXEC;
#endif
        const int alias_descriptor = ::open(candidate.c_str(), alias_flags);
        if (alias_descriptor < 0) continue;
        struct stat observed{};
        const bool inspected = ::fstat(alias_descriptor, &observed) == 0;
        (void)::close(alias_descriptor);
        if (inspected
            && observed.st_dev == admitted.st_dev
            && observed.st_ino == admitted.st_ino) {
            return candidate.string();
        }
    }
    throw std::runtime_error(
        "Exact snapshot IC ingestion requires /proc/self/fd or /dev/fd to expose the opened file object");
}
#endif

} // namespace

VerifiedSnapshotSource::VerifiedSnapshotSource(
    const std::filesystem::path& path,
    std::string expected_sha256) {
    if (!expected_sha256.empty() && !is_canonical_sha256(expected_sha256)) {
        throw std::invalid_argument(
            "Expected snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
    }
#if defined(_WIN32)
    (void)path;
    throw std::runtime_error(
        "Exact snapshot IC object binding is not implemented for Windows; refusing pathname-only ingestion");
#else
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    descriptor_ = ::open(path.c_str(), flags);
    if (descriptor_ < 0) {
        throw system_error(
            "Could not open snapshot IC source '" + path.string() + "'");
    }

    try {
        struct stat admitted{};
        if (::fstat(descriptor_, &admitted) != 0) {
            throw system_error("Could not inspect opened snapshot IC source");
        }
        if (!S_ISREG(admitted.st_mode)) {
            throw std::invalid_argument(
                "Snapshot IC source must resolve to a regular file object");
        }
        acquire_optional_shared_lock(descriptor_);

        descriptor_path_ = descriptor_alias(descriptor_, admitted);
        sha256_ = sha256_file(descriptor_path_);
        if (!expected_sha256.empty() && sha256_ != expected_sha256) {
            throw std::runtime_error(
                "Snapshot IC source disagrees with the operator-admitted SHA-256: expected="
                + expected_sha256 + ", observed=" + sha256_);
        }
        rewind_regular_file(descriptor_);
        hdf5_file_ = H5FileHandle::checked(
            H5Fopen(
                descriptor_path_.c_str(),
                H5F_ACC_RDONLY,
                H5P_DEFAULT),
            "open admitted snapshot IC object " + descriptor_path_);
    } catch (...) {
        close();
        throw;
    }
#endif
}

VerifiedSnapshotSource::~VerifiedSnapshotSource() noexcept {
    close();
}

VerifiedSnapshotSource::VerifiedSnapshotSource(
    VerifiedSnapshotSource&& other) noexcept
    : hdf5_file_(std::move(other.hdf5_file_)),
      descriptor_(std::exchange(other.descriptor_, -1)),
      descriptor_path_(std::move(other.descriptor_path_)),
      sha256_(std::move(other.sha256_)) {}

VerifiedSnapshotSource& VerifiedSnapshotSource::operator=(
    VerifiedSnapshotSource&& other) noexcept {
    if (this != &other) {
        close();
        hdf5_file_ = std::move(other.hdf5_file_);
        descriptor_ = std::exchange(other.descriptor_, -1);
        descriptor_path_ = std::move(other.descriptor_path_);
        sha256_ = std::move(other.sha256_);
    }
    return *this;
}

const std::string& VerifiedSnapshotSource::hdf5_read_path() const {
    if (descriptor_ < 0 || descriptor_path_.empty()) {
        throw std::logic_error(
            "Verified snapshot source is empty or has already been moved");
    }
#if !defined(_WIN32)
    // macOS fdescfs opens /dev/fd/N with dup-like shared-offset semantics.
    // Hashing through the alias can therefore leave the owned descriptor at
    // EOF. Rewind immediately before every consumer opens the alias so HDF5
    // and path-based hash readers always start at byte zero.
    rewind_regular_file(descriptor_);
#endif
    return descriptor_path_;
}

void VerifiedSnapshotSource::verify_unchanged() const {
    if (descriptor_ < 0 || descriptor_path_.empty() || sha256_.empty()) {
        throw std::logic_error(
            "Verified snapshot source is empty or has already been moved");
    }
#if !defined(_WIN32)
    rewind_regular_file(descriptor_);
#endif
    const std::string observed = sha256_file(descriptor_path_);
    if (observed != sha256_) {
        throw std::runtime_error(
            "Snapshot IC file object changed during ingestion: expected sha256="
            + sha256_ + ", observed=" + observed);
    }
}

void VerifiedSnapshotSource::close() noexcept {
    hdf5_file_.reset();
#if !defined(_WIN32)
    if (descriptor_ >= 0) {
        (void)::close(descriptor_);
    }
#endif
    descriptor_ = -1;
    descriptor_path_.clear();
    sha256_.clear();
}

} // namespace cosmo_nbody::io
