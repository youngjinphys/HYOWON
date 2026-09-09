#include "cosmo_nbody/io/exact_hdf5_file_identity.hpp"

#include "cosmo_nbody/io/content_hash.hpp"

#include <hdf5.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cosmo_nbody::io {
namespace {

std::runtime_error system_error(
    const std::string& context,
    const char* operation,
    const std::filesystem::path& path,
    int error_number) {
    return std::runtime_error(
        context + ": " + operation + ": " + path.string() + ": "
        + std::error_code(error_number, std::generic_category()).message());
}

#if !defined(_WIN32)
bool same_file_identity(
    const struct stat& lhs,
    const struct stat& rhs) noexcept {
    const bool common = lhs.st_dev == rhs.st_dev
        && lhs.st_ino == rhs.st_ino
        && lhs.st_mode == rhs.st_mode
        && lhs.st_size == rhs.st_size;
#if defined(__APPLE__)
    return common
        && lhs.st_mtimespec.tv_sec == rhs.st_mtimespec.tv_sec
        && lhs.st_mtimespec.tv_nsec == rhs.st_mtimespec.tv_nsec
        && lhs.st_ctimespec.tv_sec == rhs.st_ctimespec.tv_sec
        && lhs.st_ctimespec.tv_nsec == rhs.st_ctimespec.tv_nsec;
#else
    return common
        && lhs.st_mtim.tv_sec == rhs.st_mtim.tv_sec
        && lhs.st_mtim.tv_nsec == rhs.st_mtim.tv_nsec
        && lhs.st_ctim.tv_sec == rhs.st_ctim.tv_sec
        && lhs.st_ctim.tv_nsec == rhs.st_ctim.tv_nsec;
#endif
}
#endif

} // namespace

struct ExactHdf5FileIdentity::Impl {
    H5FileHandle file;
    std::filesystem::path requested_path;
    std::filesystem::path descriptor_path;
    std::string context;
#if !defined(_WIN32)
    int bound_descriptor{-1};
    int hdf5_descriptor{-1};
    struct stat before{};
#endif

    ~Impl() {
#if !defined(_WIN32)
        if (bound_descriptor >= 0) (void)::close(bound_descriptor);
#endif
    }
};

ExactHdf5FileIdentity::ExactHdf5FileIdentity(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ExactHdf5FileIdentity::ExactHdf5FileIdentity(
    ExactHdf5FileIdentity&&) noexcept = default;
ExactHdf5FileIdentity& ExactHdf5FileIdentity::operator=(
    ExactHdf5FileIdentity&&) noexcept = default;
ExactHdf5FileIdentity::~ExactHdf5FileIdentity() = default;

ExactHdf5FileIdentity ExactHdf5FileIdentity::open_readonly(
    const std::filesystem::path& path,
    std::string context) {
    if (path.empty() || context.empty()) {
        throw std::invalid_argument(
            "Exact HDF5 identity requires a non-empty path and context");
    }
#if defined(_WIN32)
    auto impl = std::make_unique<Impl>();
    impl->requested_path = std::filesystem::absolute(path).lexically_normal();
    impl->context = std::move(context);
    impl->file = H5FileHandle::checked(
        H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
        impl->context + ": open HDF5 object without exact POSIX identity");
    return ExactHdf5FileIdentity(std::move(impl));
#else
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifndef O_NOFOLLOW
    throw std::runtime_error(
        context + ": exact HDF5 identity requires O_NOFOLLOW: " + path.string());
#else
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0) {
        throw system_error(context, "could not bind file", path, errno);
    }

    auto impl = std::make_unique<Impl>();
    impl->bound_descriptor = descriptor;
    impl->requested_path = std::filesystem::absolute(path).lexically_normal();
    impl->descriptor_path = std::filesystem::path("/dev/fd")
        / std::to_string(descriptor);
    impl->context = std::move(context);

    if (::fstat(descriptor, &impl->before) != 0) {
        throw system_error(
            impl->context, "could not inspect bound file", path, errno);
    }
    if (!S_ISREG(impl->before.st_mode) || impl->before.st_size <= 0) {
        throw std::runtime_error(
            impl->context + ": bound HDF5 input must be a non-empty regular file: "
            + path.string());
    }

    impl->file = H5FileHandle::checked(
        H5Fopen(
            impl->descriptor_path.string().c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT),
        impl->context + ": open exact HDF5 object");

    auto access = H5PropertyHandle::checked(
        H5Fget_access_plist(impl->file.get()),
        impl->context + ": HDF5 access properties");
    const hid_t driver = H5Pget_driver(access.get());
    if (driver != H5FD_SEC2) {
        throw std::runtime_error(
            impl->context + ": exact HDF5 identity requires SEC2/POSIX driver");
    }

    void* raw_handle = nullptr;
    check_hdf5(
        H5Fget_vfd_handle(impl->file.get(), H5P_DEFAULT, &raw_handle),
        impl->context + ": get HDF5 POSIX descriptor");
    if (raw_handle == nullptr) {
        throw std::runtime_error(
            impl->context + ": HDF5 SEC2 driver returned no descriptor");
    }
    impl->hdf5_descriptor = *static_cast<int*>(raw_handle);
    if (impl->hdf5_descriptor < 0) {
        throw std::runtime_error(
            impl->context + ": HDF5 SEC2 driver returned invalid descriptor");
    }
    struct stat hdf5_stat{};
    if (::fstat(impl->hdf5_descriptor, &hdf5_stat) != 0) {
        throw system_error(
            impl->context, "could not inspect HDF5 object", path, errno);
    }
    if (!same_file_identity(impl->before, hdf5_stat)) {
        throw std::runtime_error(
            impl->context + ": HDF5 opened a different file object: "
            + path.string());
    }
    return ExactHdf5FileIdentity(std::move(impl));
#endif
}

hid_t ExactHdf5FileIdentity::file_id() const noexcept {
    return impl_ ? impl_->file.get() : -1;
}

const std::filesystem::path& ExactHdf5FileIdentity::requested_path() const noexcept {
    static const std::filesystem::path empty;
    return impl_ ? impl_->requested_path : empty;
}

bool ExactHdf5FileIdentity::exact_identity_available() const noexcept {
#if defined(_WIN32)
    return false;
#else
    return impl_ && impl_->bound_descriptor >= 0 && impl_->hdf5_descriptor >= 0;
#endif
}

std::uint64_t ExactHdf5FileIdentity::size_bytes() const {
    if (!impl_ || impl_->file.get() < 0) {
        throw std::logic_error("Exact HDF5 identity binding is not open");
    }
#if defined(_WIN32)
    throw std::runtime_error(
        impl_->context + ": exact HDF5 object size is unavailable on this platform");
#else
    if (impl_->before.st_size <= 0) {
        throw std::runtime_error(
            impl_->context + ": exact HDF5 object size is invalid");
    }
    using StatSize = decltype(impl_->before.st_size);
    if constexpr (std::numeric_limits<StatSize>::max()
                  > std::numeric_limits<std::uint64_t>::max()) {
        if (impl_->before.st_size
            > static_cast<StatSize>(std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                impl_->context + ": exact HDF5 object size exceeds uint64_t");
        }
    }
    return static_cast<std::uint64_t>(impl_->before.st_size);
#endif
}

void ExactHdf5FileIdentity::read_raw_exact_at(
    std::uint64_t offset,
    std::span<std::byte> destination) const {
    if (!impl_ || impl_->file.get() < 0) {
        throw std::logic_error("Exact HDF5 identity binding is not open");
    }
#if defined(_WIN32)
    (void)offset;
    (void)destination;
    throw std::runtime_error(
        impl_->context + ": exact HDF5 raw reads are unavailable on this platform");
#else
    const std::uint64_t size = size_bytes();
    if (offset > size
        || destination.size() > size - offset
        || offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())) {
        throw std::out_of_range(
            impl_->context + ": exact HDF5 raw read exceeds admitted object");
    }
    std::size_t completed = 0U;
    while (completed < destination.size()) {
        const std::uint64_t absolute = offset
            + static_cast<std::uint64_t>(completed);
        if (absolute > static_cast<std::uint64_t>(
                std::numeric_limits<off_t>::max())) {
            throw std::overflow_error(
                impl_->context + ": exact HDF5 raw offset exceeds off_t");
        }
        const ssize_t count = ::pread(
            impl_->bound_descriptor,
            destination.data() + completed,
            destination.size() - completed,
            static_cast<off_t>(absolute));
        if (count > 0) {
            completed += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        throw system_error(
            impl_->context,
            count == 0 ? "unexpected EOF during exact raw read"
                       : "could not read exact raw bytes",
            impl_->requested_path,
            count == 0 ? EIO : errno);
    }
#endif
}

void ExactHdf5FileIdentity::finish_identity() const {
    if (!impl_ || impl_->file.get() < 0) {
        throw std::logic_error("Exact HDF5 identity binding is not open");
    }
#if defined(_WIN32)
    throw std::runtime_error(
        impl_->context + ": exact HDF5 identity is unavailable on this platform");
#else
    struct stat bound_after{};
    struct stat hdf5_after{};
    struct stat path_after{};
    if (::fstat(impl_->bound_descriptor, &bound_after) != 0
        || ::fstat(impl_->hdf5_descriptor, &hdf5_after) != 0
        || ::lstat(impl_->requested_path.c_str(), &path_after) != 0
        || !same_file_identity(impl_->before, bound_after)
        || !same_file_identity(bound_after, hdf5_after)
        || !same_file_identity(hdf5_after, path_after)) {
        throw std::runtime_error(
            impl_->context
            + ": HDF5 input changed or its pathname was replaced during admission: "
            + impl_->requested_path.string());
    }
#endif
}

std::string ExactHdf5FileIdentity::finish_sha256() const {
    if (!impl_ || impl_->file.get() < 0) {
        throw std::logic_error("Exact HDF5 identity binding is not open");
    }
#if defined(_WIN32)
    throw std::runtime_error(
        impl_->context + ": exact HDF5 identity is unavailable on this platform");
#else
    const std::string digest = sha256_file(impl_->descriptor_path);
    finish_identity();
    return digest;
#endif
}

} // namespace cosmo_nbody::io
