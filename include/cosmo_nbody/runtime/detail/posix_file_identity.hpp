#pragma once

#include <cstdint>
#include <sys/stat.h>

namespace cosmo_nbody::runtime::detail {

struct PosixFileIdentity {
    std::uintmax_t device{0};
    std::uintmax_t inode{0};
    std::uintmax_t mode{0};
    std::intmax_t size{0};
    std::int64_t modification_seconds{0};
    std::int64_t modification_nanoseconds{0};

    friend bool operator==(const PosixFileIdentity&, const PosixFileIdentity&) = default;
};

inline PosixFileIdentity posix_file_identity(const struct stat& status) noexcept {
#if defined(__APPLE__)
    const auto modified = status.st_mtimespec;
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
    const auto modified = status.st_mtim;
#else
#error "nanosecond POSIX file identity is unsupported on this platform"
#endif
    return PosixFileIdentity{
        static_cast<std::uintmax_t>(status.st_dev),
        static_cast<std::uintmax_t>(status.st_ino),
        static_cast<std::uintmax_t>(status.st_mode),
        static_cast<std::intmax_t>(status.st_size),
        static_cast<std::int64_t>(modified.tv_sec),
        static_cast<std::int64_t>(modified.tv_nsec),
    };
}

inline bool same_posix_file_identity(
    const struct stat& left,
    const struct stat& right) noexcept {
    return posix_file_identity(left) == posix_file_identity(right);
}

} // namespace cosmo_nbody::runtime::detail
