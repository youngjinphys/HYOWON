#include "cosmo_nbody/runtime/scratch_file_reservation.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace cosmo_nbody::runtime {

std::filesystem::path prepare_scratch_directory(
    const std::string& directory_text,
    std::string_view label) {
    std::error_code error;
    const bool explicitly_configured = !directory_text.empty();
    std::filesystem::path requested = explicitly_configured
        ? std::filesystem::path(directory_text)
        : std::filesystem::temp_directory_path(error);
    if (error) {
        throw std::runtime_error(
            std::string(label) + " scratch temporary-directory lookup failed: "
            + error.message());
    }

    std::filesystem::path absolute =
        std::filesystem::absolute(requested, error).lexically_normal();
    if (error) {
        throw std::runtime_error(
            std::string(label) + " scratch directory resolution failed for '"
            + requested.string() + "': " + error.message());
    }

    std::filesystem::create_directories(absolute, error);
    if (error) {
        throw std::runtime_error(
            std::string(label) + " scratch directory creation failed for '"
            + absolute.string() + "': " + error.message());
    }

    const auto status = std::filesystem::symlink_status(absolute, error);
    if (error) {
        throw std::runtime_error(
            std::string(label) + " scratch directory inspection failed for '"
            + absolute.string() + "': " + error.message());
    }
    if (std::filesystem::is_symlink(status)) {
        throw std::invalid_argument(
            std::string(label)
            + " scratch directory must not be a symbolic link: "
            + absolute.string());
    }
    if (!std::filesystem::is_directory(status)) {
        throw std::invalid_argument(
            std::string(label) + " scratch path is not a directory: "
            + absolute.string());
    }

    const std::filesystem::path canonical =
        std::filesystem::canonical(absolute, error);
    if (error) {
        throw std::runtime_error(
            std::string(label) + " scratch canonicalization failed for '"
            + absolute.string() + "': " + error.message());
    }
    // Compare against the canonicalized parent so operating-system ancestor
    // links (macOS /var -> /private/var, /tmp -> /private/tmp) stay usable;
    // a symlinked final component still diverges here and is also rejected
    // by the symlink_status check above.
    const std::filesystem::path parent_resolved =
        std::filesystem::canonical(absolute.parent_path(), error);
    if (error) {
        throw std::runtime_error(
            std::string(label)
            + " scratch parent canonicalization failed for '"
            + absolute.string() + "': " + error.message());
    }
    if (explicitly_configured
        && canonical != parent_resolved / absolute.filename()) {
        throw std::invalid_argument(
            std::string(label)
            + " scratch directory must not traverse symbolic links: configured='"
            + absolute.string() + "' canonical='" + canonical.string() + "'");
    }
    return canonical;
}

void reserve_scratch_file_space(
    int fd,
    std::size_t bytes,
    std::string_view label) {
#if defined(__unix__) || defined(__APPLE__)
    if (fd < 0) {
        throw std::invalid_argument(
            std::string(label)
            + " scratch reservation received an invalid file descriptor");
    }
    if (bytes == 0) return;
    if (bytes > static_cast<std::size_t>(
            std::numeric_limits<off_t>::max())) {
        throw std::overflow_error(
            std::string(label) + " scratch reservation exceeds off_t range");
    }
    const off_t length = static_cast<off_t>(bytes);

#if defined(__APPLE__)
    fstore_t store{};
    store.fst_flags = F_ALLOCATECONTIG;
    store.fst_posmode = F_PEOFPOSMODE;
    store.fst_offset = 0;
    store.fst_length = length;
    if (::fcntl(fd, F_PREALLOCATE, &store) == -1) {
        store.fst_flags = F_ALLOCATEALL;
        if (::fcntl(fd, F_PREALLOCATE, &store) == -1) {
            throw std::runtime_error(
                std::string(label) + " scratch space reservation failed: "
                + std::strerror(errno));
        }
    }
#else
    int status = 0;
    do {
        status = ::posix_fallocate(fd, 0, length);
    } while (status == EINTR);
    if (status != 0) {
        throw std::runtime_error(
            std::string(label) + " scratch space reservation failed: "
            + std::strerror(status));
    }
#endif

    // F_PREALLOCATE does not change logical length on macOS. The same-length
    // truncate is harmless after posix_fallocate and keeps this helper exact for
    // every supported POSIX path.
    if (::ftruncate(fd, length) != 0) {
        throw std::runtime_error(
            std::string(label) + " scratch length publication failed: "
            + std::strerror(errno));
    }
#else
    (void)fd;
    (void)bytes;
    (void)label;
    throw std::runtime_error(
        "Scratch file reservation requires a POSIX platform");
#endif
}

void advise_mutable_scratch_mapping(
    void* mapping,
    std::size_t bytes,
    std::string_view label) {
#if defined(__linux__)
    // Apply before first touch so the file-backed mapping is not populated under
    // a different VMA policy. MADV_RANDOM changes read-ahead behavior without
    // changing MAP_SHARED backing or arithmetic.
    if (::madvise(mapping, bytes, MADV_RANDOM) != 0) {
        const int error = errno;
        throw std::runtime_error(
            std::string(label) + " scratch MADV_RANDOM request failed: "
            + std::strerror(error));
    }
    std::clog << "[scratch] label=" << label << " bytes=" << bytes
              << " mmap_advice=MADV_RANDOM result=accepted\n";
#else
    (void)mapping;
    std::clog << "[scratch] label=" << label << " bytes=" << bytes
              << " mmap_advice=kernel_default reason=non_linux_platform\n";
#endif
}

} // namespace cosmo_nbody::runtime
