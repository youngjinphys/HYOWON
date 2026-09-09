#include "cosmo_nbody/io/durable_file_publication.hpp"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <stdio.h>
#endif
#endif

namespace cosmo_nbody::io {
namespace {

std::atomic<unsigned long long> staging_counter{0};

void require_named_path(const std::filesystem::path& path) {
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument(
            "Durable file publication path must name a file");
    }
}

bool exists_no_follow(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error) {
        if (error == std::errc::no_such_file_or_directory) return false;
        throw std::runtime_error(
            "Cannot inspect durable file path " + path.string()
            + ": " + error.message());
    }
    return std::filesystem::exists(status);
}

void require_existing_directory_no_symlink(
    const std::filesystem::path& path,
    std::string_view role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(
            std::string(role)
            + " parent must be an existing non-symlink directory: "
            + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
}

std::filesystem::path canonical_parent(
    const std::filesystem::path& requested_path,
    std::string_view role) {
    const std::filesystem::path requested_parent = requested_path.has_parent_path()
        ? requested_path.parent_path()
        : std::filesystem::path{"."};
    require_existing_directory_no_symlink(requested_parent, role);
    std::error_code error;
    const auto parent = std::filesystem::canonical(requested_parent, error);
    if (error) {
        throw std::runtime_error(
            "Cannot canonicalize " + std::string(role) + " parent: "
            + error.message());
    }
    return parent;
}

void require_replaceable_destination(
    const std::filesystem::path& path,
    std::string_view role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error) {
        if (error == std::errc::no_such_file_or_directory) return;
        throw std::runtime_error(
            "Cannot inspect " + std::string(role) + " destination "
            + path.string() + ": " + error.message());
    }
    if (!std::filesystem::exists(status)) return;
    if (std::filesystem::is_symlink(status)
        || !std::filesystem::is_regular_file(status)) {
        throw std::runtime_error(
            std::string(role)
            + " destination must be absent or a regular non-symlink file: "
            + path.string());
    }
}

void require_absent_destination(
    const std::filesystem::path& path,
    std::string_view role) {
    if (exists_no_follow(path)) {
        throw DurableFilePublicationCollision(
            std::string(role) + " destination already exists: "
            + path.string());
    }
}

void require_destination_policy(
    const std::filesystem::path& path,
    std::string_view role,
    DurableFilePublicationPolicy policy) {
    if (policy == DurableFilePublicationPolicy::RequireAbsent) {
        require_absent_destination(path, role);
        return;
    }
    require_replaceable_destination(path, role);
}

unsigned long long process_identifier() {
#ifdef _WIN32
    return static_cast<unsigned long long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long long>(::getpid());
#endif
}

std::filesystem::path unique_stage_path(
    const std::filesystem::path& parent,
    const std::filesystem::path& filename) {
    for (unsigned int attempt = 0; attempt < 1024; ++attempt) {
        const auto counter = staging_counter.fetch_add(
            1, std::memory_order_relaxed);
        const auto ticks = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
        std::ostringstream name;
        name << '.' << filename.string()
             << ".file-stage."
             << process_identifier() << '.'
             << static_cast<unsigned long long>(ticks) << '.'
             << counter;
        const auto candidate = parent / name.str();
        if (!exists_no_follow(candidate)) return candidate;
    }
    throw std::runtime_error(
        "Could not allocate a unique sibling file staging path");
}

#ifdef _WIN32
std::string windows_error_message(DWORD error) {
    return "Windows error "
        + std::to_string(static_cast<unsigned long>(error));
}

void sync_regular_file(const std::filesystem::path& path) {
    HANDLE handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "Cannot open staged file for durability sync: " + path.string()
            + ": " + windows_error_message(GetLastError()));
    }
    const BOOL flushed = FlushFileBuffers(handle);
    const DWORD flush_error = flushed ? ERROR_SUCCESS : GetLastError();
    CloseHandle(handle);
    if (!flushed) {
        throw std::runtime_error(
            "Cannot flush staged file to storage: " + path.string()
            + ": " + windows_error_message(flush_error));
    }
}

void atomic_publish(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string_view role,
    DurableFilePublicationPolicy policy) {
    DWORD flags = MOVEFILE_WRITE_THROUGH;
    if (policy == DurableFilePublicationPolicy::ReplaceExisting) {
        flags |= MOVEFILE_REPLACE_EXISTING;
    }
    if (MoveFileExW(source.c_str(), destination.c_str(), flags)) return;

    const DWORD error = GetLastError();
    if (policy == DurableFilePublicationPolicy::RequireAbsent
        && (error == ERROR_ALREADY_EXISTS || error == ERROR_FILE_EXISTS)) {
        throw DurableFilePublicationCollision(
            std::string(role) + " destination already exists: "
            + destination.string());
    }
    throw std::runtime_error(
        "Atomic file publication failed: " + windows_error_message(error));
}

void sync_parent_directory(const std::filesystem::path&) {
    // MOVEFILE_WRITE_THROUGH is the strongest portable request available here.
}
#else
int close_fd(int descriptor) noexcept {
#if defined(__linux__)
    // Linux releases the descriptor before reporting most close errors,
    // including EINTR, so retrying can close a newly reused descriptor.
    return ::close(descriptor);
#else
    // Darwin's ordinary close() can be interrupted before descriptor release.
    int status = 0;
    do {
        status = ::close(descriptor);
    } while (status != 0 && errno == EINTR);
    return status;
#endif
}

void close_fd_noexcept(int descriptor) noexcept {
    if (descriptor < 0) return;
    (void)close_fd(descriptor);
}

void sync_regular_file(const std::filesystem::path& path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Cannot open staged file for fsync: " + path.string()
            + ": " + std::strerror(errno));
    }
    int status = 0;
    do {
        status = ::fsync(descriptor);
    } while (status != 0 && errno == EINTR);
    const int sync_error = status == 0 ? 0 : errno;
    close_fd_noexcept(descriptor);
    if (status != 0) {
        throw std::runtime_error(
            "Cannot fsync staged file: " + path.string()
            + ": " + std::strerror(sync_error));
    }
}

void atomic_replace(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
    int status = 0;
    do {
        status = ::rename(source.c_str(), destination.c_str());
    } while (status != 0 && errno == EINTR);
    if (status != 0) {
        throw std::runtime_error(
            "Atomic file replacement failed: "
            + std::string(std::strerror(errno)));
    }
}

void atomic_publish_absent(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string_view role) {
#if defined(__linux__) && defined(SYS_renameat2)
    long status = 0;
    do {
        status = ::syscall(
            SYS_renameat2,
            AT_FDCWD,
            source.c_str(),
            AT_FDCWD,
            destination.c_str(),
            1U); // Linux RENAME_NOREPLACE
    } while (status != 0 && errno == EINTR);
    if (status == 0) return;
    const int error = errno;
    if (error == EEXIST) {
        throw DurableFilePublicationCollision(
            std::string(role) + " destination already exists: "
            + destination.string());
    }
    if (error == ENOSYS || error == EINVAL
#ifdef EOPNOTSUPP
        || error == EOPNOTSUPP
#endif
    ) {
        throw std::runtime_error(
            std::string(role)
            + " requires atomic no-replace rename, but the active Linux runtime/filesystem does not support it: "
            + destination.string());
    }
    throw std::runtime_error(
        "Atomic create-only file publication failed: "
        + std::string(std::strerror(error)));
#elif defined(__APPLE__)
    int status = 0;
    do {
        status = ::renamex_np(
            source.c_str(), destination.c_str(), RENAME_EXCL);
    } while (status != 0 && errno == EINTR);
    if (status == 0) return;
    const int error = errno;
    if (error == EEXIST) {
        throw DurableFilePublicationCollision(
            std::string(role) + " destination already exists: "
            + destination.string());
    }
    if (error == EINVAL
#ifdef ENOTSUP
        || error == ENOTSUP
#endif
#ifdef EOPNOTSUPP
        || error == EOPNOTSUPP
#endif
    ) {
        throw std::runtime_error(
            std::string(role)
            + " requires atomic exclusive rename, but the active macOS volume does not support it: "
            + destination.string());
    }
    throw std::runtime_error(
        "Atomic create-only file publication failed: "
        + std::string(std::strerror(error)));
#else
    (void)source;
    (void)destination;
    throw std::runtime_error(
        std::string(role)
        + " requires an atomic no-replace rename primitive that is not implemented on this platform");
#endif
}

void atomic_publish(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string_view role,
    DurableFilePublicationPolicy policy) {
    if (policy == DurableFilePublicationPolicy::RequireAbsent) {
        atomic_publish_absent(source, destination, role);
        return;
    }
    atomic_replace(source, destination);
}

void sync_parent_directory(const std::filesystem::path& parent) {
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int descriptor = ::open(parent.c_str(), flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Cannot open file publication parent for fsync: "
            + parent.string() + ": " + std::strerror(errno));
    }
    int status = 0;
    do {
        status = ::fsync(descriptor);
    } while (status != 0 && errno == EINTR);
    const int sync_error = status == 0 ? 0 : errno;
    close_fd_noexcept(descriptor);
    if (status != 0) {
        throw std::runtime_error(
            "Cannot fsync file publication parent: "
            + parent.string() + ": " + std::strerror(sync_error));
    }
}
#endif

std::uintmax_t require_regular_nonempty_file(
    const std::filesystem::path& path,
    std::string_view role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_regular_file(status)) {
        throw std::runtime_error(
            std::string(role)
            + " is not a regular non-symlink file: " + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
    const std::uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error || bytes == 0) {
        throw std::runtime_error(
            std::string(role) + " is empty or unreadable: " + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
    return bytes;
}

void require_exact_published_size(
    const std::filesystem::path& path,
    std::uintmax_t expected_size,
    std::string_view role) {
    const std::uintmax_t observed = require_regular_nonempty_file(path, role);
    if (observed != expected_size) {
        throw std::runtime_error(
            std::string(role) + " changed size during atomic publication: "
            + path.string());
    }
}

} // namespace

void sync_regular_file_and_parent_durable(
    const std::filesystem::path& requested_path,
    std::string_view role) {
    require_named_path(requested_path);
    if (role.empty()) {
        throw std::invalid_argument(
            "Durable existing-file sync requires a non-empty role");
    }
    const auto parent = canonical_parent(requested_path, role);
    const auto path = parent / requested_path.filename();
    (void)require_regular_nonempty_file(path, role);
    sync_regular_file(path);
    sync_parent_directory(parent);
}

DurableFilePublication::DurableFilePublication(
    const std::filesystem::path& requested_path,
    std::string role,
    DurableFilePublicationPolicy policy)
    : role_(std::move(role)),
      policy_(policy) {
    require_named_path(requested_path);
    if (role_.empty()) {
        throw std::invalid_argument(
            "Durable file publication requires a non-empty role");
    }
    parent_ = canonical_parent(requested_path, role_);
    final_path_ = parent_ / requested_path.filename();
    require_destination_policy(final_path_, role_, policy_);
    stage_path_ = unique_stage_path(parent_, requested_path.filename());
}

DurableFilePublication::~DurableFilePublication() noexcept {
    if (published_.final_visible() || published_.discarded()) return;
    try {
        discard_unpublished();
    } catch (...) {
        // Best effort only: a destructor must not convert cleanup trouble into
        // process termination. The canonical pathname was never published.
    }
}

void DurableFilePublication::discard_unpublished() {
    if (published_.final_visible() || published_.committed()) {
        throw std::logic_error(
            "Cannot discard a durable file publication after the canonical pathname became visible");
    }
    if (published_.discarded()) return;

    std::error_code status_error;
    const auto status = std::filesystem::symlink_status(
        stage_path_, status_error);
    bool stage_exists = false;
    if (status_error) {
        if (status_error != std::errc::no_such_file_or_directory) {
            throw std::runtime_error(
                "Cannot inspect unpublished " + role_ + " stage "
                + stage_path_.string() + ": " + status_error.message());
        }
    } else {
        stage_exists = std::filesystem::exists(status);
    }

    if (stage_exists) {
        if (std::filesystem::is_symlink(status)
            || !std::filesystem::is_regular_file(status)) {
            throw std::runtime_error(
                "Refusing to remove unpublished " + role_
                + " stage after its object type changed: "
                + stage_path_.string());
        }
        std::error_code remove_error;
        if (!std::filesystem::remove(stage_path_, remove_error)
            || remove_error) {
            throw std::runtime_error(
                "Cannot remove unpublished " + role_ + " stage "
                + stage_path_.string()
                + (remove_error ? ": " + remove_error.message() : ""));
        }
        sync_parent_directory(parent_);
    }
    published_.mark_discarded();
}

void DurableFilePublication::publish_nonempty() {
    if (published_.final_visible() || published_.committed()
        || published_.discarded()) {
        throw std::logic_error(
            "Durable file publication already changed the canonical pathname");
    }

    published_.require_hdf5_close_integrity();
    const std::uintmax_t expected_size =
        require_regular_nonempty_file(stage_path_, role_ + " staging file");
    sync_regular_file(stage_path_);
    // Keep this preflight because it produces precise type/collision diagnostics
    // before the syscall. RequireAbsent correctness does not depend on it: the
    // publication syscall itself is no-replace and closes the TOCTOU window.
    require_destination_policy(final_path_, role_, policy_);

    atomic_publish(stage_path_, final_path_, role_, policy_);
    published_.mark_final_visible();
    require_exact_published_size(final_path_, expected_size, role_);
    sync_parent_directory(parent_);
    published_.mark_committed();
}

} // namespace cosmo_nbody::io
