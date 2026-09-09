#include "cosmo_nbody/io/durable_file_publication.hpp"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <stdio.h>
#endif
#endif

namespace cosmo_nbody::io {
namespace {

#ifndef _WIN32
std::atomic<unsigned long long> directory_stage_counter{0};

std::filesystem::path canonical_parent(
    const std::filesystem::path& requested_path,
    std::string_view role) {
    const auto requested_parent = requested_path.has_parent_path()
        ? requested_path.parent_path()
        : std::filesystem::path{"."};
    std::error_code error;
    const auto parent = std::filesystem::canonical(
        requested_parent, error);
    if (error) {
        throw std::runtime_error(
            "Cannot canonicalize " + std::string(role)
            + " parent: " + error.message());
    }
    const auto status = std::filesystem::symlink_status(parent, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(
            std::string(role)
            + " parent must resolve to an existing directory: "
            + requested_parent.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
    return parent;
}

void close_descriptor_noexcept(int descriptor) noexcept {
    if (descriptor < 0) return;
#if defined(__linux__)
    (void)::close(descriptor);
#else
    while (::close(descriptor) != 0 && errno == EINTR) {
    }
#endif
}

void sync_descriptor(int descriptor, std::string_view role) {
    int status = 0;
    do {
        status = ::fsync(descriptor);
    } while (status != 0 && errno == EINTR);
    if (status != 0) {
        throw std::runtime_error(
            "Cannot fsync " + std::string(role)
            + ": " + std::strerror(errno));
    }
}

bool same_identity(
    const struct stat& status,
    std::uintmax_t device,
    std::uintmax_t inode) {
    return S_ISDIR(status.st_mode)
        && static_cast<std::uintmax_t>(status.st_dev) == device
        && static_cast<std::uintmax_t>(status.st_ino) == inode;
}

void require_parent_path_identity(
    const std::filesystem::path& parent,
    std::uintmax_t device,
    std::uintmax_t inode,
    std::string_view role) {
    struct stat status{};
    const int inspect_status = ::lstat(parent.c_str(), &status);
    if (inspect_status != 0
        || !same_identity(status, device, inode)) {
        const int error = inspect_status != 0 ? errno : EIO;
        throw std::runtime_error(
            std::string(role)
            + " parent pathname no longer names the opened directory: "
            + parent.string() + ": " + std::strerror(error));
    }
}

void require_absent_at(
    int parent,
    const std::filesystem::path& name,
    const std::filesystem::path& display_path,
    std::string_view role) {
    struct stat status{};
    if (::fstatat(
            parent, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0) {
        throw DurableFilePublicationCollision(
            std::string(role) + " destination already exists: "
            + display_path.string());
    }
    const int error = errno;
    if (error != ENOENT) {
        throw std::runtime_error(
            "Cannot inspect " + std::string(role) + " destination "
            + display_path.string() + ": " + std::strerror(error));
    }
}

int open_expected_directory(
    int parent,
    const std::filesystem::path& name,
    std::uintmax_t device,
    std::uintmax_t inode,
    std::string_view role) {
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::openat(parent, name.c_str(), flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Cannot open " + std::string(role)
            + " directory: " + std::strerror(errno));
    }
    struct stat status{};
    const int inspect_status = ::fstat(descriptor, &status);
    if (inspect_status != 0
        || !same_identity(status, device, inode)) {
        const int error = inspect_status != 0 ? errno : EIO;
        close_descriptor_noexcept(descriptor);
        throw std::runtime_error(
            std::string(role)
            + " directory identity changed: " + std::strerror(error));
    }
    return descriptor;
}

std::vector<std::string> directory_entries(
    int descriptor,
    std::string_view role) {
    const int scan_descriptor = ::dup(descriptor);
    if (scan_descriptor < 0) {
        throw std::runtime_error(
            "Cannot scan " + std::string(role)
            + ": " + std::strerror(errno));
    }
    struct DirectoryStreamCloser {
        void operator()(DIR* stream) const noexcept {
            if (stream != nullptr) (void)::closedir(stream);
        }
    };
    std::unique_ptr<DIR, DirectoryStreamCloser> stream(
        ::fdopendir(scan_descriptor));
    if (!stream) {
        const int error = errno;
        close_descriptor_noexcept(scan_descriptor);
        throw std::runtime_error(
            "Cannot scan " + std::string(role)
            + ": " + std::strerror(error));
    }
    std::vector<std::string> entries;
    errno = 0;
    while (const dirent* entry = ::readdir(stream.get())) {
        const std::string_view name{entry->d_name};
        if (name != "." && name != "..") entries.emplace_back(name);
    }
    const int scan_error = errno;
    stream.reset();
    if (scan_error != 0) {
        throw std::runtime_error(
            "Cannot scan " + std::string(role)
            + ": " + std::strerror(scan_error));
    }
    return entries;
}

void require_flat_nonempty_directory(
    int directory,
    std::string_view role) {
    const auto entries = directory_entries(directory, role);
    if (entries.empty()) {
        throw std::runtime_error(
            std::string(role) + " directory is empty");
    }
    for (const auto& name : entries) {
        struct stat status{};
        if (::fstatat(
                directory,
                name.c_str(),
                &status,
                AT_SYMLINK_NOFOLLOW) != 0) {
            throw std::runtime_error(
                "Cannot inspect " + std::string(role)
                + " entry " + name + ": " + std::strerror(errno));
        }
        if (!S_ISREG(status.st_mode)) {
            throw std::runtime_error(
                std::string(role)
                + " requires regular-file analyzer products only; non-regular entry found: "
                + name);
        }
    }
}

void remove_flat_entries(
    int directory,
    std::string_view role) {
    for (const auto& name : directory_entries(directory, role)) {
        struct stat status{};
        if (::fstatat(
                directory, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            if (errno == ENOENT) continue;
            throw std::runtime_error(
                "Cannot inspect unpublished " + std::string(role)
                + " entry " + name + ": " + std::strerror(errno));
        }
        if (S_ISDIR(status.st_mode)) {
            throw std::runtime_error(
                "Refusing recursive cleanup of unexpected directory in unpublished "
                + std::string(role) + " stage: " + name);
        }
        if (::unlinkat(directory, name.c_str(), 0) != 0
            && errno != ENOENT) {
            throw std::runtime_error(
                "Cannot remove unpublished " + std::string(role)
                + " entry " + name + ": " + std::strerror(errno));
        }
    }
}
#endif

} // namespace

DurableDirectoryPublication::DurableDirectoryPublication(
    const std::filesystem::path& requested_path,
    std::string role)
    : role_(std::move(role)) {
    auto named_path = requested_path.lexically_normal();
    if (!named_path.empty() && named_path.filename().empty()
        && named_path.parent_path() != named_path) {
        named_path = named_path.parent_path();
    }
    if (named_path.empty() || named_path.filename().empty()) {
        throw std::invalid_argument(
            "Durable directory publication path must name a directory");
    }
    if (role_.empty()) {
        throw std::invalid_argument(
            "Durable directory publication requires a non-empty role");
    }
#ifdef _WIN32
    throw std::runtime_error(
        "Durable directory publication requires a qualified POSIX no-replace directory rename primitive");
#else
    parent_ = canonical_parent(named_path, role_);
    final_path_ = parent_ / named_path.filename();

    struct stat before{};
    const int parent_inspect_status = ::lstat(parent_.c_str(), &before);
    if (parent_inspect_status != 0
        || !S_ISDIR(before.st_mode)) {
        const int error = parent_inspect_status != 0 ? errno : ENOTDIR;
        throw std::runtime_error(
            "Cannot inspect " + role_ + " parent: "
            + std::strerror(error));
    }
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(parent_.c_str(), flags);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Cannot open " + role_ + " parent: "
            + std::strerror(errno));
    }
    parent_descriptor_ = descriptor;
    try {
        struct stat after{};
        const int inspect_status = ::fstat(descriptor, &after);
        if (inspect_status != 0
            || !S_ISDIR(after.st_mode)
            || before.st_dev != after.st_dev
            || before.st_ino != after.st_ino) {
            const int error = inspect_status != 0 ? errno : EIO;
            throw std::runtime_error(
                role_ + " parent changed while its identity was acquired: "
                + std::strerror(error));
        }
        parent_device_ = static_cast<std::uintmax_t>(after.st_dev);
        parent_inode_ = static_cast<std::uintmax_t>(after.st_ino);
        const bool shared_write =
            (after.st_mode & (S_IWGRP | S_IWOTH)) != 0;
        const bool sticky = (after.st_mode & S_ISVTX) != 0;
        if (shared_write && !sticky) {
            throw std::runtime_error(
                role_
                + " parent is writable by another POSIX principal without sticky-directory rename protection");
        }
        require_absent_at(
            descriptor, final_path_.filename(), final_path_, role_);

#if !((defined(__linux__) && defined(SYS_renameat2)) || defined(__APPLE__))
        throw std::runtime_error(
            role_
            + " requires a qualified atomic no-replace directory rename primitive on this platform");
#endif
        for (unsigned int attempt = 0; attempt < 1024; ++attempt) {
            const auto counter = directory_stage_counter.fetch_add(
                1, std::memory_order_relaxed);
            const auto ticks = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
            std::ostringstream name;
            name << ".hyowon-directory-stage." << ::getpid() << '.'
                 << static_cast<unsigned long long>(ticks) << '.'
                 << counter;
            const auto candidate_name = std::filesystem::path{name.str()};
            if (::mkdirat(descriptor, candidate_name.c_str(), S_IRWXU) == 0) {
                struct stat identity{};
                const int inspect_status = ::fstatat(
                        descriptor,
                        candidate_name.c_str(),
                        &identity,
                        AT_SYMLINK_NOFOLLOW);
                if (inspect_status != 0
                    || !S_ISDIR(identity.st_mode)) {
                    const int error = inspect_status != 0 ? errno : EIO;
                    throw std::runtime_error(
                        "Cannot establish " + role_
                        + " staging-directory identity: "
                        + std::strerror(error));
                }
                stage_name_ = candidate_name;
                stage_path_ = parent_ / stage_name_;
                stage_device_ = static_cast<std::uintmax_t>(identity.st_dev);
                stage_inode_ = static_cast<std::uintmax_t>(identity.st_ino);
                return;
            }
            if (errno != EEXIST) {
                throw std::runtime_error(
                    "Cannot create exclusive " + role_
                    + " staging directory: " + std::strerror(errno));
            }
        }
        throw std::runtime_error(
            "Could not exclusively create a sibling " + role_
            + " staging directory");
    } catch (...) {
        close_descriptor_noexcept(descriptor);
        parent_descriptor_ = -1;
        throw;
    }
#endif
}

DurableDirectoryPublication::~DurableDirectoryPublication() noexcept {
#ifndef _WIN32
    if (!final_visible_ && !discarded_) {
        try {
            discard_unpublished();
        } catch (...) {
        }
    }
    close_descriptor_noexcept(parent_descriptor_);
    parent_descriptor_ = -1;
#endif
}

void DurableDirectoryPublication::discard_unpublished() {
    if (final_visible_ || committed_) {
        throw std::logic_error(
            "Cannot discard a directory publication after the canonical pathname became visible");
    }
    if (discarded_) return;
#ifdef _WIN32
    throw std::runtime_error(
        "Durable directory publication is unsupported on Windows");
#else
    const int parent = parent_descriptor_;
    const int stage = open_expected_directory(
        parent,
        stage_name_,
        stage_device_,
        stage_inode_,
        role_ + " staging");
    try {
        remove_flat_entries(stage, role_);
    } catch (...) {
        close_descriptor_noexcept(stage);
        throw;
    }
    close_descriptor_noexcept(stage);

    const int verified = open_expected_directory(
        parent,
        stage_name_,
        stage_device_,
        stage_inode_,
        role_ + " staging");
    close_descriptor_noexcept(verified);
    if (::unlinkat(parent, stage_name_.c_str(), AT_REMOVEDIR) != 0) {
        throw std::runtime_error(
            "Cannot remove unpublished " + role_ + " stage: "
            + std::strerror(errno));
    }
    sync_descriptor(parent, role_ + " parent");
    discarded_ = true;
#endif
}

void DurableDirectoryPublication::publish_nonempty() {
    if (final_visible_ || committed_ || discarded_) {
        throw std::logic_error(
            "Durable directory publication already changed the canonical pathname");
    }
#ifdef _WIN32
    throw std::runtime_error(
        "Durable directory publication is unsupported on Windows");
#else
    require_no_hdf5_close_failures_since(
        hdf5_close_failure_epoch_, "durable directory publication");
    const int parent = parent_descriptor_;
    require_parent_path_identity(
        parent_, parent_device_, parent_inode_, role_);
    const int stage = open_expected_directory(
        parent,
        stage_name_,
        stage_device_,
        stage_inode_,
        role_ + " staging");
    try {
        require_flat_nonempty_directory(stage, role_ + " staging");
        sync_descriptor(stage, role_ + " staging directory");
    } catch (...) {
        close_descriptor_noexcept(stage);
        throw;
    }
    close_descriptor_noexcept(stage);

    int status = -1;
#if defined(__linux__) && defined(SYS_renameat2)
    do {
        status = static_cast<int>(::syscall(
            SYS_renameat2,
            parent, stage_name_.c_str(),
            parent, final_path_.filename().c_str(),
            1U)); // Linux RENAME_NOREPLACE
    } while (status != 0 && errno == EINTR);
#elif defined(__APPLE__)
    unsigned int flags = RENAME_EXCL;
#ifdef RENAME_NOFOLLOW_ANY
    flags |= RENAME_NOFOLLOW_ANY;
#endif
    do {
        status = ::renameatx_np(
            parent, stage_name_.c_str(),
            parent, final_path_.filename().c_str(),
            flags);
    } while (status != 0 && errno == EINTR);
#else
    throw std::runtime_error(
        role_ + " requires an atomic no-replace directory rename primitive that is not implemented on this platform");
#endif
    if (status != 0) {
        const int error = errno;
        struct stat visible{};
        final_visible_ = ::fstatat(
                             parent,
                             final_path_.filename().c_str(),
                             &visible,
                             AT_SYMLINK_NOFOLLOW) == 0
            && same_identity(visible, stage_device_, stage_inode_);
        struct stat remaining_stage{};
        const bool stage_absent = ::fstatat(
                                      parent,
                                      stage_name_.c_str(),
                                      &remaining_stage,
                                      AT_SYMLINK_NOFOLLOW) != 0
            && errno == ENOENT;
        if (final_visible_ && stage_absent) {
            // Network filesystems can report an error after the server has
            // completed the rename. Exact source identity plus source absence
            // is sufficient to continue with post-publication verification.
            final_visible_ = true;
        } else if (error == EEXIST) {
            throw DurableFilePublicationCollision(
                role_ + " destination already exists: "
                + final_path_.string());
        } else if (error == ENOSYS || error == EINVAL
#ifdef EOPNOTSUPP
            || error == EOPNOTSUPP
#endif
        ) {
            throw std::runtime_error(
                role_
                + " requires atomic no-replace directory rename, but the active runtime/filesystem does not support it: "
                + final_path_.string());
        } else {
            throw std::runtime_error(
                "Atomic create-only directory publication failed: "
                + std::string(std::strerror(error)));
        }
    }
    final_visible_ = true;

    require_parent_path_identity(
        parent_, parent_device_, parent_inode_, role_);
    const int published = open_expected_directory(
        parent,
        final_path_.filename(),
        stage_device_,
        stage_inode_,
        role_);
    close_descriptor_noexcept(published);
    sync_descriptor(parent, role_ + " parent");
    committed_ = true;
#endif
}

} // namespace cosmo_nbody::io
