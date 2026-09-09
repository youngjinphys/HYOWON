#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/io/durable_file_publication.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cosmo_nbody::io {
namespace {

#ifdef _WIN32
void write_complete_stage(
    const std::filesystem::path& stage_path,
    std::string_view content,
    std::string_view role) {
    HANDLE handle = CreateFileW(
        stage_path.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "Failed to exclusively create durable staged "
            + std::string(role) + ": Windows error "
            + std::to_string(GetLastError()));
    }

    std::size_t offset = 0U;
    while (offset < content.size()) {
        const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
            content.size() - offset,
            static_cast<std::size_t>(std::numeric_limits<DWORD>::max())));
        DWORD written = 0U;
        if (!WriteFile(
                handle,
                content.data() + offset,
                request,
                &written,
                nullptr)
            || written == 0U) {
            const DWORD error = GetLastError();
            CloseHandle(handle);
            throw std::runtime_error(
                "Failed to write durable staged " + std::string(role)
                + ": Windows error " + std::to_string(error));
        }
        offset += written;
    }

    if (!FlushFileBuffers(handle)) {
        const DWORD error = GetLastError();
        CloseHandle(handle);
        throw std::runtime_error(
            "Failed to flush durable staged " + std::string(role)
            + ": Windows error " + std::to_string(error));
    }
    if (!CloseHandle(handle)) {
        throw std::runtime_error(
            "Failed to close durable staged " + std::string(role)
            + ": Windows error " + std::to_string(GetLastError()));
    }
}
#else
int close_descriptor(int descriptor) noexcept {
#if defined(__linux__)
    // Linux releases the descriptor before reporting most close errors,
    // including EINTR. Retrying may therefore close a descriptor that another
    // thread has already acquired under the same numeric value.
    return ::close(descriptor);
#else
    // Darwin's ordinary close() is a pthread cancellation point and can report
    // EINTR before the descriptor has actually been closed. Preserve its
    // established retry contract instead of applying Linux semantics globally.
    int status = 0;
    do {
        status = ::close(descriptor);
    } while (status != 0 && errno == EINTR);
    return status;
#endif
}

void close_descriptor_noexcept(int descriptor) noexcept {
    if (descriptor < 0) return;
    (void)close_descriptor(descriptor);
}

void write_complete_stage(
    const std::filesystem::path& stage_path,
    std::string_view content,
    std::string_view role) {
    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(stage_path.c_str(), flags, 0600);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Failed to exclusively create durable staged "
            + std::string(role) + ": " + std::strerror(errno));
    }

    std::size_t offset = 0U;
    while (offset < content.size()) {
        const ssize_t written = ::write(
            descriptor,
            content.data() + offset,
            content.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            const int error = written < 0 ? errno : EIO;
            close_descriptor_noexcept(descriptor);
            throw std::runtime_error(
                "Failed to write durable staged " + std::string(role)
                + ": " + std::strerror(error));
        }
        offset += static_cast<std::size_t>(written);
    }

    int sync_status = 0;
    do {
        sync_status = ::fsync(descriptor);
    } while (sync_status != 0 && errno == EINTR);
    const int sync_error = sync_status == 0 ? 0 : errno;
    if (sync_status != 0) {
        close_descriptor_noexcept(descriptor);
        throw std::runtime_error(
            "Failed to flush durable staged " + std::string(role)
            + ": " + std::strerror(sync_error));
    }

    const int close_status = close_descriptor(descriptor);
    const int close_error = close_status == 0 ? 0 : errno;
    if (close_status != 0) {
        throw std::runtime_error(
            "Failed to close durable staged " + std::string(role)
            + ": " + std::strerror(close_error));
    }
}
#endif

void write_text_durable(
    const std::filesystem::path& path,
    std::string_view content,
    std::string_view role,
    DurableFilePublicationPolicy policy,
    const DurableTextStagingValidator& validator) {
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument(
            "Durable text publication path must name a file");
    }
    if (content.empty() || role.empty()) {
        throw std::invalid_argument(
            "Durable text publication requires non-empty content and role");
    }

    DurableFilePublication publication(
        path,
        std::string(role),
        policy);
    write_complete_stage(publication.staging_path(), content, role);
    if (validator) validator(publication.staging_path());
    publication.publish_nonempty();
}

} // namespace

void write_text_durable_atomic(
    const std::filesystem::path& path,
    std::string_view content,
    std::string_view role) {
    write_text_durable(
        path,
        content,
        role,
        DurableFilePublicationPolicy::ReplaceExisting,
        {});
}

void write_text_durable_atomic_validated(
    const std::filesystem::path& path,
    std::string_view content,
    std::string_view role,
    const DurableTextStagingValidator& validator) {
    write_text_durable(
        path,
        content,
        role,
        DurableFilePublicationPolicy::RequireAbsent,
        validator);
}

} // namespace cosmo_nbody::io
