#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/scratch_file_reservation.hpp"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace cosmo_nbody::runtime {
namespace {

std::string safe_label(std::string label) {
    if (label.empty()) label = "scratch";
    for (char& character : label) {
        const bool valid =
            (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9')
            || character == '_' || character == '-';
        if (!valid) character = '_';
    }
    return label;
}

} // namespace

void RawScratchBuffer::FreeDeleter::operator()(
    std::byte* pointer) const noexcept {
    std::free(pointer);
}

RawScratchBuffer::RawScratchBuffer(
    std::size_t bytes,
    config::ScratchMode mode,
    const std::string& directory_text,
    const std::string& label)
    : bytes_(bytes),
      label_(safe_label(label)) {
    if (bytes_ == 0) return;

    const bool request_file = config::uses_file_backed_scratch(mode);
#if defined(__unix__) || defined(__APPLE__)
    if (request_file) {
        const std::filesystem::path directory = prepare_scratch_directory(
            directory_text, "Raw");

        std::string pattern =
            (directory / ("cosmo_" + label_ + "_XXXXXX")).string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        fd_ = ::mkstemp(writable.data());
        if (fd_ < 0) {
            throw std::runtime_error(
                "mkstemp failed for raw scratch: "
                + std::string(std::strerror(errno)));
        }
        int unlink_status = 0;
        do {
            unlink_status = ::unlink(writable.data());
        } while (unlink_status != 0 && errno == EINTR);
        if (unlink_status != 0) {
            const int error = errno;
            // Construction has not completed: no destructor will close fd_.
            // Release it before formatting an error, which can itself allocate.
            (void)::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "unlink failed for raw scratch; empty file may remain at '"
                + std::string(writable.data()) + "': " + std::strerror(error));
        }
        try {
            // The backing-store allocation itself is authoritative. A separate
            // statvfs-style free-space prediction with an arbitrary reserve can
            // disagree with quotas, concurrent allocations, filesystem policy,
            // or provider semantics and must not reject an otherwise valid run.
            reserve_scratch_file_space(fd_, bytes_, "Raw");
        } catch (...) {
            ::close(fd_);
            fd_ = -1;
            throw;
        }
        void* mapping = ::mmap(
            nullptr,
            bytes_,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd_,
            0);
        if (mapping == MAP_FAILED) {
            const int error = errno;
            (void)::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "mmap failed for raw scratch: "
                + std::string(std::strerror(error)));
        }
        try {
            advise_mutable_scratch_mapping(mapping, bytes_, label_);
        } catch (...) {
            (void)::munmap(mapping, bytes_);
            (void)::close(fd_);
            fd_ = -1;
            throw;
        }
        // mmap owns an independent reference to the unlinked backing file.
        // Retaining one descriptor per live buffer is unnecessary and can
        // exhaust RLIMIT_NOFILE while ample mapping/backing capacity remains.
        (void)::close(fd_);
        fd_ = -1;
        data_ = mapping;
        file_backed_ = true;
        return;
    }
#else
    if (request_file) {
        throw std::runtime_error(
            "File-backed raw scratch requires a POSIX mmap platform");
    }
#endif

    auto* allocation = static_cast<std::byte*>(std::malloc(bytes_));
    if (!allocation) throw std::bad_alloc();
    heap_.reset(allocation);
    data_ = allocation;
}

RawScratchBuffer::~RawScratchBuffer() {
    if (bytes_ == 0) return;
#if defined(__unix__) || defined(__APPLE__)
    if (file_backed_ && data_) {
        (void)::munmap(data_, bytes_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        (void)::close(fd_);
        fd_ = -1;
    }
#endif
    heap_.reset();
    data_ = nullptr;
}

} // namespace cosmo_nbody::runtime
