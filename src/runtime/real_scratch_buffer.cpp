#include "cosmo_nbody/runtime/real_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/scratch_file_reservation.hpp"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace cosmo_nbody {
namespace runtime {

namespace {

std::size_t checked_bytes(std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max()
            / sizeof(core::Real)) {
        throw std::overflow_error(
            "Scratch buffer byte size overflows size_t");
    }
    return elements * sizeof(core::Real);
}

} // namespace

RealScratchBuffer::RealScratchBuffer(
    std::size_t elements,
    const config::MemoryPolicyParams& policy,
    const std::string& label)
    : size_(elements),
      bytes_(checked_bytes(elements)) {
    if (size_ == 0) return;

    const bool request_file = config::uses_file_backed_scratch(
        policy.ic_scratch_mode);
#if defined(__unix__) || defined(__APPLE__)
    if (request_file) {
        const std::filesystem::path directory = prepare_scratch_directory(
            policy.scratch_directory, "IC");

        std::string safe_label = label.empty() ? "scratch" : label;
        for (char& c : safe_label) {
            const bool ok =
                (c >= 'a' && c <= 'z')
                || (c >= 'A' && c <= 'Z')
                || (c >= '0' && c <= '9')
                || c == '_' || c == '-';
            if (!ok) c = '_';
        }
        std::string pattern =
            (directory / ("cosmo_" + safe_label + "_XXXXXX")).string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');

        fd_ = ::mkstemp(writable.data());
        if (fd_ < 0) {
            throw std::runtime_error(
                "mkstemp failed for IC scratch: "
                + std::string(std::strerror(errno)));
        }

        // MAP_SHARED is deliberate for explicit Disk mode. Dirty pages remain
        // reclaimable to the backing file; MAP_PRIVATE would turn writes into
        // anonymous COW pages and contradict the operator's explicit request.
        (void)::unlink(writable.data());
        try {
            // The filesystem's actual reservation operation is the allocation
            // authority. Do not precede it with a guessed free-space reserve or
            // percentage gate that can reject valid quota/filesystem states.
            reserve_scratch_file_space(fd_, bytes_, "IC");
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
            const std::string message = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "mmap failed for IC scratch: " + message);
        }
        try {
            advise_mutable_scratch_mapping(mapping, bytes_, safe_label);
        } catch (...) {
            (void)::munmap(mapping, bytes_);
            (void)::close(fd_);
            fd_ = -1;
            throw;
        }
        data_ = static_cast<core::Real*>(mapping);
        file_backed_ = true;
        return;
    }
#else
    if (request_file) {
        throw std::runtime_error(
            "memory.ic_scratch_mode=disk requires a POSIX mmap platform");
    }
#endif

    // Anonymous allocation is the RAM-first path. The allocation itself is the
    // runtime authority; the operating system remains free to apply its normal
    // compression/swap policy to cold pages.
    heap_.reset(new core::Real[size_]);
    data_ = heap_.get();
}

RealScratchBuffer::~RealScratchBuffer() {
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
}

} // namespace runtime
} // namespace cosmo_nbody
