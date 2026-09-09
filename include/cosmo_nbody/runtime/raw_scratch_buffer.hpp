#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace cosmo_nbody::runtime {

// Untyped exact storage used by scratch structures whose element width is part
// of their representation. Disk mode changes only page backing; it
// does not quantize, reorder, or transform stored bytes.
class RawScratchBuffer {
public:
    RawScratchBuffer(
        std::size_t bytes,
        config::ScratchMode mode,
        const std::string& directory,
        const std::string& label);
    ~RawScratchBuffer();

    RawScratchBuffer(const RawScratchBuffer&) = delete;
    RawScratchBuffer& operator=(const RawScratchBuffer&) = delete;
    RawScratchBuffer(RawScratchBuffer&&) = delete;
    RawScratchBuffer& operator=(RawScratchBuffer&&) = delete;

    std::size_t size_bytes() const noexcept { return bytes_; }
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    bool file_backed() const noexcept { return file_backed_; }

private:
    struct FreeDeleter {
        void operator()(std::byte* pointer) const noexcept;
    };

    std::size_t bytes_{0};
    void* data_{nullptr};
    bool file_backed_{false};
    std::string label_;
    std::unique_ptr<std::byte, FreeDeleter> heap_;

#if defined(__unix__) || defined(__APPLE__)
    int fd_{-1};
#endif
};

} // namespace cosmo_nbody::runtime
