#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace cosmo_nbody {
namespace runtime {

class RealScratchBuffer {
public:
    RealScratchBuffer(
        std::size_t elements,
        const config::MemoryPolicyParams& policy,
        const std::string& label);
    ~RealScratchBuffer();

    RealScratchBuffer(const RealScratchBuffer&) = delete;
    RealScratchBuffer& operator=(const RealScratchBuffer&) = delete;
    RealScratchBuffer(RealScratchBuffer&&) = delete;
    RealScratchBuffer& operator=(RealScratchBuffer&&) = delete;

    std::size_t size() const noexcept { return size_; }
    core::Real* data() noexcept { return data_; }
    const core::Real* data() const noexcept { return data_; }
    core::Real& operator[](std::size_t index) { return data_[index]; }
    const core::Real& operator[](std::size_t index) const { return data_[index]; }
    bool file_backed() const noexcept { return file_backed_; }

private:
    std::size_t size_{0};
    std::size_t bytes_{0};
    core::Real* data_{nullptr};
    bool file_backed_{false};
    std::unique_ptr<core::Real[]> heap_;

#if defined(__unix__) || defined(__APPLE__)
    int fd_{-1};
#endif
};

} // namespace runtime
} // namespace cosmo_nbody
