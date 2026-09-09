#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <stdexcept>

namespace cosmo_nbody::analysis {

// Minimal periodic geometry for analysis, independent of simulation settings.
class PeriodicDomain {
public:
    explicit PeriodicDomain(core::Real box_size) : box_size_(box_size) {
        if (!std::isfinite(box_size_) || box_size_ <= 0.0) {
            throw std::invalid_argument(
                "Periodic analysis box size must be finite and positive");
        }
    }

    core::Real box_size() const noexcept { return box_size_; }

private:
    core::Real box_size_;
};

} // namespace cosmo_nbody::analysis
