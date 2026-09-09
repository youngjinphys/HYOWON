// Plummer softening for TreePM short range:
// phi=-Gm/sqrt(r^2+eps^2), f=Gmr/(r^2+eps^2)^(3/2).
// eps is comoving, has no compact support, and does not additionally soften PM.
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <stdexcept>

namespace cosmo_nbody {
namespace gravity {

class SofteningKernel {
public:
    explicit SofteningKernel(core::Real eps)
        : eps_(eps) {
        if (!std::isfinite(eps_) || eps_ <= 0.0) {
            throw std::invalid_argument(
                "Plummer softening epsilon must be finite and positive");
        }
    }

    core::Real epsilon() const noexcept { return eps_; }

    // 1/(r^2+eps^2)^(3/2). A rounded far-field zero must not later be rescaled
    // by a source mass that could make the joint expression representable.
    core::Real force_factor(core::Real r2) const {
        validate_squared_radius(r2);
        if (std::isinf(r2)) return 0.0;
        const core::Real softened_radius = std::hypot(std::sqrt(r2), eps_);
        const core::Real inverse = 1.0 / softened_radius;
        const core::Real value = inverse * inverse * inverse;
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                "Plummer softening force factor is not representable");
        }
        return value;
    }

    // 1/sqrt(r^2+eps^2); far-field limit is zero.
    core::Real potential_factor(core::Real r2) const {
        validate_squared_radius(r2);
        if (std::isinf(r2)) return 0.0;
        const core::Real value = 1.0 / std::hypot(std::sqrt(r2), eps_);
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::overflow_error(
                "Plummer softening potential factor is not representable");
        }
        return value;
    }

private:
    static void validate_squared_radius(core::Real r2) {
        if (std::isnan(r2) || r2 < 0.0) {
            throw std::invalid_argument(
                "Plummer softening requires a non-negative squared radius");
        }
    }

    core::Real eps_;
};

} // namespace gravity
} // namespace cosmo_nbody
