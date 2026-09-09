#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <stdexcept>

namespace cosmo_nbody::mesh {

// Supported PM discretizations:
// - Pure PM: discrete Poisson Laplacian + centered real-space gradient; CIC
//   deconvolution is an explicit coordinate.
// - TreePM long range: continuum -k^2 Green function, Gaussian split, spectral
//   gradient, and W_CIC^-2 compensation for deposition plus interpolation.
// Named constructors prevent unsupported hybrid combinations.
enum class PMForceMethodKind {
    PurePMDiscrete,
    TreePMSpectralLongRange,
};

class PMForceMethod {
public:
    static PMForceMethod pure_pm(bool deconvolve_cic) noexcept {
        return PMForceMethod(
            PMForceMethodKind::PurePMDiscrete,
            core::Real{0.0},
            deconvolve_cic);
    }

    static PMForceMethod treepm_long_range(core::Real split_scale) {
        if (!std::isfinite(split_scale) || split_scale <= 0.0) {
            throw std::invalid_argument(
                "TreePM long-range PM method requires a finite positive split scale");
        }
        return PMForceMethod(
            PMForceMethodKind::TreePMSpectralLongRange,
            split_scale,
            true);
    }

    [[nodiscard]] PMForceMethodKind kind() const noexcept { return kind_; }
    [[nodiscard]] bool is_treepm_long_range() const noexcept {
        return kind_ == PMForceMethodKind::TreePMSpectralLongRange;
    }
    [[nodiscard]] bool uses_spectral_gradient() const noexcept {
        return is_treepm_long_range();
    }
    [[nodiscard]] bool deconvolves_cic() const noexcept {
        return deconvolve_cic_;
    }
    [[nodiscard]] core::Real split_scale() const noexcept {
        return split_scale_;
    }

private:
    constexpr PMForceMethod(
        PMForceMethodKind kind,
        core::Real split_scale,
        bool deconvolve_cic) noexcept
        : kind_(kind),
          split_scale_(split_scale),
          deconvolve_cic_(deconvolve_cic) {}

    PMForceMethodKind kind_{PMForceMethodKind::PurePMDiscrete};
    core::Real split_scale_{0.0};
    bool deconvolve_cic_{false};
};

} // namespace cosmo_nbody::mesh
