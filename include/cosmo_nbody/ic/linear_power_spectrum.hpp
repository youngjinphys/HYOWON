#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/cubic_spline.hpp"

namespace cosmo_nbody {
namespace ic {

struct LinearPowerNormalizationDiagnostics {
    bool available{false};
    core::Real declared_sigma8_z0{0.0};
    core::Real represented_support_sigma8_z0{0.0};
    core::Real represented_support_relative_difference{0.0};
    core::Real support_k_min_h_Mpc{0.0};
    core::Real support_k_max_h_Mpc{0.0};
};

class LinearPowerSpectrum {
public:
    explicit LinearPowerSpectrum(const config::SimulationParameters& config);

    // P(k) at the configured start redshift, in (Mpc/h)^3. The exact zero
    // mode returns zero. Negative wavenumbers are invalid because k is a
    // magnitude; positive queries outside the exact tabulated support are
    // rejected rather than delegated to CubicSpline's generic
    // linear-extrapolation policy.
    core::Real evaluate(core::Real k) const;

    // Dimensionless power Delta^2(k) = k^3 P(k) / (2 pi^2), after growth
    // rescaling the input table to the configured start redshift. The same
    // non-negative magnitude-domain validation as evaluate() applies.
    core::Real dimensionless_power(core::Real k) const;

    // Diagnostic only. The represented-support value integrates the production
    // spline only over the exact tabulated k support. It therefore does not
    // bound interpolation error or the omitted low/high-k tails and must not be
    // promoted to a scientific PASS/FAIL criterion by itself.
    const LinearPowerNormalizationDiagnostics& normalization_diagnostics() const noexcept {
        return normalization_diagnostics_;
    }

private:
    math::CubicSpline log_spline_;
    core::Real k_min_{0.0};
    core::Real k_max_{0.0};
    LinearPowerNormalizationDiagnostics normalization_diagnostics_;
};

} // namespace ic
} // namespace cosmo_nbody
