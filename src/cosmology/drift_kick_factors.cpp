#include "cosmo_nbody/cosmology/drift_kick_factors.hpp"

#include "cosmo_nbody/math/roundoff_resolved_gauss_legendre_16.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace cosmo_nbody {
namespace cosmology {

namespace {

void validate_interval(core::Real a1, core::Real a2) {
    if (!std::isfinite(a1) || !std::isfinite(a2) || a1 <= 0.0 || a2 <= 0.0) {
        throw std::invalid_argument(
            "Drift/kick scale factors must be finite and positive");
    }
}

void require_finite_factor(core::Real value, const char* label) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            std::string("Drift/kick quadrature produced a non-finite ") + label);
    }
}

core::Real stable_positive_log_ratio(
    core::Real upper,
    core::Real lower) {
    const core::Real relative = (upper - lower) / lower;
    const core::Real value = std::isfinite(relative)
        ? std::log1p(relative)
        : std::log(upper) - std::log(lower);
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "Drift/kick logarithmic interval is non-finite");
    }
    if (value <= 0.0) {
        throw std::underflow_error(
            "Drift/kick logarithmic interval was erased by rounding");
    }
    return value;
}

core::Real scale_factor_from_relative_log(
    core::Real base,
    core::Real relative_log) {
    const core::Real ratio = std::exp(relative_log);
    core::Real value = base * ratio;
    if (!std::isfinite(ratio) || !std::isfinite(value) || value <= 0.0) {
        value = std::exp(std::log(base) + relative_log);
    }
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::overflow_error(
            "Drift/kick logarithmic quadrature scale factor is not representable");
    }
    return value;
}

template <typename ScaleFactorIntegrand>
core::Real roundoff_resolved_relative_log_integral(
    core::Real a1,
    core::Real a2,
    const ScaleFactorIntegrand& integrand,
    const char* label) {
    const bool forward = a2 > a1;
    const core::Real base = forward ? a1 : a2;
    const core::Real upper = forward ? a2 : a1;
    const core::Real logarithmic_width = stable_positive_log_ratio(upper, base);

    // Use u=ln(a/base) so the lower endpoint is exact zero and nearby epochs
    // avoid cancellation from separately rounded absolute logarithms.
    const core::Real integral = math::roundoff_resolved_gauss_legendre_16(
        [&](core::Real relative_log) {
            return integrand(
                scale_factor_from_relative_log(base, relative_log));
        },
        core::Real{0.0},
        logarithmic_width,
        label);
    return forward ? integral : -integral;
}

} // namespace

DriftKickIntegrator::DriftKickIntegrator(const CosmologyModel& cosmo_model)
    : cosmo_model_(cosmo_model) {
}

core::Real DriftKickIntegrator::drift_factor(core::Real a1, core::Real a2) {
    validate_interval(a1, a2);
    if (a1 == a2) return 0.0;

    const core::Real result = compute_drift(a1, a2);
    require_finite_factor(result, "drift factor");
    return result;
}

core::Real DriftKickIntegrator::kick_factor(core::Real a1, core::Real a2) {
    validate_interval(a1, a2);
    if (a1 == a2) return 0.0;

    const core::Real result = compute_kick(a1, a2);
    require_finite_factor(result, "kick factor");
    return result;
}

core::Real DriftKickIntegrator::compute_drift(core::Real a1, core::Real a2) const {
    return roundoff_resolved_relative_log_integral(
        a1,
        a2,
        [&](core::Real a) {
            const core::Real a_h = a * cosmo_model_.H(a);
            const core::Real value = 1.0 / (a * a_h);
            if (!std::isfinite(a_h) || a_h <= 0.0
                || !std::isfinite(value)) {
                throw std::overflow_error(
                    "Drift integrand is not finite and representable");
            }
            return value;
        },
        "drift factor");
}

core::Real DriftKickIntegrator::compute_kick(core::Real a1, core::Real a2) const {
    return roundoff_resolved_relative_log_integral(
        a1,
        a2,
        [&](core::Real a) {
            const core::Real a_h = a * cosmo_model_.H(a);
            const core::Real value = 1.0 / a_h;
            if (!std::isfinite(a_h) || a_h <= 0.0
                || !std::isfinite(value)) {
                throw std::overflow_error(
                    "Kick integrand is not finite and representable");
            }
            return value;
        },
        "kick factor");
}

} // namespace cosmology
} // namespace cosmo_nbody
