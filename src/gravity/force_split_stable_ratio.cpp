#include "cosmo_nbody/gravity/force_split.hpp"
#include "force_split_gaussian.hpp"

#include "cosmo_nbody/math/floating_environment.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace cosmo_nbody::gravity {

namespace {

constexpr core::Real ASYMPTOTIC_GAUSSIAN_Q = 64.0;
using WorkingReal = std::conditional_t<
    (std::numeric_limits<long double>::digits
        > std::numeric_limits<core::Real>::digits),
    long double,
    core::Real>;

// Keep pi in the actual working format. Some library implementations
// materialize std::numbers::pi_v<long double> from a binary64 constant.
constexpr WorkingReal WORKING_PI = static_cast<WorkingReal>(
    3.141592653589793238462643383279502884L);

struct TwoSum {
    WorkingReal value{0};
    WorkingReal error{0};
};

// Error-free transformation of one floating addition under the strict
// round-to-nearest environment required by the caller. This recovers the
// rounding residue of a+b without inventing an empirical cancellation band.
TwoSum two_sum(WorkingReal a, WorkingReal b) noexcept {
    const WorkingReal value = a + b;
    const WorkingReal b_virtual = value - a;
    const WorkingReal error =
        (a - (value - b_virtual)) + (b - b_virtual);
    return {value, error};
}

WorkingReal gaussian_long_complement_log(
    core::Real radius,
    core::Real split_scale) {
    const WorkingReal q =
        static_cast<WorkingReal>(radius)
        / static_cast<WorkingReal>(split_scale);
    if (!std::isfinite(q)
        || q >= static_cast<WorkingReal>(ASYMPTOTIC_GAUSSIAN_Q)) {
        return WorkingReal{0.0};
    }

    if (q < WorkingReal{1.0}) {
        const WorkingReal log_q =
            std::log(static_cast<WorkingReal>(radius))
            - std::log(static_cast<WorkingReal>(split_scale));
        const WorkingReal coefficient =
            detail::gaussian_long_complement_over_q3(q);
        if (!std::isfinite(log_q)
            || !std::isfinite(coefficient)
            || !(coefficient > WorkingReal{0.0})) {
            throw std::overflow_error(
                "TreePM Gaussian long-complement logarithm is invalid");
        }
        return WorkingReal{3.0} * log_q + std::log(coefficient);
    }

    const WorkingReal gaussian_term =
        (q / std::sqrt(WORKING_PI))
        * std::exp(-WorkingReal{0.25} * q * q);
    const WorkingReal short_multiplier =
        std::erfc(WorkingReal{0.5} * q) + gaussian_term;
    if (!std::isfinite(short_multiplier)
        || short_multiplier < WorkingReal{0.0}
        || short_multiplier >= WorkingReal{1.0}) {
        throw std::runtime_error(
            "TreePM Gaussian short multiplier is invalid in ratio evaluation");
    }
    const WorkingReal log_complement = std::log1p(-short_multiplier);
    if (!std::isfinite(log_complement)
        || log_complement > WorkingReal{0.0}) {
        throw std::runtime_error(
            "TreePM Gaussian long-complement logarithm is invalid");
    }
    return log_complement;
}

WorkingReal plummer_to_newtonian_log_factor(
    core::Real radius,
    core::Real eps) {
    const WorkingReal r = static_cast<WorkingReal>(radius);
    const WorkingReal e = static_cast<WorkingReal>(eps);
    const WorkingReal ratio = e / r;

    // log[(r^2+eps^2)^(3/2)/r^3]
    //   = 3/2 log[1+(eps/r)^2].
    const WorkingReal sqrt_max =
        std::sqrt(std::numeric_limits<WorkingReal>::max());
    if (std::isfinite(ratio) && ratio <= sqrt_max) {
        const WorkingReal value =
            WorkingReal{1.5} * std::log1p(ratio * ratio);
        if (!std::isfinite(value) || value < WorkingReal{0.0}) {
            throw std::overflow_error(
                "TreePM Plummer/Newtonian logarithmic factor is invalid");
        }
        return value;
    }

    const WorkingReal inverse_ratio = r / e;
    const WorkingReal log_ratio = std::log(e) - std::log(r);
    const WorkingReal value =
        WorkingReal{3.0} * log_ratio
        + WorkingReal{1.5}
            * std::log1p(inverse_ratio * inverse_ratio);
    if (!std::isfinite(value) || value < WorkingReal{0.0}) {
        throw std::overflow_error(
            "TreePM Plummer/Newtonian logarithmic factor is invalid");
    }
    return value;
}

core::Real representable_signed_fraction(
    long double value,
    std::string_view role) {
    const long double maximum = static_cast<long double>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || std::abs(value) > maximum) {
        throw std::overflow_error(
            std::string(role) + " is not representable");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result)) {
        throw std::overflow_error(
            std::string(role) + " rounded outside core::Real");
    }
    if (value != 0.0L && result == 0.0) {
        throw std::underflow_error(
            std::string(role)
            + " is non-zero but underflows core::Real before dimensional scaling");
    }
    return result;
}

} // namespace

core::Real ForceSplitKernel::short_range_force_fraction_of_plummer(
    core::Real r,
    core::Real eps) const {
    if (!std::isfinite(r) || !(r > 0.0)
        || !std::isfinite(eps) || !(eps > 0.0)) {
        throw std::invalid_argument(
            "TreePM dimensionless force ratio requires finite positive radius and epsilon");
    }
    math::require_strict_floating_environment(
        "TreePM dimensionless force ratio");

    // f_long/f_Plummer = (1-M_short(q))
    //   * (1+(eps/r)^2)^(3/2).
    // Work with logarithms. two_sum retains the exact rounding residue of the
    // addition of the already-evaluated logarithms, so a near-zero physical
    // correction is not classified by an empirical N*epsilon threshold.
    const WorkingReal log_long_complement =
        gaussian_long_complement_log(r, split_scale_);
    const WorkingReal log_softening_factor =
        plummer_to_newtonian_log_factor(r, eps);
    const TwoSum log_ratio = two_sum(
        log_long_complement, log_softening_factor);
    if (!std::isfinite(log_ratio.value)
        || !std::isfinite(log_ratio.error)) {
        throw std::overflow_error(
            "TreePM dimensionless long/plummer logarithm is invalid");
    }

    // expm1(s+e) = expm1(s) + exp(s)*expm1(e). Evaluating the
    // compensation separately preserves a two-sum residue that would be lost
    // by materializing s+e again; no accuracy gate is introduced here.
    const WorkingReal primary = std::expm1(log_ratio.value);
    const WorkingReal correction =
        std::exp(log_ratio.value) * std::expm1(log_ratio.error);
    if (!std::isfinite(primary) || !std::isfinite(correction)) {
        throw std::overflow_error(
            "TreePM dimensionless force ratio exponential is invalid");
    }
    const TwoSum expm1_ratio = two_sum(primary, correction);
    const WorkingReal fraction =
        -(expm1_ratio.value + expm1_ratio.error);
    return representable_signed_fraction(
        static_cast<long double>(fraction),
        "TreePM dimensionless short-force fraction");
}

} // namespace cosmo_nbody::gravity
