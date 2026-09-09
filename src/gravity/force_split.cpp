#include "cosmo_nbody/gravity/force_split.hpp"
#include "force_split_gaussian.hpp"

#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/math/exact_signed_sum.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>

namespace cosmo_nbody {
namespace gravity {

namespace {

// At q=64 the Gaussian tail is far below the smallest representable double.
// Handling the asymptotic branch explicitly avoids forming q^2 when r/r_s is
// finite-but-enormous or overflows even though the physical limit is regular.
constexpr core::Real ASYMPTOTIC_GAUSSIAN_Q = 64.0;

core::Real nonnegative_real(long double value, const char* label) {
    const long double maximum = static_cast<long double>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || value < 0.0L || value > maximum) {
        throw std::overflow_error(label);
    }
    // A positive long-double value may legitimately underflow to zero when the
    // repository scalar is double. Zero is the correct representable limit.
    return static_cast<core::Real>(value);
}

core::Real inverse_cube(core::Real radius) {
    if (std::isnan(radius) || radius <= 0.0) {
        throw std::invalid_argument(
            "TreePM inverse-cube radius must be positive and not NaN");
    }
    if (std::isinf(radius)) return 0.0;
    const core::Real inverse = 1.0 / radius;
    const core::Real value = inverse * inverse * inverse;
    if (!std::isfinite(value) || value < 0.0) {
        throw std::overflow_error(
            "TreePM Newtonian inverse-cube factor is not representable");
    }
    return value;
}

core::Real inverse_hypot(core::Real lhs, core::Real rhs) {
    const long double lhs_wide = std::abs(static_cast<long double>(lhs));
    const long double rhs_wide = std::abs(static_cast<long double>(rhs));
    const long double scale = std::max(lhs_wide, rhs_wide);
    if (!std::isfinite(scale) || scale <= 0.0L) {
        throw std::overflow_error(
            "TreePM softened inverse radius scale is invalid");
    }
    const long double normalized_lhs = lhs_wide / scale;
    const long double normalized_rhs = rhs_wide / scale;
    const long double normalized_norm = std::sqrt(
        normalized_lhs * normalized_lhs
        + normalized_rhs * normalized_rhs);
    const long double value = (1.0L / scale) / normalized_norm;
    return nonnegative_real(
        value,
        "TreePM softened inverse radius is not representable");
}

math::ScaledPositiveProduct scaled_plummer_force_factor(
    const core::Vec3& displacement,
    core::Real eps) {
    const core::Real scale = std::max({
        std::abs(displacement.x),
        std::abs(displacement.y),
        std::abs(displacement.z),
        eps});
    if (!std::isfinite(scale) || !(scale > 0.0)) {
        throw std::overflow_error(
            "TreePM scaled Plummer force scale is invalid");
    }

    const core::Real nx = displacement.x / scale;
    const core::Real ny = displacement.y / scale;
    const core::Real nz = displacement.z / scale;
    const core::Real neps = eps / scale;
    const core::Real normalized_squared =
        nx * nx + ny * ny + nz * nz + neps * neps;
    const core::Real normalized_radius = std::sqrt(normalized_squared);
    if (!std::isfinite(normalized_radius) || !(normalized_radius > 0.0)) {
        throw std::overflow_error(
            "TreePM scaled Plummer normalized radius is invalid");
    }

    const std::array<core::Real, 1> numerators{core::Real{1.0}};
    const std::array<core::Real, 6> denominators{
        scale, scale, scale,
        normalized_radius, normalized_radius, normalized_radius};
    return math::ScaledPositiveProduct::from_quotient(
        numerators,
        denominators,
        "TreePM scaled Plummer force factor");
}

} // namespace

core::Real ForceSplitKernel::near_cutoff_shifted_potential(
    core::Real radius,
    core::Real eps) const {
    const core::Real cutoff = cutoff_radius_;
    if (!(radius < cutoff)) return 0.0;
    const core::Real radial_gap = cutoff - radius;
    const core::Real relative_gap = radial_gap / cutoff;
    // Simpson's absolute remainder is O(h^5).  At a force-zero cutoff the
    // shifted potential is O(h^2), so its relative truncation scale is O(h^3),
    // while direct subtraction has an O(epsilon/h^2) cancellation scale.
    // Balancing those leading scales gives h=epsilon^(1/5) and an
    // O(epsilon^(3/5)) transition scale. This switch follows the local
    // truncation/cancellation balance itself; no repository regression
    // tolerance is used to choose the transition.
    const core::Real local_limit = std::pow(
        std::numeric_limits<core::Real>::epsilon(),
        core::Real{0.2});
    if (!std::isfinite(relative_gap)
        || relative_gap < 0.0
        || relative_gap > local_limit) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }

    // Since dA/dr = -r f_short(r),
    //   A(r)-A(r_cut) = integral_r^r_cut s f_short(s) ds.
    // A two-endpoint trapezoid loses O(sqrt(epsilon)) relative accuracy when
    // the cutoff force jump is nearly zero. Simpson's rule remains fourth order
    // for ordinary points and third order relative to the quadratic potential
    // gap at that degenerate boundary. Form each final node contribution with
    // the scale-safe product path so no standalone force factor or r^-3
    // needs to be representable, then sum the signed binary64 contributions
    // exactly.
    const core::Real midpoint = std::midpoint(radius, cutoff);
    const auto contribution = [&] (
        core::Real node,
        core::Real simpson_weight,
        const char* role) {
        const core::Real signed_fraction =
            short_range_force_fraction_of_plummer(node, eps);
        if (signed_fraction == 0.0) return core::Real{0.0};

        const core::Real scale = std::max(node, eps);
        if (!std::isfinite(scale) || !(scale > 0.0)) {
            throw std::overflow_error(
                "TreePM near-cutoff Simpson scale is invalid");
        }
        const core::Real normalized_radius = std::hypot(
            node / scale, eps / scale);
        if (!std::isfinite(normalized_radius)
            || !(normalized_radius > 0.0)) {
            throw std::overflow_error(
                "TreePM near-cutoff Simpson normalized radius is invalid");
        }

        const std::array<core::Real, 4> numerators{
            radial_gap,
            simpson_weight,
            node,
            std::abs(signed_fraction)};
        const std::array<core::Real, 7> denominators{
            core::Real{6.0},
            scale, scale, scale,
            normalized_radius, normalized_radius, normalized_radius};
        const core::Real magnitude = math::scaled_positive_product_quotient(
            numerators,
            denominators,
            role);
        return std::signbit(signed_fraction) ? -magnitude : magnitude;
    };

    math::ExactSignedDoubleSum quadrature;
    quadrature.add(contribution(
        radius, core::Real{1.0},
        "TreePM near-cutoff Simpson lower contribution"));
    quadrature.add(contribution(
        midpoint, core::Real{4.0},
        "TreePM near-cutoff Simpson midpoint contribution"));
    quadrature.add(contribution(
        cutoff, core::Real{1.0},
        "TreePM near-cutoff Simpson upper contribution"));
    return quadrature.value();
}

ForceSplitKernel::ForceSplitKernel(
    core::Real split_scale,
    core::Real cutoff_multiplier)
    : split_scale_(split_scale),
      cutoff_radius_(cutoff_multiplier * split_scale) {
    if (!std::isfinite(split_scale_) || split_scale_ <= 0.0) {
        throw std::invalid_argument(
            "TreePM split scale r_s must be finite and positive");
    }
    if (!std::isfinite(cutoff_multiplier) || cutoff_multiplier <= 0.0
        || !std::isfinite(cutoff_radius_)
        || cutoff_radius_ <= 0.0) {
        throw std::invalid_argument(
            "TreePM cutoff multiplier/radius must be finite and positive");
    }
}

core::Real ForceSplitKernel::short_range_potential_multiplier(
    core::Real r) const {
    if (std::isnan(r) || r < 0.0) {
        throw std::invalid_argument(
            "short-range split radius must be non-negative and not NaN");
    }
    if (std::isinf(r)) return 0.0;
    const core::Real q = r / split_scale_;
    if (!std::isfinite(q)) return 0.0;
    return std::erfc(0.5 * q);
}

core::Real ForceSplitKernel::short_range_force_multiplier(
    core::Real r) const {
    if (std::isnan(r) || r < 0.0) {
        throw std::invalid_argument(
            "short-range split radius must be non-negative and not NaN");
    }
    if (std::isinf(r)) return 0.0;
    const core::Real q = r / split_scale_;
    if (!std::isfinite(q) || q >= ASYMPTOTIC_GAUSSIAN_Q) {
        return 0.0;
    }
    const core::Real value =
        std::erfc(0.5 * q)
        + (q / std::sqrt(std::numbers::pi))
            * std::exp(-0.25 * q * q);
    constexpr core::Real roundoff_allowance =
        64.0 * std::numeric_limits<core::Real>::epsilon();
    if (!std::isfinite(value) || value < 0.0
        || value > 1.0 + roundoff_allowance) {
        throw std::runtime_error(
            "TreePM short-range force multiplier is invalid");
    }
    return std::clamp(value, core::Real{0.0}, core::Real{1.0});
}

core::Real ForceSplitKernel::short_range_potential_correction(
    core::Real r,
    core::Real eps) const {
    if (std::isnan(r) || r < 0.0) {
        throw std::invalid_argument(
            "TreePM short-potential radius must be non-negative and not NaN");
    }
    if (!std::isfinite(eps) || eps <= 0.0) {
        throw std::invalid_argument(
            "TreePM short-potential epsilon must be finite and positive");
    }
    if (r >= cutoff_radius_) return 0.0;

    const core::Real local_value =
        near_cutoff_shifted_potential(r, eps);
    if (std::isfinite(local_value)) return local_value;

    const auto raw_correction = [&](core::Real radius) {
        const core::Real softened = inverse_hypot(radius, eps);
        const core::Real q = radius / split_scale_;
        core::Real long_potential = 0.0;
        if (q < 1e-4) {
            const core::Real q2 = q * q;
            const core::Real series = core::Real{1.0}
                - q2 / core::Real{12.0}
                + q2 * q2 / core::Real{160.0}
                - q2 * q2 * q2 / core::Real{2688.0};
            long_potential =
                (series / std::sqrt(std::numbers::pi)) / split_scale_;
        } else {
            long_potential = std::erf(0.5 * q) / radius;
        }
        const core::Real value = softened - long_potential;
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                "TreePM short-potential correction is non-finite");
        }
        return value;
    };

    const core::Real value =
        raw_correction(r) - raw_correction(cutoff_radius_);
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "TreePM cutoff-shifted short potential is non-finite");
    }
    return value;
}

core::Real ForceSplitKernel::short_range_potential_correction(
    const core::Vec3& displacement,
    core::Real eps) const {
    if (!std::isfinite(displacement.x)
        || !std::isfinite(displacement.y)
        || !std::isfinite(displacement.z)) {
        throw std::invalid_argument(
            "TreePM vector short-potential displacement must be finite");
    }
    if (!std::isfinite(eps) || !(eps > 0.0)) {
        throw std::invalid_argument(
            "TreePM vector short-potential epsilon must be finite and positive");
    }

    if (!core::scale_safe_norm3_less(
            displacement.x,
            displacement.y,
            displacement.z,
            cutoff_radius_)) {
        return 0.0;
    }

    const core::Real rounded_radius = core::scale_safe_norm3(
        displacement.x, displacement.y, displacement.z);
    if (!std::isfinite(rounded_radius) || rounded_radius < 0.0) {
        throw std::overflow_error(
            "TreePM vector short-potential radius is invalid");
    }
    if (rounded_radius < cutoff_radius_) {
        return short_range_potential_correction(rounded_radius, eps);
    }

    // Rare exact-inside / rounded-on-cutoff fallback. Since
    // dA/d(r^2) = -0.5 * f_tree(r),
    //   A(r)-A(r_cut) = 0.5 * integral[f_tree(sqrt(u)), du]
    // over u in [r^2,r_cut^2]. The interval exists only because the rounded
    // binary64 norm lost a sub-ulp radial gap. Preserve the exact first-order
    // scale as a dyadic mantissa/exponent so neither the squared gap nor force
    // factors need to fit standalone doubles.
    const core::detail::ExactBinary64PositiveDyadic squared_gap =
        core::detail::exact_binary64_norm3_squared_gap_below_radius(
            displacement.x,
            displacement.y,
            displacement.z,
            cutoff_radius_);
    // First-order in the exact squared-radius gap is the only reconstruction
    // available when hypot has already rounded onto r_cut. A zero or unresolved
    // cutoff force jump does not imply a zero potential correction: the leading
    // contribution can be quadratic and still be representable. Likewise, if
    // the O((Delta u)^2) scale is not smaller than the first-order term, this
    // implementation has no controlled reconstruction. Fail closed rather than
    // silently replacing a representable higher-order correction with zero.
    const core::Real signed_jump_fraction =
        short_range_force_fraction_of_plummer(cutoff_radius_, eps);
    const core::Real jump_fraction = std::abs(signed_jump_fraction);
    if (!(jump_fraction > 0.0)) {
        throw std::runtime_error(
            "TreePM rounded-on-cutoff potential requires higher-order reconstruction at a zero cutoff force jump");
    }
    const long double log2_relative_gap =
        std::log2(static_cast<long double>(squared_gap.mantissa))
        + static_cast<long double>(squared_gap.exponent)
        - 2.0L * std::log2(static_cast<long double>(cutoff_radius_));
    const long double log2_second_over_first =
        log2_relative_gap
        - std::log2(static_cast<long double>(jump_fraction));
    if (!std::isfinite(log2_second_over_first)) {
        throw std::runtime_error(
            "TreePM rounded-on-cutoff potential conditioning estimate is non-finite");
    }
    if (log2_second_over_first >= 0.0L) {
        throw std::runtime_error(
            "TreePM rounded-on-cutoff potential first-order reconstruction is not leading");
    }

    const math::ScaledPositiveProduct half_squared_gap =
        math::ScaledPositiveProduct::from_binary_scale(
            core::Real{0.5} * squared_gap.mantissa,
            squared_gap.exponent,
            "TreePM rounded-on-cutoff squared-radius gap");
    const core::Vec3 cutoff_displacement{cutoff_radius_, 0.0, 0.0};
    const math::ScaledPositiveProduct plummer_cutoff =
        scaled_plummer_force_factor(cutoff_displacement, eps);
    const core::Real magnitude = plummer_cutoff.multiplied_by_product_and_factor(
        half_squared_gap,
        jump_fraction,
        "TreePM rounded-on-cutoff short-potential correction");
    return std::signbit(signed_jump_fraction) ? -magnitude : magnitude;
}

core::Real ForceSplitKernel::long_range_newtonian_force_factor(
    core::Real r) const {
    if (std::isnan(r) || r < 0.0) {
        throw std::invalid_argument(
            "long-range split radius must be non-negative and not NaN");
    }
    if (std::isinf(r)) return 0.0;
    const core::Real q = r / split_scale_;

    // In the large-q limit the short-range multiplier is zero and the
    // complementary long-range coefficient is exactly Newtonian. Evaluate the
    // inverse cube without ever forming r^3, which may overflow even when the
    // representable result simply underflows to zero.
    if (!std::isfinite(q) || q >= ASYMPTOTIC_GAUSSIAN_Q) {
        return inverse_cube(r);
    }

    // Direct subtraction of the two O(q) terms loses significant digits before
    // q reaches the old fixed small-q switch. For q<1, evaluate the convergent
    // representation of (1-M_short)/q^3 and apply r_s^-3 separately. The helper
    // stops when the working representation no longer changes, rather than at a
    // repository-owned q or error threshold.
    if (q < 1.0) {
        const long double coefficient_wide =
            detail::gaussian_long_complement_over_q3(
                static_cast<long double>(q));
        const core::Real coefficient = nonnegative_real(
            coefficient_wide,
            "TreePM Gaussian long-force coefficient is invalid");
        const std::array<core::Real, 1> numerators{coefficient};
        const std::array<core::Real, 3> denominators{
            split_scale_, split_scale_, split_scale_};
        try {
            return math::scaled_positive_product_quotient(
                numerators,
                denominators,
                "TreePM small-r long-range force factor");
        } catch (const std::underflow_error&) {
            return 0.0;
        }
    }

    const core::Real one_minus_short =
        std::erf(0.5 * q)
        - (q / std::sqrt(std::numbers::pi))
            * std::exp(-0.25 * q * q);
    const core::Real value = one_minus_short * inverse_cube(r);
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(
            "TreePM long-range Newtonian force factor is invalid");
    }
    return value;
}

core::Vec3 ForceSplitKernel::scale_safe_short_range_acceleration(
    const core::Vec3& displacement,
    core::Real source_mass,
    core::Real eps) const {
    if (!std::isfinite(displacement.x)
        || !std::isfinite(displacement.y)
        || !std::isfinite(displacement.z)) {
        throw std::invalid_argument(
            "TreePM scaled short force requires finite displacement components");
    }
    if (!std::isfinite(source_mass) || !(source_mass > 0.0)
        || !std::isfinite(eps) || !(eps > 0.0)) {
        throw std::invalid_argument(
            "TreePM scaled short force requires finite positive mass and softening");
    }
    if (displacement.x == 0.0
        && displacement.y == 0.0
        && displacement.z == 0.0) {
        return {};
    }

    const core::Real radius = core::scale_safe_norm3(
        displacement.x, displacement.y, displacement.z);
    if (!std::isfinite(radius) || !(radius > 0.0)) {
        throw std::overflow_error(
            "TreePM scaled short-force radius is not representable");
    }

    const core::Real signed_fraction =
        short_range_force_fraction_of_plummer(radius, eps);
    if (signed_fraction == 0.0) return {};

    // Normalize the softened Plummer denominator before cubing it.  With
    // s=max(|dx|,|dy|,|dz|,eps), T=sum[(component/s)^2]+(eps/s)^2 lies in [1,4]
    // and
    //   (r^2+eps^2)^(3/2) = s^3 * T^(3/2).
    // Feed every positive factor to the scale-safe product evaluator at once so
    // no standalone r^-3 coefficient or intermediate G*m*dx product needs to be
    // representable.
    const core::Real scale = std::max({
        std::abs(displacement.x),
        std::abs(displacement.y),
        std::abs(displacement.z),
        eps});
    if (!std::isfinite(scale) || !(scale > 0.0)) {
        throw std::overflow_error(
            "TreePM scaled short-force Plummer scale is invalid");
    }
    const core::Real nx = displacement.x / scale;
    const core::Real ny = displacement.y / scale;
    const core::Real nz = displacement.z / scale;
    const core::Real neps = eps / scale;
    const core::Real normalized_squared =
        nx * nx + ny * ny + nz * nz + neps * neps;
    const core::Real normalized_radius = std::sqrt(normalized_squared);
    if (!std::isfinite(normalized_radius) || !(normalized_radius > 0.0)) {
        throw std::overflow_error(
            "TreePM scaled short-force normalized Plummer radius is invalid");
    }
    const std::array<core::Real, 6> denominators{
        scale, scale, scale,
        normalized_radius, normalized_radius, normalized_radius};
    const core::Real fraction_magnitude = std::abs(signed_fraction);

    const auto component = [&](core::Real delta, const char* role) {
        if (delta == 0.0) return core::Real{0.0};
        const std::array<core::Real, 4> numerators{
            cosmology::units::G,
            source_mass,
            std::abs(delta),
            fraction_magnitude};
        core::Real magnitude = 0.0;
        try {
            magnitude = math::scaled_positive_product_quotient(
                numerators,
                denominators,
                role);
        } catch (const std::underflow_error&) {
            // short_range_force_fraction_of_plummer() already requires gradual
            // underflow. A residual that still cannot be materialized after all
            // scale factors are combined rounds physically to zero in core::Real.
            return core::Real{0.0};
        }
        const bool negative =
            std::signbit(delta) != std::signbit(signed_fraction);
        return negative ? -magnitude : magnitude;
    };

    core::Vec3 result{
        component(displacement.x, "TreePM scaled short-force x"),
        component(displacement.y, "TreePM scaled short-force y"),
        component(displacement.z, "TreePM scaled short-force z")};
    if (!std::isfinite(result.x)
        || !std::isfinite(result.y)
        || !std::isfinite(result.z)) {
        throw std::overflow_error(
            "TreePM scaled short-force acceleration is non-finite");
    }
    return result;
}

core::Real ForceSplitKernel::long_range_filter(
    core::Real k_squared) const {
    if (!std::isfinite(k_squared) || k_squared < 0.0) {
        throw std::invalid_argument(
            "TreePM long-range filter requires finite k^2 >= 0");
    }
    const long double exponent =
        -static_cast<long double>(k_squared)
        * static_cast<long double>(split_scale_)
        * static_cast<long double>(split_scale_);
    if (!std::isfinite(exponent)) {
        if (exponent < 0.0L) return 0.0;
        throw std::overflow_error(
            "TreePM long-range filter exponent is invalid");
    }
    if (exponent < core::real_round_to_zero_log_threshold()) return 0.0;
    const core::Real value = static_cast<core::Real>(std::exp(exponent));
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::runtime_error(
            "TreePM long-range filter produced an invalid value");
    }
    return value;
}

bool ForceSplitKernel::inside_cutoff_radius(core::Real radius) const {
    if (std::isnan(radius) || radius < 0.0) {
        throw std::invalid_argument(
            "TreePM cutoff radius check requires r >= 0 and not NaN");
    }
    return radius < cutoff_radius_;
}

} // namespace gravity
} // namespace cosmo_nbody
