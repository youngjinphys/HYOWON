#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/floating_environment.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::math {

// Binary mantissa/exponent representation of a strictly positive product ratio.
// The represented value need not itself fit core::Real. This allows a large
// invariant normalization to be prepared once and combined later with a small
// per-sample factor whose final product is representable.
class ScaledPositiveProduct {
public:
    static ScaledPositiveProduct from_quotient(
        std::span<const core::Real> numerators,
        std::span<const core::Real> denominators,
        std::string_view role) {
        require_round_to_nearest(role);
        if (numerators.empty()) {
            throw std::invalid_argument(
                std::string(role) + " requires at least one numerator factor");
        }

        ScaledPositiveProduct product;
        for (const core::Real factor : numerators) {
            product.consume(factor, false, role);
        }
        for (const core::Real factor : denominators) {
            product.consume(factor, true, role);
        }
        return product;
    }

    // Construct directly from a positive binary scale mantissa*2^exponent.
    // Keeping the exponent separate lets exact dyadic geometry feed later
    // physical products without first underflowing or overflowing a double.
    static ScaledPositiveProduct from_binary_scale(
        core::Real mantissa,
        long long exponent,
        std::string_view role) {
        if (!std::isfinite(mantissa) || !(mantissa > 0.0)) {
            throw std::invalid_argument(
                std::string(role)
                + " binary-scale mantissa must be finite and positive");
        }
        ScaledPositiveProduct product;
        product.mantissa_ = static_cast<long double>(mantissa);
        product.exponent_ = exponent;
        int normalization_exponent = 0;
        product.mantissa_ = std::frexp(
            product.mantissa_, &normalization_exponent);
        checked_add_exponent(
            product.exponent_, normalization_exponent, role);
        return product;
    }

    core::Real value(std::string_view role) const {
        return materialize(mantissa_, exponent_, role);
    }

    core::Real multiplied_by(
        core::Real factor,
        std::string_view role) const {
        require_round_to_nearest(role);
        long double mantissa = mantissa_;
        long long exponent = exponent_;
        (void)consume_into(mantissa, exponent, factor, false, role);
        return materialize(mantissa, exponent, role);
    }

    // Evaluate sqrt(this * numerator_factor / denominator_factor) without
    // materializing the squared quantity first. This is an exceptional-range
    // primitive: callers can retain their ordinary binary64 path and use this
    // only when that intermediate underflows or overflows even though its square
    // root remains representable.
    core::Real square_root_multiplied_by_quotient(
        core::Real numerator_factor,
        core::Real denominator_factor,
        std::string_view role) const {
        require_round_to_nearest(role);
        long double mantissa = mantissa_;
        long long exponent = exponent_;
        (void)consume_into(
            mantissa, exponent, numerator_factor, false, role);
        (void)consume_into(
            mantissa, exponent, denominator_factor, true, role);

        long long root_exponent = exponent / 2;
        long long remainder_exponent = exponent % 2;
        if (remainder_exponent < 0) {
            remainder_exponent += 2;
            --root_exponent;
        }
        const long double root_mantissa = std::sqrt(std::scalbn(
            mantissa, static_cast<int>(remainder_exponent)));
        if (!std::isfinite(root_mantissa) || !(root_mantissa > 0.0L)) {
            throw std::overflow_error(
                std::string(role) + " square-root mantissa is invalid");
        }
        return materialize(root_mantissa, root_exponent, role);
    }

    // Evaluate this * rhs * factor while all dimensional binary exponents stay
    // separate from the bounded working-format mantissas. This is needed when
    // an exact dyadic geometry factor and a physical coefficient are each
    // outside the standalone binary64 range even though their final product is
    // representable. Within the common materializer's supported exponent
    // range, a mathematically non-zero final value below binary64's gradual-
    // underflow range has the representable limit zero.
    core::Real multiplied_by_product_and_factor(
        const ScaledPositiveProduct& rhs,
        core::Real factor,
        std::string_view role) const {
        require_strict_floating_environment(role);
        long double mantissa = mantissa_ * rhs.mantissa_;
        long long exponent = exponent_;
        checked_add_exponent(exponent, rhs.exponent_, role);

        int normalization_exponent = 0;
        mantissa = std::frexp(mantissa, &normalization_exponent);
        checked_add_exponent(exponent, normalization_exponent, role);
        if (!std::isfinite(mantissa) || !(mantissa > 0.0L)) {
            throw std::overflow_error(
                std::string(role)
                + " scaled positive-product composition is invalid");
        }

        (void)consume_into(mantissa, exponent, factor, false, role);
        try {
            return materialize(mantissa, exponent, role);
        } catch (const std::underflow_error&) {
            return 0.0;
        }
    }

    // Evaluate (this - rhs) * scale without materializing either positive term
    // or the positive scale. This is the primitive needed when exact dyadic
    // geometry supplies a mantissa/exponent gap that may itself lie outside the
    // standalone binary64 range.
    //
    // The stored mantissas are working-format approximations of factor products.
    // If their rounding envelopes overlap, their ordering (and therefore the
    // cancellation residual) is not numerically identifiable from this state.
    // Fail closed rather than silently turning a representable residual into
    // zero or assigning it the wrong sign.
    core::Real signed_difference_scaled_by_product(
        const ScaledPositiveProduct& rhs,
        const ScaledPositiveProduct& scale,
        std::string_view role) const {
        // This path intentionally maps a mathematically sub-binary64 final
        // residual to zero. That classification is only meaningful when the
        // executing thread uses round-to-nearest with gradual underflow.
        require_strict_floating_environment(role);
        const int ordering = compare_normalized_checked(
            rhs.mantissa_, rhs.exponent_, rhs.rounded_operation_count_, role);
        if (ordering == 0) return 0.0;

        long double mantissa = 0.0L;
        long long exponent = 0;
        positive_difference_components(
            rhs, ordering, mantissa, exponent, role);

        mantissa *= scale.mantissa_;
        checked_add_exponent(exponent, scale.exponent_, role);
        int normalization_exponent = 0;
        mantissa = std::frexp(mantissa, &normalization_exponent);
        checked_add_exponent(exponent, normalization_exponent, role);
        if (!std::isfinite(mantissa) || !(mantissa > 0.0L)) {
            throw std::overflow_error(
                std::string(role)
                + " scaled positive-product composition is invalid");
        }

        try {
            const core::Real magnitude = materialize(mantissa, exponent, role);
            return ordering > 0 ? magnitude : -magnitude;
        } catch (const std::underflow_error&) {
            return 0.0;
        }
    }

    // Evaluate (this - rhs) * product(numerators) / product(denominators)
    // without materializing either positive term or their dimensional scale.
    // A cancellation whose sign is unresolved at the working precision is
    // rejected rather than guessed. A final magnitude below binary64's
    // round-to-nearest range is returned as zero; a non-zero unrepresentable
    // overflow remains an error.
    core::Real signed_difference_scaled_by_quotient(
        const ScaledPositiveProduct& rhs,
        std::span<const core::Real> numerators,
        std::span<const core::Real> denominators,
        std::string_view role) const {
        require_round_to_nearest(role);
        ScaledPositiveProduct scale;
        for (const core::Real factor : numerators) {
            scale.consume(factor, false, role);
        }
        for (const core::Real factor : denominators) {
            scale.consume(factor, true, role);
        }
        return signed_difference_scaled_by_product(rhs, scale, role);
    }

    // Compare the represented positive value to an ordinary positive binary64
    // value without materializing the represented ratio. Return -1, 0, or +1.
    // If accumulated factor rounding is large enough to make the ordering
    // unresolved, reject the comparison instead of treating rounded equality
    // as mathematical equality.
    int compare_to(
        core::Real rhs,
        std::string_view role) const {
        require_round_to_nearest(role);
        if (!std::isfinite(rhs) || rhs <= 0.0) {
            throw std::invalid_argument(
                std::string(role)
                + " comparison target must be finite and strictly positive");
        }
        int rhs_exponent = 0;
        const core::Real rhs_mantissa = std::frexp(rhs, &rhs_exponent);
        return compare_normalized_checked(
            static_cast<long double>(rhs_mantissa),
            static_cast<long long>(rhs_exponent),
            0,
            role);
    }

    // Compare two scaled products directly. Neither represented value needs to
    // fit core::Real. Exact working-format constructions retain ordinary
    // exponent/mantissa ordering; rounded constructions fail closed when their
    // conservative relative-error envelopes overlap.
    int compare_to(const ScaledPositiveProduct& rhs) const {
        constexpr std::string_view role = "scaled positive-product comparison";
        require_round_to_nearest(role);
        return compare_normalized_checked(
            rhs.mantissa_,
            rhs.exponent_,
            rhs.rounded_operation_count_,
            role);
    }

    // Return the positive real cube root without first materializing the
    // represented product/ratio. This matters when the ratio itself lies
    // outside core::Real even though its cube root is representable. Split the
    // binary exponent as e = 3q + r with r in [0,2], take cbrt(m*2^r), then
    // apply 2^q. No dimensional intermediate is squared or cubed.
    core::Real cube_root_value(std::string_view role) const {
        require_round_to_nearest(role);
        long long quotient_exponent = exponent_ / 3;
        long long remainder_exponent = exponent_ % 3;
        if (remainder_exponent < 0) {
            remainder_exponent += 3;
            --quotient_exponent;
        }
        const long double root_mantissa = std::cbrt(
            std::scalbn(mantissa_, static_cast<int>(remainder_exponent)));
        if (!std::isfinite(root_mantissa) || root_mantissa <= 0.0L) {
            throw std::overflow_error(
                std::string(role) + " cube-root mantissa is invalid");
        }
        return materialize(root_mantissa, quotient_exponent, role);
    }

private:
    static long double relative_roundoff_bound(
        std::size_t rounded_operations,
        std::string_view role) {
        if (rounded_operations == 0) return 0.0L;

        // Each recorded operation is a normalized positive multiply or divide
        // whose FMA residual proves that the working-format result was rounded.
        // Under round-to-nearest its relative error is bounded by unit roundoff
        // u=epsilon/2. The standard product bound gamma_n=n*u/(1-n*u)
        // encloses the accumulated relative error. Refuse pathological chains
        // once even this conservative first-order envelope ceases to be small.
        const long double unit_roundoff =
            0.5L * std::numeric_limits<long double>::epsilon();
        const long double n_u =
            static_cast<long double>(rounded_operations) * unit_roundoff;
        if (!std::isfinite(n_u) || n_u >= 0.25L) {
            throw std::overflow_error(
                std::string(role)
                + " accumulated product rounding is too large to bound reliably");
        }
        return n_u / (1.0L - n_u);
    }

    // For signed long long exponents with exponent < common_exponent, unsigned
    // subtraction yields the exact mathematical positive distance modulo 2^N.
    // The true distance is in [1,2^N-1], so no information is lost even across
    // LLONG_MIN/LLONG_MAX. Avoid converting huge adjacent integers to floating
    // point, where binary64 long double ABIs can round them to the same value.
    static unsigned long long exponent_distance(
        long long exponent,
        long long common_exponent) noexcept {
        return static_cast<unsigned long long>(common_exponent)
            - static_cast<unsigned long long>(exponent);
    }

    static long double align_mantissa_to_exponent(
        long double mantissa,
        long long exponent,
        long long common_exponent) noexcept {
        if (exponent >= common_exponent) return mantissa;
        const unsigned long long distance =
            exponent_distance(exponent, common_exponent);
        if (distance > static_cast<unsigned long long>(
                std::numeric_limits<int>::max())) {
            return 0.0L;
        }
        return std::scalbn(mantissa, -static_cast<int>(distance));
    }

    int compare_normalized_checked(
        long double rhs_mantissa,
        long long rhs_exponent,
        std::size_t rhs_rounded_operations,
        std::string_view role) const {
        require_round_to_nearest(role);
        if (rounded_operation_count_ == 0
            && rhs_rounded_operations == 0) {
            if (exponent_ < rhs_exponent) return -1;
            if (exponent_ > rhs_exponent) return 1;
            if (mantissa_ < rhs_mantissa) return -1;
            if (mantissa_ > rhs_mantissa) return 1;
            return 0;
        }

        const long long common_exponent =
            std::max(exponent_, rhs_exponent);
        const long double lhs_aligned = align_mantissa_to_exponent(
            mantissa_, exponent_, common_exponent);
        const long double rhs_aligned = align_mantissa_to_exponent(
            rhs_mantissa, rhs_exponent, common_exponent);
        if (!std::isfinite(lhs_aligned) || lhs_aligned < 0.0L
            || !std::isfinite(rhs_aligned) || rhs_aligned < 0.0L) {
            throw std::overflow_error(
                std::string(role)
                + " normalized product comparison is invalid");
        }

        const long double scale = std::max(lhs_aligned, rhs_aligned);
        if (!(scale > 0.0L)) {
            throw std::overflow_error(
                std::string(role)
                + " normalized product comparison underflowed both operands");
        }
        const long double gap = std::abs(lhs_aligned - rhs_aligned);
        const long double lhs_gamma = relative_roundoff_bound(
            rounded_operation_count_, role);
        const long double rhs_gamma = relative_roundoff_bound(
            rhs_rounded_operations, role);
        const long double arithmetic_guard =
            8.0L * std::numeric_limits<long double>::epsilon() * scale;
        const long double uncertainty =
            lhs_aligned * lhs_gamma
            + rhs_aligned * rhs_gamma
            + arithmetic_guard;
        if (!std::isfinite(uncertainty) || gap <= uncertainty) {
            throw std::overflow_error(
                std::string(role)
                + " ordering is unresolved at the working precision");
        }
        return lhs_aligned < rhs_aligned ? -1 : 1;
    }

    void positive_difference_components(
        const ScaledPositiveProduct& rhs,
        int ordering,
        long double& mantissa,
        long long& exponent,
        std::string_view role) const {
        if (ordering == 0) {
            mantissa = 0.0L;
            exponent = 0;
            return;
        }
        const ScaledPositiveProduct& larger = ordering > 0 ? *this : rhs;
        const ScaledPositiveProduct& smaller = ordering > 0 ? rhs : *this;
        mantissa = larger.mantissa_;
        exponent = larger.exponent_;

        const long double aligned_smaller = align_mantissa_to_exponent(
            smaller.mantissa_,
            smaller.exponent_,
            larger.exponent_);
        mantissa -= aligned_smaller;
        if (!std::isfinite(mantissa) || !(mantissa > 0.0L)) {
            throw std::overflow_error(
                std::string(role)
                + " scaled positive-product difference is unresolved");
        }

        int normalization_exponent = 0;
        mantissa = std::frexp(mantissa, &normalization_exponent);
        checked_add_exponent(exponent, normalization_exponent, role);
    }

    static void checked_add_exponent(
        long long& exponent,
        long long increment,
        std::string_view role) {
        if ((increment > 0
                && exponent > std::numeric_limits<long long>::max() - increment)
            || (increment < 0
                && exponent < std::numeric_limits<long long>::min() - increment)) {
            throw std::overflow_error(
                std::string(role) + " binary exponent accumulation overflowed");
        }
        exponent += increment;
    }

    // Return true iff the normalized working-format multiply/divide rounded.
    // The FMA residual is evaluated on bounded mantissas, so this is independent
    // of the dimensional exponent and remains usable on ABIs where long double
    // has only binary64 precision.
    static bool consume_into(
        long double& mantissa,
        long long& exponent,
        core::Real factor,
        bool divide,
        std::string_view role) {
        if (!std::isfinite(factor) || factor <= 0.0) {
            throw std::invalid_argument(
                std::string(role)
                + " factors must be finite and strictly positive");
        }

        int factor_exponent = 0;
        const core::Real factor_mantissa = std::frexp(
            factor, &factor_exponent);
        const long double factor_wide =
            static_cast<long double>(factor_mantissa);
        const long double before = mantissa;
        long double residual = 0.0L;
        if (divide) {
            mantissa = before / factor_wide;
            residual = std::fma(mantissa, factor_wide, -before);
            checked_add_exponent(exponent, -factor_exponent, role);
        } else {
            mantissa = before * factor_wide;
            residual = std::fma(before, factor_wide, -mantissa);
            checked_add_exponent(exponent, factor_exponent, role);
        }

        int normalization_exponent = 0;
        mantissa = std::frexp(mantissa, &normalization_exponent);
        checked_add_exponent(exponent, normalization_exponent, role);
        if (!std::isfinite(mantissa) || mantissa <= 0.0L
            || !std::isfinite(residual)) {
            throw std::overflow_error(
                std::string(role) + " normalized mantissa is invalid");
        }
        return residual != 0.0L;
    }

    void consume(
        core::Real factor,
        bool divide,
        std::string_view role) {
        if (consume_into(mantissa_, exponent_, factor, divide, role)) {
            if (rounded_operation_count_
                == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error(
                    std::string(role)
                    + " rounded-operation count overflowed");
            }
            ++rounded_operation_count_;
        }
    }

    static core::Real materialize(
        long double mantissa,
        long long exponent,
        std::string_view role) {
        require_round_to_nearest(role);
        if (exponent < static_cast<long long>(
                std::numeric_limits<int>::min())
            || exponent > static_cast<long long>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                std::string(role) + " binary exponent is not representable");
        }

        const long double wide = std::scalbn(
            mantissa, static_cast<int>(exponent));
        if (wide == 0.0L) {
            throw std::underflow_error(
                std::string(role) + " result underflowed the working format");
        }
        if (!std::isfinite(wide)
            || wide < 0.0L
            || wide > static_cast<long double>(
                std::numeric_limits<core::Real>::max())) {
            throw std::overflow_error(
                std::string(role) + " result is not representable");
        }

        const core::Real result = static_cast<core::Real>(wide);
        if (!std::isfinite(result) || result <= 0.0) {
            throw std::underflow_error(
                std::string(role) + " result underflowed core::Real");
        }
        return result;
    }

    long double mantissa_{1.0L};
    long long exponent_{0};
    std::size_t rounded_operation_count_{0};
};

// Evaluate product(numerators) / product(denominators) without materializing
// an overflowing or underflowing intermediate. Every factor must be finite and
// strictly positive. A result that is not representable in core::Real is
// rejected rather than rounded to infinity or zero.
inline core::Real scaled_positive_product_quotient(
    std::span<const core::Real> numerators,
    std::span<const core::Real> denominators,
    std::string_view role) {
    return ScaledPositiveProduct::from_quotient(
        numerators, denominators, role).value(role);
}

} // namespace cosmo_nbody::math
