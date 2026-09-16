// Periodic geometry shared by force, halo, domain, and analysis code. wrap maps
// to [0,L); minimum_image(dx,L) maps to [-L/2,L/2). Endpoint-aware displacement
// instead rounds the mathematically selected route and can therefore equal
// either represented half-box endpoint. Directional uniqueness is an exact
// endpoint property, not a property of that rounded descriptive component.
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace cosmo_nbody {
namespace math {

enum class BoundaryMode {
    Open,
    PeriodicMinimumImage,
    PeriodicMesh // Used for force grids where minimum-image is implicit via FFT
};

// Map finite x to [0,L) without the overflow-prone x-L*floor(x/L) form.
inline core::Real wrap(core::Real x, core::Real L) {
    if (!std::isfinite(x) || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (x >= 0.0 && x < L) return x;

    core::Real wrapped = std::fmod(x, L);
    if (!std::isfinite(wrapped)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (wrapped < 0.0) wrapped += L;
    // Preserve the canonical half-open interval if reduction rounds to L.
    if (wrapped >= L) wrapped = 0.0;
    return wrapped == 0.0 ? core::Real{0.0} : wrapped;
}

// Map finite dx to [-L/2,L/2); +L/2 ties map to -L/2.
inline core::Real minimum_image(core::Real dx, core::Real L) {
    if (!std::isfinite(dx) || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    const core::Real half_L = 0.5 * L;
    if (!std::isfinite(half_L) || half_L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (dx >= -half_L && dx < half_L) return dx;

    core::Real wrapped = std::remainder(dx, L);
    if (!std::isfinite(wrapped)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    if (wrapped >= half_L) wrapped -= L;
    else if (wrapped < -half_L) wrapped += L;
    return wrapped == 0.0 ? core::Real{0.0} : wrapped;
}

namespace detail {

enum class WrappedMinimumImageRoute {
    Direct,
    AcrossBoundary,
    CutLocus,
};

inline bool valid_wrapped_coordinate(
    core::Real coordinate,
    core::Real L) noexcept {
    return std::isfinite(coordinate)
        && coordinate >= 0.0
        && coordinate < L;
}

inline bool valid_wrapped_vec3(const core::Vec3& value, core::Real L) noexcept {
    return valid_wrapped_coordinate(value.x, L)
        && valid_wrapped_coordinate(value.y, L)
        && valid_wrapped_coordinate(value.z, L);
}

// One adjacent binary64 on each side encloses the exact result of a correctly
// rounded nonnegative operation, including gradual underflow and overflow.
// Integer stepping is the nonnegative specialization of nextafter; +infinity
// steps down to DBL_MAX, while zero and +infinity saturate outwards. No relative
// epsilon estimate, normal-exponent assumption, or host long-double ABI is used.
inline core::Real nonnegative_round_down(core::Real value) noexcept {
    if (value == 0.0) return 0.0;
    return core::portable_bit_cast<core::Real>(
        core::portable_bit_cast<std::uint64_t>(value) - 1U);
}

inline core::Real nonnegative_round_up(core::Real value) noexcept {
    if (value == 0.0) return std::numeric_limits<core::Real>::denorm_min();
    if (std::isinf(value)) return value;
    return core::portable_bit_cast<core::Real>(
        core::portable_bit_cast<std::uint64_t>(value) + 1U);
}

// Interval filter for already validated canonical endpoints. Subtraction gives
// [down(high-low),up(high-low)]; the across route first encloses L-high and then
// its sum with low. min is monotone in both arguments, as are square and sum on
// nonnegative values, so the final intervals enclose the exact torus norm^2 and
// radius^2. Only disjoint intervals decide a sign. Overlap (including ties) goes
// to the existing exact integer predicate. This requires the same ordinary
// IEEE-754 operations as the exact binary64 contract, without fast-math
// reassociation or flush-to-zero arithmetic.
inline std::optional<int> filtered_wrapped_distance_to_radius_compare(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L,
    core::Real radius) noexcept {
    core::Real norm_lower = 0.0;
    core::Real norm_upper = 0.0;
    const core::Real first[3] = {from.x, from.y, from.z};
    const core::Real second[3] = {to.x, to.y, to.z};
    for (int axis = 0; axis < 3; ++axis) {
        const core::Real high = std::max(first[axis], second[axis]);
        const core::Real low = std::min(first[axis], second[axis]);
        const core::Real direct = high - low;
        const core::Real across_first = L - high;
        const core::Real lower = std::min(
            nonnegative_round_down(direct),
            nonnegative_round_down(
                nonnegative_round_down(across_first) + low));
        // An axis alone outside the radius proves the three-dimensional norm
        // outside. Validation above must precede this early exit.
        if (lower > radius) return 1;
        const core::Real upper = std::min(
            nonnegative_round_up(direct),
            nonnegative_round_up(
                nonnegative_round_up(across_first) + low));
        norm_lower = nonnegative_round_down(
            norm_lower + nonnegative_round_down(lower * lower));
        norm_upper = nonnegative_round_up(
            norm_upper + nonnegative_round_up(upper * upper));
    }
    const core::Real radius_squared = radius * radius;
    if (norm_upper < nonnegative_round_down(radius_squared)) return -1;
    if (norm_lower > nonnegative_round_up(radius_squared)) return 1;
    return std::nullopt;
}

inline void add_direct_axis_distance_squared(
    core::Real high,
    core::Real low,
    core::detail::ExactBinary64SquareAccumulator& positive,
    core::detail::ExactBinary64SquareAccumulator& negative) {
    // (high-low)^2 = high^2 + low^2 - 2 high low.
    positive.add_square(high);
    positive.add_square(low);
    negative.add_product(high, low);
    negative.add_product(high, low);
}

inline void add_across_axis_distance_squared(
    core::Real L,
    core::Real high,
    core::Real low,
    core::detail::ExactBinary64SquareAccumulator& positive,
    core::detail::ExactBinary64SquareAccumulator& negative) {
    // (L-high+low)^2 = L^2 + high^2 + low^2
    //                      + 2 L low - 2 L high - 2 high low.
    positive.add_square(L);
    positive.add_square(high);
    positive.add_square(low);
    positive.add_product(L, low);
    positive.add_product(L, low);
    negative.add_product(L, high);
    negative.add_product(L, high);
    negative.add_product(high, low);
    negative.add_product(high, low);
}

inline int compare_signed_exact_squares(
    const core::detail::ExactBinary64SquareAccumulator& lhs_positive,
    const core::detail::ExactBinary64SquareAccumulator& lhs_negative,
    const core::detail::ExactBinary64SquareAccumulator& rhs_positive,
    const core::detail::ExactBinary64SquareAccumulator& rhs_negative) {
    auto positive = lhs_positive;
    auto negative = lhs_negative;
    positive.add_accumulator(rhs_negative);
    negative.add_accumulator(rhs_positive);
    return positive.compare(negative);
}

// Classify the mathematical shortest route between represented canonical
// endpoints. Normal binary64 box sizes use the exact half-box threshold and
// Sterbenz subtraction. If L/2 is itself not representable (possible only at the
// extreme subnormal end of the declared finite-positive domain), compare the
// direct and across-boundary squared routes exactly instead of allowing rounded
// half-box arithmetic to decide topology.
inline std::optional<WrappedMinimumImageRoute>
wrapped_minimum_image_route(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(L) || L <= 0.0
        || !valid_wrapped_coordinate(from, L)
        || !valid_wrapped_coordinate(to, L)) {
        return std::nullopt;
    }

    const core::Real high = from > to ? from : to;
    const core::Real low = from > to ? to : from;
    const core::Real half_L = 0.5 * L;
    if (std::isfinite(half_L) && half_L > 0.0
        && half_L + half_L == L) {
        if (high < half_L) return WrappedMinimumImageRoute::Direct;
        const core::Real threshold = high - half_L;
        if (low > threshold) return WrappedMinimumImageRoute::Direct;
        if (low < threshold) return WrappedMinimumImageRoute::AcrossBoundary;
        return WrappedMinimumImageRoute::CutLocus;
    }

    core::detail::ExactBinary64SquareAccumulator direct_positive;
    core::detail::ExactBinary64SquareAccumulator direct_negative;
    core::detail::ExactBinary64SquareAccumulator across_positive;
    core::detail::ExactBinary64SquareAccumulator across_negative;
    add_direct_axis_distance_squared(
        high, low, direct_positive, direct_negative);
    add_across_axis_distance_squared(
        L, high, low, across_positive, across_negative);
    const int relation = compare_signed_exact_squares(
        direct_positive,
        direct_negative,
        across_positive,
        across_negative);
    if (relation < 0) return WrappedMinimumImageRoute::Direct;
    if (relation > 0) return WrappedMinimumImageRoute::AcrossBoundary;
    return WrappedMinimumImageRoute::CutLocus;
}

inline void add_exact_wrapped_axis_distance_squared(
    core::Real from,
    core::Real to,
    core::Real L,
    core::detail::ExactBinary64SquareAccumulator& positive,
    core::detail::ExactBinary64SquareAccumulator& negative) {
    if (from == to) return;

    const auto route = wrapped_minimum_image_route(from, to, L);
    if (!route.has_value()) {
        throw std::invalid_argument(
            "Exact periodic distance requires canonical finite endpoints and a positive box");
    }
    const core::Real high = from > to ? from : to;
    const core::Real low = from > to ? to : from;

    if (*route == WrappedMinimumImageRoute::AcrossBoundary) {
        add_across_axis_distance_squared(L, high, low, positive, negative);
        return;
    }

    // Direct and exact-cut-locus routes have the same squared value. Express it
    // from represented endpoints rather than materializing a possibly
    // unrepresentable half-box value.
    add_direct_axis_distance_squared(high, low, positive, negative);
}

inline void add_exact_wrapped_distance_squared(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L,
    core::detail::ExactBinary64SquareAccumulator& positive,
    core::detail::ExactBinary64SquareAccumulator& negative) {
    add_exact_wrapped_axis_distance_squared(
        from.x, to.x, L, positive, negative);
    add_exact_wrapped_axis_distance_squared(
        from.y, to.y, L, positive, negative);
    add_exact_wrapped_axis_distance_squared(
        from.z, to.z, L, positive, negative);
}

} // namespace detail

// Unsigned distance for two canonical endpoints. This is a descriptive rounded
// component distance. Boundary and ordering decisions must use the exact
// comparison helpers below instead.
inline core::Real minimum_image_distance_wrapped(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0
        || from < 0.0 || from >= L || to < 0.0 || to >= L) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    const core::Real high = from > to ? from : to;
    const core::Real low = from > to ? to : from;
    const core::Real distance = std::fmin(high - low, (L - high) + low);
    return distance == 0.0 ? core::Real{0.0} : distance;
}

// Compare the exact mathematical three-dimensional torus distance of represented
// canonical binary64 endpoints with a represented radius. Return -1, 0, or +1
// according as d<r, d=r, or d>r. Disjoint outward-rounded intervals may certify
// the sign; overlapping intervals use exact integer arithmetic. A rounded
// minimum-image component or norm alone never decides the result.
inline int minimum_image_distance_to_radius_compare_wrapped(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L,
    core::Real radius) {
    if (!std::isfinite(L) || L <= 0.0
        || !std::isfinite(radius) || radius < 0.0
        || !detail::valid_wrapped_vec3(from, L)
        || !detail::valid_wrapped_vec3(to, L)) {
        throw std::invalid_argument(
            "Exact periodic distance comparison requires canonical finite endpoints, a positive box, and a finite non-negative radius");
    }

    if (const auto filtered = detail::filtered_wrapped_distance_to_radius_compare(
            from, to, L, radius)) {
        return *filtered;
    }

    core::detail::ExactBinary64SquareAccumulator positive;
    core::detail::ExactBinary64SquareAccumulator negative;
    detail::add_exact_wrapped_distance_squared(
        from, to, L, positive, negative);
    negative.add_square(radius);
    return positive.compare(negative);
}

// Compare two exact torus distances measured from the same canonical origin.
// Return -1, 0, or +1 according as d(origin,lhs) is smaller, equal, or larger
// than d(origin,rhs). The two signed dyadic expressions are combined before a
// single exact integer comparison; no square root or rounded radius participates.
inline int minimum_image_distances_compare_wrapped(
    const core::Vec3& origin,
    const core::Vec3& lhs,
    const core::Vec3& rhs,
    core::Real L) {
    if (!std::isfinite(L) || L <= 0.0
        || !detail::valid_wrapped_vec3(origin, L)
        || !detail::valid_wrapped_vec3(lhs, L)
        || !detail::valid_wrapped_vec3(rhs, L)) {
        throw std::invalid_argument(
            "Exact periodic distance ordering requires canonical finite endpoints and a positive box");
    }

    core::detail::ExactBinary64SquareAccumulator lhs_positive;
    core::detail::ExactBinary64SquareAccumulator lhs_negative;
    core::detail::ExactBinary64SquareAccumulator rhs_positive;
    core::detail::ExactBinary64SquareAccumulator rhs_negative;
    detail::add_exact_wrapped_distance_squared(
        origin, lhs, L, lhs_positive, lhs_negative);
    detail::add_exact_wrapped_distance_squared(
        origin, rhs, L, rhs_positive, rhs_negative);
    return detail::compare_signed_exact_squares(
        lhs_positive,
        lhs_negative,
        rhs_positive,
        rhs_negative);
}

// Monotone, range-safe sorting key: round the exact squared distance once to a
// 53-bit significand with a separate exponent. Correct rounding is monotone, so
// *different* keys prove exact order. Equal keys are not proof of equal distance:
// sorting and shell grouping must resolve them with the exact comparator above.
// Zero is represented by {0,0}; positive keys have mantissa in [0.5,1).
inline core::detail::ExactBinary64PositiveDyadic
minimum_image_squared_distance_key_wrapped(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L) {
    if (!std::isfinite(L) || L <= 0.0
        || !detail::valid_wrapped_vec3(from, L)
        || !detail::valid_wrapped_vec3(to, L)) {
        throw std::invalid_argument(
            "Exact periodic distance key requires canonical finite endpoints and a positive box");
    }
    core::detail::ExactBinary64SquareAccumulator positive;
    core::detail::ExactBinary64SquareAccumulator negative;
    detail::add_exact_wrapped_distance_squared(from, to, L, positive, negative);
    if (positive.compare(negative) == 0) return {};
    return positive.positive_difference(negative);
}

inline int minimum_image_squared_distance_keys_compare(
    const core::detail::ExactBinary64PositiveDyadic& lhs,
    const core::detail::ExactBinary64PositiveDyadic& rhs) noexcept {
    if (lhs.mantissa == 0.0 || rhs.mantissa == 0.0)
        return lhs.mantissa < rhs.mantissa ? -1 : lhs.mantissa > rhs.mantissa ? 1 : 0;
    if (lhs.exponent != rhs.exponent) return lhs.exponent < rhs.exponent ? -1 : 1;
    return lhs.mantissa < rhs.mantissa ? -1 : lhs.mantissa > rhs.mantissa ? 1 : 0;
}

// Range-safe logarithmic projection of one strictly positive exact torus
// distance. The exact squared distance is rounded only once to a binary64
// significand plus a separate exponent; the logarithm is then evaluated without
// ever constructing rounded minimum-image components or an intermediate square
// root. Coincident endpoints return nullopt.
inline std::optional<long double> minimum_image_distance_log_wrapped(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L) {
    if (!std::isfinite(L) || L <= 0.0
        || !detail::valid_wrapped_vec3(from, L)
        || !detail::valid_wrapped_vec3(to, L)) {
        throw std::invalid_argument(
            "Exact periodic distance logarithm requires canonical finite endpoints and a positive box");
    }

    core::detail::ExactBinary64SquareAccumulator positive;
    core::detail::ExactBinary64SquareAccumulator negative;
    detail::add_exact_wrapped_distance_squared(
        from, to, L, positive, negative);
    const int relation = positive.compare(negative);
    if (relation < 0) {
        throw std::logic_error(
            "Exact periodic squared distance became negative");
    }
    if (relation == 0) return std::nullopt;
    const auto squared = positive.positive_difference(negative);
    const long double log_squared =
        std::log(static_cast<long double>(squared.mantissa))
        + static_cast<long double>(squared.exponent) * std::log(2.0L);
    if (!std::isfinite(log_squared)) {
        throw std::overflow_error(
            "Exact periodic distance logarithm is not representable");
    }
    return 0.5L * log_squared;
}

// Closed three-dimensional torus-radius predicate for represented canonical
// binary64 endpoints. Exact-comparison failures are not a geometric "outside"
// result; propagate them so membership-changing callers fail loudly.
inline bool minimum_image_distance_leq_wrapped(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L,
    core::Real radius) {
    return minimum_image_distance_to_radius_compare_wrapped(
        from, to, L, radius) <= 0;
}

// Compute candidate-reference from two already wrapped endpoints without first
// forming a near-L subtraction. The exact side of the cut locus is classified
// from represented endpoint comparisons. A unique route is then rounded once as
// a signed component; unlike a decision predicate, that descriptive component
// may round to +/-L/2 even though its exact route is unique.
inline core::Real minimum_image_displacement_wrapped(
    core::Real from,
    core::Real to,
    core::Real L) {
    const auto route = detail::wrapped_minimum_image_route(from, to, L);
    if (!route.has_value()) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }

    const bool forward = to >= from;
    const core::Real high = forward ? to : from;
    const core::Real low = forward ? from : to;
    core::Real magnitude = 0.0;
    core::Real result = 0.0;
    if (*route == detail::WrappedMinimumImageRoute::CutLocus) {
        // For an exact tie either route has the same mathematical magnitude.
        // Use the represented endpoint subtraction as the descriptive value.
        magnitude = high - low;
        if (!std::isfinite(magnitude) || magnitude < 0.0) {
            return std::numeric_limits<core::Real>::quiet_NaN();
        }
        return magnitude == 0.0 ? core::Real{0.0} : -magnitude;
    }
    if (*route == detail::WrappedMinimumImageRoute::Direct) {
        magnitude = high - low;
        if (!std::isfinite(magnitude) || magnitude < 0.0) {
            return std::numeric_limits<core::Real>::quiet_NaN();
        }
        result = forward ? magnitude : -magnitude;
    } else {
        magnitude = (L - high) + low;
        if (!std::isfinite(magnitude) || magnitude < 0.0) {
            return std::numeric_limits<core::Real>::quiet_NaN();
        }
        result = forward ? -magnitude : magnitude;
    }
    return result == 0.0 ? core::Real{0.0} : result;
}

// Preserve a directly representable small difference for arbitrary finite input.
// Production particle/node coordinates are canonical [0,L) values and therefore
// take the endpoint-aware path above. Noncanonical callers retain the historical
// small-difference fast path; otherwise each endpoint is wrapped independently.
inline core::Real minimum_image_displacement(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }

    if (from >= 0.0 && from < L && to >= 0.0 && to < L) {
        return minimum_image_displacement_wrapped(from, to, L);
    }

    const core::Real half_L = 0.5 * L;
    const core::Real direct = to - from;
    if (std::isfinite(half_L) && half_L > 0.0
        && std::isfinite(direct)
        && direct >= -half_L && direct < half_L) {
        return direct == 0.0 ? core::Real{0.0} : direct;
    }

    const core::Real wrapped_from = wrap(from, L);
    const core::Real wrapped_to = wrap(to, L);
    if (!std::isfinite(wrapped_from) || !std::isfinite(wrapped_to)) {
        return std::numeric_limits<core::Real>::quiet_NaN();
    }
    return minimum_image_displacement_wrapped(wrapped_from, wrapped_to, L);
}

// Exact half-box separation has two equally short directions and is therefore
// not directionally unique. A unique exact route that merely rounds to L/2 must
// remain unique, so this predicate classifies represented endpoints directly
// instead of inferring topology from a rounded displacement value.
inline bool minimum_image_displacement_is_directionally_unique(
    core::Real from,
    core::Real to,
    core::Real L) {
    if (!std::isfinite(from) || !std::isfinite(to)
        || !std::isfinite(L) || L <= 0.0) {
        return false;
    }

    if (from >= 0.0 && from < L && to >= 0.0 && to < L) {
        const auto route = detail::wrapped_minimum_image_route(from, to, L);
        return route.has_value()
            && *route != detail::WrappedMinimumImageRoute::CutLocus;
    }

    const core::Real wrapped_from = wrap(from, L);
    const core::Real wrapped_to = wrap(to, L);
    const auto route = detail::wrapped_minimum_image_route(
        wrapped_from, wrapped_to, L);
    return route.has_value()
        && *route != detail::WrappedMinimumImageRoute::CutLocus;
}

inline core::Vec3 wrap(const core::Vec3& x, core::Real L) {
    return { wrap(x.x, L), wrap(x.y, L), wrap(x.z, L) };
}

inline core::Vec3 minimum_image(const core::Vec3& dx, core::Real L) {
    return { minimum_image(dx.x, L), minimum_image(dx.y, L), minimum_image(dx.z, L) };
}

inline core::Vec3 minimum_image_displacement(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L) {
    return {
        minimum_image_displacement(from.x, to.x, L),
        minimum_image_displacement(from.y, to.y, L),
        minimum_image_displacement(from.z, to.z, L)};
}

inline bool minimum_image_displacement_is_directionally_unique(
    const core::Vec3& from,
    const core::Vec3& to,
    core::Real L) {
    return minimum_image_displacement_is_directionally_unique(from.x, to.x, L)
        && minimum_image_displacement_is_directionally_unique(from.y, to.y, L)
        && minimum_image_displacement_is_directionally_unique(from.z, to.z, L);
}

} // namespace math
} // namespace cosmo_nbody
