// Real and Accum are binary64. Position is comoving Mpc/h; Momentum stores
// p=a^2 dx/dt; PeculiarVelocity stores a dx/dt=p/a in km/s; Acceleration stores g.
#pragma once

#include "cosmo_nbody/core/exact_binary64_norm.hpp"

#include <cstdint>
#include <cmath>
#include <limits>

namespace cosmo_nbody {
namespace core {

using Real = double;
using Accum = double;

using ParticleId = std::uint64_t;
using GridIndex = std::int64_t;
using LocalIndex = std::int64_t;

// Log threshold below which positive wide values round to zero when narrowed to
// binary64 round-to-nearest; values at/above it must use the actual conversion.
inline long double real_round_to_zero_log_threshold() {
    return std::log(static_cast<long double>(
        std::numeric_limits<Real>::denorm_min()))
        - std::log(2.0L);
}

// Range-safe 3-vector norm. Compose two-argument hypot because some libc++
// three-argument implementations use an unscaled intermediate sum of squares.
template <typename Floating>
inline Floating scale_safe_norm3(Floating x, Floating y, Floating z) {
    return std::hypot(std::hypot(x, y), z);
}

namespace detail {

template <typename Floating>
inline constexpr bool has_binary64_semantics =
    std::numeric_limits<Floating>::is_iec559
    && std::numeric_limits<Floating>::radix == 2
    && std::numeric_limits<Floating>::digits == 53
    && std::numeric_limits<Floating>::max_exponent == 1024
    && std::numeric_limits<Floating>::min_exponent == -1021;

template <typename Floating>
inline long double normalized_norm3_squared(
    Floating x,
    Floating y,
    Floating z,
    Floating radius,
    bool& valid,
    bool& zero_radius_exact_match) {
    const long double wide_x = static_cast<long double>(x);
    const long double wide_y = static_cast<long double>(y);
    const long double wide_z = static_cast<long double>(z);
    const long double wide_radius = static_cast<long double>(radius);
    valid = std::isfinite(wide_x)
        && std::isfinite(wide_y)
        && std::isfinite(wide_z)
        && std::isfinite(wide_radius)
        && wide_radius >= 0.0L;
    zero_radius_exact_match = false;
    if (!valid) return 0.0L;

    const long double abs_x = std::abs(wide_x);
    const long double abs_y = std::abs(wide_y);
    const long double abs_z = std::abs(wide_z);
    if (abs_x > wide_radius
        || abs_y > wide_radius
        || abs_z > wide_radius) {
        valid = false;
        return 0.0L;
    }
    if (wide_radius == 0.0L) {
        zero_radius_exact_match =
            abs_x == 0.0L && abs_y == 0.0L && abs_z == 0.0L;
        return 0.0L;
    }

    const long double normalized_x = abs_x / wide_radius;
    const long double normalized_y = abs_y / wide_radius;
    const long double normalized_z = abs_z / wide_radius;
    return normalized_x * normalized_x
        + normalized_y * normalized_y
        + normalized_z * normalized_z;
}

} // namespace detail

// Compare ||x|| <= radius without dimensional squaring. Binary64 inputs use exact
// dyadic squared values; wider types use a normalized range-safe comparison.
template <typename Floating>
inline bool scale_safe_norm3_leq(
    Floating x,
    Floating y,
    Floating z,
    Floating radius) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
        || !std::isfinite(radius) || radius < Floating{0}) {
        return false;
    }
    if constexpr (detail::has_binary64_semantics<Floating>) {
        return detail::exact_binary64_norm3_compare(
            static_cast<double>(x),
            static_cast<double>(y),
            static_cast<double>(z),
            static_cast<double>(radius)) <= 0;
    }

    bool valid = false;
    bool zero_radius_exact_match = false;
    const long double normalized_squared = detail::normalized_norm3_squared(
        x, y, z, radius, valid, zero_radius_exact_match);
    if (!valid) return false;
    if (static_cast<long double>(radius) == 0.0L) {
        return zero_radius_exact_match;
    }
    return normalized_squared <= 1.0L;
}

// Strict r<h counterpart; binary64 exact comparison preserves boundary membership.
template <typename Floating>
inline bool scale_safe_norm3_less(
    Floating x,
    Floating y,
    Floating z,
    Floating radius) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
        || !std::isfinite(radius) || radius < Floating{0}) {
        return false;
    }
    if constexpr (detail::has_binary64_semantics<Floating>) {
        return detail::exact_binary64_norm3_compare(
            static_cast<double>(x),
            static_cast<double>(y),
            static_cast<double>(z),
            static_cast<double>(radius)) < 0;
    }

    bool valid = false;
    bool zero_radius_exact_match = false;
    const long double normalized_squared = detail::normalized_norm3_squared(
        x, y, z, radius, valid, zero_radius_exact_match);
    if (!valid || static_cast<long double>(radius) == 0.0L) return false;
    return normalized_squared < 1.0L;
}

struct Vec3 {
    Real x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(Real x_, Real y_, Real z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    Vec3 operator-(const Vec3& rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    Vec3 operator*(Real scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    Vec3 operator/(Real scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    Vec3& operator+=(const Vec3& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
    Vec3& operator-=(const Vec3& rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
    Vec3& operator*=(Real scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    Vec3& operator/=(Real scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

    Real dot(const Vec3& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }

    // norm2() intentionally exposes direct squaring; norm() remains range-safe.
    Real norm2() const { return x * x + y * y + z * z; }
    Real norm() const { return scale_safe_norm3(x, y, z); }
};

inline Vec3 operator*(Real scalar, const Vec3& v) {
    return v * scalar;
}

// Distinct wrappers prevent silent mixing at module boundaries.
struct Position {
    Vec3 vec;
};

struct Momentum {
    Vec3 vec;
};

struct PeculiarVelocity {
    Vec3 vec;
};

struct Acceleration {
    Vec3 vec;
};

} // namespace core
} // namespace cosmo_nbody
