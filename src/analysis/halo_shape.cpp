#include "cosmo_nbody/analysis/halo_shape.hpp"
#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "membership_validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

namespace {
using WideReal = long double;
using WideVec3 = std::array<WideReal, 3>;
using WideMatrix3 = std::array<WideVec3, 3>;

class NumericalResolutionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PeriodicCutLocusError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct CompensatedWideSum {
    WideReal sum{0.0L};
    WideReal correction{0.0L};

    void add(WideReal value) {
        if (!std::isfinite(value)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer tensor contribution is non-finite");
        }
        const WideReal updated = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            correction += (sum - updated) + value;
        } else {
            correction += (value - updated) + sum;
        }
        if (!std::isfinite(updated) || !std::isfinite(correction)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer tensor accumulation overflowed");
        }
        sum = updated;
    }

    WideReal value() const {
        const WideReal result = sum + correction;
        if (!std::isfinite(result)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer compensated tensor sum is non-finite");
        }
        return result;
    }
};

struct ExactUInt128 {
    std::uint64_t high{0};
    std::uint64_t low{0};
};

ExactUInt128 multiply_u64(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    const std::uint64_t lhs_low = static_cast<std::uint32_t>(lhs);
    const std::uint64_t lhs_high = lhs >> 32U;
    const std::uint64_t rhs_low = static_cast<std::uint32_t>(rhs);
    const std::uint64_t rhs_high = rhs >> 32U;
    const std::uint64_t low_low = lhs_low * rhs_low;
    const std::uint64_t low_high = lhs_low * rhs_high;
    const std::uint64_t high_low = lhs_high * rhs_low;
    const std::uint64_t high_high = lhs_high * rhs_high;
    std::uint64_t low = low_low;
    std::uint64_t carry = 0;
    std::uint64_t previous = low;
    low += low_high << 32U;
    if (low < previous) ++carry;
    previous = low;
    low += high_low << 32U;
    if (low < previous) ++carry;
    return {
        high_high + (low_high >> 32U) + (high_low >> 32U) + carry,
        low};
}

struct DecodedBinary64 {
    std::uint64_t mantissa{0};
    int exponent{-1074};
    bool negative{false};
};

DecodedBinary64 decode_binary64(core::Real value) {
    static_assert(std::numeric_limits<core::Real>::is_iec559);
    static_assert(std::numeric_limits<core::Real>::digits == 53);
    static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "HaloShapeAnalyzer exact rank predicate requires finite binary64");
    }
    const std::uint64_t bits = core::portable_bit_cast<std::uint64_t>(value);
    const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
    const std::uint64_t fraction =
        bits & ((std::uint64_t{1} << 52U) - 1U);
    if (exponent_bits == 0) {
        return {fraction, -1074, (bits >> 63U) != 0};
    }
    return {
        (std::uint64_t{1} << 52U) | fraction,
        static_cast<int>(exponent_bits) - 1075,
        (bits >> 63U) != 0};
}

class ExactDyadicPredicateSum {
public:
    void add_product(
        core::Real first,
        core::Real second,
        bool negate = false) {
        const auto lhs = decode_binary64(first);
        const auto rhs = decode_binary64(second);
        if (lhs.mantissa == 0 || rhs.mantissa == 0) return;
        const ExactUInt128 product = multiply_u64(lhs.mantissa, rhs.mantissa);
        const std::array<std::uint64_t, 3> words{
            product.low, product.high, 0};
        add_words(
            words,
            lhs.exponent + rhs.exponent - minimum_exponent,
            negate != (lhs.negative != rhs.negative));
    }

    void add_triple_product(
        core::Real first,
        core::Real second,
        core::Real third,
        bool negate = false) {
        const auto lhs = decode_binary64(first);
        const auto middle = decode_binary64(second);
        const auto rhs = decode_binary64(third);
        if (lhs.mantissa == 0
            || middle.mantissa == 0
            || rhs.mantissa == 0) {
            return;
        }
        const ExactUInt128 first_product =
            multiply_u64(lhs.mantissa, middle.mantissa);
        const ExactUInt128 low_product =
            multiply_u64(first_product.low, rhs.mantissa);
        const ExactUInt128 high_product =
            multiply_u64(first_product.high, rhs.mantissa);
        const std::uint64_t middle_word =
            low_product.high + high_product.low;
        const std::uint64_t carry =
            middle_word < low_product.high ? 1U : 0U;
        if (high_product.high
            > std::numeric_limits<std::uint64_t>::max() - carry) {
            throw std::overflow_error(
                "HaloShapeAnalyzer exact triple-product mantissa overflowed");
        }
        const std::array<std::uint64_t, 3> words{
            low_product.low,
            middle_word,
            high_product.high + carry};
        add_words(
            words,
            lhs.exponent + middle.exponent + rhs.exponent
                - minimum_exponent,
            negate != (lhs.negative ^ middle.negative ^ rhs.negative));
    }

    bool is_zero() const noexcept { return positive_ == negative_; }

    std::optional<WideReal> log_absolute_value() const {
        if (is_zero()) return std::nullopt;
        const auto magnitude_order = [&]() noexcept {
            for (std::size_t offset = 0; offset < limb_count; ++offset) {
                const std::size_t index = limb_count - 1 - offset;
                if (positive_[index] > negative_[index]) return 1;
                if (positive_[index] < negative_[index]) return -1;
            }
            return 0;
        }();
        const Limbs& larger = magnitude_order > 0 ? positive_ : negative_;
        const Limbs& smaller = magnitude_order > 0 ? negative_ : positive_;
        Limbs magnitude{};
        std::uint64_t borrow = 0;
        for (std::size_t index = 0; index < limb_count; ++index) {
            const std::uint64_t first_difference =
                larger[index] - smaller[index];
            const bool first_borrow = larger[index] < smaller[index];
            magnitude[index] = first_difference - borrow;
            const bool second_borrow = first_difference < borrow;
            borrow = (first_borrow || second_borrow) ? 1U : 0U;
        }
        if (borrow != 0) {
            throw std::logic_error(
                "HaloShapeAnalyzer exact dyadic magnitude subtraction underflowed");
        }

        std::size_t leading_limb = limb_count;
        while (leading_limb > 0 && magnitude[leading_limb - 1] == 0) {
            --leading_limb;
        }
        if (leading_limb == 0) {
            throw std::logic_error(
                "HaloShapeAnalyzer exact dyadic non-zero magnitude vanished");
        }
        --leading_limb;
        const std::uint64_t leading_word = magnitude[leading_limb];
        unsigned leading_bit = 63U;
        while (((leading_word >> leading_bit) & 1U) == 0U) --leading_bit;
        WideReal significand = std::ldexp(
            static_cast<WideReal>(leading_word),
            -static_cast<int>(leading_bit));
        if (leading_limb > 0) {
            significand += std::ldexp(
                static_cast<WideReal>(magnitude[leading_limb - 1]),
                -static_cast<int>(leading_bit) - 64);
        }
        if (leading_limb > 1) {
            significand += std::ldexp(
                static_cast<WideReal>(magnitude[leading_limb - 2]),
                -static_cast<int>(leading_bit) - 128);
        }
        const int binary_exponent = minimum_exponent
            + static_cast<int>(64U * leading_limb + leading_bit);
        const WideReal result = std::log(significand)
            + static_cast<WideReal>(binary_exponent) * std::log(2.0L);
        if (!std::isfinite(result)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer exact dyadic logarithm is non-finite");
        }
        return result;
    }

private:
    // A triple product of finite binary64 values lies on the 2^-3222
    // dyadic grid. Its largest bit is below 6294 on that grid; summing six
    // determinant terms needs at most three carry bits. 100 limbs cover bits
    // [0,6399], including every subnormal/normal exponent combination.
    static constexpr int minimum_exponent = -3222;
    static constexpr std::size_t limb_count = 100;
    using Limbs = std::array<std::uint64_t, limb_count>;

    static void add_word(
        Limbs& target,
        std::size_t index,
        std::uint64_t word) {
        while (word != 0) {
            if (index >= target.size()) {
                throw std::overflow_error(
                    "HaloShapeAnalyzer exact dyadic rank accumulator overflowed");
            }
            const std::uint64_t previous = target[index];
            target[index] += word;
            word = target[index] < previous ? 1U : 0U;
            ++index;
        }
    }

    void add_words(
        const std::array<std::uint64_t, 3>& words,
        int shift,
        bool negative) {
        if (shift < 0) {
            throw std::logic_error(
                "HaloShapeAnalyzer exact dyadic rank shift is negative");
        }
        Limbs& target = negative ? negative_ : positive_;
        const std::size_t index = static_cast<std::size_t>(shift) / 64U;
        const unsigned offset = static_cast<unsigned>(shift) % 64U;
        for (std::size_t word_index = 0;
             word_index < words.size();
             ++word_index) {
            add_word(target, index + word_index, words[word_index] << offset);
            if (offset != 0) {
                add_word(
                    target,
                    index + word_index + 1,
                    words[word_index] >> (64U - offset));
            }
        }
    }

    Limbs positive_{};
    Limbs negative_{};
};

bool exactly_collinear(core::Vec3 first, core::Vec3 second) {
    for (const auto [lhs_a, lhs_b, rhs_a, rhs_b] : {
             std::array<core::Real, 4>{first.y, second.z, first.z, second.y},
             std::array<core::Real, 4>{first.z, second.x, first.x, second.z},
             std::array<core::Real, 4>{first.x, second.y, first.y, second.x}}) {
        ExactDyadicPredicateSum component;
        component.add_product(lhs_a, lhs_b);
        component.add_product(rhs_a, rhs_b, true);
        if (!component.is_zero()) return false;
    }
    return true;
}

bool exactly_coplanar(
    core::Vec3 first,
    core::Vec3 second,
    core::Vec3 third) {
    ExactDyadicPredicateSum determinant;
    determinant.add_triple_product(first.x, second.y, third.z);
    determinant.add_triple_product(first.y, second.z, third.x);
    determinant.add_triple_product(first.z, second.x, third.y);
    determinant.add_triple_product(first.z, second.y, third.x, true);
    determinant.add_triple_product(first.y, second.x, third.z, true);
    determinant.add_triple_product(first.x, second.z, third.y, true);
    return determinant.is_zero();
}

struct ExactDisplacementSpan {
    std::size_t rank{0};
    core::Vec3 first{};
    core::Vec3 second{};

    void add(core::Vec3 displacement) {
        if (rank == 3
            || (displacement.x == 0.0
                && displacement.y == 0.0
                && displacement.z == 0.0)) {
            return;
        }
        if (rank == 0) {
            first = displacement;
            rank = 1;
            return;
        }
        if (rank == 1) {
            if (!exactly_collinear(first, displacement)) {
                second = displacement;
                rank = 2;
            }
            return;
        }
        if (!exactly_coplanar(first, second, displacement)) rank = 3;
    }
};

core::Real checked_real(WideReal value, const char* label) {
    const WideReal maximum = static_cast<WideReal>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || value > maximum || value < -maximum) {
        throw NumericalResolutionError(label);
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result)
        || (value != 0.0L && result == 0.0)) {
        throw NumericalResolutionError(label);
    }
    return result;
}

bool finite_vec3(const core::Vec3& value) {
    return std::isfinite(value.x)
        && std::isfinite(value.y)
        && std::isfinite(value.z);
}

WideReal wide_dot(const core::Vec3& lhs, const core::Vec3& rhs) {
    const WideReal scale = std::max({
        std::abs(static_cast<WideReal>(lhs.x)),
        std::abs(static_cast<WideReal>(lhs.y)),
        std::abs(static_cast<WideReal>(lhs.z))});
    if (scale == 0.0L) return 0.0L;
    const WideReal result = scale * (
        (static_cast<WideReal>(lhs.x) / scale)
            * static_cast<WideReal>(rhs.x)
        + (static_cast<WideReal>(lhs.y) / scale)
            * static_cast<WideReal>(rhs.y)
        + (static_cast<WideReal>(lhs.z) / scale)
            * static_cast<WideReal>(rhs.z));
    if (!std::isfinite(result)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer projected displacement is non-finite");
    }
    return result;
}

core::Vec3 normalize(core::Vec3 v) {
    if (!finite_vec3(v)) {
        throw std::invalid_argument(
            "HaloShapeAnalyzer cannot normalize a non-finite vector");
    }
    const core::Real norm = core::scale_safe_norm3(v.x, v.y, v.z);
    if (!std::isfinite(norm)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer vector norm is non-finite");
    }
    if (norm == 0.0) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer cannot normalize an exact zero vector");
    }
    return v / norm;
}

core::Vec3 cross(const core::Vec3& lhs, const core::Vec3& rhs) {
    const WideReal x = static_cast<WideReal>(lhs.y)
            * static_cast<WideReal>(rhs.z)
        - static_cast<WideReal>(lhs.z)
            * static_cast<WideReal>(rhs.y);
    const WideReal y = static_cast<WideReal>(lhs.z)
            * static_cast<WideReal>(rhs.x)
        - static_cast<WideReal>(lhs.x)
            * static_cast<WideReal>(rhs.z);
    const WideReal z = static_cast<WideReal>(lhs.x)
            * static_cast<WideReal>(rhs.y)
        - static_cast<WideReal>(lhs.y)
            * static_cast<WideReal>(rhs.x);
    return {
        checked_real(x, "HaloShapeAnalyzer cross-product x is not representable"),
        checked_real(y, "HaloShapeAnalyzer cross-product y is not representable"),
        checked_real(z, "HaloShapeAnalyzer cross-product z is not representable")};
}

std::pair<core::Vec3, core::Vec3> deterministic_perpendicular_basis(
    core::Vec3 axis) {
    axis = normalize(axis);
    const core::Real ax = std::abs(axis.x);
    const core::Real ay = std::abs(axis.y);
    const core::Real az = std::abs(axis.z);
    core::Vec3 reference{1.0, 0.0, 0.0};
    if (ay <= ax && ay <= az) {
        reference = {0.0, 1.0, 0.0};
    } else if (az <= ax && az <= ay) {
        reference = {0.0, 0.0, 1.0};
    }
    core::Vec3 first = normalize(cross(axis, reference));
    core::Vec3 second = normalize(cross(axis, first));
    return {first, second};
}

struct EigenSystem3 {
    std::array<core::Real, 3> values{0.0, 0.0, 0.0};
    std::array<core::Vec3, 3> vectors{
        core::Vec3{1.0, 0.0, 0.0},
        core::Vec3{0.0, 1.0, 0.0},
        core::Vec3{0.0, 0.0, 1.0}};
    core::Real axis_ratio_b_over_a{0.0};
    core::Real axis_ratio_c_over_a{0.0};
    bool solver_indeterminate{false};
};

core::Vec3 canonicalize_axis(core::Vec3 axis) noexcept {
    const core::Real ax = std::abs(axis.x);
    const core::Real ay = std::abs(axis.y);
    const core::Real az = std::abs(axis.z);
    core::Real chosen = axis.x;
    if (ay > ax && ay >= az) {
        chosen = axis.y;
    } else if (az > ax && az > ay) {
        chosen = axis.z;
    }
    if (chosen < 0.0) axis *= -1.0;
    return axis;
}

void append_row_to_qr(WideMatrix3& triangular, WideVec3 row) {
    for (std::size_t pivot = 0; pivot < 3; ++pivot) {
        const WideReal diagonal = triangular[pivot][pivot];
        const WideReal incoming = row[pivot];
        const WideReal radius = std::hypot(diagonal, incoming);
        if (!std::isfinite(radius)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer streaming QR norm is non-finite");
        }
        if (radius == 0.0L) continue;
        const WideReal cosine = diagonal / radius;
        const WideReal sine = incoming / radius;
        for (std::size_t column = pivot + 1; column < 3; ++column) {
            const WideReal upper = triangular[pivot][column];
            const WideReal lower = row[column];
            const WideReal rotated_upper = cosine * upper + sine * lower;
            const WideReal rotated_lower = -sine * upper + cosine * lower;
            if (!std::isfinite(rotated_upper)
                || !std::isfinite(rotated_lower)) {
                throw NumericalResolutionError(
                    "HaloShapeAnalyzer streaming QR rotation is non-finite");
            }
            triangular[pivot][column] = rotated_upper;
            row[column] = rotated_lower;
        }
        triangular[pivot][pivot] = radius;
        row[pivot] = 0.0L;
    }
}

struct ColumnPairMoments {
    WideReal first_norm2{0.0L};
    WideReal second_norm2{0.0L};
    WideReal cross{0.0L};
};

ColumnPairMoments column_pair_moments(
    const WideMatrix3& matrix,
    std::size_t first,
    std::size_t second) {
    CompensatedWideSum first_norm2;
    CompensatedWideSum second_norm2;
    CompensatedWideSum cross;
    for (std::size_t row = 0; row < 3; ++row) {
        first_norm2.add(matrix[row][first] * matrix[row][first]);
        second_norm2.add(matrix[row][second] * matrix[row][second]);
        cross.add(matrix[row][first] * matrix[row][second]);
    }
    return {
        first_norm2.value(),
        second_norm2.value(),
        cross.value()};
}

WideReal representable_jacobi_tangent(
    WideReal first_norm2,
    WideReal second_norm2,
    WideReal cross) {
    if (cross == 0.0L) return 0.0L;
    const WideReal delta = 0.5L * (second_norm2 - first_norm2);
    const WideReal denominator = delta + std::copysign(
        std::hypot(delta, cross),
        delta == 0.0L ? WideReal{1.0L} : delta);
    if (!std::isfinite(denominator) || denominator == 0.0L) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer one-sided Jacobi denominator is not representable");
    }
    const WideReal tangent = cross / denominator;
    if (!std::isfinite(tangent)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer one-sided Jacobi rotation is not representable");
    }
    return tangent;
}

void rotate_columns(
    WideMatrix3& matrix,
    std::size_t first,
    std::size_t second,
    WideReal cosine,
    WideReal sine) {
    for (std::size_t row = 0; row < 3; ++row) {
        const WideReal first_value = matrix[row][first];
        const WideReal second_value = matrix[row][second];
        const WideReal rotated_first =
            cosine * first_value - sine * second_value;
        const WideReal rotated_second =
            sine * first_value + cosine * second_value;
        if (!std::isfinite(rotated_first)
            || !std::isfinite(rotated_second)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer one-sided Jacobi rotation is non-finite");
        }
        matrix[row][first] = rotated_first;
        matrix[row][second] = rotated_second;
    }
}

struct WideSingularSystem3 {
    std::array<WideReal, 3> singular_values{};
    WideMatrix3 right_vectors{
        WideVec3{1.0L, 0.0L, 0.0L},
        WideVec3{0.0L, 1.0L, 0.0L},
        WideVec3{0.0L, 0.0L, 1.0L}};
};

WideReal normalized_offdiagonal_gram_energy(
    const WideMatrix3& matrix) {
    WideReal matrix_scale = 0.0L;
    for (const auto& row : matrix) {
        for (const WideReal value : row) {
            matrix_scale = std::max(matrix_scale, std::abs(value));
        }
    }
    if (matrix_scale == 0.0L) return 0.0L;

    CompensatedWideSum total_norm2;
    for (const auto& row : matrix) {
        for (const WideReal value : row) {
            const WideReal scaled = value / matrix_scale;
            total_norm2.add(scaled * scaled);
        }
    }
    CompensatedWideSum offdiagonal_energy;
    for (const auto [first, second] : {
             std::pair<std::size_t, std::size_t>{0, 1},
             {0, 2},
             {1, 2}}) {
        CompensatedWideSum cross;
        for (std::size_t row = 0; row < 3; ++row) {
            cross.add(
                (matrix[row][first] / matrix_scale)
                * (matrix[row][second] / matrix_scale));
        }
        const WideReal coupling = cross.value();
        offdiagonal_energy.add(coupling * coupling);
    }
    const WideReal norm2 = total_norm2.value();
    return offdiagonal_energy.value() / (norm2 * norm2);
}

WideSingularSystem3 one_sided_svd(WideMatrix3 matrix) {
    WideMatrix3 right_vectors{
        WideVec3{1.0L, 0.0L, 0.0L},
        WideVec3{0.0L, 1.0L, 0.0L},
        WideVec3{0.0L, 0.0L, 1.0L}};
    for (;;) {
        std::size_t first = 0;
        std::size_t second = 0;
        WideReal selected_tangent = 0.0L;
        WideMatrix3 selected_matrix = matrix;
        WideReal selected_energy = normalized_offdiagonal_gram_energy(matrix);
        const auto consider_pair = [&](std::size_t candidate_first,
                                       std::size_t candidate_second) {
            const auto moments = column_pair_moments(
                matrix, candidate_first, candidate_second);
            const WideReal magnitude = std::abs(moments.cross);
            if (magnitude == 0.0L) return;
            const WideReal tangent = representable_jacobi_tangent(
                moments.first_norm2,
                moments.second_norm2,
                moments.cross);
            if (tangent == 0.0L) return;
            const WideReal cosine =
                1.0L / std::hypot(1.0L, tangent);
            const WideReal sine = tangent * cosine;
            WideMatrix3 candidate = matrix;
            rotate_columns(
                candidate,
                candidate_first,
                candidate_second,
                cosine,
                sine);
            const WideReal candidate_energy =
                normalized_offdiagonal_gram_energy(candidate);
            if (candidate_energy < selected_energy) {
                first = candidate_first;
                second = candidate_second;
                selected_tangent = tangent;
                selected_matrix = candidate;
                selected_energy = candidate_energy;
            }
        };
        consider_pair(0, 1);
        consider_pair(0, 2);
        consider_pair(1, 2);
        if (selected_tangent == 0.0L) break;

        const WideReal cosine =
            1.0L / std::hypot(1.0L, selected_tangent);
        const WideReal sine = selected_tangent * cosine;
        if (!std::isfinite(cosine) || !std::isfinite(sine)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer one-sided Jacobi coefficients are non-finite");
        }
        matrix = selected_matrix;
        rotate_columns(right_vectors, first, second, cosine, sine);
    }

    WideSingularSystem3 result;
    std::array<std::size_t, 3> order{0, 1, 2};
    std::array<WideReal, 3> singular_values{};
    for (std::size_t column = 0; column < 3; ++column) {
        singular_values[column] = std::hypot(
            matrix[0][column],
            std::hypot(matrix[1][column], matrix[2][column]));
        if (!std::isfinite(singular_values[column])) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer singular value is non-finite");
        }
    }
    std::stable_sort(
        order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            return singular_values[lhs] > singular_values[rhs];
        });
    for (std::size_t index = 0; index < 3; ++index) {
        result.singular_values[index] = singular_values[order[index]];
        for (std::size_t row = 0; row < 3; ++row) {
            result.right_vectors[row][index] =
                right_vectors[row][order[index]];
        }
    }
    return result;
}

struct WideInterval {
    WideReal lower{0.0L};
    WideReal upper{0.0L};
};

WideReal round_down(WideReal value) noexcept {
    return std::nextafter(
        value, -std::numeric_limits<WideReal>::infinity());
}

WideReal round_up(WideReal value) noexcept {
    return std::nextafter(
        value, std::numeric_limits<WideReal>::infinity());
}

WideInterval interval_product(WideReal lhs, WideReal rhs) {
    if (lhs == 0.0L || rhs == 0.0L) return {};
    const WideReal product = lhs * rhs;
    if (!std::isfinite(product)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer interval product is non-finite");
    }
    return {round_down(product), round_up(product)};
}

WideInterval interval_sum(WideInterval lhs, WideInterval rhs) {
    const WideReal lower = lhs.lower + rhs.lower;
    const WideReal upper = lhs.upper + rhs.upper;
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer interval sum is non-finite");
    }
    return {round_down(lower), round_up(upper)};
}

WideInterval interval_difference(WideInterval lhs, WideInterval rhs) {
    const WideReal lower = lhs.lower - rhs.upper;
    const WideReal upper = lhs.upper - rhs.lower;
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer interval difference is non-finite");
    }
    return {round_down(lower), round_up(upper)};
}

WideInterval interval_scale(WideInterval interval, WideReal scalar) {
    if (scalar == 0.0L) return {};
    const WideReal first = interval.lower * scalar;
    const WideReal second = interval.upper * scalar;
    if (!std::isfinite(first) || !std::isfinite(second)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer scaled interval is non-finite");
    }
    return {
        round_down(std::min(first, second)),
        round_up(std::max(first, second))};
}

WideInterval interval_quotient_by_positive(
    WideInterval numerator,
    WideInterval denominator) {
    if (!(denominator.lower > 0.0L)
        || !std::isfinite(denominator.upper)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer interval denominator is not strictly positive");
    }
    WideReal lower = 0.0L;
    WideReal upper = 0.0L;
    if (numerator.lower >= 0.0L) {
        lower = numerator.lower / denominator.upper;
        upper = numerator.upper / denominator.lower;
    } else if (numerator.upper <= 0.0L) {
        lower = numerator.lower / denominator.lower;
        upper = numerator.upper / denominator.upper;
    } else {
        lower = numerator.lower / denominator.lower;
        upper = numerator.upper / denominator.lower;
    }
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer interval quotient is non-finite");
    }
    return {round_down(lower), round_up(upper)};
}

WideReal interval_absolute_upper(WideInterval interval) noexcept {
    return round_up(std::max(std::abs(interval.lower), std::abs(interval.upper)));
}

WideReal interval_frobenius_upper(
    const std::array<WideInterval, 9>& matrix) {
    WideReal squared_norm = 0.0L;
    for (const auto interval : matrix) {
        const WideReal magnitude = interval_absolute_upper(interval);
        const WideReal square = round_up(magnitude * magnitude);
        squared_norm = round_up(squared_norm + square);
    }
    const WideReal result = round_up(std::sqrt(squared_norm));
    if (!std::isfinite(result)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer interval Frobenius norm is non-finite");
    }
    return result;
}

bool certify_singular_system(
    const WideMatrix3& triangular,
    const std::array<core::Vec3, 3>& axes,
    core::Real axis_ratio_b_over_a,
    core::Real axis_ratio_c_over_a,
    std::size_t exact_rank) {
    std::array<WideInterval, 9> gram{};
    WideInterval trace{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            WideInterval entry{};
            for (std::size_t inner = 0; inner < 3; ++inner) {
                entry = interval_sum(
                    entry,
                    interval_product(
                        triangular[inner][row],
                        triangular[inner][column]));
            }
            gram[3 * row + column] = entry;
            if (row == column) trace = interval_sum(trace, entry);
        }
    }
    if (!(trace.lower > 0.0L)) return false;
    for (auto& entry : gram) {
        entry = interval_quotient_by_positive(entry, trace);
    }

    const auto axis_component = [&](std::size_t row, std::size_t column) {
        const core::Vec3& axis = axes[column];
        if (row == 0) return static_cast<WideReal>(axis.x);
        if (row == 1) return static_cast<WideReal>(axis.y);
        return static_cast<WideReal>(axis.z);
    };
    const WideReal q = static_cast<WideReal>(axis_ratio_b_over_a);
    const WideReal s = static_cast<WideReal>(axis_ratio_c_over_a);
    const WideReal denominator = 1.0L + q * q + s * s;
    if (!std::isfinite(denominator) || !(denominator > 0.0L)) return false;
    const std::array<WideReal, 3> centers{
        1.0L / denominator,
        q * q / denominator,
        s * s / denominator};

    std::array<WideInterval, 9> orthogonality_error{};
    std::array<WideInterval, 9> eigen_residual{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            WideInterval inner_product{};
            WideInterval action{};
            for (std::size_t inner = 0; inner < 3; ++inner) {
                inner_product = interval_sum(
                    inner_product,
                    interval_product(
                        axis_component(inner, row),
                        axis_component(inner, column)));
                action = interval_sum(
                    action,
                    interval_scale(
                        gram[3 * row + inner],
                        axis_component(inner, column)));
            }
            const WideInterval identity{
                row == column ? 1.0L : 0.0L,
                row == column ? 1.0L : 0.0L};
            orthogonality_error[3 * row + column] =
                interval_difference(inner_product, identity);
            eigen_residual[3 * row + column] = interval_difference(
                action,
                interval_product(
                    axis_component(row, column), centers[column]));
        }
    }

    const WideReal delta =
        interval_frobenius_upper(orthogonality_error);
    if (!(delta < 1.0L)) return false;
    const WideReal denominator_lower =
        round_down(std::sqrt(round_down(1.0L - delta)));
    if (!(denominator_lower > 0.0L)) return false;
    const WideReal beta = round_up(
        interval_frobenius_upper(eigen_residual) / denominator_lower);
    if (!std::isfinite(beta)) return false;

    const auto separated = [&](std::size_t first, std::size_t second) {
        const WideReal gap = round_down(
            std::abs(centers[first] - centers[second]));
        return gap > round_up(2.0L * beta);
    };
    if (exact_rank == 3) {
        return round_down(centers[2] - beta) > 0.0L
            && separated(0, 1)
            && separated(1, 2);
    }
    if (exact_rank == 2) {
        return centers[1] > round_up(2.0L * beta)
            && separated(0, 1);
    }
    if (exact_rank == 1) {
        return centers[0] > round_up(2.0L * beta);
    }
    return false;
}

core::Vec3 narrowed_axis(
    const WideMatrix3& right_vectors,
    std::size_t column) {
    const WideReal norm = std::hypot(
        right_vectors[0][column],
        std::hypot(
            right_vectors[1][column], right_vectors[2][column]));
    if (!std::isfinite(norm) || norm == 0.0L) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer singular vector has no representable direction");
    }
    return canonicalize_axis(normalize({
        checked_real(
            right_vectors[0][column] / norm,
            "HaloShapeAnalyzer singular vector x is not representable"),
        checked_real(
            right_vectors[1][column] / norm,
            "HaloShapeAnalyzer singular vector y is not representable"),
        checked_real(
            right_vectors[2][column] / norm,
            "HaloShapeAnalyzer singular vector z is not representable")}));
}

std::array<core::Vec3, 3> narrowed_orthonormal_axes(
    const WideMatrix3& right_vectors) {
    core::Vec3 major = narrowed_axis(right_vectors, 0);
    core::Vec3 intermediate = narrowed_axis(right_vectors, 1);
    const WideReal projection = wide_dot(intermediate, major);
    intermediate = normalize({
        checked_real(
            static_cast<WideReal>(intermediate.x)
                - projection * static_cast<WideReal>(major.x),
            "HaloShapeAnalyzer orthogonalized intermediate x is not representable"),
        checked_real(
            static_cast<WideReal>(intermediate.y)
                - projection * static_cast<WideReal>(major.y),
            "HaloShapeAnalyzer orthogonalized intermediate y is not representable"),
        checked_real(
            static_cast<WideReal>(intermediate.z)
                - projection * static_cast<WideReal>(major.z),
            "HaloShapeAnalyzer orthogonalized intermediate z is not representable")});
    intermediate = canonicalize_axis(intermediate);
    core::Vec3 minor = normalize(cross(major, intermediate));
    const core::Vec3 raw_minor = narrowed_axis(right_vectors, 2);
    if (wide_dot(minor, raw_minor) < 0.0L) minor *= -1.0;
    return {major, intermediate, minor};
}

EigenSystem3 psd_eigensystem_from_qr(
    const WideMatrix3& triangular,
    WideReal displacement_scale,
    WideReal maximum_log_weight,
    WideReal log_total_member_mass,
    std::size_t proven_rank_upper_bound) {
    auto singular = one_sided_svd(triangular);
    for (std::size_t index = proven_rank_upper_bound; index < 3; ++index) {
        singular.singular_values[index] = 0.0L;
    }
    if (singular.singular_values[0] == 0.0L) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer mass tensor has zero spatial extent");
    }

    EigenSystem3 result;
    result.vectors = narrowed_orthonormal_axes(singular.right_vectors);
    try {
        result.axis_ratio_b_over_a = checked_real(
            singular.singular_values[1] / singular.singular_values[0],
            "HaloShapeAnalyzer b/a is not representable");
        result.axis_ratio_c_over_a = checked_real(
            singular.singular_values[2] / singular.singular_values[0],
            "HaloShapeAnalyzer c/a is not representable");
    } catch (const NumericalResolutionError&) {
        result.solver_indeterminate = true;
        return result;
    }
    if (!certify_singular_system(
            triangular,
            result.vectors,
            result.axis_ratio_b_over_a,
            result.axis_ratio_c_over_a,
            proven_rank_upper_bound)) {
        result.solver_indeterminate = true;
        return result;
    }
    const WideReal log_leading_eigenvalue =
        2.0L * std::log(singular.singular_values[0])
        + 2.0L * std::log(displacement_scale)
        + maximum_log_weight
        - log_total_member_mass;
    if (!std::isfinite(log_leading_eigenvalue)) {
        result.solver_indeterminate = true;
        return result;
    }
    const WideReal leading_eigenvalue = std::exp(log_leading_eigenvalue);
    if (!std::isfinite(leading_eigenvalue)
        || leading_eigenvalue <= 0.0L) {
        result.solver_indeterminate = true;
        return result;
    }
    try {
        result.values[0] = checked_real(
            leading_eigenvalue,
            "HaloShapeAnalyzer leading eigenvalue is not representable");
    } catch (const NumericalResolutionError&) {
        result.solver_indeterminate = true;
        return result;
    }
    try {
        result.values[1] = checked_real(
            static_cast<WideReal>(result.values[0])
                * static_cast<WideReal>(result.axis_ratio_b_over_a)
                * static_cast<WideReal>(result.axis_ratio_b_over_a),
            "HaloShapeAnalyzer middle eigenvalue is not representable");
        result.values[2] = checked_real(
            static_cast<WideReal>(result.values[0])
                * static_cast<WideReal>(result.axis_ratio_c_over_a)
                * static_cast<WideReal>(result.axis_ratio_c_over_a),
            "HaloShapeAnalyzer smallest eigenvalue is not representable");
    } catch (const NumericalResolutionError&) {
        result.solver_indeterminate = true;
        return result;
    }

    const bool largest_repeated = result.axis_ratio_b_over_a == 1.0;
    const bool smallest_repeated =
        result.axis_ratio_b_over_a == result.axis_ratio_c_over_a;
    if (largest_repeated && smallest_repeated) {
        // A completely degenerate tensor has no preferred orientation. Identity
        // is a deterministic computational basis; equal axis ratios make the
        // reduced-tensor metric invariant under any rotation of this basis.
        result.vectors[0] = {1.0, 0.0, 0.0};
        result.vectors[1] = {0.0, 1.0, 0.0};
        result.vectors[2] = {0.0, 0.0, 1.0};
        return result;
    }
    if (!largest_repeated && smallest_repeated) {
        const auto [first, second] =
            deterministic_perpendicular_basis(result.vectors[0]);
        result.vectors[1] = first;
        result.vectors[2] = second;
        return result;
    }
    if (largest_repeated && !smallest_repeated) {
        const auto [first, second] =
            deterministic_perpendicular_basis(result.vectors[2]);
        result.vectors[0] = first;
        result.vectors[1] = second;
        return result;
    }
    return result;
}

std::optional<WideReal> log_ellipsoidal_radius_squared(
    const core::Vec3& displacement,
    const core::Vec3& major_axis,
    const core::Vec3& intermediate_axis,
    const core::Vec3& minor_axis,
    core::Real q,
    core::Real s) {
    if (displacement.x == 0.0
        && displacement.y == 0.0
        && displacement.z == 0.0) {
        return std::nullopt;
    }
    if (!(q > 0.0) || !(s > 0.0)
        || !std::isfinite(q) || !std::isfinite(s)) {
        throw std::invalid_argument(
            "HaloShapeAnalyzer ellipsoidal metric requires positive finite axis ratios");
    }

    const WideReal displacement_scale = std::max({
        std::abs(static_cast<WideReal>(displacement.x)),
        std::abs(static_cast<WideReal>(displacement.y)),
        std::abs(static_cast<WideReal>(displacement.z))});
    struct ProjectionMagnitude {
        bool exact_zero{false};
        bool bounded_fast_path{false};
        WideReal normalized_projection{0.0L};
        std::optional<WideReal> exact_log_absolute{};
    };
    const auto projected_magnitude = [&](const core::Vec3& axis) {
        const std::array<WideReal, 3> normalized_components{
            static_cast<WideReal>(displacement.x) / displacement_scale,
            static_cast<WideReal>(displacement.y) / displacement_scale,
            static_cast<WideReal>(displacement.z) / displacement_scale};
        const std::array<WideReal, 3> axis_components{
            static_cast<WideReal>(axis.x),
            static_cast<WideReal>(axis.y),
            static_cast<WideReal>(axis.z)};
        CompensatedWideSum approximate_projection;
        WideReal absolute_terms_upper = 0.0L;
        for (std::size_t component = 0; component < 3; ++component) {
            const WideReal term = normalized_components[component]
                * axis_components[component];
            approximate_projection.add(term);
            const WideReal term_upper = std::nextafter(
                std::abs(term),
                std::numeric_limits<WideReal>::infinity());
            absolute_terms_upper = std::nextafter(
                absolute_terms_upper + term_upper,
                std::numeric_limits<WideReal>::infinity());
        }
        // Three scalings, three products, and the compensated three-term sum
        // use fewer than sixteen rounded arithmetic operations on any path.
        // The standard gamma_n bound selects the fast path; ambiguous cases
        // fall back to the exact dyadic accumulator below.
        constexpr WideReal operation_count = 16.0L;
        const WideReal unit_roundoff =
            std::numeric_limits<WideReal>::epsilon() / 2.0L;
        const WideReal accumulated_roundoff = std::nextafter(
            operation_count * unit_roundoff,
            std::numeric_limits<WideReal>::infinity());
        const WideReal gamma_denominator = std::nextafter(
            1.0L - accumulated_roundoff,
            -std::numeric_limits<WideReal>::infinity());
        if (!(gamma_denominator > 0.0L)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer projection error bound is not representable");
        }
        const WideReal gamma = std::nextafter(
            accumulated_roundoff / gamma_denominator,
            std::numeric_limits<WideReal>::infinity());
        const WideReal forward_error_bound = std::nextafter(
            gamma * absolute_terms_upper,
            std::numeric_limits<WideReal>::infinity());
        const WideReal approximate = approximate_projection.value();
        const WideReal relative_error_bound = approximate == 0.0L
            ? std::numeric_limits<WideReal>::infinity()
            : std::nextafter(
                forward_error_bound / std::abs(approximate),
                std::numeric_limits<WideReal>::infinity());
        if (relative_error_bound
            <= static_cast<WideReal>(
                std::numeric_limits<core::Real>::epsilon())) {
            return ProjectionMagnitude{false, true, approximate, std::nullopt};
        }

        ExactDyadicPredicateSum projection;
        projection.add_product(displacement.x, axis.x);
        projection.add_product(displacement.y, axis.y);
        projection.add_product(displacement.z, axis.z);
        const auto exact_log = projection.log_absolute_value();
        if (!exact_log.has_value()) {
            return ProjectionMagnitude{true, true, 0.0L, std::nullopt};
        }
        return ProjectionMagnitude{false, false, 0.0L, exact_log};
    };
    const std::array<ProjectionMagnitude, 3> projection{
        projected_magnitude(major_axis),
        projected_magnitude(intermediate_axis),
        projected_magnitude(minor_axis)};
    const std::array<WideReal, 3> log_axis_ratio{
        0.0L,
        std::log(static_cast<WideReal>(q)),
        std::log(static_cast<WideReal>(s))};
    const std::array<WideReal, 3> axis_ratio{
        1.0L,
        static_cast<WideReal>(q),
        static_cast<WideReal>(s)};

    bool direct_norm_available = true;
    std::array<WideReal, 3> direct_terms{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (projection[axis].exact_zero) continue;
        if (!projection[axis].bounded_fast_path) {
            direct_norm_available = false;
            break;
        }
        direct_terms[axis] = projection[axis].normalized_projection
            / axis_ratio[axis];
        if (!std::isfinite(direct_terms[axis])) {
            direct_norm_available = false;
            break;
        }
    }
    if (direct_norm_available) {
        const WideReal normalized_radius = std::hypot(
            direct_terms[0],
            std::hypot(direct_terms[1], direct_terms[2]));
        if (std::isfinite(normalized_radius) && normalized_radius > 0.0L) {
            const WideReal result = 2.0L * (
                std::log(displacement_scale)
                + std::log(normalized_radius));
            if (std::isfinite(result)) return result;
        }
    }

    std::array<WideReal, 3> log_terms{
        -std::numeric_limits<WideReal>::infinity(),
        -std::numeric_limits<WideReal>::infinity(),
        -std::numeric_limits<WideReal>::infinity()};
    WideReal maximum_log_term = -std::numeric_limits<WideReal>::infinity();
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (projection[axis].exact_zero) continue;
        WideReal log_projection = 0.0L;
        if (projection[axis].bounded_fast_path) {
            log_projection = std::log(displacement_scale)
                + std::log(std::abs(projection[axis].normalized_projection));
        } else if (projection[axis].exact_log_absolute.has_value()) {
            log_projection = *projection[axis].exact_log_absolute;
        } else {
            throw std::logic_error(
                "HaloShapeAnalyzer projection magnitude lost its representation");
        }
        log_terms[axis] = 2.0L * (
            log_projection - log_axis_ratio[axis]);
        if (!std::isfinite(log_terms[axis])) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer log ellipsoidal-radius term is non-finite");
        }
        maximum_log_term = std::max(maximum_log_term, log_terms[axis]);
    }
    if (!std::isfinite(maximum_log_term)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer non-zero displacement has no representable projection");
    }

    CompensatedWideSum scaled_sum;
    for (const WideReal log_term : log_terms) {
        if (!std::isfinite(log_term)) continue;
        scaled_sum.add(std::exp(log_term - maximum_log_term));
    }
    const WideReal result = maximum_log_term + std::log(scaled_sum.value());
    if (!std::isfinite(result)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer log ellipsoidal radius is non-finite");
    }
    return result;
}

core::Vec3 member_displacement(
    const core::ParticleStore& particles,
    core::Vec3 center,
    core::Real box_size,
    std::size_t index) {
    if (index >= particles.num_owned_particles()) {
        throw std::out_of_range(
            "HaloShapeAnalyzer member index must refer to an owned particle");
    }
    const core::Vec3 position{
        particles.get_positions_x()[index],
        particles.get_positions_y()[index],
        particles.get_positions_z()[index]};
    if (!finite_vec3(position)) {
        throw std::invalid_argument(
            "HaloShapeAnalyzer member position must be finite");
    }
    if (!math::minimum_image_displacement_is_directionally_unique(
            center, position, box_size)) {
        throw PeriodicCutLocusError(
            "HaloShapeAnalyzer cannot unwrap a member on the exact periodic cut locus");
    }
    return math::minimum_image_displacement(center, position, box_size);
}

WideReal reduced_weight_residual(
    const core::ParticleStore& particles,
    core::Vec3 center,
    core::Real box_size,
    std::span<const std::size_t> member_indices,
    const core::Vec3& previous_major_axis,
    const core::Vec3& previous_intermediate_axis,
    const core::Vec3& previous_minor_axis,
    core::Real previous_q,
    core::Real previous_s,
    const core::Vec3& current_major_axis,
    const core::Vec3& current_intermediate_axis,
    const core::Vec3& current_minor_axis,
    core::Real current_q,
    core::Real current_s) {
    WideReal minimum_log_change = std::numeric_limits<WideReal>::infinity();
    WideReal maximum_log_change = -std::numeric_limits<WideReal>::infinity();
    for (const std::size_t index : member_indices) {
        const core::Vec3 displacement =
            member_displacement(particles, center, box_size, index);
        const auto previous_log_radius = log_ellipsoidal_radius_squared(
            displacement,
            previous_major_axis,
            previous_intermediate_axis,
            previous_minor_axis,
            previous_q,
            previous_s);
        if (!previous_log_radius.has_value()) continue;
        const auto current_log_radius = log_ellipsoidal_radius_squared(
            displacement,
            current_major_axis,
            current_intermediate_axis,
            current_minor_axis,
            current_q,
            current_s);
        if (!current_log_radius.has_value()) {
            throw std::logic_error(
                "HaloShapeAnalyzer center classification changed between metrics");
        }
        const WideReal log_change =
            *current_log_radius - *previous_log_radius;
        minimum_log_change = std::min(minimum_log_change, log_change);
        maximum_log_change = std::max(maximum_log_change, log_change);
    }
    if (!std::isfinite(minimum_log_change)
        || !std::isfinite(maximum_log_change)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer reduced metric has no directional member");
    }
    const WideReal residual = maximum_log_change - minimum_log_change;
    if (!std::isfinite(residual) || residual < 0.0L) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer reduced relative-weight residual is invalid");
    }
    return residual;
}

} // namespace

HaloShapeResult HaloShapeAnalyzer::compute(
    const core::ParticleStore& particles,
    core::Vec3 center,
    core::Real box_size,
    std::span<const std::size_t> member_indices,
    const HaloShapeOptions& options) {
    if (!(box_size > 0.0) || !std::isfinite(box_size)) {
        throw std::invalid_argument("HaloShapeAnalyzer box_size must be positive and finite");
    }
    if (!finite_vec3(center)) {
        throw std::invalid_argument("HaloShapeAnalyzer center must be finite");
    }
    if (options.max_iterations < 1) {
        throw std::invalid_argument("HaloShapeAnalyzer max_iterations must be positive");
    }
    if (!(options.convergence_tolerance > 0.0) ||
        !std::isfinite(options.convergence_tolerance)) {
        throw std::invalid_argument(
            "HaloShapeAnalyzer convergence_tolerance must be positive and finite");
    }

    const auto membership_path = detail::require_unique_owned_members(
        particles,
        member_indices,
        "HaloShapeAnalyzer member index must refer to an owned particle",
        "HaloShapeAnalyzer member indices must be unique");

    // FoF catalogs already arrive in strictly increasing stable ParticleID
    // order and therefore retain the allocation-free fast path. General callers
    // are permitted to provide any unique index order; canonicalize that
    // fallback once so every tensor iteration sees the same arithmetic order.
    // Compensated floating-point summation improves accuracy but is not, by
    // itself, permutation invariant over a wide dynamic range.
    std::vector<std::size_t> canonical_members;
    if (membership_path == detail::MembershipValidationPath::GeneralUnique) {
        canonical_members.assign(member_indices.begin(), member_indices.end());
        const auto ids = particles.get_ids();
        std::sort(
            canonical_members.begin(),
            canonical_members.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                if (ids[lhs] != ids[rhs]) return ids[lhs] < ids[rhs];
                return lhs < rhs;
            });
        member_indices = std::span<const std::size_t>(canonical_members);
    }

    ExactDisplacementSpan exact_member_span;
    std::size_t member_directional_particle_count = 0;
    bool periodic_cut_locus_ambiguous = false;
    for (const std::size_t index : member_indices) {
        const core::Real mass = particles.mass_at(index);
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "HaloShapeAnalyzer member mass must be finite and positive");
        }
        core::Vec3 displacement{};
        try {
            displacement =
                member_displacement(particles, center, box_size, index);
        } catch (const PeriodicCutLocusError&) {
            periodic_cut_locus_ambiguous = true;
            continue;
        }
        if (displacement.x != 0.0
            || displacement.y != 0.0
            || displacement.z != 0.0) {
            ++member_directional_particle_count;
        }
        exact_member_span.add(displacement);
    }

    if (periodic_cut_locus_ambiguous) {
        HaloShapeResult unavailable;
        unavailable.particle_count = member_indices.size();
        unavailable.directional_particle_count =
            member_directional_particle_count;
        unavailable.periodic_cut_locus_ambiguous = true;
        return unavailable;
    }

    if (member_indices.size() < 3) {
        HaloShapeResult fallback;
        fallback.particle_count = member_indices.size();
        fallback.directional_particle_count =
            member_directional_particle_count;
        fallback.converged = false;
        fallback.iterations = 0;
        return fallback;
    }

    if (exact_member_span.rank == 0) {
        HaloShapeResult unavailable;
        unavailable.particle_count = member_indices.size();
        unavailable.directional_particle_count = 0;
        unavailable.zero_spatial_extent = true;
        return unavailable;
    }

    if (options.use_reduced_tensor) {
        if (exact_member_span.rank < 3) {
            HaloShapeResult unavailable;
            unavailable.particle_count = member_indices.size();
            unavailable.directional_particle_count =
                member_directional_particle_count;
            unavailable.rank_deficient_reduced_metric = true;
            return unavailable;
        }
    }

    core::Vec3 major_axis{1.0, 0.0, 0.0};
    core::Vec3 intermediate_axis{0.0, 1.0, 0.0};
    core::Vec3 minor_axis{0.0, 0.0, 1.0};
    core::Real q = 1.0;
    core::Real s = 1.0;
    const auto solve_once = [&](core::Vec3 solve_major_axis,
                                core::Vec3 solve_intermediate_axis,
                                core::Vec3 solve_minor_axis,
                                core::Real solve_q,
                                core::Real solve_s,
                                const HaloShapeOptions& solve_options) {
        try {
            return compute_once(
                particles,
                center,
                box_size,
                member_indices,
                solve_major_axis,
                solve_intermediate_axis,
                solve_minor_axis,
                solve_q,
                solve_s,
                solve_options);
        } catch (const NumericalResolutionError&) {
            HaloShapeResult unresolved;
            unresolved.particle_count = member_indices.size();
            unresolved.directional_particle_count =
                member_directional_particle_count;
            unresolved.solver_indeterminate = true;
            return unresolved;
        }
    };
    // Bootstrap reduced-tensor iteration from an unreduced eigenbasis so the
    // first reweighting is not biased by an arbitrary axis choice.
    if (options.use_reduced_tensor) {
        HaloShapeOptions boot = options;
        boot.use_reduced_tensor = false;
        const auto seed = solve_once(
            major_axis,
            intermediate_axis,
            minor_axis,
            q,
            s,
            boot);
        if (seed.solver_indeterminate) {
            return seed;
        }
        major_axis = seed.major_axis;
        intermediate_axis = seed.intermediate_axis;
        minor_axis = seed.minor_axis;
        q = seed.axis_ratio_b_over_a;
        s = seed.axis_ratio_c_over_a;
        if (q == 0.0 || s == 0.0) {
            HaloShapeResult unavailable;
            unavailable.particle_count = member_indices.size();
            unavailable.directional_particle_count =
                seed.directional_particle_count;
            unavailable.solver_indeterminate = true;
            return unavailable;
        }
    }
    HaloShapeResult result;
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        const core::Vec3 previous_major_axis = major_axis;
        const core::Vec3 previous_intermediate_axis = intermediate_axis;
        const core::Vec3 previous_minor_axis = minor_axis;
        const core::Real previous_q = q;
        const core::Real previous_s = s;
        result = solve_once(
            major_axis,
            intermediate_axis,
            minor_axis,
            q,
            s,
            options);
        result.iterations = iteration;
        if (result.solver_indeterminate) {
            result.converged = false;
            break;
        }
        if (!options.use_reduced_tensor) {
            result.converged = true;
            break;
        }
        q = result.axis_ratio_b_over_a;
        s = result.axis_ratio_c_over_a;
        major_axis = result.major_axis;
        intermediate_axis = result.intermediate_axis;
        minor_axis = result.minor_axis;
        if (q == 0.0 || s == 0.0) {
            result.solver_indeterminate = true;
            result.converged = false;
            break;
        }
        try {
            result.converged = reduced_weight_residual(
                particles,
                center,
                box_size,
                member_indices,
                previous_major_axis,
                previous_intermediate_axis,
                previous_minor_axis,
                previous_q,
                previous_s,
                major_axis,
                intermediate_axis,
                minor_axis,
                q,
                s)
                <= static_cast<WideReal>(options.convergence_tolerance);
        } catch (const NumericalResolutionError&) {
            result = HaloShapeResult{};
            result.particle_count = member_indices.size();
            result.directional_particle_count =
                member_directional_particle_count;
            result.solver_indeterminate = true;
            result.iterations = iteration;
            break;
        }
        if (result.converged) break;
    }
    return result;
}

HaloShapeResult HaloShapeAnalyzer::compute_once(
    const core::ParticleStore& particles,
    core::Vec3 center,
    core::Real box_size,
    std::span<const std::size_t> member_indices,
    core::Vec3 major_axis,
    core::Vec3 intermediate_axis,
    core::Vec3 minor_axis,
    core::Real q,
    core::Real s,
    const HaloShapeOptions& options) {
    if (options.use_reduced_tensor
        && (!std::isfinite(q) || !std::isfinite(s)
            || s <= 0.0 || q < s || q > 1.0)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer reduced tensor requires 0 < s <= q <= 1");
    }
    major_axis = normalize(major_axis);
    intermediate_axis = normalize(intermediate_axis);
    minor_axis = normalize(minor_axis);

    const auto tensor_log_weight = [&](const core::Vec3& r, core::Real mass)
        -> std::optional<WideReal> {
        const WideReal log_mass = std::log(static_cast<WideReal>(mass));
        if (!std::isfinite(log_mass)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer logarithmic member mass is invalid");
        }
        if (!options.use_reduced_tensor) return log_mass;
        const auto log_radius = log_ellipsoidal_radius_squared(
            r,
            major_axis,
            intermediate_axis,
            minor_axis,
            q,
            s);
        if (!log_radius.has_value()) return std::nullopt;
        const WideReal result = log_mass - *log_radius;
        if (!std::isfinite(result)) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer logarithmic reduced tensor weight is non-finite");
        }
        return result;
    };

    WideReal maximum_log_weight =
        -std::numeric_limits<WideReal>::infinity();
    WideReal maximum_log_member_mass =
        -std::numeric_limits<WideReal>::infinity();
    WideReal displacement_scale = 0.0L;
    ExactDisplacementSpan displacement_span;
    std::size_t directional_particle_count = 0;
    for (const std::size_t index : member_indices) {
        const core::Vec3 r = member_displacement(particles, center, box_size, index);
        if (!finite_vec3(r)) {
            throw std::invalid_argument(
                "HaloShapeAnalyzer periodic displacement must be finite");
        }
        const core::Real m = particles.mass_at(index);
        if (!std::isfinite(m) || m <= 0.0) {
            throw std::invalid_argument(
                "HaloShapeAnalyzer member mass must be finite and positive");
        }
        const WideReal log_mass = std::log(static_cast<WideReal>(m));
        maximum_log_member_mass =
            std::max(maximum_log_member_mass, log_mass);
        const WideReal row_scale = std::max({
            std::abs(static_cast<WideReal>(r.x)),
            std::abs(static_cast<WideReal>(r.y)),
            std::abs(static_cast<WideReal>(r.z))});
        if (row_scale > 0.0L) {
            ++directional_particle_count;
            displacement_scale = std::max(displacement_scale, row_scale);
            displacement_span.add(r);
        }
        const auto log_weight = tensor_log_weight(r, m);
        if (!log_weight.has_value()) continue;
        maximum_log_weight = std::max(maximum_log_weight, *log_weight);
    }

    if (!std::isfinite(maximum_log_weight)) {
        throw std::invalid_argument(
            "HaloShapeAnalyzer tensor accumulation is non-positive");
    }
    if (!(displacement_scale > 0.0L) || displacement_span.rank == 0) {
        throw std::runtime_error(
            "HaloShapeAnalyzer mass tensor has zero spatial extent");
    }

    WideMatrix3 triangular{};
    CompensatedWideSum scaled_total_member_mass_accumulator;
    for (const std::size_t index : member_indices) {
        const core::Vec3 r = member_displacement(particles, center, box_size, index);
        const core::Real m = particles.mass_at(index);
        const WideReal log_mass = std::log(static_cast<WideReal>(m));
        const WideReal scaled_member_mass =
            std::exp(log_mass - maximum_log_member_mass);
        if (scaled_member_mass == 0.0L) {
            HaloShapeResult unresolved;
            unresolved.particle_count = member_indices.size();
            unresolved.directional_particle_count =
                directional_particle_count;
            unresolved.solver_indeterminate = true;
            return unresolved;
        }
        scaled_total_member_mass_accumulator.add(scaled_member_mass);
        const auto log_weight = tensor_log_weight(r, m);
        if (!log_weight.has_value()) continue;
        const WideReal scaled_weight =
            std::exp(*log_weight - maximum_log_weight);
        if (scaled_weight == 0.0L) {
            HaloShapeResult unresolved;
            unresolved.particle_count = member_indices.size();
            unresolved.directional_particle_count =
                directional_particle_count;
            unresolved.solver_indeterminate = true;
            return unresolved;
        }
        if (!std::isfinite(scaled_weight) || scaled_weight > 1.0L) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer scaled tensor weight is invalid");
        }
        const WideReal row_weight = std::sqrt(scaled_weight);
        if (!std::isfinite(row_weight) || row_weight <= 0.0L) {
            throw NumericalResolutionError(
                "HaloShapeAnalyzer scaled square-root weight is invalid");
        }
        WideVec3 row{
            row_weight * static_cast<WideReal>(r.x) / displacement_scale,
            row_weight * static_cast<WideReal>(r.y) / displacement_scale,
            row_weight * static_cast<WideReal>(r.z) / displacement_scale};
        const std::array<core::Real, 3> displacement_components{r.x, r.y, r.z};
        for (std::size_t component = 0; component < 3; ++component) {
            if (displacement_components[component] != 0.0
                && row[component] == 0.0L) {
                HaloShapeResult unresolved;
                unresolved.particle_count = member_indices.size();
                unresolved.directional_particle_count =
                    directional_particle_count;
                unresolved.solver_indeterminate = true;
                return unresolved;
            }
        }
        append_row_to_qr(triangular, row);
    }

    const WideReal scaled_total_member_mass =
        scaled_total_member_mass_accumulator.value();
    if (!(scaled_total_member_mass >= 1.0L)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer scaled total member mass is invalid");
    }
    const WideReal log_total_member_mass = maximum_log_member_mass
        + std::log(scaled_total_member_mass);
    if (!std::isfinite(log_total_member_mass)) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer total member mass is not representable in log space");
    }
    const auto eigen = psd_eigensystem_from_qr(
        triangular,
        displacement_scale,
        maximum_log_weight,
        log_total_member_mass,
        displacement_span.rank);
    HaloShapeResult result;
    if (eigen.solver_indeterminate) {
        result.particle_count = member_indices.size();
        result.directional_particle_count = directional_particle_count;
        result.solver_indeterminate = true;
        return result;
    }
    result.lambda_a = eigen.values[0];
    result.lambda_b = eigen.values[1];
    result.lambda_c = eigen.values[2];
    result.major_axis = eigen.vectors[0];
    result.intermediate_axis = eigen.vectors[1];
    result.minor_axis = eigen.vectors[2];
    result.axis_ratio_b_over_a = eigen.axis_ratio_b_over_a;
    result.axis_ratio_c_over_a = eigen.axis_ratio_c_over_a;
    const core::Real q2 = result.axis_ratio_b_over_a
        * result.axis_ratio_b_over_a;
    const core::Real s2 = result.axis_ratio_c_over_a
        * result.axis_ratio_c_over_a;
    const core::Real denominator = 1.0 - s2;
    result.triaxiality = denominator > 0.0
        ? (1.0 - q2) / denominator : 0.0;
    if (!std::isfinite(result.axis_ratio_b_over_a)
        || !std::isfinite(result.axis_ratio_c_over_a)
        || !std::isfinite(result.triaxiality)
        || result.axis_ratio_b_over_a < 0.0
        || result.axis_ratio_b_over_a > 1.0
        || result.axis_ratio_c_over_a < 0.0
        || result.axis_ratio_c_over_a > result.axis_ratio_b_over_a
        || result.triaxiality < 0.0
        || result.triaxiality > 1.0) {
        throw NumericalResolutionError(
            "HaloShapeAnalyzer derived axis ratios are invalid");
    }
    result.particle_count = member_indices.size();
    result.directional_particle_count = directional_particle_count;
    result.converged = true;
    return result;
}

} // namespace analysis
} // namespace cosmo_nbody
