#include "cosmo_nbody/analysis/symmetric_eigensystem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::analysis {
namespace {

using WideReal = long double;
using WideVec3 = std::array<WideReal, 3>;
using MatrixState = std::array<WideReal, 6>;

core::Real checked_real(WideReal value, const char* label) {
    const WideReal maximum = static_cast<WideReal>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || value > maximum || value < -maximum) {
        throw std::overflow_error(
            std::string(label) + " is outside the representable finite range");
    }
    return static_cast<core::Real>(value);
}

core::Real checked_eigenvalue(WideReal value, const char* label) {
    const core::Real narrowed = checked_real(value, label);
    if (value != 0.0L && narrowed == 0.0) {
        throw std::overflow_error(
            std::string(label) + " underflows the representable nonzero range");
    }
    return narrowed;
}

void require_finite_matrix(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22,
    core::Real eigenvalue = 0.0) {
    for (const core::Real value : {
             a00, a01, a02, a11, a12, a22, eigenvalue}) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Symmetric 3x3 eigensystem requires finite inputs");
        }
    }
}

core::Vec3 canonicalize(core::Vec3 axis) noexcept {
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

struct FloatingExpansion {
    std::array<core::Real, 96> components{};
    std::size_t size{0};
};

bool grow_expansion(FloatingExpansion& expansion, core::Real addend) noexcept {
    if (!std::isfinite(addend)) return false;
    std::array<core::Real, 96> result{};
    std::size_t result_size = 0;
    core::Real accumulator = addend;
    for (std::size_t index = 0; index < expansion.size; ++index) {
        const core::Real value = expansion.components[index];
        const core::Real sum = accumulator + value;
        if (!std::isfinite(sum)) return false;
        const core::Real value_virtual = sum - accumulator;
        const core::Real accumulator_virtual = sum - value_virtual;
        const core::Real value_roundoff = value - value_virtual;
        const core::Real accumulator_roundoff =
            accumulator - accumulator_virtual;
        const core::Real roundoff =
            accumulator_roundoff + value_roundoff;
        if (roundoff != 0.0) {
            if (result_size >= result.size()) return false;
            result[result_size++] = roundoff;
        }
        accumulator = sum;
    }
    if (accumulator != 0.0) {
        if (result_size >= result.size()) return false;
        result[result_size++] = accumulator;
    }
    expansion.components = result;
    expansion.size = result_size;
    return true;
}

bool add_exact_product(
    FloatingExpansion& expansion,
    core::Real lhs,
    core::Real rhs,
    core::Real sign = 1.0) noexcept {
    if (lhs == 0.0 || rhs == 0.0) return true;
    const core::Real rounded = lhs * rhs;
    if (!std::isfinite(rounded)
        || rounded == 0.0
        || std::fpclassify(rounded) == FP_SUBNORMAL) {
        return false;
    }
    constexpr int exact_residual_exponent_floor =
        std::numeric_limits<core::Real>::min_exponent
        + std::numeric_limits<core::Real>::digits - 1;
    if (std::ilogb(std::abs(rounded)) < exact_residual_exponent_floor) {
        return false;
    }

    const core::Real residual = std::fma(lhs, rhs, -rounded);
    if (!std::isfinite(residual)
        || std::fpclassify(residual) == FP_SUBNORMAL) {
        return false;
    }
    if (sign < 0.0) {
        return grow_expansion(expansion, -residual)
            && grow_expansion(expansion, -rounded);
    }
    return grow_expansion(expansion, residual)
        && grow_expansion(expansion, rounded);
}

bool add_exact_triple_product(
    FloatingExpansion& expansion,
    core::Real lhs,
    core::Real middle,
    core::Real rhs,
    core::Real sign) noexcept {
    FloatingExpansion product;
    if (!add_exact_product(product, lhs, middle)) return false;
    FloatingExpansion triple;
    for (std::size_t index = 0; index < product.size; ++index) {
        if (!add_exact_product(
                triple, product.components[index], rhs, sign)) {
            return false;
        }
    }
    for (std::size_t index = 0; index < triple.size; ++index) {
        if (!grow_expansion(expansion, triple.components[index])) return false;
    }
    return true;
}

struct ExactSymmetricInvariants {
    bool known{false};
    bool determinant_zero{false};
    bool second_coefficient_zero{false};
};

ExactSymmetricInvariants exact_symmetric_invariants(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22) noexcept {
    std::array<core::Real, 6> entries{
        a00, a01, a02, a11, a12, a22};
    core::Real maximum = 0.0;
    for (const core::Real value : entries) {
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum == 0.0) return {true, true, true};

    int maximum_exponent = 0;
    (void)std::frexp(maximum, &maximum_exponent);
    const int shift = -maximum_exponent;
    for (core::Real& value : entries) {
        if (value == 0.0) continue;
        const core::Real scaled = std::scalbn(value, shift);
        if (!std::isfinite(scaled)
            || scaled == 0.0
            || std::scalbn(scaled, -shift) != value) {
            return {};
        }
        value = scaled;
    }
    a00 = entries[0];
    a01 = entries[1];
    a02 = entries[2];
    a11 = entries[3];
    a12 = entries[4];
    a22 = entries[5];

    FloatingExpansion determinant;
    if (!add_exact_triple_product(determinant, a00, a11, a22, 1.0)
        || !add_exact_triple_product(determinant, a01, a12, a02, 1.0)
        || !add_exact_triple_product(determinant, a02, a01, a12, 1.0)
        || !add_exact_triple_product(determinant, a02, a11, a02, -1.0)
        || !add_exact_triple_product(determinant, a01, a01, a22, -1.0)
        || !add_exact_triple_product(determinant, a00, a12, a12, -1.0)) {
        return {};
    }

    FloatingExpansion second_coefficient;
    if (!add_exact_product(second_coefficient, a00, a11)
        || !add_exact_product(second_coefficient, a00, a22)
        || !add_exact_product(second_coefficient, a11, a22)
        || !add_exact_product(second_coefficient, a01, a01, -1.0)
        || !add_exact_product(second_coefficient, a02, a02, -1.0)
        || !add_exact_product(second_coefficient, a12, a12, -1.0)) {
        return {};
    }
    return {
        true,
        determinant.size == 0,
        second_coefficient.size == 0};
}

struct WideEigensystem3 {
    std::array<WideReal, 3> values{};
    std::array<WideVec3, 3> vectors{
        WideVec3{1.0L, 0.0L, 0.0L},
        WideVec3{0.0L, 1.0L, 0.0L},
        WideVec3{0.0L, 0.0L, 1.0L}};
};

MatrixState matrix_state(const WideReal matrix[3][3]) noexcept {
    return {
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][1], matrix[1][2], matrix[2][2]};
}

WideReal representable_jacobi_tangent(
    WideReal app,
    WideReal aqq,
    WideReal apq) {
    if (apq == 0.0L) return 0.0L;
    const WideReal delta = 0.5L * (aqq - app);
    const WideReal denominator = delta + std::copysign(
        std::hypot(delta, apq),
        delta == 0.0L ? WideReal{1.0L} : delta);
    if (!std::isfinite(denominator) || denominator == 0.0L) {
        throw std::overflow_error(
            "Symmetric Jacobi rotation denominator is not representable");
    }
    const WideReal tangent = apq / denominator;
    if (!std::isfinite(tangent)) {
        throw std::overflow_error(
            "Symmetric Jacobi rotation is not representable");
    }
    // A nonzero represented coupling whose rotation rounds to exactly zero has
    // no representable effect on the diagonalization at this precision. This is
    // a representation boundary, not an empirical epsilon threshold.
    return tangent;
}

void sort_eigensystem(WideEigensystem3& system) {
    std::array<std::size_t, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return system.values[lhs] > system.values[rhs];
    });
    const auto values = system.values;
    const auto vectors = system.vectors;
    for (std::size_t index = 0; index < order.size(); ++index) {
        system.values[index] = values[order[index]];
        system.vectors[index] = vectors[order[index]];
    }
}

void restore_proven_exact_zeros(
    WideEigensystem3& system,
    const ExactSymmetricInvariants& invariants) noexcept {
    if (!invariants.known || !invariants.determinant_zero) return;
    if (invariants.second_coefficient_zero) {
        std::size_t nonzero_index = 0;
        if (std::abs(system.values[1]) > std::abs(system.values[nonzero_index])) {
            nonzero_index = 1;
        }
        if (std::abs(system.values[2]) > std::abs(system.values[nonzero_index])) {
            nonzero_index = 2;
        }
        for (std::size_t index = 0; index < system.values.size(); ++index) {
            if (index != nonzero_index) system.values[index] = 0.0L;
        }
    } else {
        std::size_t zero_index = 0;
        if (std::abs(system.values[1]) < std::abs(system.values[zero_index])) {
            zero_index = 1;
        }
        if (std::abs(system.values[2]) < std::abs(system.values[zero_index])) {
            zero_index = 2;
        }
        system.values[zero_index] = 0.0L;
    }
}

WideEigensystem3 jacobi_eigensystem(
    WideReal a00,
    WideReal a01,
    WideReal a02,
    WideReal a11,
    WideReal a12,
    WideReal a22) {
    WideReal matrix[3][3]{
        {a00, a01, a02},
        {a01, a11, a12},
        {a02, a12, a22}};
    WideReal vectors[3][3]{
        {1.0L, 0.0L, 0.0L},
        {0.0L, 1.0L, 0.0L},
        {0.0L, 0.0L, 1.0L}};
    std::set<MatrixState> visited;

    for (;;) {
        const MatrixState state = matrix_state(matrix);
        if (!visited.insert(state).second) {
            throw std::runtime_error(
                "Symmetric Jacobi eigensystem entered a represented-state cycle");
        }

        int p = -1;
        int q = -1;
        WideReal selected_tangent = 0.0L;
        WideReal largest_resolvable_coupling = 0.0L;
        const auto consider_pair = [&](int candidate_p, int candidate_q) {
            const WideReal apq = matrix[candidate_p][candidate_q];
            const WideReal magnitude = std::abs(apq);
            if (magnitude == 0.0L) return;
            const WideReal tangent = representable_jacobi_tangent(
                matrix[candidate_p][candidate_p],
                matrix[candidate_q][candidate_q],
                apq);
            if (tangent == 0.0L) return;
            if (p < 0 || magnitude > largest_resolvable_coupling) {
                p = candidate_p;
                q = candidate_q;
                selected_tangent = tangent;
                largest_resolvable_coupling = magnitude;
            }
        };
        consider_pair(0, 1);
        consider_pair(0, 2);
        consider_pair(1, 2);
        if (p < 0) break;

        const WideReal app = matrix[p][p];
        const WideReal aqq = matrix[q][q];
        const WideReal apq = matrix[p][q];
        const WideReal tangent = selected_tangent;
        const WideReal cosine = 1.0L / std::hypot(1.0L, tangent);
        const WideReal sine = tangent * cosine;
        if (!std::isfinite(cosine) || !std::isfinite(sine)) {
            throw std::overflow_error(
                "Symmetric Jacobi rotation coefficients are not representable");
        }

        const WideReal new_app = app - tangent * apq;
        const WideReal new_aqq = aqq + tangent * apq;
        if (!std::isfinite(new_app) || !std::isfinite(new_aqq)) {
            throw std::overflow_error(
                "Symmetric Jacobi diagonal update is not representable");
        }
        matrix[p][p] = new_app;
        matrix[q][q] = new_aqq;
        matrix[p][q] = 0.0L;
        matrix[q][p] = 0.0L;

        for (int row = 0; row < 3; ++row) {
            if (row == p || row == q) continue;
            const WideReal arp = matrix[row][p];
            const WideReal arq = matrix[row][q];
            const WideReal rotated_p = cosine * arp - sine * arq;
            const WideReal rotated_q = sine * arp + cosine * arq;
            if (!std::isfinite(rotated_p) || !std::isfinite(rotated_q)) {
                throw std::overflow_error(
                    "Symmetric Jacobi rotation produced a non-finite coupling");
            }
            matrix[row][p] = rotated_p;
            matrix[p][row] = rotated_p;
            matrix[row][q] = rotated_q;
            matrix[q][row] = rotated_q;
        }

        for (int row = 0; row < 3; ++row) {
            const WideReal vrp = vectors[row][p];
            const WideReal vrq = vectors[row][q];
            const WideReal rotated_p = cosine * vrp - sine * vrq;
            const WideReal rotated_q = sine * vrp + cosine * vrq;
            if (!std::isfinite(rotated_p) || !std::isfinite(rotated_q)) {
                throw std::overflow_error(
                    "Symmetric Jacobi eigenvector rotation is not representable");
            }
            vectors[row][p] = rotated_p;
            vectors[row][q] = rotated_q;
        }
    }

    WideEigensystem3 result;
    for (std::size_t index = 0; index < 3; ++index) {
        result.values[index] = matrix[index][index];
        result.vectors[index] = {
            vectors[0][index], vectors[1][index], vectors[2][index]};
    }
    return result;
}

WideEigensystem3 normalized_eigensystem(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22) {
    const WideReal scale = std::max({
        std::abs(static_cast<WideReal>(a00)),
        std::abs(static_cast<WideReal>(a01)),
        std::abs(static_cast<WideReal>(a02)),
        std::abs(static_cast<WideReal>(a11)),
        std::abs(static_cast<WideReal>(a12)),
        std::abs(static_cast<WideReal>(a22))});
    if (scale == 0.0L) return {};

    WideEigensystem3 result = jacobi_eigensystem(
        static_cast<WideReal>(a00) / scale,
        static_cast<WideReal>(a01) / scale,
        static_cast<WideReal>(a02) / scale,
        static_cast<WideReal>(a11) / scale,
        static_cast<WideReal>(a12) / scale,
        static_cast<WideReal>(a22) / scale);
    restore_proven_exact_zeros(
        result,
        exact_symmetric_invariants(a00, a01, a02, a11, a12, a22));
    for (WideReal& value : result.values) value *= scale;
    sort_eigensystem(result);
    return result;
}

core::Vec3 normalized_core_vector(const WideVec3& wide) {
    const WideReal norm = std::hypot(wide[0], std::hypot(wide[1], wide[2]));
    if (!std::isfinite(norm) || norm == 0.0L) {
        throw std::runtime_error(
            "Symmetric Jacobi eigenvector has no representable direction");
    }
    core::Vec3 result{
        checked_real(wide[0] / norm, "Symmetric eigenvector x"),
        checked_real(wide[1] / norm, "Symmetric eigenvector y"),
        checked_real(wide[2] / norm, "Symmetric eigenvector z")};
    const core::Real narrowed_norm = core::scale_safe_norm3(
        result.x, result.y, result.z);
    if (!std::isfinite(narrowed_norm) || narrowed_norm == 0.0) {
        throw std::runtime_error(
            "Symmetric eigenvector normalization is not representable");
    }
    result /= narrowed_norm;
    return canonicalize(result);
}

} // namespace

SymmetricEigenvalues3 symmetric_eigenvalues_3x3(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22) {
    require_finite_matrix(a00, a01, a02, a11, a12, a22);
    const WideEigensystem3 system = normalized_eigensystem(
        a00, a01, a02, a11, a12, a22);
    return {
        checked_eigenvalue(system.values[0], "Largest symmetric eigenvalue"),
        checked_eigenvalue(system.values[1], "Middle symmetric eigenvalue"),
        checked_eigenvalue(system.values[2], "Smallest symmetric eigenvalue")};
}

std::optional<core::Vec3> symmetric_eigenvector_3x3(
    core::Real a00,
    core::Real a01,
    core::Real a02,
    core::Real a11,
    core::Real a12,
    core::Real a22,
    core::Real eigenvalue) {
    require_finite_matrix(a00, a01, a02, a11, a12, a22, eigenvalue);
    const WideEigensystem3 system = normalized_eigensystem(
        a00, a01, a02, a11, a12, a22);

    std::array<core::Real, 3> represented_values{
        checked_eigenvalue(system.values[0], "Largest symmetric eigenvalue"),
        checked_eigenvalue(system.values[1], "Middle symmetric eigenvalue"),
        checked_eigenvalue(system.values[2], "Smallest symmetric eigenvalue")};
    std::size_t matching_index = 0;
    std::size_t matching_count = 0;
    for (std::size_t index = 0; index < represented_values.size(); ++index) {
        if (represented_values[index] == eigenvalue) {
            matching_index = index;
            ++matching_count;
        }
    }
    // Repeated eigenvalues do not define a unique direction. Requiring an exact
    // match to a represented eigenvalue also prevents an externally rounded or
    // approximate lambda from silently selecting a nearby eigendirection.
    if (matching_count != 1) return std::nullopt;
    return normalized_core_vector(system.vectors[matching_index]);
}

} // namespace cosmo_nbody::analysis
