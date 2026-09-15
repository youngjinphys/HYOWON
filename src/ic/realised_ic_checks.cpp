#include "cosmo_nbody/ic/realised_ic_checks.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace ic {

namespace {

std::size_t checked_particle_cube(std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument(
            "Realised IC validation requires particles_per_dimension >= 1");
    }
    if (n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "Realised IC particle count N^3 overflows size_t");
    }
    const std::size_t n2 = n * n;
    if (n2 > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "Realised IC particle count N^3 overflows size_t");
    }
    return n2 * n;
}

core::Real periodic_displacement(
    core::Real realised,
    core::Real lattice,
    core::Real box_size) {
    core::Real delta = realised - lattice;
    delta -= box_size * std::nearbyint(delta / box_size);
    return delta;
}

// Non-negative scale+sumsq accumulator with compensated dimensionless merging.
// Blocks merge in index order for OpenMP-team-independent reduction without
// forming x*x.
class ScaledSquaredNorm {
public:
    void add(long double value) noexcept {
        const long double magnitude = std::abs(value);
        if (magnitude == 0.0L) return;
        if (scale_ < magnitude) {
            const long double ratio = scale_ / magnitude;
            const long double previous = total_sumsq() * ratio * ratio;
            scale_ = magnitude;
            sumsq_ = 1.0L;
            correction_ = 0.0L;
            add_sumsq(previous);
        } else {
            const long double ratio = magnitude / scale_;
            add_sumsq(ratio * ratio);
        }
    }

    void merge(const ScaledSquaredNorm& other) noexcept {
        if (other.scale_ == 0.0L) return;
        if (scale_ == 0.0L) {
            *this = other;
            return;
        }
        if (scale_ < other.scale_) {
            const long double ratio = scale_ / other.scale_;
            const long double previous = total_sumsq() * ratio * ratio;
            scale_ = other.scale_;
            sumsq_ = other.sumsq_;
            correction_ = other.correction_;
            add_sumsq(previous);
        } else {
            const long double ratio = other.scale_ / scale_;
            add_sumsq(other.total_sumsq() * ratio * ratio);
        }
    }

    [[nodiscard]] long double root_mean_square(
        long double sample_count) const noexcept {
        if (scale_ == 0.0L) return 0.0L;
        return scale_ * std::sqrt(total_sumsq() / sample_count);
    }

private:
    void add_sumsq(long double value) noexcept {
        const long double next = sumsq_ + value;
        if (std::abs(sumsq_) >= std::abs(value)) {
            correction_ += (sumsq_ - next) + value;
        } else {
            correction_ += (value - next) + sumsq_;
        }
        sumsq_ = next;
    }

    [[nodiscard]] long double total_sumsq() const noexcept {
        return sumsq_ + correction_;
    }

    long double scale_{0.0L};
    long double sumsq_{1.0L};
    long double correction_{0.0L};
};

core::Real representable_summary_value(
    long double value,
    const char* label) {
    if (!std::isfinite(value)
        || value < 0.0L
        || value > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::runtime_error(
            std::string("Generated initial-condition ") + label
            + " is not representable");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result) || (value != 0.0L && result == 0.0)) {
        throw std::runtime_error(
            std::string("Generated initial-condition ") + label
            + " underflows core::Real");
    }
    return result;
}

core::Real representable_signed_summary_value(
    long double value,
    const char* label) {
    if (!std::isfinite(value)
        || value < -static_cast<long double>(
            std::numeric_limits<core::Real>::max())
        || value > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::runtime_error(
            std::string("Generated initial-condition ") + label
            + " is not representable");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result) || (value != 0.0L && result == 0.0)) {
        throw std::runtime_error(
            std::string("Generated initial-condition ") + label
            + " underflows core::Real");
    }
    return result;
}

struct StructuralAccumulator {
    ScaledSquaredNorm displacement_norm;
    ScaledSquaredNorm momentum_norm;
    long double displacement_max{0.0L};
    long double momentum_max{0.0L};
    long double forward_cell_edge_determinant_min{
        std::numeric_limits<long double>::infinity()};
    long double forward_cell_edge_determinant_max{
        -std::numeric_limits<long double>::infinity()};
    std::size_t forward_cell_edge_determinant_evaluated_count{0};
    std::size_t forward_cell_edge_determinant_unevaluable_count{0};
    std::size_t forward_cell_edge_determinant_nonpositive_count{0};
    bool non_finite{false};
    bool out_of_box{false};
};

} // namespace

RealisedICSummary validate_realised_lattice_ic(
    std::size_t particles_per_dimension,
    core::Real box_size_Mpc_h,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> mom_x,
    std::span<const core::Real> mom_y,
    std::span<const core::Real> mom_z) {
    if (!std::isfinite(box_size_Mpc_h) || box_size_Mpc_h <= 0.0) {
        throw std::invalid_argument(
            "Realised IC validation requires a finite positive box size");
    }

    const std::size_t expected = checked_particle_cube(
        particles_per_dimension);
    if (pos_x.size() != expected || pos_y.size() != expected
        || pos_z.size() != expected || mom_x.size() != expected
        || mom_y.size() != expected || mom_z.size() != expected) {
        throw std::invalid_argument(
            "Realised IC phase-space arrays must all have N^3 elements");
    }

    const core::Real spacing = box_size_Mpc_h
        / static_cast<core::Real>(particles_per_dimension);
    const bool forward_cell_edge_requested = particles_per_dimension >= 3;
    const long double inverse_spacing = 1.0L
        / static_cast<long double>(spacing);
    if (!std::isfinite(inverse_spacing) || inverse_spacing <= 0.0L) {
        throw std::overflow_error(
            "Realised IC lattice spacing is not representable");
    }
    const std::size_t n2 = particles_per_dimension
        * particles_per_dimension;
    // Fixed blocks make the merge order independent of the OpenMP team. The
    // block-local scaled sums retain a full-population validation; no particle
    // sampling is introduced.
    constexpr std::size_t block_size = 1U << 18U;
    const std::size_t block_count = (expected - 1U) / block_size + 1U;
    std::vector<StructuralAccumulator> blocks(block_count);
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(block_count >= 2)
#endif
    for (std::size_t block = 0; block < block_count; ++block) {
        auto& accumulator = blocks[block];
        const std::size_t begin = block * block_size;
        const std::size_t end = begin + std::min(block_size, expected - begin);
        for (std::size_t index = begin; index < end; ++index) {
            const core::Real x = pos_x[index];
            const core::Real y = pos_y[index];
            const core::Real z = pos_z[index];
            const core::Real px = mom_x[index];
            const core::Real py = mom_y[index];
            const core::Real pz = mom_z[index];

            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
                || !std::isfinite(px) || !std::isfinite(py)
                || !std::isfinite(pz)) {
                accumulator.non_finite = true;
                continue;
            }
            if (x < 0.0 || x >= box_size_Mpc_h
                || y < 0.0 || y >= box_size_Mpc_h
                || z < 0.0 || z >= box_size_Mpc_h) {
                accumulator.out_of_box = true;
                continue;
            }

            const std::size_t i = index / n2;
            const std::size_t remainder = index % n2;
            const std::size_t j = remainder / particles_per_dimension;
            const std::size_t k = remainder % particles_per_dimension;
            const core::Real qx = static_cast<core::Real>(i) * spacing;
            const core::Real qy = static_cast<core::Real>(j) * spacing;
            const core::Real qz = static_cast<core::Real>(k) * spacing;

            const long double dx = static_cast<long double>(
                periodic_displacement(x, qx, box_size_Mpc_h));
            const long double dy = static_cast<long double>(
                periodic_displacement(y, qy, box_size_Mpc_h));
            const long double dz = static_cast<long double>(
                periodic_displacement(z, qz, box_size_Mpc_h));
            const long double lpx = static_cast<long double>(px);
            const long double lpy = static_cast<long double>(py);
            const long double lpz = static_cast<long double>(pz);

            accumulator.displacement_norm.add(dx);
            accumulator.displacement_norm.add(dy);
            accumulator.displacement_norm.add(dz);
            accumulator.momentum_norm.add(lpx);
            accumulator.momentum_norm.add(lpy);
            accumulator.momentum_norm.add(lpz);
            accumulator.displacement_max = std::max(
                accumulator.displacement_max,
                core::scale_safe_norm3(dx, dy, dz));
            accumulator.momentum_max = std::max(
                accumulator.momentum_max,
                core::scale_safe_norm3(lpx, lpy, lpz));

            if (forward_cell_edge_requested) {
                const std::size_t i_plus = i + 1 == particles_per_dimension
                    ? 0 : i + 1;
                const std::size_t j_plus = j + 1 == particles_per_dimension
                    ? 0 : j + 1;
                const std::size_t k_plus = k + 1 == particles_per_dimension
                    ? 0 : k + 1;
                const std::size_t x_plus =
                    (i_plus * particles_per_dimension + j)
                    * particles_per_dimension + k;
                const std::size_t y_plus =
                    (i * particles_per_dimension + j_plus)
                    * particles_per_dimension + k;
                const std::size_t z_plus =
                    (i * particles_per_dimension + j)
                    * particles_per_dimension + k_plus;

                const core::Vec3 origin{x, y, z};
                const core::Vec3 plus_points[3] = {
                    {pos_x[x_plus], pos_y[x_plus], pos_z[x_plus]},
                    {pos_x[y_plus], pos_y[y_plus], pos_z[y_plus]},
                    {pos_x[z_plus], pos_y[z_plus], pos_z[z_plus]},
                };
                long double columns[3][3]{};
                bool unique = true;
                for (int axis = 0; axis < 3; ++axis) {
                    unique = unique
                        && math::minimum_image_displacement_is_directionally_unique(
                            origin, plus_points[axis], box_size_Mpc_h);
                    const core::Vec3 separation = math::minimum_image_displacement(
                        origin, plus_points[axis], box_size_Mpc_h);
                    columns[axis][0] =
                        static_cast<long double>(separation.x) * inverse_spacing;
                    columns[axis][1] =
                        static_cast<long double>(separation.y) * inverse_spacing;
                    columns[axis][2] =
                        static_cast<long double>(separation.z) * inverse_spacing;
                }
                if (!unique
                    || !std::isfinite(columns[0][0])
                    || !std::isfinite(columns[0][1])
                    || !std::isfinite(columns[0][2])
                    || !std::isfinite(columns[1][0])
                    || !std::isfinite(columns[1][1])
                    || !std::isfinite(columns[1][2])
                    || !std::isfinite(columns[2][0])
                    || !std::isfinite(columns[2][1])
                    || !std::isfinite(columns[2][2])) {
                    ++accumulator.forward_cell_edge_determinant_unevaluable_count;
                    continue;
                }

                const long double a00 = columns[0][0];
                const long double a10 = columns[0][1];
                const long double a20 = columns[0][2];
                const long double a01 = columns[1][0];
                const long double a11 = columns[1][1];
                const long double a21 = columns[1][2];
                const long double a02 = columns[2][0];
                const long double a12 = columns[2][1];
                const long double a22 = columns[2][2];
                const long double determinant =
                    a00 * (a11 * a22 - a12 * a21)
                    - a01 * (a10 * a22 - a12 * a20)
                    + a02 * (a10 * a21 - a11 * a20);
                if (!std::isfinite(determinant)) {
                    ++accumulator.forward_cell_edge_determinant_unevaluable_count;
                    continue;
                }
                ++accumulator.forward_cell_edge_determinant_evaluated_count;
                accumulator.forward_cell_edge_determinant_min = std::min(
                    accumulator.forward_cell_edge_determinant_min, determinant);
                accumulator.forward_cell_edge_determinant_max = std::max(
                    accumulator.forward_cell_edge_determinant_max, determinant);
                if (determinant <= 0.0L) {
                    ++accumulator.forward_cell_edge_determinant_nonpositive_count;
                }
            }
        }
    }

    ScaledSquaredNorm displacement_norm;
    ScaledSquaredNorm momentum_norm;
    long double displacement_max = 0.0L;
    long double momentum_max = 0.0L;
    long double forward_cell_edge_determinant_min =
        std::numeric_limits<long double>::infinity();
    long double forward_cell_edge_determinant_max =
        -std::numeric_limits<long double>::infinity();
    std::size_t forward_cell_edge_determinant_evaluated_count = 0;
    std::size_t forward_cell_edge_determinant_unevaluable_count = 0;
    std::size_t forward_cell_edge_determinant_nonpositive_count = 0;
    for (const auto& block : blocks) {
        if (block.non_finite) {
            throw std::runtime_error(
                "Generated initial conditions contain non-finite phase-space values");
        }
        if (block.out_of_box) {
            throw std::runtime_error(
                "Generated initial-condition positions must lie in [0, L)");
        }
        displacement_norm.merge(block.displacement_norm);
        momentum_norm.merge(block.momentum_norm);
        displacement_max = std::max(displacement_max, block.displacement_max);
        momentum_max = std::max(momentum_max, block.momentum_max);

        if (block.forward_cell_edge_determinant_evaluated_count != 0U) {
            forward_cell_edge_determinant_min = std::min(
                forward_cell_edge_determinant_min,
                block.forward_cell_edge_determinant_min);
            forward_cell_edge_determinant_max = std::max(
                forward_cell_edge_determinant_max,
                block.forward_cell_edge_determinant_max);
        }
        if (block.forward_cell_edge_determinant_evaluated_count
                > std::numeric_limits<std::size_t>::max()
                    - forward_cell_edge_determinant_evaluated_count
            || block.forward_cell_edge_determinant_unevaluable_count
                > std::numeric_limits<std::size_t>::max()
                    - forward_cell_edge_determinant_unevaluable_count
            || block.forward_cell_edge_determinant_nonpositive_count
                > std::numeric_limits<std::size_t>::max()
                    - forward_cell_edge_determinant_nonpositive_count) {
            throw std::overflow_error(
                "Generated initial-condition cell-edge diagnostic count overflow");
        }
        forward_cell_edge_determinant_evaluated_count +=
            block.forward_cell_edge_determinant_evaluated_count;
        forward_cell_edge_determinant_unevaluable_count +=
            block.forward_cell_edge_determinant_unevaluable_count;
        forward_cell_edge_determinant_nonpositive_count +=
            block.forward_cell_edge_determinant_nonpositive_count;
    }

    if (forward_cell_edge_requested) {
        if (forward_cell_edge_determinant_evaluated_count > expected
            || forward_cell_edge_determinant_unevaluable_count
                != expected - forward_cell_edge_determinant_evaluated_count) {
            throw std::logic_error(
                "Generated initial-condition cell-edge diagnostic population accounting is inconsistent");
        }
        if (forward_cell_edge_determinant_nonpositive_count
            > forward_cell_edge_determinant_evaluated_count) {
            throw std::logic_error(
                "Generated initial-condition nonpositive cell-edge count exceeds evaluated count");
        }
    } else if (forward_cell_edge_determinant_evaluated_count != 0U
        || forward_cell_edge_determinant_unevaluable_count != 0U
        || forward_cell_edge_determinant_nonpositive_count != 0U) {
        throw std::logic_error(
            "Generated initial-condition cell-edge diagnostic ran when not requested");
    }

    const long double count = static_cast<long double>(expected);
    const long double displacement_rms =
        displacement_norm.root_mean_square(count);
    const long double momentum_rms =
        momentum_norm.root_mean_square(count);

    RealisedICSummary summary;
    summary.particle_count = expected;
    summary.displacement_rms_Mpc_h = representable_summary_value(
        displacement_rms, "displacement RMS");
    summary.displacement_max_Mpc_h = representable_summary_value(
        displacement_max, "maximum displacement");
    summary.momentum_rms = representable_summary_value(
        momentum_rms, "momentum RMS");
    summary.momentum_max = representable_summary_value(
        momentum_max, "maximum momentum");
    summary.forward_cell_edge_determinant_requested =
        forward_cell_edge_requested;
    summary.forward_cell_edge_determinant_evaluated_count =
        forward_cell_edge_determinant_evaluated_count;
    summary.forward_cell_edge_determinant_unevaluable_count =
        forward_cell_edge_determinant_unevaluable_count;
    summary.forward_cell_edge_determinant_nonpositive_count =
        forward_cell_edge_determinant_nonpositive_count;
    if (forward_cell_edge_determinant_evaluated_count != 0U) {
        summary.forward_cell_edge_determinant_min =
            representable_signed_summary_value(
                forward_cell_edge_determinant_min,
                "minimum forward cell-edge determinant");
        summary.forward_cell_edge_determinant_max =
            representable_signed_summary_value(
                forward_cell_edge_determinant_max,
                "maximum forward cell-edge determinant");
    }
    return summary;
}

} // namespace ic
} // namespace cosmo_nbody
