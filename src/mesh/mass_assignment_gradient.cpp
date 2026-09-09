#include "cosmo_nbody/mesh/mass_assignment.hpp"

#include "cic_coordinate.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cosmo_nbody {
namespace mesh {

void CICMassAssignment::require_finite_full_mesh_field(
    const RealField& field) const {
    if (geom_.local_n0() != geom_.grid_size()
        || geom_.local_0_start() != 0) {
        throw std::invalid_argument(
            "CIC full-mesh validation requires a full periodic mesh");
    }
    if (field.size() != geom_.padded_real_size()) {
        throw std::invalid_argument(
            "CIC centered-gradient potential size does not match mesh geometry");
    }

    const std::size_t n = geom_.grid_size();
    int invalid = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) reduction(|: invalid)
#endif
    for (std::size_t ix = 0; ix < n; ++ix) {
        for (std::size_t iy = 0; iy < n; ++iy) {
            for (std::size_t iz = 0; iz < n; ++iz) {
                if (!std::isfinite(field[geom_.real_index(ix, iy, iz)])) {
                    invalid = 1;
                }
            }
        }
    }
    if (invalid != 0) {
        throw std::invalid_argument(
            "CIC centered-gradient potential must be finite");
    }
}

void CICMassAssignment::interpolate_centered_gradient3_add_full_mesh(
    const RealField& potential,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> out_x,
    std::span<core::Real> out_y,
    std::span<core::Real> out_z) const {
    const std::size_t count = pos_x.size();
    if (pos_y.size() != count || pos_z.size() != count
        || out_x.size() != count || out_y.size() != count
        || out_z.size() != count) {
        throw std::invalid_argument(
            "CIC all-axis centered-gradient component sizes must match");
    }
    if (geom_.local_n0() != geom_.grid_size()
        || geom_.local_0_start() != 0) {
        throw std::invalid_argument(
            "All-axis centered-gradient CIC gather requires a full periodic mesh");
    }
    if (potential.size() != geom_.padded_real_size()) {
        throw std::invalid_argument(
            "CIC centered-gradient potential size does not match mesh geometry");
    }
    if (!std::isfinite(dx_) || dx_ <= 0.0) {
        throw std::runtime_error(
            "CIC centered-gradient mesh spacing is invalid");
    }

    std::size_t first_invalid_position = count;
    std::size_t first_invalid_output = count;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        reduction(min: first_invalid_position, first_invalid_output) \
        if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t particle = 0; particle < count; ++particle) {
        if (!std::isfinite(pos_x[particle])
            || !std::isfinite(pos_y[particle])
            || !std::isfinite(pos_z[particle])) {
            first_invalid_position = std::min(
                first_invalid_position, particle);
        }
        if (!std::isfinite(out_x[particle])
            || !std::isfinite(out_y[particle])
            || !std::isfinite(out_z[particle])) {
            first_invalid_output = std::min(
                first_invalid_output, particle);
        }
    }
    if (first_invalid_position <= first_invalid_output
        && first_invalid_position != count) {
        throw std::invalid_argument(
            "CIC centered-gradient gather positions must be finite");
    }
    if (first_invalid_output != count) {
        throw std::invalid_argument(
            "CIC centered-gradient input acceleration must be finite");
    }
    require_finite_full_mesh_field(potential);

    const core::Real box_size = geom_.box_size();
    const core::Real two_dx = 2.0 * dx_;
    const std::size_t N = geom_.grid_size();
    int lost_nonzero_gradient_product = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        reduction(|: lost_nonzero_gradient_product)
#endif
    for (std::size_t particle = 0; particle < count; ++particle) {
        const auto x_coordinate = detail::cic_coordinate_1d(
            pos_x[particle], box_size, dx_, N);
        const auto y_coordinate = detail::cic_coordinate_1d(
            pos_y[particle], box_size, dx_, N);
        const auto z_coordinate = detail::cic_coordinate_1d(
            pos_z[particle], box_size, dx_, N);
        const auto ix = x_coordinate.base;
        const auto iy = y_coordinate.base;
        const auto iz = z_coordinate.base;
        const core::Real wx[2] = {
            1.0 - x_coordinate.fraction,
            x_coordinate.fraction};
        const core::Real wy[2] = {
            1.0 - y_coordinate.fraction,
            y_coordinate.fraction};
        const core::Real wz[2] = {
            1.0 - z_coordinate.fraction,
            z_coordinate.fraction};
        core::Real gathered_x = 0.0;
        core::Real gathered_y = 0.0;
        core::Real gathered_z = 0.0;

        for (int ox = 0; ox < 2; ++ox) {
            const std::size_t gx = geom_.wrap_index(ix + ox);
            const std::size_t gx_plus = geom_.wrap_index(
                static_cast<std::int64_t>(gx) + 1);
            const std::size_t gx_minus = geom_.wrap_index(
                static_cast<std::int64_t>(gx) - 1);
            for (int oy = 0; oy < 2; ++oy) {
                const std::size_t gy = geom_.wrap_index(iy + oy);
                const std::size_t gy_plus = geom_.wrap_index(
                    static_cast<std::int64_t>(gy) + 1);
                const std::size_t gy_minus = geom_.wrap_index(
                    static_cast<std::int64_t>(gy) - 1);
                for (int oz = 0; oz < 2; ++oz) {
                    const std::size_t gz = geom_.wrap_index(iz + oz);
                    const std::size_t gz_plus = geom_.wrap_index(
                        static_cast<std::int64_t>(gz) + 1);
                    const std::size_t gz_minus = geom_.wrap_index(
                        static_cast<std::int64_t>(gz) - 1);
                    const auto weight_result = detail::cic_weight_3d(
                        wx[ox], wy[oy], wz[oz]);
                    const core::Real weight = weight_result.value;
                    const bool mathematical_weight_nonzero =
                        weight != 0.0 || weight_result.lost_nonzero_product;
                    // Form the centered quotient directly. Materializing 0.5/dx
                    // first can overflow for a subnormal but valid mesh spacing
                    // even when the final gradient is finite. This grouping also
                    // matches the distributed centered-gradient implementation.
                    const core::Real difference_x =
                        potential[geom_.real_index(gx_plus, gy, gz)]
                        - potential[geom_.real_index(gx_minus, gy, gz)];
                    const core::Real difference_y =
                        potential[geom_.real_index(gx, gy_plus, gz)]
                        - potential[geom_.real_index(gx, gy_minus, gz)];
                    const core::Real difference_z =
                        potential[geom_.real_index(gx, gy, gz_plus)]
                        - potential[geom_.real_index(gx, gy, gz_minus)];
                    const core::Real gradient_x = -difference_x / two_dx;
                    const core::Real gradient_y = -difference_y / two_dx;
                    const core::Real gradient_z = -difference_z / two_dx;
                    if (mathematical_weight_nonzero
                        && ((difference_x != 0.0 && gradient_x == 0.0)
                            || (difference_y != 0.0 && gradient_y == 0.0)
                            || (difference_z != 0.0 && gradient_z == 0.0))) {
                        lost_nonzero_gradient_product = 1;
                    }
                    if (weight_result.lost_nonzero_product
                        && (gradient_x != 0.0
                            || gradient_y != 0.0
                            || gradient_z != 0.0)) {
                        lost_nonzero_gradient_product = 1;
                    }
                    const core::Real contribution_x = weight * gradient_x;
                    const core::Real contribution_y = weight * gradient_y;
                    const core::Real contribution_z = weight * gradient_z;
                    if (weight != 0.0
                        && ((gradient_x != 0.0 && contribution_x == 0.0)
                            || (gradient_y != 0.0 && contribution_y == 0.0)
                            || (gradient_z != 0.0 && contribution_z == 0.0))) {
                        lost_nonzero_gradient_product = 1;
                    }
                    gathered_x += contribution_x;
                    gathered_y += contribution_y;
                    gathered_z += contribution_z;
                }
            }
        }
        out_x[particle] += gathered_x;
        out_y[particle] += gathered_y;
        out_z[particle] += gathered_z;
    }
    if (lost_nonzero_gradient_product != 0) {
        throw std::underflow_error(
            "CIC centered-gradient gather lost a non-zero CIC weight, gradient, or weighted contribution in core::Real");
    }

    int invalid_output = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: invalid_output) \
        if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t particle = 0; particle < count; ++particle) {
        invalid_output |= !std::isfinite(out_x[particle])
            || !std::isfinite(out_y[particle])
            || !std::isfinite(out_z[particle]);
    }
    if (invalid_output != 0) {
        throw std::overflow_error(
            "CIC centered-gradient gathered acceleration is non-finite");
    }
}

} // namespace mesh
} // namespace cosmo_nbody
