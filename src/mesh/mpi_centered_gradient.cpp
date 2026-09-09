#include "cosmo_nbody/mesh/mpi_centered_gradient.hpp"

#include "cic_coordinate.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::mesh {
namespace {

std::size_t checked_slab_gradient_product(
    std::size_t lhs,
    std::size_t rhs,
    const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(
            std::string("Distributed centered-gradient ") + label
            + " overflows size_t");
    }
    return lhs * rhs;
}

struct GatheredSlabGradient {
    std::array<core::Real, 3> value{0.0, 0.0, 0.0};
    bool lost_nonzero_product{false};
};

} // namespace

void interpolate_centered_gradient3_add_slab(
    const MeshGeometry& geometry,
    const RealField& potential,
    std::span<const PotentialHaloPlane> halo_plan,
    std::span<const core::Real> halo_values,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> out_x,
    std::span<core::Real> out_y,
    std::span<core::Real> out_z,
    bool validate_only) {
    const std::size_t count = pos_x.size();
    if (pos_y.size() != count || pos_z.size() != count
        || out_x.size() != count || out_y.size() != count
        || out_z.size() != count) {
        throw std::invalid_argument(
            "Distributed centered-gradient component sizes must match");
    }

    const std::size_t n = geometry.grid_size();
    if (n < 2 || geometry.local_n0() == 0
        || geometry.local_0_start() > n - geometry.local_n0()) {
        throw std::invalid_argument(
            "Distributed centered-gradient geometry is invalid");
    }
    if (potential.size() != geometry.real_size()) {
        throw std::invalid_argument(
            "Distributed centered-gradient potential size does not match geometry");
    }
    const core::Real dx = geometry.cell_size();
    if (!std::isfinite(dx) || dx <= 0.0) {
        throw std::runtime_error(
            "Distributed centered-gradient mesh spacing is invalid");
    }

    const std::size_t plane_cells = checked_slab_gradient_product(
        n, n, "plane size");
    std::array<std::size_t, 3> expected_planes{};
    std::size_t expected_count = 0;
    const std::size_t candidates[3] = {
        (geometry.local_0_start() + n - 1) % n,
        (geometry.local_0_start() + geometry.local_n0()) % n,
        (geometry.local_0_start() + geometry.local_n0() + 1) % n};
    for (const std::size_t candidate : candidates) {
        bool duplicate = false;
        for (std::size_t index = 0; index < expected_count; ++index) {
            duplicate = duplicate || expected_planes[index] == candidate;
        }
        if (!duplicate) expected_planes[expected_count++] = candidate;
    }
    if (halo_plan.size() != expected_count) {
        throw std::invalid_argument(
            "Distributed centered-gradient halo plan has the wrong size");
    }
    for (std::size_t index = 0; index < expected_count; ++index) {
        if (halo_plan[index].global_plane != expected_planes[index]) {
            throw std::invalid_argument(
                "Distributed centered-gradient halo plan has the wrong plane order");
        }
    }
    const std::size_t expected_halo_values = checked_slab_gradient_product(
        expected_count, plane_cells, "halo storage");
    if (halo_values.size() != expected_halo_values) {
        throw std::invalid_argument(
            "Distributed centered-gradient halo storage has the wrong size");
    }

    const std::size_t local_begin = geometry.local_0_start();
    const std::size_t local_end = local_begin + geometry.local_n0();
    const auto plane_data = [&](std::size_t global_x) noexcept
        -> const core::Real* {
        if (global_x >= local_begin && global_x < local_end) {
            return potential.data() + (global_x - local_begin) * plane_cells;
        }
        for (std::size_t slot = 0; slot < halo_plan.size(); ++slot) {
            if (halo_plan[slot].global_plane == global_x) {
                return halo_values.data() + slot * plane_cells;
            }
        }
        return nullptr;
    };

    const core::Real box_size = geometry.box_size();
    // Validate all particle-local inputs and prove that every potential plane
    // needed by both CIC support nodes and their x derivatives is available.
    for (std::size_t particle = 0; particle < count; ++particle) {
        if (!std::isfinite(pos_x[particle])
            || !std::isfinite(pos_y[particle])
            || !std::isfinite(pos_z[particle])) {
            throw std::invalid_argument(
                "Distributed centered-gradient positions must be finite");
        }
        if (!std::isfinite(out_x[particle])
            || !std::isfinite(out_y[particle])
            || !std::isfinite(out_z[particle])) {
            throw std::invalid_argument(
                "Distributed centered-gradient input acceleration must be finite");
        }
        const auto x_coordinate = detail::cic_coordinate_1d(
            pos_x[particle], box_size, dx, n);
        const std::size_t gx0 = geometry.wrap_index(x_coordinate.base);
        if (gx0 < local_begin || gx0 >= local_end) {
            throw std::invalid_argument(
                "Distributed centered-gradient particle is outside its owning slab");
        }
        for (int offset = 0; offset < 2; ++offset) {
            const std::size_t gx = geometry.wrap_index(
                x_coordinate.base + offset);
            const std::size_t plus = geometry.wrap_index(
                static_cast<std::int64_t>(gx) + 1);
            const std::size_t minus = geometry.wrap_index(
                static_cast<std::int64_t>(gx) - 1);
            if (!plane_data(gx) || !plane_data(plus) || !plane_data(minus)) {
                throw std::logic_error(
                    "Distributed centered-gradient halo does not cover CIC stencil support");
            }
        }
    }

    for (const core::Real value : potential) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Distributed centered-gradient potential must be finite");
        }
    }
    for (const core::Real value : halo_values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Distributed centered-gradient halo potential must be finite");
        }
    }

    const core::Real two_dx = 2.0 * dx;
    if (!std::isfinite(two_dx) || two_dx <= 0.0) {
        throw std::overflow_error(
            "Distributed centered-gradient two-cell spacing is not representable");
    }
    const auto gathered_particle = [&](std::size_t particle) {
        const auto x_coordinate = detail::cic_coordinate_1d(
            pos_x[particle], box_size, dx, n);
        const auto y_coordinate = detail::cic_coordinate_1d(
            pos_y[particle], box_size, dx, n);
        const auto z_coordinate = detail::cic_coordinate_1d(
            pos_z[particle], box_size, dx, n);
        const core::Real wx[2] = {
            1.0 - x_coordinate.fraction,
            x_coordinate.fraction};
        const core::Real wy[2] = {
            1.0 - y_coordinate.fraction,
            y_coordinate.fraction};
        const core::Real wz[2] = {
            1.0 - z_coordinate.fraction,
            z_coordinate.fraction};
        GatheredSlabGradient gathered;

        for (int ox = 0; ox < 2; ++ox) {
            const std::size_t gx = geometry.wrap_index(
                x_coordinate.base + ox);
            const std::size_t gx_plus = geometry.wrap_index(
                static_cast<std::int64_t>(gx) + 1);
            const std::size_t gx_minus = geometry.wrap_index(
                static_cast<std::int64_t>(gx) - 1);
            const core::Real* plane = plane_data(gx);
            const core::Real* plus_plane = plane_data(gx_plus);
            const core::Real* minus_plane = plane_data(gx_minus);
            for (int oy = 0; oy < 2; ++oy) {
                const std::size_t gy = geometry.wrap_index(
                    y_coordinate.base + oy);
                const std::size_t gy_plus = geometry.wrap_index(
                    static_cast<std::int64_t>(gy) + 1);
                const std::size_t gy_minus = geometry.wrap_index(
                    static_cast<std::int64_t>(gy) - 1);
                for (int oz = 0; oz < 2; ++oz) {
                    const std::size_t gz = geometry.wrap_index(
                        z_coordinate.base + oz);
                    const std::size_t gz_plus = geometry.wrap_index(
                        static_cast<std::int64_t>(gz) + 1);
                    const std::size_t gz_minus = geometry.wrap_index(
                        static_cast<std::int64_t>(gz) - 1);
                    const auto weight_result = detail::cic_weight_3d(
                        wx[ox], wy[oy], wz[oz]);
                    const core::Real weight = weight_result.value;
                    const bool mathematical_weight_nonzero =
                        weight != 0.0 || weight_result.lost_nonzero_product;
                    const core::Real difference_x =
                        plus_plane[gy * n + gz]
                        - minus_plane[gy * n + gz];
                    const core::Real difference_y =
                        plane[gy_plus * n + gz]
                        - plane[gy_minus * n + gz];
                    const core::Real difference_z =
                        plane[gy * n + gz_plus]
                        - plane[gy * n + gz_minus];
                    const core::Real gradient_x = -difference_x / two_dx;
                    const core::Real gradient_y = -difference_y / two_dx;
                    const core::Real gradient_z = -difference_z / two_dx;
                    if (mathematical_weight_nonzero
                        && ((difference_x != 0.0 && gradient_x == 0.0)
                            || (difference_y != 0.0 && gradient_y == 0.0)
                            || (difference_z != 0.0 && gradient_z == 0.0))) {
                        gathered.lost_nonzero_product = true;
                    }
                    if (weight_result.lost_nonzero_product
                        && (gradient_x != 0.0
                            || gradient_y != 0.0
                            || gradient_z != 0.0)) {
                        gathered.lost_nonzero_product = true;
                    }
                    const core::Real contribution_x = weight * gradient_x;
                    const core::Real contribution_y = weight * gradient_y;
                    const core::Real contribution_z = weight * gradient_z;
                    if (weight != 0.0
                        && ((gradient_x != 0.0 && contribution_x == 0.0)
                            || (gradient_y != 0.0 && contribution_y == 0.0)
                            || (gradient_z != 0.0 && contribution_z == 0.0))) {
                        gathered.lost_nonzero_product = true;
                    }
                    gathered.value[0] += contribution_x;
                    gathered.value[1] += contribution_y;
                    gathered.value[2] += contribution_z;
                }
            }
        }
        return gathered;
    };

    // No output is changed until every final addition has been proven finite and
    // the slab arithmetic has not silently erased a non-zero CIC contribution.
    for (std::size_t particle = 0; particle < count; ++particle) {
        const auto gathered = gathered_particle(particle);
        if (gathered.lost_nonzero_product) {
            throw std::underflow_error(
                "Distributed centered-gradient gather lost a non-zero CIC weight, gradient, or weighted contribution in core::Real");
        }
        if (!std::isfinite(out_x[particle] + gathered.value[0])
            || !std::isfinite(out_y[particle] + gathered.value[1])
            || !std::isfinite(out_z[particle] + gathered.value[2])) {
            throw std::overflow_error(
                "Distributed centered-gradient gathered acceleration is non-finite");
        }
    }
    if (validate_only) return;

    for (std::size_t particle = 0; particle < count; ++particle) {
        const auto gathered = gathered_particle(particle);
        if (gathered.lost_nonzero_product) {
            throw std::logic_error(
                "Distributed centered-gradient arithmetic changed after validation");
        }
        out_x[particle] += gathered.value[0];
        out_y[particle] += gathered.value[1];
        out_z[particle] += gathered.value[2];
    }
}

} // namespace cosmo_nbody::mesh
