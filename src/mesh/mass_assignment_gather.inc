#include "cosmo_nbody/runtime/thread_policy.hpp"

namespace detail {

struct CICGatherSupport {
    std::size_t grid_size{0};
    std::size_t local_start{0};
    std::size_t local_end{0};
    bool decomposed{false};
};

struct CICGatherValue {
    core::Real value{0.0};
    bool lost_nonzero_product{false};
    bool escaped_support{false};
};

CICGatherSupport prepare_cic_gather_support(
    const MeshGeometry& geom,
    core::Real cell_size,
    const RealField& field,
    const RealField* right_ghost,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z) {
    require_position_sizes(pos_x, pos_y, pos_z);
    require_finite_positions(pos_x, pos_y, pos_z);
    if (field.size() != geom.padded_real_size()) {
        throw std::invalid_argument(
            "CIC interpolation field size does not match mesh geometry");
    }

    const std::size_t N = geom.grid_size();
    if (N == 0) {
        throw std::logic_error("CIC interpolation mesh must be non-empty");
    }
    if (N > std::numeric_limits<std::size_t>::max() / N) {
        throw std::overflow_error(
            "CIC interpolation ghost-plane size overflows size_t");
    }
    const std::size_t ghost_plane_size = N * N;
    if (right_ghost && right_ghost->size() != ghost_plane_size) {
        throw std::invalid_argument(
            "CIC right ghost plane has incorrect size");
    }

    const std::size_t local_start = geom.local_0_start();
    const std::size_t local_end = local_start + geom.local_n0();
    const bool decomposed = geom.local_n0() < N;
    if (decomposed && !right_ghost) {
        throw std::invalid_argument(
            "Decomposed CIC interpolation requires a right ghost plane");
    }

    // Domain decomposition assigns each particle by the wrapped base x-cell.
    // CIC gather therefore needs only that local base plane and its right
    // neighbour. Prove this once before any caller mutates output or accumulates
    // a diagnostic so both gather APIs share the same ownership definition.
    if (decomposed) {
        const core::Real L = geom.box_size();
        std::size_t first_foreign = pos_x.size();
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) reduction(min: first_foreign) \
            if(runtime::should_use_host_parallel_team(pos_x.size()))
#endif
        for (std::size_t particle = 0; particle < pos_x.size(); ++particle) {
            const auto coordinate = cic_coordinate_1d(
                pos_x[particle], L, cell_size, N);
            const std::size_t base =
                static_cast<std::size_t>(coordinate.base);
            if (base < local_start || base >= local_end) {
                first_foreign = std::min(first_foreign, particle);
            }
        }
        if (first_foreign != pos_x.size()) {
            throw std::invalid_argument(
                "CIC particle base cell is outside the local mesh slab");
        }
    }

    return CICGatherSupport{N, local_start, local_end, decomposed};
}

CICGatherValue gather_cic_particle_value(
    const MeshGeometry& geom,
    core::Real cell_size,
    const CICGatherSupport& support,
    const RealField& field,
    const RealField* right_ghost,
    core::Real x,
    core::Real y,
    core::Real z) {
    const core::Real L = geom.box_size();
    const std::size_t N = support.grid_size;
    const auto x_coordinate = cic_coordinate_1d(x, L, cell_size, N);
    const auto y_coordinate = cic_coordinate_1d(y, L, cell_size, N);
    const auto z_coordinate = cic_coordinate_1d(z, L, cell_size, N);
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

    CICGatherValue result;
    for (int x_offset = 0; x_offset < 2; ++x_offset) {
        const std::size_t gx = geom.wrap_index(ix + x_offset);
        const bool is_local =
            gx >= support.local_start && gx < support.local_end;
        const bool is_right_ghost = gx == geom.wrap_index(
            static_cast<std::int64_t>(support.local_end));

        for (int y_offset = 0; y_offset < 2; ++y_offset) {
            const std::size_t gy = geom.wrap_index(iy + y_offset);
            for (int z_offset = 0; z_offset < 2; ++z_offset) {
                const std::size_t gz = geom.wrap_index(iz + z_offset);
                const auto weight_result = cic_weight_3d(
                    wx[x_offset], wy[y_offset], wz[z_offset]);
                const core::Real weight = weight_result.value;
                const std::size_t plane_index = gy * N + gz;
                core::Real field_value = 0.0;
                if (is_local) {
                    const std::size_t local_x = gx - support.local_start;
                    field_value = field[
                        geom.real_index(local_x, gy, gz)];
                } else if (is_right_ghost && right_ghost) {
                    field_value = (*right_ghost)[plane_index];
                } else {
                    result.escaped_support = true;
                    return result;
                }
                if (weight_result.lost_nonzero_product
                    && field_value != core::Real{0.0}) {
                    result.lost_nonzero_product = true;
                }
                const core::Real contribution = field_value * weight;
                if (field_value != core::Real{0.0}
                    && weight != core::Real{0.0}
                    && contribution == core::Real{0.0}) {
                    result.lost_nonzero_product = true;
                }
                result.value += contribution;
            }
        }
    }
    return result;
}

} // namespace detail

void CICMassAssignment::interpolate(
    const RealField& field,
    const RealField* right_ghost,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> out_values) const {
    interpolate_impl(
        field,
        right_ghost,
        pos_x,
        pos_y,
        pos_z,
        out_values,
        false);
}

void CICMassAssignment::interpolate_add(
    const RealField& field,
    const RealField* right_ghost,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> out_values) const {
    interpolate_impl(
        field,
        right_ghost,
        pos_x,
        pos_y,
        pos_z,
        out_values,
        true);
}

core::Accum CICMassAssignment::interpolate_mass_weighted_sum(
    const RealField& field,
    const RealField* right_ghost,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    core::Accum scale) const {
    if (!std::isfinite(scale)) {
        throw std::invalid_argument(
            "CIC mass-weighted reduction scale must be finite");
    }
    if (uniform_mass.has_value()) {
        if (!std::isfinite(*uniform_mass) || *uniform_mass < 0.0) {
            throw std::invalid_argument(
                "CIC mass-weighted reduction uniform mass must be finite and non-negative");
        }
    } else {
        if (masses.size() != pos_x.size()) {
            throw std::invalid_argument(
                "CIC mass-weighted reduction mass array size mismatch");
        }
        int invalid_mass = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) reduction(|: invalid_mass) \
            if(runtime::should_use_host_parallel_team(masses.size()))
#endif
        for (std::size_t index = 0; index < masses.size(); ++index) {
            if (!std::isfinite(masses[index]) || masses[index] < 0.0) {
                invalid_mass = 1;
            }
        }
        if (invalid_mass != 0) {
            throw std::invalid_argument(
                "CIC mass-weighted reduction masses must be finite and non-negative");
        }
    }

    const detail::CICGatherSupport support =
        detail::prepare_cic_gather_support(
            geom_, dx_, field, right_ghost, pos_x, pos_y, pos_z);
    const core::Accum result = math::deterministic_blocked_sum(
        pos_x.size(),
        [&](std::size_t particle) -> core::Accum {
            const detail::CICGatherValue gathered =
                detail::gather_cic_particle_value(
                    geom_,
                    dx_,
                    support,
                    field,
                    right_ghost,
                    pos_x[particle],
                    pos_y[particle],
                    pos_z[particle]);
            if (gathered.escaped_support
                || gathered.lost_nonzero_product
                || !std::isfinite(gathered.value)) {
                return std::numeric_limits<core::Accum>::quiet_NaN();
            }

            const core::Real mass = uniform_mass.has_value()
                ? *uniform_mass : masses[particle];
            const core::Accum scaled_mass =
                scale * static_cast<core::Accum>(mass);
            if (scale != core::Accum{0.0}
                && mass != core::Real{0.0}
                && scaled_mass == core::Accum{0.0}) {
                return std::numeric_limits<core::Accum>::quiet_NaN();
            }
            const core::Accum weighted =
                scaled_mass * static_cast<core::Accum>(gathered.value);
            if (scaled_mass != core::Accum{0.0}
                && gathered.value != core::Real{0.0}
                && weighted == core::Accum{0.0}) {
                return std::numeric_limits<core::Accum>::quiet_NaN();
            }
            return weighted;
        });
    if (!std::isfinite(result)) {
        throw std::underflow_error(
            "CIC mass-weighted reduction lost a non-zero interpolation or scaled mass-field product in the accumulation format");
    }
    return result;
}

core::Accum CICMassAssignment::interpolate_mass_weighted_sum_full_mesh(
    const RealField& field,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass) const {
    require_position_sizes(pos_x, pos_y, pos_z);
    require_finite_positions(pos_x, pos_y, pos_z);
    if (geom_.local_n0() != geom_.grid_size()
        || geom_.local_0_start() != 0) {
        throw std::invalid_argument(
            "CIC mass-weighted sum requires a full periodic mesh");
    }
    if (field.size() != geom_.padded_real_size()) {
        throw std::invalid_argument(
            "CIC mass-weighted sum field size does not match mesh geometry");
    }
    if (uniform_mass.has_value()) {
        if (!std::isfinite(*uniform_mass) || *uniform_mass < 0.0) {
            throw std::invalid_argument(
                "CIC mass-weighted sum uniform mass must be finite and non-negative");
        }
    } else {
        if (masses.size() != pos_x.size()) {
            throw std::invalid_argument(
                "CIC mass-weighted sum mass array size mismatch");
        }
        int invalid_mass = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) reduction(|: invalid_mass) \
            if(runtime::should_use_host_parallel_team(masses.size()))
#endif
        for (std::size_t index = 0; index < masses.size(); ++index) {
            if (!std::isfinite(masses[index]) || masses[index] < 0.0) {
                invalid_mass = 1;
            }
        }
        if (invalid_mass != 0) {
            throw std::invalid_argument(
                "CIC mass-weighted sum masses must be finite and non-negative");
        }
    }

    const core::Real L = geom_.box_size();
    const std::size_t N = geom_.grid_size();
    const core::Accum result = math::deterministic_blocked_sum(
        pos_x.size(),
        [&](std::size_t particle) {
            const auto x_coordinate = cic_coordinate_1d(
                pos_x[particle], L, dx_, N);
            const auto y_coordinate = cic_coordinate_1d(
                pos_y[particle], L, dx_, N);
            const auto z_coordinate = cic_coordinate_1d(
                pos_z[particle], L, dx_, N);
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
            const core::Real mass = uniform_mass.has_value()
                ? *uniform_mass : masses[particle];
            const core::Accum wide_mass = static_cast<core::Accum>(mass);

            core::Accum value = 0.0;
            bool lost_nonzero_product = false;
            for (int x_offset = 0; x_offset < 2; ++x_offset) {
                const std::size_t gx = geom_.wrap_index(ix + x_offset);
                for (int y_offset = 0; y_offset < 2; ++y_offset) {
                    const std::size_t gy = geom_.wrap_index(iy + y_offset);
                    for (int z_offset = 0; z_offset < 2; ++z_offset) {
                        const std::size_t gz = geom_.wrap_index(iz + z_offset);
                        const core::Accum field_value =
                            static_cast<core::Accum>(field[
                                geom_.real_index(gx, gy, gz)]);
                        const auto weight_result = detail::cic_weight_3d(
                            wx[x_offset], wy[y_offset], wz[z_offset]);
                        if (weight_result.lost_nonzero_product
                            && field_value != core::Accum{0.0}
                            && wide_mass != core::Accum{0.0}) {
                            lost_nonzero_product = true;
                        }
                        const core::Accum weight = static_cast<core::Accum>(
                            weight_result.value);
                        const core::Accum contribution = field_value * weight;
                        if (field_value != core::Accum{0.0}
                            && weight != core::Accum{0.0}
                            && contribution == core::Accum{0.0}
                            && wide_mass != core::Accum{0.0}) {
                            lost_nonzero_product = true;
                        }
                        value += contribution;
                    }
                }
            }
            if (lost_nonzero_product) {
                return std::numeric_limits<core::Accum>::quiet_NaN();
            }
            const core::Accum weighted = wide_mass * value;
            if (wide_mass != core::Accum{0.0}
                && value != core::Accum{0.0}
                && weighted == core::Accum{0.0}) {
                return std::numeric_limits<core::Accum>::quiet_NaN();
            }
            return weighted;
        });
    if (!std::isfinite(result)) {
        throw std::underflow_error(
            "CIC mass-weighted sum lost a non-zero interpolation or mass-field product in the accumulation format");
    }
    return result;
}

void CICMassAssignment::interpolate_impl(
    const RealField& field,
    const RealField* right_ghost,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> out_values,
    bool add) const {
    if (pos_x.size() != out_values.size()) {
        throw std::invalid_argument(
            "CIC output size must match position count");
    }
    const detail::CICGatherSupport support =
        detail::prepare_cic_gather_support(
            geom_, dx_, field, right_ghost, pos_x, pos_y, pos_z);

    int lost_nonzero_interpolation_product = 0;
    int escaped_support = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        reduction(|: lost_nonzero_interpolation_product, escaped_support) \
        if(runtime::should_use_host_parallel_team(pos_x.size()))
#endif
    for (std::size_t particle = 0; particle < pos_x.size(); ++particle) {
        const detail::CICGatherValue gathered =
            detail::gather_cic_particle_value(
                geom_,
                dx_,
                support,
                field,
                right_ghost,
                pos_x[particle],
                pos_y[particle],
                pos_z[particle]);
        lost_nonzero_interpolation_product |= gathered.lost_nonzero_product;
        escaped_support |= gathered.escaped_support;
        if (add) {
            out_values[particle] += gathered.value;
        } else {
            out_values[particle] = gathered.value;
        }
    }
    if (escaped_support != 0) {
        throw std::logic_error(
            "CIC interpolation support escaped the admitted local/right planes");
    }
    if (lost_nonzero_interpolation_product != 0) {
        throw std::underflow_error(
            "CIC interpolation lost a non-zero CIC weight or field-weight product in core::Real");
    }
}

} // namespace mesh
} // namespace cosmo_nbody
