#include "cosmo_nbody/mesh/mass_assignment.hpp"

#include "cic_coordinate.hpp"
#include "cosmo_nbody/math/deterministic_sum.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cosmo_nbody::mesh {

core::Accum CICMassAssignment::interpolate_mass_weighted_directional_derivative(
    const RealField& field,
    const RealField* right_ghost,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<const core::Real> direction_x,
    std::span<const core::Real> direction_y,
    std::span<const core::Real> direction_z) const {
    const std::size_t count = pos_x.size();
    if (pos_y.size() != count || pos_z.size() != count
        || direction_x.size() != count
        || direction_y.size() != count
        || direction_z.size() != count) {
        throw std::invalid_argument(
            "CIC directional-derivative component sizes must match");
    }
    if (!uniform_mass.has_value() && masses.size() != count) {
        throw std::invalid_argument(
            "CIC directional-derivative mass count differs from particle count");
    }
    if (uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass < 0.0)) {
        throw std::invalid_argument(
            "CIC directional-derivative uniform mass must be finite and non-negative");
    }
    if (!std::isfinite(dx_) || dx_ <= 0.0) {
        throw std::logic_error(
            "CIC directional derivative requires finite positive cell size");
    }
    if (field.size() != geom_.padded_real_size()) {
        throw std::invalid_argument(
            "CIC directional-derivative field size does not match mesh geometry");
    }

    const std::size_t N = geom_.grid_size();
    if (N == 0 || N > std::numeric_limits<std::size_t>::max() / N) {
        throw std::overflow_error(
            "CIC directional-derivative mesh extent is invalid");
    }
    const std::size_t ghost_plane_size = N * N;
    if (right_ghost && right_ghost->size() != ghost_plane_size) {
        throw std::invalid_argument(
            "CIC directional-derivative right ghost plane has incorrect size");
    }
    const std::size_t local_start = geom_.local_0_start();
    const std::size_t local_end = local_start + geom_.local_n0();
    const bool decomposed = geom_.local_n0() < N;
    if (decomposed && !right_ghost) {
        throw std::invalid_argument(
            "Decomposed CIC directional derivative requires a right ghost plane");
    }

    const core::Real L = geom_.box_size();
    return math::deterministic_blocked_sum(
        count,
        [&](std::size_t particle) -> core::Accum {
            const core::Real mass = uniform_mass.has_value()
                ? *uniform_mass : masses[particle];
            if (!std::isfinite(mass) || mass < 0.0
                || !std::isfinite(direction_x[particle])
                || !std::isfinite(direction_y[particle])
                || !std::isfinite(direction_z[particle])) {
                throw std::invalid_argument(
                    "CIC directional derivative requires finite direction and non-negative mass");
            }

            const auto x_coordinate = detail::cic_coordinate_1d(
                pos_x[particle], L, dx_, N);
            const auto y_coordinate = detail::cic_coordinate_1d(
                pos_y[particle], L, dx_, N);
            const auto z_coordinate = detail::cic_coordinate_1d(
                pos_z[particle], L, dx_, N);
            const std::size_t base_x = static_cast<std::size_t>(x_coordinate.base);
            if (decomposed && (base_x < local_start || base_x >= local_end)) {
                throw std::invalid_argument(
                    "CIC directional-derivative particle base cell is outside the local mesh slab");
            }

            const core::Real wx[2] = {
                1.0 - x_coordinate.fraction,
                x_coordinate.fraction};
            const core::Real wy[2] = {
                1.0 - y_coordinate.fraction,
                y_coordinate.fraction};
            const core::Real wz[2] = {
                1.0 - z_coordinate.fraction,
                z_coordinate.fraction};

            core::Real values[2][2][2]{};
            for (int x_offset = 0; x_offset < 2; ++x_offset) {
                const std::size_t gx = geom_.wrap_index(
                    x_coordinate.base + x_offset);
                const bool is_local = gx >= local_start && gx < local_end;
                const bool is_right_ghost = gx == geom_.wrap_index(
                    static_cast<std::int64_t>(local_end));
                for (int y_offset = 0; y_offset < 2; ++y_offset) {
                    const std::size_t gy = geom_.wrap_index(
                        y_coordinate.base + y_offset);
                    for (int z_offset = 0; z_offset < 2; ++z_offset) {
                        const std::size_t gz = geom_.wrap_index(
                            z_coordinate.base + z_offset);
                        core::Real value = 0.0;
                        if (is_local) {
                            value = field[geom_.real_index(
                                gx - local_start, gy, gz)];
                        } else if (is_right_ghost && right_ghost) {
                            value = (*right_ghost)[gy * N + gz];
                        } else {
                            throw std::logic_error(
                                "CIC directional-derivative stencil escaped local/right-ghost support");
                        }
                        if (!std::isfinite(value)) {
                            throw std::invalid_argument(
                                "CIC directional-derivative potential must be finite");
                        }
                        values[x_offset][y_offset][z_offset] = value;
                    }
                }
            }

            long double gradient_x = 0.0L;
            long double gradient_y = 0.0L;
            long double gradient_z = 0.0L;
            for (int y_offset = 0; y_offset < 2; ++y_offset) {
                for (int z_offset = 0; z_offset < 2; ++z_offset) {
                    gradient_x +=
                        (static_cast<long double>(values[1][y_offset][z_offset])
                         - static_cast<long double>(values[0][y_offset][z_offset]))
                        * static_cast<long double>(wy[y_offset])
                        * static_cast<long double>(wz[z_offset]);
                }
            }
            for (int x_offset = 0; x_offset < 2; ++x_offset) {
                for (int z_offset = 0; z_offset < 2; ++z_offset) {
                    gradient_y +=
                        (static_cast<long double>(values[x_offset][1][z_offset])
                         - static_cast<long double>(values[x_offset][0][z_offset]))
                        * static_cast<long double>(wx[x_offset])
                        * static_cast<long double>(wz[z_offset]);
                }
            }
            for (int x_offset = 0; x_offset < 2; ++x_offset) {
                for (int y_offset = 0; y_offset < 2; ++y_offset) {
                    gradient_z +=
                        (static_cast<long double>(values[x_offset][y_offset][1])
                         - static_cast<long double>(values[x_offset][y_offset][0]))
                        * static_cast<long double>(wx[x_offset])
                        * static_cast<long double>(wy[y_offset]);
                }
            }
            const long double inverse_dx =
                1.0L / static_cast<long double>(dx_);
            gradient_x *= inverse_dx;
            gradient_y *= inverse_dx;
            gradient_z *= inverse_dx;

            const long double directional =
                static_cast<long double>(direction_x[particle]) * gradient_x
                + static_cast<long double>(direction_y[particle]) * gradient_y
                + static_cast<long double>(direction_z[particle]) * gradient_z;
            const long double wide_term =
                static_cast<long double>(mass) * directional;
            if (!std::isfinite(wide_term)) {
                throw std::overflow_error(
                    "CIC directional-derivative contribution is non-finite");
            }
            const core::Accum term = static_cast<core::Accum>(wide_term);
            if (wide_term != 0.0L && term == core::Accum{0.0}) {
                throw std::underflow_error(
                    "CIC directional-derivative contribution is nonzero but not representable");
            }
            if (!std::isfinite(term)) {
                throw std::overflow_error(
                    "CIC directional-derivative contribution is not representable");
            }
            return term;
        });
}

} // namespace cosmo_nbody::mesh
