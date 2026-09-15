#include "cosmo_nbody/validation/conservation_checks.hpp"

#include "cosmo_nbody/math/deterministic_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::validation {
namespace {

core::Real checked_real(long double value, const char* context) {
    if (!std::isfinite(value)
        || value > static_cast<long double>(std::numeric_limits<core::Real>::max())
        || value < -static_cast<long double>(std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(std::string(context) + " is not representable");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (value != 0.0L && result == core::Real{0.0}) {
        throw std::underflow_error(std::string(context) + " underflowed core::Real");
    }
    return result;
}

int reduction_size(bool mpi_enabled, const char* context) {
    if (!mpi_enabled) return 1;
#ifndef COSMO_NBODY_HAS_MPI
    (void)context;
    throw std::logic_error("Conservation force-energy reduction requires MPI");
#else
    runtime::require_active_mpi_main_thread(context);
    int size = 0;
    const int status = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (status != MPI_SUCCESS || size < 1) {
        (void)MPI_Abort(MPI_COMM_WORLD, status == MPI_SUCCESS ? 1 : status);
        std::abort();
    }
    return size;
#endif
}

core::Real rank_order_sum(
    core::Accum local_value,
    bool mpi_enabled,
    const char* context) {
    const int size = reduction_size(mpi_enabled, context);
    core::Real local = 0.0;
    std::vector<core::Real> gathered;
    std::exception_ptr preparation_exception;
    try {
        local = checked_real(static_cast<long double>(local_value), context);
        if (size > 1) gathered.resize(static_cast<std::size_t>(size));
    } catch (...) {
        preparation_exception = std::current_exception();
    }
    // Conversion can fail on just one rank, just as allocation can.
    runtime::synchronize_mpi_exception(preparation_exception, size, context);
    if (size == 1) return local;
#ifndef COSMO_NBODY_HAS_MPI
    throw std::logic_error("Conservation force-energy reduction requires MPI");
#else
    const int status = MPI_Allgather(
        &local, 1, MPI_DOUBLE,
        gathered.data(), 1, MPI_DOUBLE,
        MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
    long double sum = 0.0L;
    long double compensation = 0.0L;
    for (const core::Real value : gathered) {
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                std::string(context) + " gathered a non-finite value");
        }
        const long double wide = static_cast<long double>(value);
        const long double updated = sum + wide;
        if (std::abs(sum) >= std::abs(wide)) {
            compensation += (sum - updated) + wide;
        } else {
            compensation += (wide - updated) + sum;
        }
        sum = updated;
    }
    return checked_real(sum + compensation, context);
#endif
}

core::Accum scaled_self_derivative_term(
    core::Real mass,
    core::Accum stencil) noexcept {
    if (stencil == core::Accum{0.0}) return core::Accum{0.0};
    int mass_exponent = 0;
    int stencil_exponent = 0;
    const core::Accum mass_fraction = std::frexp(
        static_cast<core::Accum>(mass), &mass_exponent);
    const core::Accum stencil_fraction = std::frexp(stencil, &stencil_exponent);
    core::Accum coefficient = core::Accum{0.5}
        * mass_fraction * mass_fraction * stencil_fraction;
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    return std::scalbn(
        coefficient,
        2 * mass_exponent + stencil_exponent + coefficient_exponent);
}

core::Accum scaled_mass_space_self_derivative_term(
    core::Real mass,
    core::Accum stencil,
    core::Real cell_size) noexcept {
    if (stencil == core::Accum{0.0}) return core::Accum{0.0};
    int mass_exponent = 0;
    int stencil_exponent = 0;
    int cell_exponent = 0;
    const core::Accum mass_fraction = std::frexp(
        static_cast<core::Accum>(mass), &mass_exponent);
    const core::Accum stencil_fraction = std::frexp(stencil, &stencil_exponent);
    const core::Accum cell_fraction = std::frexp(
        static_cast<core::Accum>(cell_size), &cell_exponent);
    core::Accum coefficient = core::Accum{0.5}
        * mass_fraction * mass_fraction * stencil_fraction
        / (cell_fraction * cell_fraction * cell_fraction);
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    return std::scalbn(
        coefficient,
        2 * mass_exponent + stencil_exponent - 3 * cell_exponent
            + coefficient_exponent);
}

} // namespace

core::Real ConservationChecks::compute_momentum_force_contraction(
    const core::ParticleStore& particles) const {
    constexpr const char* context = "Conservation momentum-force contraction";
    const int size = reduction_size(config_.get_runtime().mpi_enabled, context);
    core::Accum local = 0.0;
    std::exception_ptr local_exception;
    try {
        const std::size_t owned = particles.num_owned_particles();
        const auto px = particles.get_momenta_x().first(owned);
        const auto py = particles.get_momenta_y().first(owned);
        const auto pz = particles.get_momenta_z().first(owned);
        const auto gx = particles.get_accelerations_x().first(owned);
        const auto gy = particles.get_accelerations_y().first(owned);
        const auto gz = particles.get_accelerations_z().first(owned);
        const auto uniform_mass = particles.get_uniform_mass();
        if (uniform_mass.has_value()
            && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
            throw std::invalid_argument(
                "Conservation diagnostic uniform mass must be finite and positive");
        }
        const auto masses = uniform_mass.has_value()
            ? std::span<const core::Real>{}
            : particles.get_masses().first(owned);

        local = math::deterministic_blocked_sum(
            owned,
            [&](std::size_t index) -> core::Accum {
                const core::Real mass = uniform_mass.has_value()
                    ? *uniform_mass : masses[index];
                if (!std::isfinite(mass) || mass <= 0.0
                    || !std::isfinite(px[index]) || !std::isfinite(py[index])
                    || !std::isfinite(pz[index]) || !std::isfinite(gx[index])
                    || !std::isfinite(gy[index]) || !std::isfinite(gz[index])) {
                    throw std::invalid_argument(
                        "Momentum-force contraction requires finite phase-space/force values and positive mass");
                }
                const long double dot =
                    static_cast<long double>(px[index]) * gx[index]
                    + static_cast<long double>(py[index]) * gy[index]
                    + static_cast<long double>(pz[index]) * gz[index];
                const long double term = static_cast<long double>(mass) * dot;
                if (!std::isfinite(term)
                    || term > static_cast<long double>(
                        std::numeric_limits<core::Accum>::max())
                    || term < -static_cast<long double>(
                        std::numeric_limits<core::Accum>::max())) {
                    throw std::overflow_error(
                        "Momentum-force contraction contribution is not representable");
                }
                const core::Accum converted = static_cast<core::Accum>(term);
                if (term != 0.0L && converted == core::Accum{0.0}) {
                    throw std::underflow_error(
                        "Momentum-force contraction contribution underflowed accumulation format");
                }
                return converted;
            });
    } catch (...) {
        local_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(local_exception, size, context);
    return rank_order_sum(
        local,
        config_.get_runtime().mpi_enabled,
        "Conservation momentum-force contraction");
}

core::Real ConservationChecks::compute_cic_self_energy_momentum_directional_comoving(
    const core::ParticleStore& particles) const {
    constexpr const char* context = "Conservation CIC self-energy directional derivative";
    const int size = reduction_size(config_.get_runtime().mpi_enabled, context);
    const std::size_t mesh_size = config_.get_box().N_mesh;
    const core::Real L = config_.get_box().L;
    core::Real dx = 0.0;
    std::exception_ptr geometry_exception;
    try {
        if (mesh_size == 0) {
            throw std::invalid_argument(
                "CIC self-energy directional derivative requires positive mesh size");
        }
        dx = L / static_cast<core::Real>(mesh_size);
        if (!std::isfinite(L) || L <= 0.0 || !std::isfinite(dx) || dx <= 0.0) {
            throw std::overflow_error(
                "CIC self-energy directional geometry is not representable");
        }
    } catch (...) {
        geometry_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(geometry_exception, size, context);

    // This routine contains collectives and must not be hidden in a local stage.
    prepare_cic_self_kernel();
    core::Accum local = 0.0;
    std::exception_ptr local_exception;
    try {
        const auto kernel = self_kernel_;
        const bool scaled_by_cell_volume = self_kernel_scaled_by_cell_volume_;
        const std::size_t owned = particles.num_owned_particles();
        const auto x = particles.get_positions_x().first(owned);
        const auto y = particles.get_positions_y().first(owned);
        const auto z = particles.get_positions_z().first(owned);
        const auto px = particles.get_momenta_x().first(owned);
        const auto py = particles.get_momenta_y().first(owned);
        const auto pz = particles.get_momenta_z().first(owned);
        const auto uniform_mass = particles.get_uniform_mass();
        if (uniform_mass.has_value()
            && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
            throw std::invalid_argument(
                "Conservation diagnostic uniform mass must be finite and positive");
        }
        const auto masses = uniform_mass.has_value()
            ? std::span<const core::Real>{}
            : particles.get_masses().first(owned);

        local = math::deterministic_blocked_sum(
            owned,
            [&](std::size_t index) -> core::Accum {
                const core::Real mass = uniform_mass.has_value()
                    ? *uniform_mass : masses[index];
                if (!std::isfinite(mass) || mass <= 0.0
                    || !std::isfinite(x[index]) || !std::isfinite(y[index])
                    || !std::isfinite(z[index]) || !std::isfinite(px[index])
                    || !std::isfinite(py[index]) || !std::isfinite(pz[index])) {
                    throw std::invalid_argument(
                        "CIC self-energy directional derivative requires finite positions/momenta and positive mass");
                }
                const std::array<core::Real, 3> position{x[index], y[index], z[index]};
                const std::array<core::Real, 3> direction{px[index], py[index], pz[index]};
                std::array<std::array<core::Real, 3>, 3> factor{};
                std::array<std::array<core::Real, 3>, 3> derivative{};
                for (std::size_t dim = 0; dim < 3; ++dim) {
                    const core::Real cell_coordinate =
                        math::wrap(position[dim], L) / dx;
                    const core::Real offset =
                        cell_coordinate - std::floor(cell_coordinate);
                    const core::Real lower = 1.0 - offset;
                    factor[dim] = {
                        offset * lower,
                        lower * lower + offset * offset,
                        offset * lower};
                    const core::Real rate = direction[dim] / dx;
                    derivative[dim] = {
                        (1.0 - 2.0 * offset) * rate,
                        (4.0 * offset - 2.0) * rate,
                        (1.0 - 2.0 * offset) * rate};
                }

                core::Accum stencil = 0.0;
                for (std::size_t ix = 0; ix < 3; ++ix) {
                    for (std::size_t iy = 0; iy < 3; ++iy) {
                        for (std::size_t iz = 0; iz < 3; ++iz) {
                            const core::Accum dweight =
                                static_cast<core::Accum>(derivative[0][ix])
                                    * factor[1][iy] * factor[2][iz]
                                + static_cast<core::Accum>(factor[0][ix])
                                    * derivative[1][iy] * factor[2][iz]
                                + static_cast<core::Accum>(factor[0][ix])
                                    * factor[1][iy] * derivative[2][iz];
                            stencil += dweight * static_cast<core::Accum>(
                                kernel[ix * 9 + iy * 3 + iz]);
                        }
                    }
                }
                const core::Accum term = scaled_by_cell_volume
                    ? scaled_mass_space_self_derivative_term(mass, stencil, dx)
                    : scaled_self_derivative_term(mass, stencil);
                if (!std::isfinite(term)) {
                    throw std::overflow_error(
                        "CIC self-energy directional contribution is non-finite");
                }
                return term;
            });

    } catch (...) {
        local_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(local_exception, size, context);

    return rank_order_sum(
        local,
        config_.get_runtime().mpi_enabled,
        "Conservation CIC self-energy directional derivative");
}

} // namespace cosmo_nbody::validation
