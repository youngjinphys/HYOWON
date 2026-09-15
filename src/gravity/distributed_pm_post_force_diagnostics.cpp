#include "cosmo_nbody/gravity/distributed_pm_solver.hpp"

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

namespace cosmo_nbody::gravity {
namespace {

void synchronize_distributed_exception(
    std::exception_ptr local_exception,
    int size,
    const char* context) {
    if (size <= 1) {
        if (local_exception) std::rethrow_exception(local_exception);
        return;
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)context;
    throw std::logic_error("Distributed PM diagnostic requires MPI");
#else
    runtime::synchronize_mpi_exception(local_exception, size, context);
#endif
}

void synchronize_distributed_invalid(
    int local_invalid,
    int size,
    const char* context) {
    // Do not allocate an exception message before all ranks enter the agreement.
    runtime::synchronize_mpi_failure(local_invalid, size, context);
}

core::Real checked_real(long double value, const char* context) {
    if (!std::isfinite(value)
        || value > static_cast<long double>(std::numeric_limits<core::Real>::max())
        || value < -static_cast<long double>(std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(std::string(context) + " is not representable");
    }
    const core::Real converted = static_cast<core::Real>(value);
    if (value != 0.0L && converted == core::Real{0.0}) {
        throw std::underflow_error(std::string(context) + " underflowed core::Real");
    }
    return converted;
}

long double rank_order_sum(
    long double local_value,
    int size,
    const char* context) {
    std::vector<long double> gathered;
    std::exception_ptr preparation_exception;
    try {
        if (!std::isfinite(local_value)) {
            throw std::overflow_error(std::string(context) + " local value is non-finite");
        }
        if (size > 1) gathered.resize(static_cast<std::size_t>(size));
    } catch (...) {
        preparation_exception = std::current_exception();
    }
    synchronize_distributed_exception(preparation_exception, size, context);
    if (size <= 1) return local_value;
#ifndef COSMO_NBODY_HAS_MPI
    throw std::logic_error("Distributed PM rank-order reduction requires MPI");
#else
    runtime::require_active_mpi_main_thread(context);
    const int status = MPI_Allgather(
        &local_value, 1, MPI_LONG_DOUBLE,
        gathered.data(), 1, MPI_LONG_DOUBLE,
        MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
    long double sum = 0.0L;
    long double compensation = 0.0L;
    for (const long double value : gathered) {
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                std::string(context) + " gathered a non-finite value");
        }
        const long double updated = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            compensation += (sum - updated) + value;
        } else {
            compensation += (value - updated) + sum;
        }
        sum = updated;
    }
    const long double result = sum + compensation;
    if (!std::isfinite(result)) {
        throw std::overflow_error(std::string(context) + " global sum is non-finite");
    }
    return result;
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

core::Real DistributedPMSolver::compute_cic_self_energy_momentum_directional_comoving(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<const core::Real> momentum_x,
    std::span<const core::Real> momentum_y,
    std::span<const core::Real> momentum_z) const {
    core::Accum local_sum = 0.0;
    std::exception_ptr local_exception;
    try {
        if (!self_kernel_ready_) {
            throw std::logic_error(
                "Distributed PM self-energy directional derivative requires an initialized kernel");
        }
        const std::size_t count = pos_x.size();
        if (pos_y.size() != count || pos_z.size() != count
            || momentum_x.size() != count
            || momentum_y.size() != count
            || momentum_z.size() != count) {
            throw std::invalid_argument(
                "Distributed PM self-energy directional components differ in size");
        }
        if (!uniform_mass.has_value() && masses.size() != count) {
            throw std::invalid_argument(
                "Distributed PM self-energy directional mass count differs from particles");
        }
        if (uniform_mass.has_value()
            && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
            throw std::invalid_argument(
                "Distributed PM self-energy directional uniform mass must be finite and positive");
        }
        const core::Real dx =
            box_size_ / static_cast<core::Real>(mesh_size_);
        if (!std::isfinite(dx) || dx <= 0.0) {
            throw std::overflow_error(
                "Distributed PM self-energy directional cell size is invalid");
        }

        local_sum = math::deterministic_blocked_sum(
            count,
            [&](std::size_t index) -> core::Accum {
                const core::Real mass = uniform_mass.has_value()
                    ? *uniform_mass : masses[index];
                if (!std::isfinite(mass) || mass <= 0.0
                    || !std::isfinite(pos_x[index])
                    || !std::isfinite(pos_y[index])
                    || !std::isfinite(pos_z[index])
                    || !std::isfinite(momentum_x[index])
                    || !std::isfinite(momentum_y[index])
                    || !std::isfinite(momentum_z[index])) {
                    throw std::invalid_argument(
                        "Distributed PM self-energy directional inputs must be finite with positive mass");
                }

                std::array<std::array<core::Real, 3>, 3> factor{};
                std::array<std::array<core::Real, 3>, 3> derivative{};
                const std::array<core::Real, 3> coordinates{
                    pos_x[index], pos_y[index], pos_z[index]};
                const std::array<core::Real, 3> direction{
                    momentum_x[index], momentum_y[index], momentum_z[index]};
                for (std::size_t dim = 0; dim < 3; ++dim) {
                    const core::Real cell_coordinate =
                        math::wrap(coordinates[dim], box_size_) / dx;
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
                            stencil += dweight
                                * static_cast<core::Accum>(
                                    self_kernel_[ix * 9 + iy * 3 + iz]);
                        }
                    }
                }
                const core::Accum term = self_kernel_scaled_by_cell_volume_
                    ? scaled_mass_space_self_derivative_term(mass, stencil, dx)
                    : scaled_self_derivative_term(mass, stencil);
                if (!std::isfinite(term)) {
                    throw std::overflow_error(
                        "Distributed PM self-energy directional contribution is non-finite");
                }
                return term;
            });

    } catch (...) {
        local_exception = std::current_exception();
    }
    synchronize_distributed_exception(
        local_exception, size_, "Distributed PM self-energy directional local stage");

    return checked_real(
        rank_order_sum(
            static_cast<long double>(local_sum),
            size_,
            "Distributed PM CIC self-energy directional derivative"),
        "Distributed PM CIC self-energy directional derivative");
}

DistributedPMPostForceEnergyDiagnostics
DistributedPMSolver::measure_post_force_energy_diagnostics(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<const core::Real> momentum_x,
    std::span<const core::Real> momentum_y,
    std::span<const core::Real> momentum_z) {
    std::exception_ptr input_exception;
    try {
        if (use_exact_gradient_) {
            throw std::logic_error(
                "Post-force energy-gradient diagnostic is defined only for the pure-PM centered-gradient path");
        }
        const std::size_t count = pos_x.size();
        if (pos_y.size() != count || pos_z.size() != count
            || momentum_x.size() != count
            || momentum_y.size() != count
            || momentum_z.size() != count) {
            throw std::invalid_argument(
                "Distributed PM post-force diagnostic component sizes must match");
        }

        if ((!uniform_mass.has_value() && masses.size() != count)
            || (uniform_mass.has_value()
                && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0))) {
            throw std::invalid_argument(
                "Distributed PM post-force mass representation is invalid");
        }
    } catch (...) {
        input_exception = std::current_exception();
    }
    synchronize_distributed_exception(
        input_exception, size_, "Distributed PM post-force input validation");

    int local_potential_invalid = 0;
    for (const core::Real value : density_) {
        local_potential_invalid |= !std::isfinite(value);
    }
    synchronize_distributed_invalid(
        local_potential_invalid,
        size_,
        "Distributed PM post-force potential is non-finite");
    if (size_ > 1) exchange_field_planes(density_);

    core::Accum local_energy = 0.0;
    core::Accum local_directional = 0.0;
    std::exception_ptr interpolation_exception;
    try {
        local_energy = mass_assignment_.interpolate_mass_weighted_sum(
            density_,
            size_ == 1 ? nullptr : &recv_right_plane_,
            pos_x, pos_y, pos_z,
            masses, uniform_mass,
            core::Accum{0.5});
        local_directional =
            mass_assignment_.interpolate_mass_weighted_directional_derivative(
                density_,
                size_ == 1 ? nullptr : &recv_right_plane_,
                pos_x, pos_y, pos_z,
                masses, uniform_mass,
                momentum_x, momentum_y, momentum_z);
    } catch (...) {
        interpolation_exception = std::current_exception();
    }
    synchronize_distributed_exception(
        interpolation_exception,
        size_,
        "Distributed PM post-force energy interpolation");

    const core::Real raw_energy = checked_real(
        rank_order_sum(
            static_cast<long double>(local_energy), size_,
            "Distributed PM post-force raw energy"),
        "Distributed PM post-force raw energy");
    const core::Real raw_directional = checked_real(
        rank_order_sum(
            static_cast<long double>(local_directional), size_,
            "Distributed PM post-force raw directional derivative"),
        "Distributed PM post-force raw directional derivative");

    // The first self-kernel construction may reuse density_ as an impulse field,
    // so all measurements requiring the force-produced potential are complete
    // before this call. No later consumer may treat density_ as that potential.
    ensure_self_kernel();
    const core::Real self_energy = compute_cic_self_energy_comoving(
        pos_x, pos_y, pos_z, masses, uniform_mass);
    const core::Real self_directional =
        compute_cic_self_energy_momentum_directional_comoving(
            pos_x, pos_y, pos_z, masses, uniform_mass,
            momentum_x, momentum_y, momentum_z);

    return DistributedPMPostForceEnergyDiagnostics{
        raw_energy,
        raw_directional,
        self_energy,
        self_directional};
}

} // namespace cosmo_nbody::gravity
