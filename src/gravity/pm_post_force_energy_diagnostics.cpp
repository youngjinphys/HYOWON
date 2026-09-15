#include "cosmo_nbody/gravity/pm_solver.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::gravity {

PMPostForceEnergyDiagnostics PMSolver::measure_post_force_energy_diagnostics_in_place(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<const core::Real> momentum_x,
    std::span<const core::Real> momentum_y,
    std::span<const core::Real> momentum_z) {
    const int mpi_size = mpi_force_stage_size(
        "PMSolver post-force energy diagnostic");
    std::exception_ptr setup_exception;
    try {
        const std::size_t count = pos_x.size();
        if (pos_y.size() != count || pos_z.size() != count
            || momentum_x.size() != count
            || momentum_y.size() != count
            || momentum_z.size() != count) {
            throw std::invalid_argument(
                "PMSolver post-force energy diagnostic component sizes must match");
        }
        if (method_.is_treepm_long_range() || method_.uses_spectral_gradient()) {
            throw std::logic_error(
                "PMSolver post-force energy diagnostic is defined only for pure PM");
        }
        if (!workspace_initialized_) {
            throw std::logic_error(
                "PMSolver post-force energy diagnostic requires a preceding force refresh");
        }
    } catch (...) {
        setup_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        setup_exception, mpi_size,
        "PMSolver post-force energy diagnostic setup");

#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (distributed_solver_) {
        const auto distributed =
            distributed_solver_->measure_post_force_energy_diagnostics(
                pos_x, pos_y, pos_z, masses, uniform_mass,
                momentum_x, momentum_y, momentum_z);
        return PMPostForceEnergyDiagnostics{
            distributed.potential_energy_comoving,
            distributed.directional_potential_energy_comoving,
            distributed.cic_self_energy_comoving,
            distributed.directional_cic_self_energy_comoving};
    }
#endif

    core::Accum local_energy = 0.0;
    core::Accum local_directional = 0.0;
    std::exception_ptr measurement_exception;
    try {
        if (!mass_assign_ || !real_buf_1_) {
            throw std::logic_error(
                "PMSolver post-force diagnostic workspace is incomplete");
        }
        const core::Accum mass_weighted =
            mass_assign_->interpolate_mass_weighted_sum_full_mesh(
                *real_buf_1_, pos_x, pos_y, pos_z, masses, uniform_mass);
        local_energy = core::Accum{0.5} * mass_weighted;
        if (mass_weighted != core::Accum{0.0}
            && local_energy == core::Accum{0.0}) {
            throw std::underflow_error(
                "PMSolver post-force raw energy underflowed its accumulation format");
        }
        local_directional =
            mass_assign_->interpolate_mass_weighted_directional_derivative(
                *real_buf_1_, nullptr,
                pos_x, pos_y, pos_z, masses, uniform_mass,
                momentum_x, momentum_y, momentum_z);
        if (!std::isfinite(local_energy) || !std::isfinite(local_directional)) {
            throw std::overflow_error(
                "PMSolver post-force energy diagnostic is non-finite");
        }
    } catch (...) {
        measurement_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        measurement_exception, mpi_size,
        "PMSolver post-force energy diagnostic measurement");

    const auto reduce_rank_order = [&] (
        core::Accum local_value,
        const char* context) -> core::Real {
        if (!mpi_global_reduce_ || mpi_size == 1) {
            const core::Real value = static_cast<core::Real>(local_value);
            if (!std::isfinite(value)) {
                throw std::overflow_error(std::string(context) + " is non-finite");
            }
            return value;
        }
#ifndef COSMO_NBODY_HAS_MPI
        (void)context;
        throw std::logic_error(
            "PMSolver rank-order post-force reduction requires MPI");
#else
        std::vector<core::Real> gathered;
        std::exception_ptr allocation_exception;
        try {
            gathered.resize(static_cast<std::size_t>(mpi_size));
        } catch (...) {
            allocation_exception = std::current_exception();
        }
        synchronize_mpi_force_stage(
            allocation_exception, mpi_size, context);
        const core::Real local = static_cast<core::Real>(local_value);
        if (MPI_Allgather(
                &local, 1, MPI_DOUBLE,
                gathered.data(), 1, MPI_DOUBLE,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(std::string(context) + " MPI_Allgather failed");
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
        const long double result = sum + compensation;
        if (!std::isfinite(result)
            || result > static_cast<long double>(
                std::numeric_limits<core::Real>::max())
            || result < -static_cast<long double>(
                std::numeric_limits<core::Real>::max())) {
            throw std::overflow_error(
                std::string(context) + " global result is not representable");
        }
        const core::Real converted = static_cast<core::Real>(result);
        if (result != 0.0L && converted == core::Real{0.0}) {
            throw std::underflow_error(
                std::string(context) + " global result underflowed core::Real");
        }
        return converted;
#endif
    };

    return PMPostForceEnergyDiagnostics{
        reduce_rank_order(
            local_energy,
            "PMSolver post-force raw potential energy"),
        reduce_rank_order(
            local_directional,
            "PMSolver post-force raw directional derivative"),
        std::nullopt,
        std::nullopt};
}

} // namespace cosmo_nbody::gravity
