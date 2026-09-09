#include "cosmo_nbody/gravity/distributed_pm_solver.hpp"

#include "cosmo_nbody/math/cic_mass_conservation.hpp"
#include "cosmo_nbody/math/density_normalization.hpp"
#include "cosmo_nbody/math/deterministic_sum.hpp"
#include "cosmo_nbody/math/mpi_exact_nonnegative_sum.hpp"
#include "cosmo_nbody/math/mpi_exact_uint64_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/mesh/mpi_centered_gradient.hpp"
#include "cosmo_nbody/mesh/mpi_potential_halo.hpp"
#include "cosmo_nbody/mesh/mpi_slab_layout.hpp"
#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace gravity {

namespace {

std::size_t checked_plane_size(std::size_t n) {
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "DistributedPMSolver plane size overflows size_t");
    }
    return n * n;
}

std::size_t checked_global_cell_count(std::size_t n) {
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "DistributedPMSolver mesh cell count overflows size_t");
    }
    const std::size_t n2 = n * n;
    if (n != 0 && n2 > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "DistributedPMSolver mesh cell count overflows size_t");
    }
    return n2 * n;
}

core::Real checked_real(long double value, const char* quantity) {
    const long double lower = static_cast<long double>(
        std::numeric_limits<core::Real>::lowest());
    const long double upper = static_cast<long double>(
        std::numeric_limits<core::Real>::max());
    if (!std::isfinite(value) || value < lower || value > upper) {
        throw std::overflow_error(
            std::string(quantity) + " is outside the core::Real range");
    }
    const core::Real converted = static_cast<core::Real>(value);
    if (!std::isfinite(converted)) {
        throw std::overflow_error(
            std::string(quantity) + " conversion is non-finite");
    }
    if (value != 0.0L && converted == 0.0) {
        throw std::underflow_error(
            std::string(quantity) + " underflows core::Real to zero");
    }
    return converted;
}

core::Accum scaled_self_energy_term(
    core::Real mass,
    core::Accum stencil) noexcept {
    if (stencil == core::Accum{0.0}) return core::Accum{0.0};

    int mass_exponent = 0;
    int stencil_exponent = 0;
    const core::Accum mass_fraction = std::frexp(
        static_cast<core::Accum>(mass), &mass_exponent);
    const core::Accum stencil_fraction =
        std::frexp(stencil, &stencil_exponent);

    core::Accum coefficient = core::Accum{0.5}
        * mass_fraction * mass_fraction * stencil_fraction;
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    return std::scalbn(
        coefficient,
        2 * mass_exponent + stencil_exponent + coefficient_exponent);
}

core::Accum scaled_mass_space_self_energy_term(
    core::Real mass,
    core::Accum mass_space_stencil,
    core::Real cell_size) noexcept {
    if (mass_space_stencil == core::Accum{0.0}) return core::Accum{0.0};

    int mass_exponent = 0;
    int stencil_exponent = 0;
    int cell_exponent = 0;
    const core::Accum mass_fraction = std::frexp(
        static_cast<core::Accum>(mass), &mass_exponent);
    const core::Accum stencil_fraction = std::frexp(
        mass_space_stencil, &stencil_exponent);
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

#ifdef COSMO_NBODY_HAS_FFTW_MPI
void validate_solver_parameters(
    core::Real box_size,
    std::size_t mesh_size,
    bool use_tree_pm,
    core::Real r_s,
    bool deconvolve_cic,
    bool use_exact_gradient,
    int mpi_size) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "DistributedPMSolver box size must be finite and positive");
    }
    if (mesh_size < 2) {
        throw std::invalid_argument(
            "DistributedPMSolver mesh size must be >= 2");
    }
    if (mpi_size < 1
        || static_cast<std::size_t>(mpi_size) > mesh_size) {
        throw std::invalid_argument(
            "DistributedPMSolver requires rank count in [1,N_mesh]");
    }
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (mesh_size > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "DistributedPMSolver mesh size exceeds uint64_t");
        }
    }
    if (mesh_size > static_cast<std::size_t>(
            std::numeric_limits<ptrdiff_t>::max())) {
        throw std::overflow_error(
            "DistributedPMSolver mesh size exceeds ptrdiff_t");
    }
    if (use_tree_pm) {
        if (!std::isfinite(r_s) || r_s <= 0.0) {
            throw std::invalid_argument(
                "Distributed TreePM split scale must be finite and positive");
        }
        if (!deconvolve_cic) {
            throw std::invalid_argument(
                "Distributed TreePM requires CIC deconvolution");
        }
        if (!use_exact_gradient) {
            throw std::invalid_argument(
                "Distributed TreePM requires the continuous spectral long-range operator");
        }
    }
}
#endif

#ifdef COSMO_NBODY_HAS_MPI
#ifdef COSMO_NBODY_HAS_FFTW_MPI
void require_solver_parameter_agreement(
    core::Real box_size,
    std::size_t mesh_size,
    bool use_tree_pm,
    core::Real r_s,
    bool deconvolve_cic,
    bool use_exact_gradient) {
    static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
    const std::uint64_t flags =
        (use_tree_pm ? std::uint64_t{1} : std::uint64_t{0})
        | (deconvolve_cic ? std::uint64_t{2} : std::uint64_t{0})
        | (use_exact_gradient ? std::uint64_t{4} : std::uint64_t{0});
    const std::array<std::uint64_t, 4> local{
        core::portable_bit_cast<std::uint64_t>(box_size),
        static_cast<std::uint64_t>(mesh_size),
        core::portable_bit_cast<std::uint64_t>(r_s),
        flags};
    std::array<std::uint64_t, 4> minimum{};
    std::array<std::uint64_t, 4> maximum{};
    const int minimum_status = MPI_Allreduce(
        local.data(),
        minimum.data(),
        static_cast<int>(local.size()),
        MPI_UINT64_T,
        MPI_MIN,
        MPI_COMM_WORLD);
    const int maximum_status = MPI_Allreduce(
        local.data(),
        maximum.data(),
        static_cast<int>(local.size()),
        MPI_UINT64_T,
        MPI_MAX,
        MPI_COMM_WORLD);
    if (minimum_status != MPI_SUCCESS || maximum_status != MPI_SUCCESS) {
        throw std::runtime_error(
            "DistributedPMSolver failed to synchronize constructor parameters");
    }
    if (minimum != maximum) {
        throw std::invalid_argument(
            "DistributedPMSolver physics parameters differ across MPI ranks");
    }
}
#endif

int checked_mpi_count(std::size_t count) {
    if (count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "DistributedPMSolver MPI plane exceeds INT_MAX values");
    }
    return static_cast<int>(count);
}

void synchronize_failure(
    int mpi_size,
    int local_failed,
    const char* context);

void synchronize_exception(
    int mpi_size,
    std::exception_ptr local_exception,
    const char* context);

std::uint64_t collective_particle_count(
    std::size_t local_count) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        const int local_invalid = local_count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())
            ? 1 : 0;
        int any_invalid = 0;
        if (MPI_Allreduce(
                &local_invalid,
                &any_invalid,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "MPI_Allreduce failed for distributed PM particle count validation");
        }
        if (any_invalid != 0) {
            throw std::overflow_error(
                "Distributed PM local particle count exceeds uint64_t");
        }
    }
    return math::mpi_exact_uint64_sum(
        static_cast<std::uint64_t>(local_count));
}
#endif

void synchronize_failure(
    int mpi_size,
    int local_failed,
    const char* context) {
    if (mpi_size == 1) {
        if (local_failed != 0) {
            throw std::runtime_error(context);
        }
        return;
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_failed;
    throw std::runtime_error(
        "Distributed PM failure synchronization requires MPI");
#else
    int any_failed = 0;
    if (MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context);
    }
    if (any_failed != 0) {
        throw std::runtime_error(context);
    }
#endif
}

void synchronize_exception(
    int mpi_size,
    std::exception_ptr local_exception,
    const char* context) {
    if (mpi_size == 1) {
        if (local_exception) std::rethrow_exception(local_exception);
        return;
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_exception;
    (void)context;
    throw std::runtime_error(
        "Distributed PM exception synchronization requires MPI");
#else
    const int local_failed = local_exception ? 1 : 0;
    int any_failed = 0;
    if (MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context);
    }
    if (any_failed != 0) {
        if (local_exception) std::rethrow_exception(local_exception);
        throw std::runtime_error(
            std::string(context) + " failed on another rank");
    }
#endif
}

long double rank_order_sum(
    long double local_value,
    int mpi_size,
    const char* context) {
    synchronize_failure(
        mpi_size,
        std::isfinite(local_value) ? 0 : 1,
        context);
    if (mpi_size == 1) return local_value;
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_value;
    throw std::runtime_error(
        "Distributed PM rank-order summation requires MPI");
#else
    std::vector<long double> rank_values;
    std::exception_ptr allocation_exception;
    try {
        rank_values.resize(static_cast<std::size_t>(mpi_size));
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    synchronize_exception(
        mpi_size, allocation_exception,
        "Distributed PM diagnostic rank-sum allocation");
    if (MPI_Allgather(
            &local_value,
            1,
            MPI_LONG_DOUBLE,
            rank_values.data(),
            1,
            MPI_LONG_DOUBLE,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allgather failed for ") + context);
    }
    long double sum = 0.0L;
    long double compensation = 0.0L;
    for (const long double value : rank_values) {
        const long double updated = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            compensation += (sum - updated) + value;
        } else {
            compensation += (value - updated) + sum;
        }
        sum = updated;
    }
    const long double total = sum + compensation;
    if (!std::isfinite(total)) {
        throw std::overflow_error(
            std::string(context) + " is non-finite after rank-order summation");
    }
    return total;
#endif
}

} // namespace

std::unique_ptr<DistributedPMSolver> DistributedPMSolver::create(
    core::Real box_size,
    std::size_t mesh_size,
    bool use_tree_pm,
    core::Real r_s,
    bool deconvolve_cic,
    bool use_exact_gradient) {
    runtime::require_active_mpi_main_thread(
        "DistributedPMSolver::create");

#if !defined(COSMO_NBODY_HAS_MPI) || !defined(COSMO_NBODY_HAS_FFTW_MPI)
    (void)box_size;
    (void)mesh_size;
    (void)use_tree_pm;
    (void)r_s;
    (void)deconvolve_cic;
    (void)use_exact_gradient;
    throw std::runtime_error(
        "DistributedPMSolver requires an MPI-enabled FFTW-MPI build");
#else
    int rank = 0;
    int size = 0;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS
        || MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
        throw std::runtime_error(
            "DistributedPMSolver failed to query MPI topology");
    }

    std::exception_ptr parameter_exception;
    try {
        validate_solver_parameters(
            box_size,
            mesh_size,
            use_tree_pm,
            r_s,
            deconvolve_cic,
            use_exact_gradient,
            size);
    } catch (...) {
        parameter_exception = std::current_exception();
    }
    synchronize_exception(
        size,
        parameter_exception,
        "DistributedPMSolver parameter validation");
    require_solver_parameter_agreement(
        box_size,
        mesh_size,
        use_tree_pm,
        r_s,
        deconvolve_cic,
        use_exact_gradient);

    const mesh::MpiFFTAllocationLayout* layout = nullptr;
    std::exception_ptr layout_exception;
    try {
        layout = &mesh::shared_mpi_fft_allocation_layout(mesh_size);
        if (layout->grid_size != mesh_size
            || layout->rank != rank
            || layout->communicator_size != size) {
            throw std::logic_error(
                "DistributedPMSolver FFTW-MPI layout topology mismatch");
        }
    } catch (...) {
        layout_exception = std::current_exception();
    }
    synchronize_exception(
        size,
        layout_exception,
        "DistributedPMSolver allocation-layout preparation");

    std::unique_ptr<DistributedPMSolver> solver;
    std::exception_ptr allocation_exception;
    try {
        solver.reset(new DistributedPMSolver(
            LocalConstructionTag{},
            box_size,
            mesh_size,
            use_tree_pm,
            r_s,
            deconvolve_cic,
            use_exact_gradient,
            *layout,
            rank,
            size));
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    synchronize_exception(
        size,
        allocation_exception,
        "DistributedPMSolver rank-local allocation");

    // This is the first stage that publishes collective FFTW-MPI plans. No
    // throwing rank-local allocation is permitted after it succeeds.
    solver->fft_.emplace(
        mesh_size,
        solver->fft_real_storage_,
        std::span<std::complex<core::Real>>(
            solver->phi_modes_.data(), solver->phi_modes_.size()));

    std::exception_ptr post_plan_exception;
    try {
        const auto& planned_layout = solver->fft_->allocation_layout();
        if (planned_layout.grid_size != solver->mesh_size_
            || planned_layout.rank != solver->rank_
            || planned_layout.communicator_size != solver->size_
            || planned_layout.local_n0 != solver->geometry_.local_n0()
            || planned_layout.local_0_start
                != solver->geometry_.local_0_start()
            || planned_layout.alloc_local_complex_elements
                != solver->geometry_.complex_size()) {
            throw std::logic_error(
                "DistributedPMSolver post-plan layout validation failed");
        }
    } catch (...) {
        post_plan_exception = std::current_exception();
    }
    synchronize_exception(
        size,
        post_plan_exception,
        "DistributedPMSolver post-plan layout validation");
    return solver;
#endif
}

DistributedPMSolver::DistributedPMSolver(
    LocalConstructionTag,
    core::Real box_size,
    std::size_t mesh_size,
    bool use_tree_pm,
    core::Real r_s,
    bool deconvolve_cic,
    bool use_exact_gradient,
    const mesh::MpiFFTAllocationLayout& layout,
    int rank,
    int size)
    : box_size_(box_size),
      mesh_size_(mesh_size),
      use_exact_gradient_(use_exact_gradient),
      rank_(rank),
      size_(size),
      geometry_(
          box_size,
          mesh_size,
          layout.local_n0,
          layout.local_0_start,
          layout.alloc_local_complex_elements),
      mass_assignment_(geometry_),
      green_(
          geometry_,
          use_tree_pm
              ? mesh::PMForceMethod::treepm_long_range(r_s)
              : mesh::PMForceMethod::pure_pm(deconvolve_cic)),
      fft_real_storage_(layout),
      density_(
          fft_real_storage_.compact_real().data(),
          fft_real_storage_.compact_real().size(),
          mesh::external_mesh_storage),
      real_force_(use_exact_gradient ? geometry_.real_size() : 0),
      phi_modes_(geometry_.complex_size()),
      force_modes_(
          use_exact_gradient
              ? std::make_unique<mesh::DestructiveComplexField>(geometry_.complex_size())
              : nullptr),
      exchange_plane_(checked_plane_size(mesh_size)),
      recv_left_plane_(checked_plane_size(mesh_size)),
      recv_right_plane_(checked_plane_size(mesh_size)),
      fft_(std::nullopt) {}

core::Real DistributedPMSolver::exchange_deposit_planes() {
    if (size_ == 1) {
        const std::array<std::span<const core::Real>, 1> local_segments{
            std::span<const core::Real>(density_.data(), density_.size())};
        return math::mpi_exact_nonnegative_sum_segments(local_segments);
    }
#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        "Distributed PM plane exchange requires MPI");
#else
    const std::array<std::span<const core::Real>, 2> pre_exchange_segments{
        std::span<const core::Real>(density_.data(), density_.size()),
        std::span<const core::Real>(exchange_plane_.data(), exchange_plane_.size())};
    const core::Real pre_exchange_mass =
        math::mpi_exact_nonnegative_sum_segments(pre_exchange_segments);

    const int left = (rank_ - 1 + size_) % size_;
    const int right = (rank_ + 1) % size_;
    const int count = checked_mpi_count(exchange_plane_.size());

    if (MPI_Sendrecv(
            exchange_plane_.data(), count, MPI_DOUBLE, right, 4101,
            recv_left_plane_.data(), count, MPI_DOUBLE, left, 4101,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Sendrecv failed for right CIC deposit overflow");
    }

    const std::size_t n = mesh_size_;
    for (std::size_t iy = 0; iy < n; ++iy) {
        for (std::size_t iz = 0; iz < n; ++iz) {
            const std::size_t plane_index = iy * n + iz;
            density_[geometry_.real_index(0, iy, iz)] +=
                recv_left_plane_[plane_index];
        }
    }

    const std::array<std::span<const core::Real>, 1> post_exchange_segments{
        std::span<const core::Real>(density_.data(), density_.size())};
    const core::Real post_exchange_mass =
        math::mpi_exact_nonnegative_sum_segments(post_exchange_segments);
    math::require_cic_transport_conservation(
        pre_exchange_mass, post_exchange_mass);
    return post_exchange_mass;
#endif
}

void DistributedPMSolver::exchange_field_planes(
    const mesh::RealField& field) {
    if (size_ == 1) return;
#ifndef COSMO_NBODY_HAS_MPI
    (void)field;
    throw std::runtime_error(
        "Distributed PM field-plane exchange requires MPI");
#else
    const int left = (rank_ - 1 + size_) % size_;
    const int right = (rank_ + 1) % size_;
    const int count = checked_mpi_count(exchange_plane_.size());
    const std::size_t n = mesh_size_;

    for (std::size_t iy = 0; iy < n; ++iy) {
        for (std::size_t iz = 0; iz < n; ++iz) {
            const std::size_t plane_index = iy * n + iz;
            exchange_plane_[plane_index] =
                field[geometry_.real_index(0, iy, iz)];
        }
    }

    if (MPI_Sendrecv(
            exchange_plane_.data(), count, MPI_DOUBLE, left, 4201,
            recv_right_plane_.data(), count, MPI_DOUBLE, right, 4201,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Sendrecv failed for right field ghost plane");
    }
#endif
}

void DistributedPMSolver::interpolate_force_field(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> acceleration) {
    std::exception_ptr local_exception;
    try {
        if (real_force_.size() != geometry_.real_size()) {
            throw std::logic_error(
                "Distributed spectral force workspace is not allocated");
        }
        if (size_ == 1) {
            mass_assignment_.interpolate_add(
                real_force_, nullptr,
                pos_x, pos_y, pos_z, acceleration);
        } else {
            exchange_field_planes(real_force_);
            mass_assignment_.interpolate_add(
                real_force_,
                &recv_right_plane_,
                pos_x, pos_y, pos_z, acceleration);
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    synchronize_exception(
        size_, local_exception, "Distributed PM force interpolation");

    int local_invalid = 0;
    for (const core::Real value : acceleration) {
        local_invalid |= !std::isfinite(value);
    }
    synchronize_failure(
        size_,
        local_invalid,
        "Distributed PM accumulated force is non-finite");
}

void DistributedPMSolver::compute_axis_spectral(
    int axis,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<core::Real> acceleration) {
    if (axis < 0 || axis > 2) {
        throw std::invalid_argument(
            "DistributedPMSolver spectral axis must be 0,1,2");
    }
    if (!force_modes_) {
        throw std::logic_error(
            "Distributed spectral gradient workspace is not allocated");
    }

    auto& force_modes = *force_modes_;
    std::fill(
        force_modes.begin(), force_modes.end(),
        std::complex<core::Real>{0.0, 0.0});
    const std::size_t n = mesh_size_;
    const std::size_t nzc = n / 2 + 1;
    int local_invalid = 0;
    for (std::size_t local_ix = 0;
         local_ix < geometry_.local_n0();
         ++local_ix) {
        const std::size_t global_ix = geometry_.global_ix(local_ix);
        core::Real kx = geometry_.k_component(global_ix);
        if (n % 2 == 0 && global_ix == n / 2) kx = 0.0;
        for (std::size_t iy = 0; iy < n; ++iy) {
            core::Real ky = geometry_.k_component(iy);
            if (n % 2 == 0 && iy == n / 2) ky = 0.0;
            for (std::size_t iz = 0; iz < nzc; ++iz) {
                core::Real kz = geometry_.k_component(iz);
                if (n % 2 == 0 && iz == n / 2) kz = 0.0;
                const core::Real k = axis == 0
                    ? kx : (axis == 1 ? ky : kz);
                const std::size_t index =
                    geometry_.complex_index(local_ix, iy, iz);
                const auto phi = phi_modes_[index];
                const std::complex<core::Real> value{
                    phi.imag() * k,
                    -phi.real() * k};
                if (!std::isfinite(value.real())
                    || !std::isfinite(value.imag())) {
                    local_invalid = 1;
                } else {
                    force_modes[index] = value;
                }
            }
        }
    }
    synchronize_failure(
        size_,
        local_invalid,
        "Distributed PM spectral force modes are non-finite");

    fft_->inverse(
        std::span<const std::complex<core::Real>>(
            force_modes.data(), force_modes.size()),
        std::span<core::Real>(
            real_force_.data(), real_force_.size()));
    interpolate_force_field(
        pos_x, pos_y, pos_z, acceleration);
}

void DistributedPMSolver::ensure_self_kernel() {
    if (self_kernel_ready_) return;

    const std::size_t global_cells = checked_global_cell_count(mesh_size_);
    const core::Real dx =
        box_size_ / static_cast<core::Real>(mesh_size_);
    if (!std::isfinite(dx) || dx <= 0.0) {
        throw std::overflow_error(
            "Distributed PM self-energy cell size is not representable");
    }

    const bool owns_origin = geometry_.local_n0() != 0
        && geometry_.local_0_start() == 0;
    int owner_count = owns_origin ? 1 : 0;
#ifdef COSMO_NBODY_HAS_MPI
    if (size_ > 1) {
        int global_owner_count = 0;
        if (MPI_Allreduce(
                &owner_count,
                &global_owner_count,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "Distributed PM self-energy origin ownership reduction failed");
        }
        owner_count = global_owner_count;
    }
#endif
    if (owner_count != 1) {
        throw std::logic_error(
            "Distributed PM self-energy origin must belong to exactly one slab");
    }

    const std::array<core::Real, 1> origin{0.0};
    const std::size_t origin_count = owns_origin ? 1U : 0U;
    const std::span<const core::Real> origin_span(
        origin.data(), origin_count);
    std::exception_ptr deposit_exception;
    try {
        mass_assignment_.deposit(
            origin_span,
            origin_span,
            origin_span,
            std::span<const core::Real>{},
            std::optional<core::Real>{1.0},
            density_,
            size_ == 1 ? nullptr : &exchange_plane_);
    } catch (...) {
        deposit_exception = std::current_exception();
    }
    synchronize_exception(
        size_, deposit_exception,
        "Distributed PM self-energy CIC impulse deposition");
    fft_real_storage_.declare_compact_overwrite_complete();

    core::Real deposited_mass = size_ > 1
        ? exchange_deposit_planes()
        : math::mpi_exact_nonnegative_sum_segments(
              std::array<std::span<const core::Real>, 1>{
                  std::span<const core::Real>(
                      density_.data(), density_.size())});
    std::exception_ptr conservation_exception;
    try {
        math::require_cic_mass_conservation(1.0, deposited_mass, 1U);
    } catch (...) {
        conservation_exception = std::current_exception();
    }
    synchronize_exception(
        size_, conservation_exception,
        "Distributed PM self-energy CIC impulse conservation");

    std::optional<math::DensityNormalization> normalization;
    std::exception_ptr normalization_exception;
    try {
        normalization.emplace(
            1.0,
            global_cells,
            dx,
            "Distributed PM self-energy impulse normalization");
    } catch (...) {
        normalization_exception = std::current_exception();
    }
    synchronize_exception(
        size_, normalization_exception,
        "Distributed PM self-energy normalization preparation");
    if (!normalization.has_value()) {
        throw std::logic_error(
            "Distributed PM self-energy normalization was not prepared");
    }

    bool scaled_by_cell_volume = false;
    int local_source_invalid = 0;
    const core::Real mean =
        core::Real{1.0} / static_cast<core::Real>(global_cells);
    if (normalization->direct_available()) {
        const core::Real inverse_cell_volume =
            normalization->inverse_cell_volume();
        for (std::size_t index = 0; index < density_.size(); ++index) {
            const core::Real value =
                (density_[index] - mean) * inverse_cell_volume;
            if (!std::isfinite(value)) {
                local_source_invalid = 1;
            } else {
                density_[index] = value;
            }
        }
    } else {
        scaled_by_cell_volume = true;
        for (std::size_t index = 0; index < density_.size(); ++index) {
            const core::Real value = density_[index] - mean;
            if (!std::isfinite(value)) {
                local_source_invalid = 1;
            } else {
                density_[index] = value;
            }
        }
    }
    synchronize_failure(
        size_, local_source_invalid,
        "Distributed PM self-energy impulse source is non-finite");

    fft_->forward_shared_destructive(
        std::span<core::Real>(density_.data(), density_.size()),
        std::span<std::complex<core::Real>>(
            phi_modes_.data(), phi_modes_.size()));
    green_.apply(phi_modes_);
    fft_->inverse(
        std::span<const std::complex<core::Real>>(
            phi_modes_.data(), phi_modes_.size()),
        std::span<core::Real>(density_.data(), density_.size()));

    int local_potential_invalid = 0;
    for (const core::Real value : density_) {
        local_potential_invalid |= !std::isfinite(value);
    }
    synchronize_failure(
        size_, local_potential_invalid,
        "Distributed PM self-energy impulse response is non-finite");

    std::array<core::Real, 27> local_kernel{};
    std::array<int, 27> local_owners{};
    const std::size_t local_start = geometry_.local_0_start();
    const std::size_t local_end = local_start + geometry_.local_n0();
    for (int lag_x = -1; lag_x <= 1; ++lag_x) {
        const std::size_t gx = static_cast<std::size_t>(
            lag_x + static_cast<int>(mesh_size_)) % mesh_size_;
        const bool local_x = gx >= local_start && gx < local_end;
        for (int lag_y = -1; lag_y <= 1; ++lag_y) {
            const std::size_t gy = static_cast<std::size_t>(
                lag_y + static_cast<int>(mesh_size_)) % mesh_size_;
            for (int lag_z = -1; lag_z <= 1; ++lag_z) {
                const std::size_t gz = static_cast<std::size_t>(
                    lag_z + static_cast<int>(mesh_size_)) % mesh_size_;
                const std::size_t kernel_index = static_cast<std::size_t>(
                    (lag_x + 1) * 9
                    + (lag_y + 1) * 3
                    + (lag_z + 1));
                if (local_x) {
                    local_kernel[kernel_index] = density_[
                        geometry_.real_index(gx - local_start, gy, gz)];
                    local_owners[kernel_index] = 1;
                }
            }
        }
    }

    std::array<core::Real, 27> kernel = local_kernel;
    std::array<int, 27> owners = local_owners;
#ifdef COSMO_NBODY_HAS_MPI
    if (size_ > 1) {
        if (MPI_Allreduce(
                local_kernel.data(), kernel.data(),
                static_cast<int>(kernel.size()), MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS
            || MPI_Allreduce(
                local_owners.data(), owners.data(),
                static_cast<int>(owners.size()), MPI_INT,
                MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "Distributed PM self-energy lag-kernel reduction failed");
        }
    }
#endif
    for (std::size_t index = 0; index < kernel.size(); ++index) {
        if (owners[index] != 1 || !std::isfinite(kernel[index])) {
            throw std::runtime_error(
                "Distributed PM self-energy lag kernel is incomplete or non-finite");
        }
    }

    self_kernel_ = kernel;
    self_kernel_scaled_by_cell_volume_ = scaled_by_cell_volume;
    self_kernel_ready_ = true;
}

core::Real DistributedPMSolver::compute_cic_self_energy_comoving(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass) const {
    if (!self_kernel_ready_) {
        throw std::logic_error(
            "Distributed PM self-energy kernel is not initialized");
    }
    if (pos_x.size() != pos_y.size() || pos_x.size() != pos_z.size()) {
        throw std::invalid_argument(
            "Distributed PM self-energy position component sizes differ");
    }
    if (!uniform_mass.has_value() && masses.size() != pos_x.size()) {
        throw std::invalid_argument(
            "Distributed PM self-energy mass count differs from particle count");
    }
    const core::Real dx =
        box_size_ / static_cast<core::Real>(mesh_size_);
    if (!std::isfinite(dx) || dx <= 0.0) {
        throw std::overflow_error(
            "Distributed PM self-energy cell size is not representable");
    }

    const core::Accum local_sum = math::deterministic_blocked_sum(
        pos_x.size(),
        [&](std::size_t index) -> core::Accum {
            const core::Real mass = uniform_mass.has_value()
                ? *uniform_mass : masses[index];
            if (!std::isfinite(mass) || mass <= 0.0) {
                return std::numeric_limits<core::Accum>::quiet_NaN();
            }
            std::array<std::array<core::Real, 3>, 3> lag_factor{};
            const std::array<core::Real, 3> coordinates{
                pos_x[index], pos_y[index], pos_z[index]};
            for (std::size_t dim = 0; dim < 3; ++dim) {
                if (!std::isfinite(coordinates[dim])) {
                    return std::numeric_limits<core::Accum>::quiet_NaN();
                }
                const core::Real cell_coordinate =
                    math::wrap(coordinates[dim], box_size_) / dx;
                const core::Real offset =
                    cell_coordinate - std::floor(cell_coordinate);
                const core::Real lower = 1.0 - offset;
                lag_factor[dim] = {
                    offset * lower,
                    lower * lower + offset * offset,
                    offset * lower};
            }
            core::Accum stencil{0.0};
            for (std::size_t lag_x = 0; lag_x < 3; ++lag_x) {
                for (std::size_t lag_y = 0; lag_y < 3; ++lag_y) {
                    for (std::size_t lag_z = 0; lag_z < 3; ++lag_z) {
                        stencil += static_cast<core::Accum>(
                            lag_factor[0][lag_x]
                            * lag_factor[1][lag_y]
                            * lag_factor[2][lag_z]
                            * self_kernel_[
                                lag_x * 9 + lag_y * 3 + lag_z]);
                    }
                }
            }
            return self_kernel_scaled_by_cell_volume_
                ? scaled_mass_space_self_energy_term(mass, stencil, dx)
                : scaled_self_energy_term(mass, stencil);
        });
    const long double global_sum = rank_order_sum(
        static_cast<long double>(local_sum),
        size_,
        "Distributed PM CIC self-energy diagnostic");
    return checked_real(
        global_sum,
        "Distributed PM CIC self-energy diagnostic");
}

DistributedPMForceDiagnostics DistributedPMSolver::collect_force_diagnostics(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass) {
    // Pure PM already has the real-space potential in density_. The exact
    // TreePM long-range path kept phi_modes_ intact while inverse-transforming
    // derivative modes, so materialize phi once here only when diagnostics are
    // requested. This retains O(N_mesh^3 / P) storage on every rank.
    if (use_exact_gradient_) {
        fft_->inverse(
            std::span<const std::complex<core::Real>>(
                phi_modes_.data(), phi_modes_.size()),
            std::span<core::Real>(density_.data(), density_.size()));
    }
    int local_potential_invalid = 0;
    for (const core::Real value : density_) {
        local_potential_invalid |= !std::isfinite(value);
    }
    synchronize_failure(
        size_, local_potential_invalid,
        "Distributed PM diagnostic potential is non-finite");

    if (size_ > 1) exchange_field_planes(density_);

    core::Accum local_mass_weighted_potential = 0.0;
    std::exception_ptr interpolation_exception;
    try {
        local_mass_weighted_potential =
            mass_assignment_.interpolate_mass_weighted_sum(
                density_,
                size_ == 1 ? nullptr : &recv_right_plane_,
                pos_x,
                pos_y,
                pos_z,
                masses,
                uniform_mass,
                core::Accum{0.5});
    } catch (...) {
        interpolation_exception = std::current_exception();
    }
    synchronize_exception(
        size_, interpolation_exception,
        "Distributed PM diagnostic potential interpolation/reduction");

    const long double global_potential = rank_order_sum(
        static_cast<long double>(local_mass_weighted_potential),
        size_,
        "Distributed PM potential-energy diagnostic");
    const core::Real potential_energy_comoving = checked_real(
        global_potential,
        "Distributed PM potential-energy diagnostic");

    ensure_self_kernel();
    const core::Real self_energy_comoving =
        compute_cic_self_energy_comoving(
            pos_x, pos_y, pos_z, masses, uniform_mass);
    return DistributedPMForceDiagnostics{
        potential_energy_comoving,
        self_energy_comoving};
}

std::optional<DistributedPMForceDiagnostics> DistributedPMSolver::compute_forces(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<core::Real> acc_x,
    std::span<core::Real> acc_y,
    std::span<core::Real> acc_z,
    bool collect_potential_energy,
    mesh::CICDepositWorkspace* deposition_workspace) {
    runtime::require_active_mpi_main_thread(
        "DistributedPMSolver::compute_forces");

    const int local_size_invalid =
        pos_x.size() != pos_y.size()
        || pos_x.size() != pos_z.size()
        || pos_x.size() != acc_x.size()
        || pos_x.size() != acc_y.size()
        || pos_x.size() != acc_z.size();
    synchronize_failure(
        size_,
        local_size_invalid,
        "DistributedPMSolver component sizes must match on every rank");

    const core::Real total_mass = math::mpi_exact_nonnegative_sum(
        pos_x.size(), masses, uniform_mass);
#ifdef COSMO_NBODY_HAS_MPI
    const std::uint64_t particle_count = collective_particle_count(
        pos_x.size());
#else
    const std::uint64_t particle_count =
        static_cast<std::uint64_t>(pos_x.size());
#endif

    std::exception_ptr deposit_exception;
    try {
        if (deposition_workspace) {
            mass_assignment_.deposit_reusing_workspace(
                pos_x,
                pos_y,
                pos_z,
                masses,
                uniform_mass,
                *deposition_workspace,
                density_,
                size_ == 1 ? nullptr : &exchange_plane_);
        } else {
            mass_assignment_.deposit(
                pos_x,
                pos_y,
                pos_z,
                masses,
                uniform_mass,
                density_,
                size_ == 1 ? nullptr : &exchange_plane_);
        }
    } catch (...) {
        deposit_exception = std::current_exception();
    }
    synchronize_exception(
        size_, deposit_exception, "Distributed PM CIC deposition");
    fft_real_storage_.declare_compact_overwrite_complete();

    core::Real deposited_mass = 0.0;
    if (size_ > 1) {
        deposited_mass = exchange_deposit_planes();
    }

    int local_deposit_invalid = 0;
    for (const core::Real value : density_) {
        local_deposit_invalid |= !std::isfinite(value) || value < 0.0;
    }
    synchronize_failure(
        size_,
        local_deposit_invalid,
        "Distributed PM deposited density is invalid");

    if (size_ == 1) {
        const std::array<std::span<const core::Real>, 1> local_segments{
            std::span<const core::Real>(density_.data(), density_.size())};
        deposited_mass =
            math::mpi_exact_nonnegative_sum_segments(local_segments);
    }
    std::exception_ptr mass_conservation_exception;
    try {
        math::require_cic_mass_conservation(
            total_mass, deposited_mass, particle_count);
    } catch (...) {
        mass_conservation_exception = std::current_exception();
    }
    synchronize_exception(
        size_,
        mass_conservation_exception,
        "Distributed PM CIC mass conservation");

    if (!std::isfinite(total_mass) || total_mass <= 0.0) {
        throw std::runtime_error(
            "Distributed PM total mass must be finite and positive");
    }

    const std::size_t global_cells = checked_global_cell_count(mesh_size_);
    const core::Real dx =
        box_size_ / static_cast<core::Real>(mesh_size_);

    std::optional<math::DensityNormalization> density_normalization;
    std::exception_ptr normalization_exception;
    try {
        density_normalization.emplace(
            total_mass,
            global_cells,
            dx,
            "Distributed PM density normalization");
    } catch (...) {
        normalization_exception = std::current_exception();
    }
    synchronize_exception(
        size_,
        normalization_exception,
        "Distributed PM density normalization preparation");
    if (!density_normalization.has_value()) {
        throw std::logic_error(
            "Distributed PM density normalization was not prepared");
    }

    int local_source_invalid = 0;
    if (density_normalization->direct_available()) {
        const core::Real mean_mass_per_cell =
            density_normalization->mean_mass_per_cell();
        const core::Real inverse_cell_volume =
            density_normalization->inverse_cell_volume();
        for (std::size_t i = 0; i < density_.size(); ++i) {
            const core::Real value =
                (density_[i] - mean_mass_per_cell) * inverse_cell_volume;
            if (!std::isfinite(value)) {
                local_source_invalid = 1;
            } else {
                density_[i] = value;
            }
        }
    } else {
        for (std::size_t i = 0; i < density_.size(); ++i) {
            try {
                density_[i] = density_normalization->normalize(
                    density_[i],
                    "Distributed PM scale-safe density source");
            } catch (...) {
                local_source_invalid = 1;
            }
        }
    }
    synchronize_failure(
        size_,
        local_source_invalid,
        "Distributed PM density contrast source is non-finite or unrepresentable");

    fft_->forward_shared_destructive(
        std::span<core::Real>(density_.data(), density_.size()),
        std::span<std::complex<core::Real>>(
            phi_modes_.data(), phi_modes_.size()));
    green_.apply(phi_modes_);

    int local_phi_invalid = 0;
    for (const auto& value : phi_modes_) {
        local_phi_invalid |= !std::isfinite(value.real())
            || !std::isfinite(value.imag());
    }
    synchronize_failure(
        size_,
        local_phi_invalid,
        "Distributed PM potential modes are non-finite");

    if (use_exact_gradient_) {
        compute_axis_spectral(0, pos_x, pos_y, pos_z, acc_x);
        compute_axis_spectral(1, pos_x, pos_y, pos_z, acc_y);
        compute_axis_spectral(2, pos_x, pos_y, pos_z, acc_z);
    } else {
        // The density source is dead after the forward transform. Reuse its slab
        // for the inverse-transformed potential. The finite-difference gradient
        // is then gathered directly from potential plus the exact periodic halo,
        // so no full real_force_ slab is retained for pure PM.
        fft_->inverse(
            std::span<const std::complex<core::Real>>(
                phi_modes_.data(), phi_modes_.size()),
            std::span<core::Real>(
                density_.data(), density_.size()));
        int local_potential_invalid = 0;
        for (const core::Real value : density_) {
            local_potential_invalid |= !std::isfinite(value);
        }
        synchronize_failure(
            size_,
            local_potential_invalid,
            "Distributed PM real-space potential is non-finite");

        if (size_ == 1) {
            std::exception_ptr gather_exception;
            try {
                mass_assignment_.interpolate_centered_gradient3_add_full_mesh(
                    density_,
                    pos_x,
                    pos_y,
                    pos_z,
                    acc_x,
                    acc_y,
                    acc_z);
            } catch (...) {
                gather_exception = std::current_exception();
            }
            synchronize_exception(
                size_,
                gather_exception,
                "Distributed PM single-rank fused gradient gather");
        } else {
            const mesh::MpiFFTCommunicatorLayout* communicator = nullptr;
            std::vector<mesh::PotentialHaloPlane> halo_plan;
            std::exception_ptr plan_exception;
            try {
                communicator =
                    &mesh::shared_mpi_fft_communicator_layout(mesh_size_);
                halo_plan = mesh::potential_halo_plan(*communicator, rank_);
            } catch (...) {
                plan_exception = std::current_exception();
            }
            synchronize_exception(
                size_,
                plan_exception,
                "Distributed PM potential-halo planning");

            const auto halo_values = mesh::exchange_potential_halo_planes(
                *communicator,
                rank_,
                std::span<const core::Real>(
                    density_.data(), density_.size()));

            // First prove every local stencil, input, and final addition without
            // changing output. Only after all ranks agree do ranks publish the
            // identical recomputation. No N-particle transaction is allocated.
            std::exception_ptr validation_exception;
            try {
                mesh::interpolate_centered_gradient3_add_slab(
                    geometry_,
                    density_,
                    halo_plan,
                    halo_values,
                    pos_x,
                    pos_y,
                    pos_z,
                    acc_x,
                    acc_y,
                    acc_z,
                    true);
            } catch (...) {
                validation_exception = std::current_exception();
            }
            synchronize_exception(
                size_,
                validation_exception,
                "Distributed PM slab fused gradient validation");

            std::exception_ptr gather_exception;
            try {
                mesh::interpolate_centered_gradient3_add_slab(
                    geometry_,
                    density_,
                    halo_plan,
                    halo_values,
                    pos_x,
                    pos_y,
                    pos_z,
                    acc_x,
                    acc_y,
                    acc_z);
            } catch (...) {
                gather_exception = std::current_exception();
            }
            synchronize_exception(
                size_,
                gather_exception,
                "Distributed PM slab fused gradient publication");
        }
    }

    if (!collect_potential_energy) return std::nullopt;
    return collect_force_diagnostics(
        pos_x, pos_y, pos_z, masses, uniform_mass);
}

} // namespace gravity
} // namespace cosmo_nbody
