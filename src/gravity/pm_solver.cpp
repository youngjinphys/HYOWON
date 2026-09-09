#include "cosmo_nbody/gravity/pm_solver.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/math/cic_mass_conservation.hpp"
#include "cosmo_nbody/math/density_normalization.hpp"
#include "cosmo_nbody/math/deterministic_sum.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/mpi_exact_nonnegative_sum.hpp"
#include "cosmo_nbody/math/mpi_exact_uint64_sum.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace gravity {
namespace {

const mesh::MeshGeometry& require_full_mesh(
    const mesh::MeshGeometry& geom) {
    if (geom.local_n0() == 0) {
        throw std::runtime_error("PMSolver requires local_n0 > 0");
    }
    if (geom.local_n0() != geom.grid_size()
        || geom.local_0_start() != 0) {
        throw std::runtime_error(
            "PMSolver serial/replicated backend requires full-mesh geometry");
    }
    return geom;
}

std::uint64_t checked_local_particle_count(std::size_t count) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "PMSolver particle count exceeds uint64_t");
        }
    }
    return static_cast<std::uint64_t>(count);
}

std::uint64_t admitted_particle_count(
    std::size_t local_count,
    bool mpi_global_reduce,
    int mpi_size) {
    std::uint64_t local = 0;
    std::exception_ptr local_exception;
    try {
        local = checked_local_particle_count(local_count);
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (!mpi_global_reduce || mpi_size == 1) {
        if (local_exception) std::rethrow_exception(local_exception);
        return local;
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)mpi_size;
    throw std::runtime_error(
        "PMSolver global particle count requires MPI");
#else
    runtime::synchronize_mpi_exception(
        local_exception,
        mpi_size,
        "PMSolver local particle-count representation");
    return math::mpi_exact_uint64_sum(local);
#endif
}

core::Real exact_local_input_mass(
    std::size_t particle_count,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass) {
    math::ExactPositiveDoubleSum sum;
    if (uniform_mass.has_value()) {
        if (!masses.empty()) {
            throw std::invalid_argument(
                "PMSolver uniform mass mode requires an empty explicit mass span");
        }
        if (!std::isfinite(*uniform_mass) || *uniform_mass < 0.0) {
            throw std::invalid_argument(
                "PMSolver uniform mass must be finite and non-negative");
        }
        sum.add_repeated(
            *uniform_mass,
            checked_local_particle_count(particle_count));
    } else {
        if (masses.size() != particle_count) {
            throw std::invalid_argument(
                "PMSolver explicit mass count differs from particle count");
        }
        for (const core::Real mass : masses) {
            if (!std::isfinite(mass) || mass < 0.0) {
                throw std::invalid_argument(
                    "PMSolver explicit masses must be finite and non-negative");
            }
            sum.add(mass);
        }
    }
    return sum.value();
}

core::Real admitted_input_mass(
    std::size_t local_count,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    bool mpi_global_reduce,
    int mpi_size) {
    if (!mpi_global_reduce || mpi_size == 1) {
        return exact_local_input_mass(local_count, masses, uniform_mass);
    }
    return math::mpi_exact_nonnegative_sum(local_count, masses, uniform_mass);
}

core::Real exact_nonnegative_mesh_mass(
    const mesh::RealField& field,
    std::size_t logical_size) {
    if (logical_size > field.size()) {
        throw std::logic_error(
            "PMSolver logical mesh exceeds deposited storage");
    }
    math::ExactPositiveDoubleSum sum;
    for (std::size_t index = 0; index < logical_size; ++index) {
        const core::Real value = field[index];
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error(
                "PMSolver deposited mesh contains a non-finite or negative mass");
        }
        sum.add(value);
    }
    return sum.value();
}

void allreduce_mass_mesh_if_requested(
    mesh::RealField& field,
    bool enabled) {
    if (!enabled) return;
#ifdef COSMO_NBODY_HAS_MPI
    runtime::require_active_mpi_main_thread(
        "PMSolver replicated mass-mesh reduction");
    int size = 1;
    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
        throw std::runtime_error(
            "PMSolver failed to query MPI size");
    }
    if (size == 1) return;

    std::size_t offset = 0;
    while (offset < field.size()) {
        const std::size_t remaining = field.size() - offset;
        const int count = static_cast<int>(std::min<std::size_t>(
            remaining,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        if (MPI_Allreduce(
                MPI_IN_PLACE,
                field.data() + offset,
                count,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "PMSolver MPI_Allreduce failed for mass mesh");
        }
        offset += static_cast<std::size_t>(count);
    }
#else
    (void)field;
    throw std::runtime_error(
        "PMSolver MPI reduction requested in a non-MPI build");
#endif
}

bool spans_overlap(
    std::span<const core::Real> lhs,
    std::span<const core::Real> rhs) noexcept {
    if (lhs.empty() || rhs.empty()) return false;
    const std::less<const core::Real*> before;
    const core::Real* lhs_end = lhs.data() + lhs.size();
    const core::Real* rhs_end = rhs.data() + rhs.size();
    return before(lhs.data(), rhs_end) && before(rhs.data(), lhs_end);
}

void require_disjoint_force_io(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::span<core::Real> acc_x,
    std::span<core::Real> acc_y,
    std::span<core::Real> acc_z) {
    const std::span<const core::Real> output_x = acc_x;
    const std::span<const core::Real> output_y = acc_y;
    const std::span<const core::Real> output_z = acc_z;
    const bool output_overlap =
        spans_overlap(output_x, output_y)
        || spans_overlap(output_x, output_z)
        || spans_overlap(output_y, output_z);
    const bool input_output_overlap =
        spans_overlap(output_x, pos_x)
        || spans_overlap(output_x, pos_y)
        || spans_overlap(output_x, pos_z)
        || spans_overlap(output_x, masses)
        || spans_overlap(output_y, pos_x)
        || spans_overlap(output_y, pos_y)
        || spans_overlap(output_y, pos_z)
        || spans_overlap(output_y, masses)
        || spans_overlap(output_z, pos_x)
        || spans_overlap(output_z, pos_y)
        || spans_overlap(output_z, pos_z)
        || spans_overlap(output_z, masses);
    if (output_overlap || input_output_overlap) {
        throw std::invalid_argument(
            "PMSolver acceleration spans must be mutually disjoint and must not overlap position or mass inputs");
    }
}

void zero_force_transaction(
    std::span<core::Real> force_x,
    std::span<core::Real> force_y,
    std::span<core::Real> force_z) noexcept {
    const std::size_t count = force_x.size();
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t index = 0; index < count; ++index) {
        force_x[index] = 0.0;
        force_y[index] = 0.0;
        force_z[index] = 0.0;
    }
}

void validate_force_commit(
    std::span<const core::Real> force_x,
    std::span<const core::Real> force_y,
    std::span<const core::Real> force_z,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z) {
    const std::size_t count = force_x.size();
    if (force_y.size() != count || force_z.size() != count
        || acceleration_x.size() != count
        || acceleration_y.size() != count
        || acceleration_z.size() != count) {
        throw std::logic_error(
            "PMSolver transactional force buffers have inconsistent sizes");
    }

    int non_finite_force = 0;
    int invalid_commit = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for reduction(|:non_finite_force,invalid_commit) \
        schedule(static) if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t index = 0; index < count; ++index) {
        const bool force_invalid =
            !std::isfinite(force_x[index])
            || !std::isfinite(force_y[index])
            || !std::isfinite(force_z[index]);
        non_finite_force |= force_invalid;
        invalid_commit |=
            !std::isfinite(acceleration_x[index])
            || !std::isfinite(acceleration_y[index])
            || !std::isfinite(acceleration_z[index])
            || !std::isfinite(acceleration_x[index] + force_x[index])
            || !std::isfinite(acceleration_y[index] + force_y[index])
            || !std::isfinite(acceleration_z[index] + force_z[index]);
    }

    if (non_finite_force != 0) {
        throw std::runtime_error(
            "PMSolver backend produced a non-finite force");
    }
    if (invalid_commit != 0) {
        throw std::overflow_error(
            "PMSolver force commit is not representable");
    }
}

void commit_force_noexcept(
    std::span<const core::Real> force_x,
    std::span<const core::Real> force_y,
    std::span<const core::Real> force_z,
    std::span<core::Real> acceleration_x,
    std::span<core::Real> acceleration_y,
    std::span<core::Real> acceleration_z) noexcept {
    const std::size_t count = force_x.size();
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t index = 0; index < count; ++index) {
        acceleration_x[index] += force_x[index];
        acceleration_y[index] += force_y[index];
        acceleration_z[index] += force_z[index];
    }
}

} // namespace

int PMSolver::mpi_force_stage_size(const char* context) const {
    if (!mpi_global_reduce_) return 1;
#ifndef COSMO_NBODY_HAS_MPI
    (void)context;
    throw std::runtime_error(
        "PMSolver MPI force stage requested in a non-MPI build");
#else
    runtime::require_active_mpi_main_thread(context);
    int size = 1;
    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size < 1) {
        throw std::runtime_error(
            std::string(context) + " failed to query MPI size");
    }
    return size;
#endif
}

void PMSolver::synchronize_mpi_force_stage(
    std::exception_ptr local_exception,
    int mpi_size,
    const char* context) const {
    if (!mpi_global_reduce_) {
        if (local_exception) std::rethrow_exception(local_exception);
        return;
    }
    runtime::synchronize_mpi_exception(
        local_exception, mpi_size, context);
}

PMSolver::PMSolver(
    const mesh::MeshGeometry& geom,
    mesh::PMForceMethod method,
    bool mpi_global_reduce,
    config::MemoryPolicyParams memory_policy)
    : geom_(geom),
      method_(method),
      mpi_global_reduce_(mpi_global_reduce),
      memory_policy_(std::move(memory_policy)) {}

void PMSolver::ensure_workspace(std::size_t particle_count) {
    if (workspace_initialized_) return;
    (void)particle_count;
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (mpi_global_reduce_) {
        auto distributed_solver = DistributedPMSolver::create(
            geom_.box_size(),
            geom_.grid_size(),
            method_.is_treepm_long_range(),
            method_.split_scale(),
            method_.deconvolves_cic(),
            method_.uses_spectral_gradient());
        distributed_solver_ = std::move(distributed_solver);
        workspace_initialized_ = true;
        return;
    }
#endif
    require_full_mesh(geom_);
    auto fft = std::make_unique<mesh::FFTBackend>(
        geom_, memory_policy_.evolution_scratch_mode,
        memory_policy_.scratch_directory);
    auto mass_assignment =
        std::make_unique<mesh::CICMassAssignment>(geom_, memory_policy_);
    auto green = std::make_unique<mesh::GreenFunction>(geom_, method_);
    std::unique_ptr<runtime::RealScratchBuffer> real_scratch1;
    std::unique_ptr<mesh::RealField> real1;
    const bool file_backed_real = config::uses_file_backed_scratch(
        memory_policy_.evolution_scratch_mode);
    if (file_backed_real) {
        config::MemoryPolicyParams scratch_policy = memory_policy_;
        scratch_policy.ic_scratch_mode = memory_policy_.evolution_scratch_mode;
        real_scratch1 = std::make_unique<runtime::RealScratchBuffer>(
            geom_.padded_real_size(), scratch_policy, "pm_real_1");
        real1 = std::make_unique<mesh::RealField>(
            real_scratch1->data(), real_scratch1->size(),
            mesh::external_mesh_storage);
    } else {
        real1 = std::make_unique<mesh::RealField>(geom_.padded_real_size());
    }

    std::unique_ptr<runtime::RealScratchBuffer> real_scratch2;
    std::unique_ptr<mesh::RealField> real2;
    if (method_.uses_spectral_gradient()) {
        if (file_backed_real) {
            config::MemoryPolicyParams scratch_policy = memory_policy_;
            scratch_policy.ic_scratch_mode = memory_policy_.evolution_scratch_mode;
            real_scratch2 = std::make_unique<runtime::RealScratchBuffer>(
                geom_.padded_real_size(), scratch_policy, "pm_real_2");
            real2 = std::make_unique<mesh::RealField>(
                real_scratch2->data(), real_scratch2->size(),
                mesh::external_mesh_storage);
        } else {
            real2 = std::make_unique<mesh::RealField>(geom_.padded_real_size());
        }
    }

    std::unique_ptr<runtime::RawScratchBuffer> complex_scratch;
    std::unique_ptr<runtime::RawScratchBuffer> phi_scratch;
    const auto complex_field = [&](std::unique_ptr<runtime::RawScratchBuffer>& backing,
                                   const char* label) {
        if (!file_backed_real) {
            return std::make_unique<mesh::ComplexField>(geom_.complex_size());
        }
        static_assert(std::is_trivially_destructible_v<std::complex<core::Real>>);
        if (geom_.complex_size() > std::numeric_limits<std::size_t>::max()
                / sizeof(std::complex<core::Real>)) {
            throw std::overflow_error("PM complex scratch byte size overflows size_t");
        }
        backing = std::make_unique<runtime::RawScratchBuffer>(
            geom_.complex_size() * sizeof(std::complex<core::Real>),
            memory_policy_.evolution_scratch_mode,
            memory_policy_.scratch_directory, label);
        auto* data = static_cast<std::complex<core::Real>*>(backing->data());
        std::uninitialized_value_construct_n(data, geom_.complex_size());
        return std::make_unique<mesh::ComplexField>(
            data, geom_.complex_size(), mesh::external_mesh_storage);
    };
    auto complex = complex_field(complex_scratch, "pm_complex");
    std::unique_ptr<mesh::ComplexField> phi;
    if (method_.uses_spectral_gradient()) {
        phi = complex_field(phi_scratch, "pm_phi");
    }

    fft_ = std::move(fft);
    mass_assign_ = std::move(mass_assignment);
    green_fn_ = std::move(green);
    real_scratch_1_ = std::move(real_scratch1);
    real_scratch_2_ = std::move(real_scratch2);
    complex_scratch_ = std::move(complex_scratch);
    phi_scratch_ = std::move(phi_scratch);
    real_buf_1_ = std::move(real1);
    real_buf_2_ = std::move(real2);
    complex_buf_ = std::move(complex);
    phi_k_buf_ = std::move(phi);
    workspace_initialized_ = true;
    std::clog << "[pm] workspace_scratch_mode="
              << config::scratch_mode_name(memory_policy_.evolution_scratch_mode)
              << " real_elements_each=" << real_buf_1_->size()
              << " real_bytes_each=" << real_buf_1_->size() * sizeof(core::Real)
              << " complex_elements_each=" << complex_buf_->size()
              << " complex_bytes_each=" << complex_buf_->size() * sizeof(std::complex<core::Real>)
              << " real_array_count=" << (method_.uses_spectral_gradient() ? 2 : 1)
              << " complex_array_count=" << (method_.uses_spectral_gradient() ? 2 : 1)
              << '\n';
}

std::optional<PMForceDiagnostics> PMSolver::compute_forces(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<core::Real> acc_x,
    std::span<core::Real> acc_y,
    std::span<core::Real> acc_z,
    bool collect_potential_energy) {
    const int mpi_size = mpi_force_stage_size(
        "PMSolver public force preflight");
    const std::size_t count = pos_x.size();
    std::exception_ptr preflight_exception;
    try {
        if (pos_y.size() != count || pos_z.size() != count
            || acc_x.size() != count || acc_y.size() != count
            || acc_z.size() != count) {
            throw std::invalid_argument(
                "PMSolver position and acceleration component sizes must match");
        }
        if (uniform_mass.has_value()) {
            if (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0) {
                throw std::invalid_argument(
                    "PMSolver uniform mass must be finite and positive");
            }
        } else if (masses.size() != count) {
            throw std::invalid_argument(
                "PMSolver explicit mass count must match particle count");
        }
        require_disjoint_force_io(
            pos_x, pos_y, pos_z, masses, acc_x, acc_y, acc_z);
    } catch (...) {
        preflight_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        preflight_exception, mpi_size, "PMSolver public force preflight");

    std::unique_ptr<core::Real[]> force_x_storage;
    std::unique_ptr<core::Real[]> force_y_storage;
    std::unique_ptr<core::Real[]> force_z_storage;
    std::exception_ptr allocation_exception;
    try {
        force_x_storage = std::unique_ptr<core::Real[]>(new core::Real[count]);
        force_y_storage = std::unique_ptr<core::Real[]>(new core::Real[count]);
        force_z_storage = std::unique_ptr<core::Real[]>(new core::Real[count]);
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        allocation_exception, mpi_size,
        "PMSolver public force transaction allocation");
    std::span<core::Real> force_x(force_x_storage.get(), count);
    std::span<core::Real> force_y(force_y_storage.get(), count);
    std::span<core::Real> force_z(force_z_storage.get(), count);
    zero_force_transaction(force_x, force_y, force_z);

    const auto diagnostics = compute_forces_in_place(
        pos_x, pos_y, pos_z, masses, uniform_mass,
        force_x, force_y, force_z, collect_potential_energy);

    std::exception_ptr validation_exception;
    try {
        validate_force_commit(
            force_x, force_y, force_z, acc_x, acc_y, acc_z);
    } catch (...) {
        validation_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        validation_exception, mpi_size,
        "PMSolver public force completion validation");
    commit_force_noexcept(force_x, force_y, force_z, acc_x, acc_y, acc_z);
    return diagnostics;
}

std::optional<PMForceDiagnostics> PMSolver::compute_forces_in_place(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    std::span<core::Real> force_x,
    std::span<core::Real> force_y,
    std::span<core::Real> force_z,
    bool collect_potential_energy,
    mesh::CICDepositWorkspace* deposition_workspace) {
    const int mpi_size = mpi_force_stage_size(
        "PMSolver private force setup");
    std::exception_ptr setup_exception;
    try {
        if (pos_x.size() != pos_y.size()
            || pos_x.size() != pos_z.size()
            || pos_x.size() != force_x.size()
            || pos_x.size() != force_y.size()
            || pos_x.size() != force_z.size()) {
            throw std::invalid_argument(
                "PMSolver private force component sizes must match");
        }
        ensure_workspace(pos_x.size());
    } catch (...) {
        setup_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        setup_exception, mpi_size, "PMSolver private force setup");
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (distributed_solver_) {
        auto distributed = distributed_solver_->compute_forces(
            pos_x, pos_y, pos_z, masses, uniform_mass,
            force_x, force_y, force_z, collect_potential_energy,
            deposition_workspace);
        if (!distributed.has_value()) return std::nullopt;
        return PMForceDiagnostics{
            distributed->potential_energy_comoving,
            distributed->cic_self_energy_comoving};
    }
#endif

    std::exception_ptr backend_setup_exception;
    try {
        if (!fft_ || !mass_assign_ || !green_fn_
            || !real_buf_1_ || !complex_buf_) {
            throw std::logic_error(
                "PMSolver serial/replicated backend is not initialized");
        }
        if (method_.uses_spectral_gradient() && (!real_buf_2_ || !phi_k_buf_)) {
            throw std::logic_error(
                "PMSolver exact-gradient workspace is incomplete");
        }
        if (method_.uses_spectral_gradient() && collect_potential_energy) {
            throw std::logic_error(
                "PM potential-energy diagnostics are not implemented for the exact-gradient/spectral force path; the request cannot be satisfied silently");
        }
    } catch (...) {
        backend_setup_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        backend_setup_exception, mpi_size,
        "PMSolver serial/replicated backend setup");

    auto& real1 = *real_buf_1_;
    auto& modes = *complex_buf_;
    const std::size_t N = geom_.grid_size();
    const core::Real dx = geom_.cell_size();

    std::exception_ptr deposit_exception;
    try {
        if (deposition_workspace) {
            mass_assign_->deposit_reusing_workspace(
                pos_x, pos_y, pos_z, masses, uniform_mass,
                *deposition_workspace, real1);
        } else {
            mass_assign_->deposit(
                pos_x, pos_y, pos_z, masses, uniform_mass, real1);
        }
    } catch (...) {
        deposit_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        deposit_exception, mpi_size, "PMSolver CIC deposition");

    core::Real input_mass = 0.0;
    std::uint64_t particle_count = 0;
    std::exception_ptr input_mass_exception;
    try {
        input_mass = admitted_input_mass(
            pos_x.size(), masses, uniform_mass,
            mpi_global_reduce_, mpi_size);
        particle_count = admitted_particle_count(
            pos_x.size(), mpi_global_reduce_, mpi_size);
        if (!std::isfinite(input_mass) || input_mass <= 0.0
            || particle_count == 0) {
            throw std::runtime_error(
                "PMSolver exact input mass identity must be finite and positive");
        }
    } catch (...) {
        input_mass_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        input_mass_exception, mpi_size,
        "PMSolver exact input mass identity");

    allreduce_mass_mesh_if_requested(real1, mpi_global_reduce_);

    std::optional<PMForceDiagnostics> diagnostics;
    std::exception_ptr post_reduction_exception;
    try {
        const core::Real deposited_mass = exact_nonnegative_mesh_mass(
            real1, geom_.real_size());
        if (!std::isfinite(deposited_mass) || deposited_mass <= 0.0) {
            throw std::runtime_error(
                "PMSolver total deposited mass must be finite and positive");
        }
        math::require_cic_mass_conservation(
            input_mass, deposited_mass, particle_count);

        if (N != 0 && N > std::numeric_limits<std::size_t>::max() / N) {
            throw std::overflow_error(
                "PMSolver mesh cell count overflows size_t");
        }
        const std::size_t n2 = N * N;
        if (N != 0 && n2 > std::numeric_limits<std::size_t>::max() / N) {
            throw std::overflow_error(
                "PMSolver mesh cell count overflows size_t");
        }
        const std::size_t num_cells = n2 * N;
        const math::DensityNormalization density_normalization(
            input_mass, num_cells, dx, "PMSolver density normalization");

        if (density_normalization.direct_available()) {
            const core::Real mean_mass_per_cell =
                density_normalization.mean_mass_per_cell();
            const core::Real inverse_cell_volume =
                density_normalization.inverse_cell_volume();
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (std::size_t local_ix = 0;
                 local_ix < geom_.local_n0(); ++local_ix) {
                for (std::size_t iy = 0; iy < N; ++iy) {
                    for (std::size_t iz = 0; iz < N; ++iz) {
                        const std::size_t idx =
                            geom_.real_index(local_ix, iy, iz);
                        real1[idx] =
                            (real1[idx] - mean_mass_per_cell)
                            * inverse_cell_volume;
                    }
                }
            }
        } else {
            int normalization_failed = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp parallel for collapse(2) schedule(static) \
                reduction(|: normalization_failed)
#endif
            for (std::size_t local_ix = 0;
                 local_ix < geom_.local_n0(); ++local_ix) {
                for (std::size_t iy = 0; iy < N; ++iy) {
                    for (std::size_t iz = 0; iz < N; ++iz) {
                        const std::size_t idx =
                            geom_.real_index(local_ix, iy, iz);
                        try {
                            real1[idx] = density_normalization.normalize(
                                real1[idx],
                                "PMSolver scale-safe density source");
                        } catch (...) {
                            normalization_failed = 1;
                        }
                    }
                }
            }
            if (normalization_failed != 0) {
                throw std::overflow_error(
                    "PMSolver scale-safe density source is not representable");
            }
        }

        if (real_scratch_1_) {
            fft_->forward_external(real1.data(), real1.size(), modes);
        } else {
            fft_->forward(real1, modes);
        }
        green_fn_->apply(modes);
        const std::size_t nz_complex = N / 2 + 1;

        if (method_.uses_spectral_gradient()) {
            auto& real2 = *real_buf_2_;
            auto& phi_modes = *phi_k_buf_;
            std::copy(
                modes.data(), modes.data() + modes.size(), phi_modes.data());

            auto do_axis = [&](int axis, std::span<core::Real> force) {
#ifdef COSMO_NBODY_HAS_OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (std::size_t local_ix = 0;
                     local_ix < geom_.local_n0(); ++local_ix) {
                    const std::size_t ix = geom_.global_ix(local_ix);
                    core::Real kx = geom_.k_component(ix);
                    if (N % 2 == 0 && ix == N / 2) kx = 0.0;
                    for (std::size_t iy = 0; iy < N; ++iy) {
                        core::Real ky = geom_.k_component(iy);
                        if (N % 2 == 0 && iy == N / 2) ky = 0.0;
                        for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                            core::Real kz = geom_.k_component(iz);
                            if (N % 2 == 0 && iz == N / 2) kz = 0.0;
                            const core::Real k =
                                axis == 0 ? kx : (axis == 1 ? ky : kz);
                            const std::size_t idx =
                                geom_.complex_index(local_ix, iy, iz);
                            const std::complex<core::Real> phi = phi_modes[idx];
                            modes[idx] = {phi.imag() * k, -phi.real() * k};
                        }
                    }
                }
                if (real_scratch_2_) {
                    fft_->inverse_external(modes, real2.data(), real2.size());
                } else {
                    fft_->inverse(modes, real2);
                }
                mass_assign_->interpolate_add(
                    real2, nullptr, pos_x, pos_y, pos_z, force);
            };

            do_axis(0, force_x);
            do_axis(1, force_y);
            do_axis(2, force_z);
        } else {
            if (real_scratch_1_) {
                fft_->inverse_external(modes, real1.data(), real1.size());
            } else {
                fft_->inverse(modes, real1);
            }
            if (collect_potential_energy) {
                const core::Accum mass_weighted_potential =
                    mass_assign_->interpolate_mass_weighted_sum_full_mesh(
                        real1, pos_x, pos_y, pos_z, masses, uniform_mass);
                const core::Accum scaled_potential_energy =
                    core::Accum{0.5} * mass_weighted_potential;
                if (mass_weighted_potential != core::Accum{0.0}
                    && scaled_potential_energy == core::Accum{0.0}) {
                    throw std::underflow_error(
                        "PMSolver potential-energy diagnostic underflowed its accumulation format");
                }
                const core::Real potential_energy_comoving =
                    static_cast<core::Real>(scaled_potential_energy);
                if (scaled_potential_energy != core::Accum{0.0}
                    && potential_energy_comoving == core::Real{0.0}) {
                    throw std::underflow_error(
                        "PMSolver potential-energy diagnostic underflowed core::Real");
                }
                if (!std::isfinite(potential_energy_comoving)) {
                    throw std::overflow_error(
                        "PMSolver potential-energy diagnostic is non-finite");
                }
                diagnostics = PMForceDiagnostics{
                    potential_energy_comoving, std::nullopt};
            }
            mass_assign_->interpolate_centered_gradient3_add_full_mesh(
                real1, pos_x, pos_y, pos_z, force_x, force_y, force_z);
        }
    } catch (...) {
        post_reduction_exception = std::current_exception();
    }
    synchronize_mpi_force_stage(
        post_reduction_exception, mpi_size,
        "PMSolver replicated backend post-reduction processing");
    return diagnostics;
}

} // namespace gravity
} // namespace cosmo_nbody
