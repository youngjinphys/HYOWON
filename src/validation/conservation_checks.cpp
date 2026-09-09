#include "cosmo_nbody/validation/conservation_checks.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/gravity/force_split.hpp"
#include "cosmo_nbody/math/conservative_axis_extent.hpp"
#include "cosmo_nbody/math/density_normalization.hpp"
#include "cosmo_nbody/math/deterministic_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/mass_assignment.hpp"
#include "cosmo_nbody/mesh/green_function.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace validation {

namespace {

bool mpi_reduce_enabled(const config::SimulationParameters& config) {
    return config.get_runtime().mpi_enabled;
}

std::size_t checked_mesh_size(const config::SimulationParameters& config) {
    const std::uint64_t configured = config.get_box().N_mesh;
    if (configured
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Conservation mesh dimension does not fit size_t");
    }
    return static_cast<std::size_t>(configured);
}

std::size_t checked_cube_cell_count(std::size_t n) {
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "Conservation mesh cell count overflows size_t");
    }
    const std::size_t n2 = n * n;
    if (n != 0 && n2 > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error(
            "Conservation mesh cell count overflows size_t");
    }
    return n2 * n;
}

#ifdef COSMO_NBODY_HAS_MPI
void require_initialized_mpi(const char* context) {
    runtime::require_active_mpi_main_thread(context);
}
#endif

void synchronize_invalid_flag(
    int local_invalid,
    bool enabled,
    const char* context) {
    if (!enabled) {
        if (local_invalid != 0) {
            throw std::runtime_error(context);
        }
        return;
    }
#ifdef COSMO_NBODY_HAS_MPI
    require_initialized_mpi(context);
    int any_invalid = 0;
    if (MPI_Allreduce(
            &local_invalid,
            &any_invalid,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context);
    }
    if (any_invalid != 0) {
        throw std::runtime_error(context);
    }
#else
    (void)local_invalid;
    throw std::runtime_error(
        "Conservation MPI reduction requested in a non-MPI build");
#endif
}

void synchronize_exception(
    std::exception_ptr local_exception,
    bool enabled,
    const char* context) {
    if (!enabled) {
        if (local_exception) std::rethrow_exception(local_exception);
        return;
    }
#ifdef COSMO_NBODY_HAS_MPI
    require_initialized_mpi(context);
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
            std::string(context) + " failed on another MPI rank");
    }
#else
    (void)local_exception;
    (void)context;
    throw std::runtime_error(
        "Conservation MPI reduction requested in a non-MPI build");
#endif
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
        throw std::overflow_error(
            std::string(quantity) + " underflows core::Real to zero");
    }
    return converted;
}

template <std::size_t Count>
core::Real checked_scaled_linear_combination(
    const std::array<core::Real, Count>& values,
    const std::array<long double, Count>& coefficients,
    const char* quantity) {
    long double scale = 0.0L;
    for (const core::Real value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                std::string(quantity) + " input is non-finite");
        }
        scale = std::max(
            scale,
            std::abs(static_cast<long double>(value)));
    }
    if (scale == 0.0L) return 0.0;

    long double sum = 0.0L;
    long double compensation = 0.0L;
    for (std::size_t index = 0; index < Count; ++index) {
        const long double coefficient = coefficients[index];
        if (!std::isfinite(coefficient)) {
            throw std::overflow_error(
                std::string(quantity) + " coefficient is non-finite");
        }
        const long double term = coefficient
            * (static_cast<long double>(values[index]) / scale);
        if (!std::isfinite(term)) {
            throw std::overflow_error(
                std::string(quantity) + " normalized term is non-finite");
        }
        const long double updated = sum + term;
        if (std::abs(sum) >= std::abs(term)) {
            compensation += (sum - updated) + term;
        } else {
            compensation += (term - updated) + sum;
        }
        if (!std::isfinite(updated) || !std::isfinite(compensation)) {
            throw std::overflow_error(
                std::string(quantity) + " accumulation is non-finite");
        }
        sum = updated;
    }
    return checked_real(scale * (sum + compensation), quantity);
}

core::Real stable_log_scale_ratio(
    core::Real current_a,
    core::Real previous_a) {
    const long double current = static_cast<long double>(current_a);
    const long double previous = static_cast<long double>(previous_a);
    const long double relative = (current - previous) / previous;
    const long double result = std::isfinite(relative) && relative <= 0.5L
        ? std::log1p(relative)
        : std::log(current) - std::log(previous);
    return checked_real(result, "Layzer-Irvine logarithmic scale interval");
}

core::Real collectively_checked_real(
    long double value,
    bool mpi_enabled,
    const char* quantity) {
    core::Real result = 0.0;
    std::exception_ptr local_exception;
    try {
        result = checked_real(value, quantity);
    } catch (...) {
        local_exception = std::current_exception();
    }
    synchronize_exception(local_exception, mpi_enabled, quantity);
    return result;
}

void allreduce_field_if_requested(mesh::RealField& field, bool enabled) {
    int local_invalid = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: local_invalid) \
        if(field.size() >= 32768)
#endif
    for (std::size_t index = 0; index < field.size(); ++index) {
        local_invalid |= !std::isfinite(field[index]);
    }
    synchronize_invalid_flag(
        local_invalid,
        enabled,
        "Conservation density mesh is non-finite before reduction");
    if (!enabled) return;

#ifdef COSMO_NBODY_HAS_MPI
    std::size_t offset = 0;
    while (offset < field.size()) {
        const int count = static_cast<int>(std::min<std::size_t>(
            field.size() - offset,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        if (MPI_Allreduce(
                MPI_IN_PLACE,
                field.data() + offset,
                count,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "MPI_Allreduce failed for conservation density mesh");
        }
        offset += static_cast<std::size_t>(count);
    }

    int global_invalid = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: global_invalid) \
        if(field.size() >= 32768)
#endif
    for (std::size_t index = 0; index < field.size(); ++index) {
        global_invalid |= !std::isfinite(field[index]);
    }
    if (global_invalid != 0) {
        throw std::overflow_error(
            "Conservation density mesh overflowed during MPI reduction");
    }
#else
    throw std::runtime_error(
        "Conservation MPI reduction requested in a non-MPI build");
#endif
}

core::Real allreduce_scalar_if_requested(
    core::Real value,
    bool enabled,
    const char* quantity) {
    synchronize_invalid_flag(
        std::isfinite(value) ? 0 : 1,
        enabled,
        quantity);
    if (!enabled) return value;
#ifdef COSMO_NBODY_HAS_MPI
    core::Real global = 0.0;
    if (MPI_Allreduce(
            &value,
            &global,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + quantity);
    }
    if (!std::isfinite(global)) {
        throw std::overflow_error(
            std::string(quantity) + " overflowed during MPI reduction");
    }
    return global;
#else
    throw std::runtime_error(
        "Conservation MPI reduction requested in a non-MPI build");
#endif
}

int allreduce_max_exponent_if_requested(
    int value,
    bool enabled,
    const char* quantity) {
    if (!enabled) return value;
#ifdef COSMO_NBODY_HAS_MPI
    require_initialized_mpi(quantity);
    int global = 0;
    if (MPI_Allreduce(
            &value,
            &global,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + quantity);
    }
    return global;
#else
    (void)value;
    (void)quantity;
    throw std::runtime_error(
        "Conservation MPI reduction requested in a non-MPI build");
#endif
}

void require_finite_li_sample(
    core::Real current_a,
    core::Real current_K,
    core::Real current_W) {
    if (!std::isfinite(current_a) || current_a <= 0.0) {
        throw std::invalid_argument(
            "Layzer-Irvine scale factor must be finite and positive");
    }
    if (!std::isfinite(current_K) || current_K < 0.0) {
        throw std::invalid_argument(
            "Layzer-Irvine kinetic energy must be finite and non-negative");
    }
    if (!std::isfinite(current_W)) {
        throw std::invalid_argument(
            "Layzer-Irvine potential energy must be finite");
    }
}

long double compensated_sum_rows(
    std::span<const long double> values) noexcept {
    long double sum = 0.0L;
    long double compensation = 0.0L;
    for (const long double value : values) {
        const long double updated = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            compensation += (sum - updated) + value;
        } else {
            compensation += (value - updated) + sum;
        }
        sum = updated;
    }
    return sum + compensation;
}

long double rank_order_sum_if_requested(
    long double local_value,
    bool enabled,
    const char* quantity) {
    synchronize_invalid_flag(
        std::isfinite(local_value) ? 0 : 1,
        enabled,
        quantity);
    if (!enabled) return local_value;
#ifdef COSMO_NBODY_HAS_MPI
    require_initialized_mpi(quantity);
    int mpi_size = 0;
    if (MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS
        || mpi_size < 1) {
        throw std::runtime_error(
            std::string("MPI_Comm_size failed for ") + quantity);
    }

    std::vector<long double> rank_values;
    std::exception_ptr allocation_exception;
    try {
        rank_values.resize(static_cast<std::size_t>(mpi_size));
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    synchronize_exception(
        allocation_exception,
        true,
        "TreePM short-potential rank gather allocation");

    if (MPI_Allgather(
            &local_value,
            1,
            MPI_LONG_DOUBLE,
            rank_values.data(),
            1,
            MPI_LONG_DOUBLE,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allgather failed for ") + quantity);
    }
    const long double total = compensated_sum_rows(rank_values);
    if (!std::isfinite(total)) {
        throw std::overflow_error(
            std::string(quantity) + " is non-finite after rank-order summation");
    }
    return total;
#else
    (void)local_value;
    (void)quantity;
    throw std::runtime_error(
        "Conservation MPI reduction requested in a non-MPI build");
#endif
}

long double treepm_short_pair_kernel_sum_for_target(
    const core::ParticleStore& particles,
    const gravity::Octree& tree,
    const gravity::ForceSplitKernel& split,
    std::size_t target_index,
    core::Real box_size,
    core::Real epsilon,
    std::vector<gravity::OctreeIndex>& stack) {
    const std::size_t owned = particles.num_owned_particles();
    if (target_index >= owned) {
        throw std::out_of_range(
            "TreePM short-potential target must be an owned particle");
    }

    const auto& nodes = tree.get_nodes();
    const auto& particle_indices = tree.get_particle_indices();
    if (nodes.empty()) return 0.0L;

    const auto positions_x = particles.get_positions_x();
    const auto positions_y = particles.get_positions_y();
    const auto positions_z = particles.get_positions_z();
    const auto ids = particles.get_ids();
    const core::Vec3 target_position{
        positions_x[target_index],
        positions_y[target_index],
        positions_z[target_index]};
    const core::ParticleId target_id = ids[target_index];
    const core::Real target_mass = particles.mass_at(target_index);

    const std::size_t stack_bound = tree.traversal_stack_bound();
    stack.clear();
    if (stack.capacity() < stack_bound) {
        stack.reserve(stack_bound);
    }
    stack.push_back(0);

    long double sum = 0.0L;
    long double compensation = 0.0L;
    while (!stack.empty()) {
        const gravity::OctreeIndex node_index = stack.back();
        stack.pop_back();
        if (static_cast<std::size_t>(node_index) >= nodes.size()) {
            throw std::logic_error(
                "TreePM short-potential traversal encountered an invalid node index");
        }
        const gravity::OctreeNode& node =
            nodes[static_cast<std::size_t>(node_index)];
        if (node.particle_end <= node.particle_begin
            || node.particle_end > particle_indices.size()) {
            throw std::logic_error(
                "TreePM short-potential tree particle range is invalid");
        }

        const core::Vec3 center_delta = math::minimum_image_displacement(
            target_position, node.geometric_center, box_size);
        const core::Real half = core::Real{0.5} * node.side_length;
        if (!std::isfinite(half) || half < 0.0) {
            throw std::logic_error(
                "TreePM short-potential tree half-width is invalid");
        }
        if (math::aabb_cutoff_relation(
                center_delta.x,
                center_delta.y,
                center_delta.z,
                half,
                split.cutoff_radius()) < 0) {
            continue;
        }

        if (node.is_leaf_node()) {
            for (std::size_t position = node.particle_begin;
                 position < node.particle_end;
                 ++position) {
                const std::size_t source_index = static_cast<std::size_t>(
                    particle_indices[position]);
                if (source_index >= particles.size()) {
                    throw std::logic_error(
                        "TreePM short-potential source index exceeds local source storage");
                }
                const core::ParticleId source_id = ids[source_index];
                if (source_index != target_index && source_id == target_id) {
                    throw std::logic_error(
                        "TreePM short-potential encountered duplicate stable particle IDs");
                }
                if (source_id <= target_id) continue;

                const core::Vec3 source_position{
                    positions_x[source_index],
                    positions_y[source_index],
                    positions_z[source_index]};
                const core::Vec3 displacement =
                    math::minimum_image_displacement(
                        target_position, source_position, box_size);
                if (!core::scale_safe_norm3_less(
                        displacement.x,
                        displacement.y,
                        displacement.z,
                        split.cutoff_radius())) {
                    continue;
                }

                const core::Real correction =
                    split.short_range_potential_correction(
                        displacement, epsilon);
                const core::Real source_mass = particles.mass_at(source_index);
                const long double term =
                    static_cast<long double>(target_mass)
                    * static_cast<long double>(source_mass)
                    * static_cast<long double>(correction);
                if (!std::isfinite(term)) {
                    throw std::overflow_error(
                        "TreePM short-potential pair kernel is non-finite");
                }
                const long double updated = sum + term;
                if (std::abs(sum) >= std::abs(term)) {
                    compensation += (sum - updated) + term;
                } else {
                    compensation += (term - updated) + sum;
                }
                if (!std::isfinite(updated) || !std::isfinite(compensation)) {
                    throw std::overflow_error(
                        "TreePM short-potential target accumulation is non-finite");
                }
                sum = updated;
            }
            continue;
        }

        const std::size_t child_count = node.occupied_child_count();
        if (stack.size() > stack_bound
            || child_count > stack_bound - stack.size()) {
            throw std::logic_error(
                "TreePM short-potential traversal exceeded the tree stack bound");
        }
        for (unsigned int octant = 0; octant < 8; ++octant) {
            if (!node.has_child(octant)) continue;
            stack.push_back(node.child_index(octant));
        }
    }
    return sum + compensation;
}

struct ScaledNonnegativeTerm {
    core::Accum fraction{0.0};
    int exponent{0};
};

ScaledNonnegativeTerm scaled_kinetic_term(
    core::Real px,
    core::Real py,
    core::Real pz,
    core::Real mass,
    core::Real scale_factor) noexcept {
    const core::Real momentum_scale = std::max({
        std::abs(px), std::abs(py), std::abs(pz)});
    if (momentum_scale == 0.0) return {};

    const core::Real sx = px / momentum_scale;
    const core::Real sy = py / momentum_scale;
    const core::Real sz = pz / momentum_scale;
    const core::Real scaled_norm_squared = sx * sx + sy * sy + sz * sz;

    int mass_exponent = 0;
    int momentum_exponent = 0;
    int norm_exponent = 0;
    int scale_factor_exponent = 0;
    const core::Real mass_fraction = std::frexp(mass, &mass_exponent);
    const core::Real momentum_fraction =
        std::frexp(momentum_scale, &momentum_exponent);
    const core::Real norm_fraction =
        std::frexp(scaled_norm_squared, &norm_exponent);
    const core::Real scale_factor_fraction =
        std::frexp(scale_factor, &scale_factor_exponent);

    core::Real coefficient = 0.5 * mass_fraction
        * momentum_fraction * momentum_fraction * norm_fraction
        / (scale_factor_fraction * scale_factor_fraction);
    int coefficient_exponent = 0;
    coefficient = std::frexp(coefficient, &coefficient_exponent);
    const int exponent = mass_exponent + 2 * momentum_exponent
        + norm_exponent - 2 * scale_factor_exponent
        + coefficient_exponent;
    return {coefficient, exponent};
}

core::Accum scaled_self_energy_term(
    core::Real mass,
    core::Accum stencil) noexcept {
    if (stencil == 0.0) return 0.0;

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
    if (mass_space_stencil == 0.0) return 0.0;

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

} // namespace

ConservationChecks::ConservationChecks(
    const config::SimulationParameters& config)
    : config_(config) {}

ConservationChecks::PotentialWorkspace::PotentialWorkspace(
    const config::SimulationParameters& parameters)
    : geometry(parameters.get_box().L, checked_mesh_size(parameters)),
      fft(geometry, parameters.get_memory_policy().evolution_scratch_mode,
          parameters.get_memory_policy().scratch_directory),
      assignment(geometry, parameters.get_memory_policy()),
      green(
          geometry,
          parameters.solver_kind() == config::SolverKind::TreePM
              ? mesh::PMForceMethod::treepm_long_range(parameters.r_s())
              : mesh::PMForceMethod::pure_pm(
                    parameters.get_gravity().deconvolve_cic)),
      field([&]() {
          auto policy = parameters.get_memory_policy();
          if (!config::uses_file_backed_scratch(policy.evolution_scratch_mode)) {
              return mesh::RealField(geometry.padded_real_size());
          }
          policy.ic_scratch_mode = policy.evolution_scratch_mode;
          real_backing = std::make_unique<runtime::RealScratchBuffer>(
              geometry.padded_real_size(), policy, "conservation_real");
          return mesh::RealField(real_backing->data(), real_backing->size(),
                                 mesh::external_mesh_storage);
      }()),
      modes([&]() {
          const auto& policy = parameters.get_memory_policy();
          if (!config::uses_file_backed_scratch(policy.evolution_scratch_mode)) {
              return mesh::ComplexField(geometry.complex_size());
          }
          static_assert(std::is_trivially_destructible_v<std::complex<core::Real>>);
          if (geometry.complex_size() > std::numeric_limits<std::size_t>::max()
                  / sizeof(std::complex<core::Real>)) {
              throw std::overflow_error("Conservation complex scratch byte size overflows size_t");
          }
          complex_backing = std::make_unique<runtime::RawScratchBuffer>(
              geometry.complex_size() * sizeof(std::complex<core::Real>),
              policy.evolution_scratch_mode, policy.scratch_directory,
              "conservation_complex");
          auto* data = static_cast<std::complex<core::Real>*>(complex_backing->data());
          std::uninitialized_value_construct_n(data, geometry.complex_size());
          return mesh::ComplexField(data, geometry.complex_size(), mesh::external_mesh_storage);
      }()) {
    std::clog << "[conservation] workspace_scratch_mode="
              << config::scratch_mode_name(parameters.get_memory_policy().evolution_scratch_mode)
              << " real_bytes=" << field.size() * sizeof(core::Real)
              << " complex_bytes=" << modes.size() * sizeof(std::complex<core::Real>)
              << '\n';
}

ConservationChecks::PotentialWorkspace&
ConservationChecks::potential_workspace() const {
    if (!potential_workspace_) {
        potential_workspace_ =
            std::make_unique<PotentialWorkspace>(config_);
    }
    return *potential_workspace_;
}

core::Real ConservationChecks::compute_treepm_short_potential_energy(
    const core::ParticleStore& particles,
    const gravity::Octree& treepm_force_tree,
    core::Real current_a) const {
    const bool mpi_enabled = mpi_reduce_enabled(config_);
    long double local_pair_kernel_sum = 0.0L;
    std::exception_ptr local_exception;
    try {
        if (config_.solver_kind() != config::SolverKind::TreePM) {
            throw std::logic_error(
                "TreePM short potential requested for a non-TreePM configuration");
        }
        if (!std::isfinite(current_a) || current_a <= 0.0) {
            throw std::invalid_argument(
                "TreePM short-potential scale factor must be finite and positive");
        }
        if (!std::isfinite(config_.eps()) || config_.eps() <= 0.0) {
            throw std::invalid_argument(
                "TreePM short-potential epsilon must be finite and positive");
        }

        const std::size_t owned = particles.num_owned_particles();
        const std::size_t source_count = particles.size();
        if (source_count < owned) {
            throw std::logic_error(
                "TreePM short-potential source population is smaller than owned targets");
        }
        if (mpi_enabled
            && particles.get_ghost_validity() != core::FieldValidity::VALID) {
            throw std::logic_error(
                "Distributed TreePM short-potential diagnostics require current r_cut ghosts");
        }
        if (treepm_force_tree.get_particle_indices().size() != source_count) {
            throw std::logic_error(
                "TreePM short-potential tree does not match the current local source population");
        }
        if (source_count != 0 && treepm_force_tree.get_nodes().empty()) {
            throw std::logic_error(
                "TreePM short-potential tree is empty for a non-empty source population");
        }

        const core::Real L = config_.get_box().L;
        if (!std::isfinite(L) || L <= 0.0) {
            throw std::invalid_argument(
                "TreePM short-potential box size must be finite and positive");
        }
        const gravity::ForceSplitKernel split(
            config_.r_s(), config_.get_gravity().cutoff_multiplier);
        if (!(split.cutoff_radius() < core::Real{0.5} * L)) {
            throw std::invalid_argument(
                "TreePM short-potential requires r_cut < L/2");
        }

        const auto x = particles.get_positions_x();
        const auto y = particles.get_positions_y();
        const auto z = particles.get_positions_z();
        for (std::size_t source = 0; source < source_count; ++source) {
            if (!std::isfinite(x[source]) || !std::isfinite(y[source])
                || !std::isfinite(z[source])
                || x[source] < 0.0 || x[source] >= L
                || y[source] < 0.0 || y[source] >= L
                || z[source] < 0.0 || z[source] >= L) {
                throw std::invalid_argument(
                    "TreePM short-potential sources must be finite and wrapped into [0,L)");
            }
            const core::Real mass = particles.mass_at(source);
            if (!std::isfinite(mass) || mass <= 0.0) {
                throw std::invalid_argument(
                    "TreePM short-potential source masses must be finite and positive");
            }
        }

        if (owned != 0) {
            constexpr std::size_t targets_per_block = 4096;
            const std::size_t block_count =
                1 + (owned - 1) / targets_per_block;
            const std::size_t worker_count =
                runtime::host_parallel_worker_capacity(block_count);
            std::vector<std::vector<gravity::OctreeIndex>> traversal_stacks(
                worker_count);
            const std::size_t stack_bound =
                treepm_force_tree.traversal_stack_bound();
            for (auto& stack : traversal_stacks) {
                stack.reserve(stack_bound);
            }
            std::vector<long double> block_sums(block_count, 0.0L);

            std::atomic<bool> worker_failed{false};
            std::exception_ptr worker_exception;
#ifdef COSMO_NBODY_HAS_OPENMP
            const int omp_worker_count = static_cast<int>(worker_count);
            #pragma omp parallel for schedule(static) num_threads(omp_worker_count) \
                if(omp_worker_count > 1)
#endif
            for (std::size_t block = 0; block < block_count; ++block) {
                if (worker_failed.load(std::memory_order_relaxed)) continue;
                try {
                    std::size_t worker_index = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
                    worker_index = static_cast<std::size_t>(omp_get_thread_num());
#endif
                    if (worker_index >= traversal_stacks.size()) {
                        throw std::logic_error(
                            "TreePM short-potential worker index exceeds scratch storage");
                    }
                    auto& stack = traversal_stacks[worker_index];
                    const std::size_t begin = block * targets_per_block;
                    const std::size_t end = begin
                        + std::min(targets_per_block, owned - begin);
                    long double sum = 0.0L;
                    long double compensation = 0.0L;
                    for (std::size_t target = begin; target < end; ++target) {
                        const long double term =
                            treepm_short_pair_kernel_sum_for_target(
                                particles,
                                treepm_force_tree,
                                split,
                                target,
                                L,
                                config_.eps(),
                                stack);
                        const long double updated = sum + term;
                        if (std::abs(sum) >= std::abs(term)) {
                            compensation += (sum - updated) + term;
                        } else {
                            compensation += (term - updated) + sum;
                        }
                        if (!std::isfinite(updated)
                            || !std::isfinite(compensation)) {
                            throw std::overflow_error(
                                "TreePM short-potential block accumulation is non-finite");
                        }
                        sum = updated;
                    }
                    block_sums[block] = sum + compensation;
                } catch (...) {
                    worker_failed.store(true, std::memory_order_relaxed);
#ifdef COSMO_NBODY_HAS_OPENMP
                    #pragma omp critical(cosmo_nbody_treepm_short_potential_failure)
#endif
                    {
                        if (!worker_exception) {
                            worker_exception = std::current_exception();
                        }
                    }
                }
            }
            if (worker_exception) std::rethrow_exception(worker_exception);
            local_pair_kernel_sum = compensated_sum_rows(block_sums);
            if (!std::isfinite(local_pair_kernel_sum)) {
                throw std::overflow_error(
                    "TreePM local short-potential pair sum is non-finite");
            }
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    synchronize_exception(
        local_exception,
        mpi_enabled,
        "TreePM short-potential local traversal");

    const long double global_pair_kernel_sum = rank_order_sum_if_requested(
        local_pair_kernel_sum,
        mpi_enabled,
        "TreePM short-potential rank sum");
    const long double energy =
        -static_cast<long double>(cosmology::units::G)
        * global_pair_kernel_sum
        / static_cast<long double>(current_a);
    return collectively_checked_real(
        energy,
        mpi_enabled,
        "TreePM short-potential energy");
}

core::Real ConservationChecks::compute_potential_energy(
    const core::ParticleStore& particles,
    core::Real current_a,
    const gravity::Octree* treepm_force_tree) const {
    if (!std::isfinite(current_a) || current_a <= 0.0) {
        throw std::invalid_argument(
            "Potential-energy scale factor must be finite and positive");
    }
    const bool treepm =
        config_.solver_kind() == config::SolverKind::TreePM;
    if (treepm && treepm_force_tree == nullptr) {
        throw std::invalid_argument(
            "TreePM potential diagnostics require the canonical tree from the current force state");
    }
    if (!treepm && treepm_force_tree != nullptr) {
        throw std::invalid_argument(
            "PM potential diagnostics must not receive a TreePM force tree");
    }

    const std::size_t N = checked_mesh_size(config_);
    const std::size_t global_cells = checked_cube_cell_count(N);
    const core::Real L = config_.get_box().L;
    const core::Real dx = L / static_cast<core::Real>(N);
    if (!std::isfinite(dx) || dx <= 0.0) {
        throw std::overflow_error(
            "Conservation mesh cell size is not representable");
    }

    const bool mpi_enabled = mpi_reduce_enabled(config_);
    PotentialWorkspace* workspace_pointer = nullptr;
    std::exception_ptr workspace_exception;
    try {
        workspace_pointer = &potential_workspace();
    } catch (...) {
        workspace_exception = std::current_exception();
    }
    synchronize_exception(
        workspace_exception,
        mpi_enabled,
        "Conservation potential workspace initialization");
    if (workspace_pointer == nullptr) {
        throw std::logic_error(
            "Conservation workspace synchronization returned no workspace");
    }

    PotentialWorkspace& workspace = *workspace_pointer;
    mesh::FFTBackend& fft = workspace.fft;
    mesh::CICMassAssignment& assignment = workspace.assignment;
    const mesh::MeshGeometry& geometry = workspace.geometry;

    const std::size_t owned = particles.num_owned_particles();
    const auto pos_x = particles.get_positions_x().first(owned);
    const auto pos_y = particles.get_positions_y().first(owned);
    const auto pos_z = particles.get_positions_z().first(owned);
    const auto masses = particles.get_uniform_mass().has_value()
        ? std::span<const core::Real>{}
        : particles.get_masses().first(owned);

    mesh::RealField& field = workspace.field;
    mesh::ComplexField& modes = workspace.modes;
    mesh::GreenFunction& green = workspace.green;

    std::exception_ptr deposit_exception;
    try {
        assignment.deposit(
            pos_x, pos_y, pos_z,
            masses,
            particles.get_uniform_mass(),
            field,
            nullptr);
    } catch (...) {
        deposit_exception = std::current_exception();
    }
    synchronize_exception(
        deposit_exception,
        mpi_enabled,
        "Conservation CIC deposition");
    allreduce_field_if_requested(field, mpi_enabled);

    const core::Accum total_mass = math::deterministic_blocked_sum(
        geometry.real_size(),
        [&](std::size_t index) { return field[index]; });
    synchronize_invalid_flag(
        !std::isfinite(total_mass) || total_mass <= 0.0,
        mpi_enabled,
        "Conservation total deposited mass must be finite and positive");

    std::optional<math::DensityNormalization> density_normalization;
    std::exception_ptr normalization_exception;
    try {
        density_normalization.emplace(
            static_cast<core::Real>(total_mass),
            global_cells,
            dx,
            "Conservation density normalization");
    } catch (...) {
        normalization_exception = std::current_exception();
    }
    synchronize_exception(
        normalization_exception,
        mpi_enabled,
        "Conservation density normalization preparation");
    if (!density_normalization.has_value()) {
        throw std::logic_error(
            "Conservation density normalization was not prepared");
    }

    int invalid_source = 0;
    if (density_normalization->direct_available()) {
        const core::Real mean = collectively_checked_real(
            static_cast<long double>(total_mass)
                / static_cast<long double>(global_cells),
            mpi_enabled,
            "Conservation mean cell mass");
        const core::Real inverse_cell_volume =
            density_normalization->inverse_cell_volume();
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) \
            reduction(|: invalid_source)
#endif
        for (std::size_t ix = 0; ix < geometry.local_n0(); ++ix) {
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const std::size_t index =
                        geometry.real_index(ix, iy, iz);
                    const core::Real candidate =
                        (field[index] - mean) * inverse_cell_volume;
                    if (!std::isfinite(candidate)) {
                        invalid_source = 1;
                    } else {
                        field[index] = candidate;
                    }
                }
            }
        }
    } else {
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) \
            reduction(|: invalid_source)
#endif
        for (std::size_t ix = 0; ix < geometry.local_n0(); ++ix) {
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const std::size_t index =
                        geometry.real_index(ix, iy, iz);
                    try {
                        field[index] = density_normalization->normalize(
                            field[index],
                            "Conservation scale-safe density source");
                    } catch (...) {
                        invalid_source = 1;
                    }
                }
            }
        }
    }
    synchronize_invalid_flag(
        invalid_source,
        mpi_enabled,
        "Conservation density contrast source is non-finite or unrepresentable");

    std::exception_ptr spectral_exception;
    try {
        fft.forward(field, modes);
        green.apply(modes);
        fft.inverse(modes, field);
    } catch (...) {
        spectral_exception = std::current_exception();
    }
    synchronize_exception(
        spectral_exception,
        mpi_enabled,
        "Conservation potential spectral solve");

    int invalid_potential = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: invalid_potential) \
        if(field.size() >= 32768)
#endif
    for (std::size_t index = 0; index < field.size(); ++index) {
        invalid_potential |= !std::isfinite(field[index]);
    }
    synchronize_invalid_flag(
        invalid_potential,
        mpi_enabled,
        "Conservation potential field is non-finite");

    std::span<core::Real> phi_at_particles;
    std::exception_ptr interpolation_exception;
    try {
        if (phi_at_particles_.size() < owned) {
            phi_at_particles_.resize(owned);
        }
        phi_at_particles = std::span<core::Real>(
            phi_at_particles_.data(), owned);
        assignment.interpolate(
            field, nullptr,
            pos_x, pos_y, pos_z,
            phi_at_particles);
    } catch (...) {
        interpolation_exception = std::current_exception();
    }
    synchronize_exception(
        interpolation_exception,
        mpi_enabled,
        "Conservation potential interpolation");

    const core::Accum local_w_com = math::deterministic_blocked_sum(
        owned,
        [&](std::size_t index) {
            return core::Accum{0.5}
                * static_cast<core::Accum>(particles.mass_at(index))
                * static_cast<core::Accum>(phi_at_particles[index]);
        });
    const core::Real local_w = collectively_checked_real(
        static_cast<long double>(local_w_com),
        mpi_enabled,
        "Conservation local potential energy");
    const core::Real global_w_com = allreduce_scalar_if_requested(
        local_w,
        mpi_enabled,
        "Conservation potential-energy scalar");
    core::Real potential = collectively_checked_real(
        static_cast<long double>(global_w_com)
            / static_cast<long double>(current_a),
        mpi_enabled,
        "Conservation long-range potential energy");
    if (treepm) {
        const core::Real short_potential =
            compute_treepm_short_potential_energy(
                particles, *treepm_force_tree, current_a);
        potential = collectively_checked_real(
            static_cast<long double>(potential)
                + static_cast<long double>(short_potential),
            mpi_enabled,
            "Conservation total TreePM potential energy");
    }
    return potential;
}

void ConservationChecks::prepare_cic_self_kernel() const {
    if (self_kernel_ready_) return;
    const std::size_t N = checked_mesh_size(config_);
    const std::size_t global_cells = checked_cube_cell_count(N);
    const core::Real L = config_.get_box().L;
    const core::Real dx = L / static_cast<core::Real>(N);
    if (!std::isfinite(dx) || dx <= 0.0) {
        throw std::overflow_error("Self-energy mesh cell size is not representable");
    }
    const bool mpi_enabled = mpi_reduce_enabled(config_);
    std::array<core::Real, 27> candidate_kernel{};
    bool candidate_scaled_by_cell_volume = false;
    std::exception_ptr kernel_exception;
    try {
        // An existing raw-potential workspace is already a required retained
        // allocation. Otherwise own the impulse mesh only for this preparation.
        std::unique_ptr<PotentialWorkspace> temporary;
        if (!potential_workspace_) {
            temporary = std::make_unique<PotentialWorkspace>(config_);
        }
        PotentialWorkspace& workspace = potential_workspace_
            ? *potential_workspace_ : *temporary;
        if (workspace.geometry.local_n0() != N) {
            throw std::invalid_argument(
                "CIC self-energy requires the replicated conservation mesh; "
                "a distributed mesh slab cannot provide the local lag stencil");
        }
        const std::array<core::Real, 1> origin{0.0};
        workspace.assignment.deposit(
            std::span<const core::Real>(origin),
            std::span<const core::Real>(origin),
            std::span<const core::Real>(origin),
            std::span<const core::Real>{},
            std::optional<core::Real>{1.0},
            workspace.field,
            nullptr);
        const math::DensityNormalization impulse_normalization(
            1.0,
            global_cells,
            dx,
            "Conservation CIC self-energy impulse normalization");
        if (impulse_normalization.direct_available()) {
            const core::Real mean =
                core::Real{1.0} / static_cast<core::Real>(global_cells);
            const core::Real inverse_cell_volume =
                impulse_normalization.inverse_cell_volume();
            for (std::size_t index = 0;
                 index < workspace.geometry.real_size();
                 ++index) {
                workspace.field[index] =
                    (workspace.field[index] - mean)
                    * inverse_cell_volume;
            }
        } else {
            // The Poisson solve is linear. If 1/dx^3 itself lies outside
            // binary64, keep the mean-subtracted unit-mass CIC impulse in
            // mass space. The resulting potential kernel is dx^3 times the
            // normal unit-density kernel and remains tagged as such until it
            // is combined with m^2/dx^3 below.
            const core::Real mean =
                core::Real{1.0} / static_cast<core::Real>(global_cells);
            for (std::size_t index = 0;
                 index < workspace.geometry.real_size();
                 ++index) {
                workspace.field[index] -= mean;
            }
            candidate_scaled_by_cell_volume = true;
        }
        workspace.fft.forward(workspace.field, workspace.modes);
        workspace.green.apply(workspace.modes);
        workspace.fft.inverse(workspace.modes, workspace.field);

        for (int lag_x = -1; lag_x <= 1; ++lag_x) {
            for (int lag_y = -1; lag_y <= 1; ++lag_y) {
                for (int lag_z = -1; lag_z <= 1; ++lag_z) {
                    const std::size_t ix =
                        static_cast<std::size_t>(
                            lag_x + static_cast<int>(N)) % N;
                    const std::size_t iy =
                        static_cast<std::size_t>(
                            lag_y + static_cast<int>(N)) % N;
                    const std::size_t iz =
                        static_cast<std::size_t>(
                            lag_z + static_cast<int>(N)) % N;
                    const core::Real value = workspace.field[
                        workspace.geometry.real_index(ix, iy, iz)];
                    if (!std::isfinite(value)) {
                        throw std::overflow_error(
                            "Self-energy impulse response is non-finite");
                    }
                    candidate_kernel[static_cast<std::size_t>(
                        (lag_x + 1) * 9
                        + (lag_y + 1) * 3
                        + (lag_z + 1))] = value;
                }
            }
        }
    } catch (...) {
        kernel_exception = std::current_exception();
    }
    synchronize_exception(
        kernel_exception,
        mpi_enabled,
        "Conservation CIC self-energy kernel initialization");
    self_kernel_ = candidate_kernel;
    self_kernel_scaled_by_cell_volume_ = candidate_scaled_by_cell_volume;
    self_kernel_ready_ = true;
}

core::Real ConservationChecks::compute_cic_self_energy(
    const core::ParticleStore& particles,
    core::Real current_a) const {
    if (!std::isfinite(current_a) || current_a <= 0.0) {
        throw std::invalid_argument(
            "Self-energy scale factor must be finite and positive");
    }

    const std::size_t N = checked_mesh_size(config_);
    (void)checked_cube_cell_count(N);
    const core::Real L = config_.get_box().L;
    const core::Real dx = L / static_cast<core::Real>(N);
    if (!std::isfinite(dx) || dx <= 0.0) {
        throw std::overflow_error(
            "Self-energy mesh cell size is not representable");
    }
    const bool mpi_enabled = mpi_reduce_enabled(config_);

    const std::size_t owned = particles.num_owned_particles();
    const auto pos_x = particles.get_positions_x().first(owned);
    const auto pos_y = particles.get_positions_y().first(owned);
    const auto pos_z = particles.get_positions_z().first(owned);
    const auto uniform_mass = particles.get_uniform_mass();
    const auto explicit_masses = uniform_mass.has_value()
        ? std::span<const core::Real>{}
        : particles.get_masses().first(owned);

    int invalid_input = 0;
    if (uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
        invalid_input = 1;
    }
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: invalid_input) \
        if(owned >= 2048)
#endif
    for (std::size_t index = 0; index < owned; ++index) {
        const core::Real mass = uniform_mass.has_value()
            ? *uniform_mass : explicit_masses[index];
        if (!std::isfinite(pos_x[index])
            || !std::isfinite(pos_y[index])
            || !std::isfinite(pos_z[index])
            || !std::isfinite(mass)
            || mass <= 0.0) {
            invalid_input = 1;
        }
    }
    std::exception_ptr input_exception;
    if (invalid_input != 0) {
        input_exception = std::make_exception_ptr(std::invalid_argument(
            "CIC self-energy positions must be finite and masses finite and positive"));
    }
    synchronize_exception(
        input_exception,
        mpi_enabled,
        "Conservation CIC self-energy input validation");

    prepare_cic_self_kernel();

    const std::array<core::Real, 27> kernel = self_kernel_;
    const bool kernel_scaled_by_cell_volume =
        self_kernel_scaled_by_cell_volume_;
    const core::Accum local_sum = math::deterministic_blocked_sum(
        owned,
        [&](std::size_t index) -> core::Accum {
            const core::Real mass = uniform_mass.has_value()
                ? *uniform_mass
                : explicit_masses[index];
            std::array<std::array<core::Real, 3>, 3> lag_factor{};
            const std::array<core::Real, 3> coordinates{
                pos_x[index], pos_y[index], pos_z[index]};
            for (std::size_t dim = 0; dim < 3; ++dim) {
                const core::Real cell_coordinate =
                    math::wrap(coordinates[dim], L) / dx;
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
                            * kernel[lag_x * 9 + lag_y * 3 + lag_z]);
                    }
                }
            }
            return kernel_scaled_by_cell_volume
                ? scaled_mass_space_self_energy_term(mass, stencil, dx)
                : scaled_self_energy_term(mass, stencil);
        });
    const core::Real local_value = collectively_checked_real(
        static_cast<long double>(local_sum),
        mpi_enabled,
        "Conservation local self-energy");
    const core::Real global_value = allreduce_scalar_if_requested(
        local_value,
        mpi_enabled,
        "Conservation self-energy scalar");
    return collectively_checked_real(
        static_cast<long double>(global_value)
            / static_cast<long double>(current_a),
        mpi_enabled,
        "Conservation peculiar self-energy");
}

core::Real ConservationChecks::compute_kinetic_energy(
    const core::ParticleStore& particles,
    core::Real current_a) const {
    if (!std::isfinite(current_a) || current_a <= 0.0) {
        throw std::invalid_argument(
            "Kinetic-energy scale factor must be finite and positive");
    }

    const std::size_t owned = particles.num_owned_particles();
    const auto momentum_x = particles.get_momenta_x().first(owned);
    const auto momentum_y = particles.get_momenta_y().first(owned);
    const auto momentum_z = particles.get_momenta_z().first(owned);
    const auto uniform_mass = particles.get_uniform_mass();
    const auto explicit_masses = uniform_mass.has_value()
        ? std::span<const core::Real>{}
        : particles.get_masses().first(owned);

    int invalid_input = 0;
    if (uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
        invalid_input = 1;
    }
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: invalid_input) \
        if(owned >= 2048)
#endif
    for (std::size_t index = 0; index < owned; ++index) {
        const core::Real mass = uniform_mass.has_value()
            ? *uniform_mass : explicit_masses[index];
        if (!std::isfinite(momentum_x[index])
            || !std::isfinite(momentum_y[index])
            || !std::isfinite(momentum_z[index])
            || !std::isfinite(mass)
            || mass <= 0.0) {
            invalid_input = 1;
        }
    }
    const bool mpi_enabled = mpi_reduce_enabled(config_);
    synchronize_invalid_flag(
        invalid_input,
        mpi_enabled,
        "Conservation kinetic inputs must be finite with positive mass");

    constexpr int no_nonzero_term = std::numeric_limits<int>::lowest();
    int local_max_exponent = no_nonzero_term;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(max: local_max_exponent) \
        if(owned >= 2048)
#endif
    for (std::size_t index = 0; index < owned; ++index) {
        const core::Real mass = uniform_mass.has_value()
            ? *uniform_mass : explicit_masses[index];
        const ScaledNonnegativeTerm term = scaled_kinetic_term(
            momentum_x[index], momentum_y[index], momentum_z[index],
            mass, current_a);
        if (term.fraction != 0.0) {
            local_max_exponent = std::max(
                local_max_exponent, term.exponent);
        }
    }
    const int global_max_exponent = allreduce_max_exponent_if_requested(
        local_max_exponent,
        mpi_enabled,
        "Conservation kinetic-energy exponent");
    if (global_max_exponent == no_nonzero_term) return 0.0;

    const core::Accum local_scaled_sum = math::deterministic_blocked_sum(
        owned,
        [&](std::size_t index) {
            const core::Real mass = uniform_mass.has_value()
                ? *uniform_mass : explicit_masses[index];
            const ScaledNonnegativeTerm term = scaled_kinetic_term(
                momentum_x[index], momentum_y[index], momentum_z[index],
                mass, current_a);
            if (term.fraction == 0.0) return core::Accum{0.0};
            return std::scalbn(
                term.fraction,
                term.exponent - global_max_exponent);
        });
    const core::Real local_scaled = collectively_checked_real(
        static_cast<long double>(local_scaled_sum),
        mpi_enabled,
        "Conservation local scaled kinetic energy");
    const core::Real global_scaled = allreduce_scalar_if_requested(
        local_scaled,
        mpi_enabled,
        "Conservation scaled kinetic-energy scalar");

    core::Real kinetic = 0.0;
    std::exception_ptr kinetic_exception;
    try {
        kinetic = std::scalbn(global_scaled, global_max_exponent);
        if (!std::isfinite(kinetic)) {
            throw std::overflow_error(
                "Conservation kinetic energy is not representable in core::Real");
        }
        if (global_scaled != 0.0 && kinetic == 0.0) {
            throw std::overflow_error(
                "Conservation kinetic energy underflows core::Real to zero");
        }
    } catch (...) {
        kinetic_exception = std::current_exception();
    }
    synchronize_exception(
        kinetic_exception,
        mpi_enabled,
        "Conservation kinetic energy");
    return kinetic;
}

void ConservationChecks::reset_layzer_irvine_state(
    core::Real current_a,
    core::Real current_K,
    core::Real current_W,
    ConservationState& state) {
    require_finite_li_sample(current_a, current_K, current_W);
    const core::Real energy = checked_scaled_linear_combination<2>(
        {current_K, current_W},
        {1.0L, 1.0L},
        "Layzer-Irvine initial energy");
    const core::Real source = checked_scaled_linear_combination<2>(
        {current_K, current_W},
        {2.0L, 1.0L},
        "Layzer-Irvine initial source");

    ConservationState candidate;
    candidate.a_prev = current_a;
    candidate.initial_E = energy;
    candidate.source_prev = source;
    candidate.last_kinetic = current_K;
    candidate.last_potential = current_W;
    candidate.last_energy = energy;
    candidate.last_source = source;
    candidate.last_residual = 0.0;
    state = candidate;
}

core::Real ConservationChecks::update_layzer_irvine_state(
    core::Real current_a,
    core::Real current_K,
    core::Real current_W,
    ConservationState& state) {
    require_finite_li_sample(current_a, current_K, current_W);
    if (!std::isfinite(state.a_prev) || state.a_prev <= 0.0) {
        throw std::logic_error(
            "Layzer-Irvine state must be reset before update");
    }
    if (!(current_a > state.a_prev)) {
        throw std::invalid_argument(
            "Layzer-Irvine updates require strictly increasing scale factor");
    }
    if (!std::isfinite(state.integrated_source)
        || !std::isfinite(state.integrated_source_compensation)
        || !std::isfinite(state.initial_E)
        || !std::isfinite(state.source_prev)) {
        throw std::logic_error(
            "Layzer-Irvine state contains a non-finite accumulator");
    }

    const core::Real energy = checked_scaled_linear_combination<2>(
        {current_K, current_W},
        {1.0L, 1.0L},
        "Layzer-Irvine current energy");
    const core::Real source = checked_scaled_linear_combination<2>(
        {current_K, current_W},
        {2.0L, 1.0L},
        "Layzer-Irvine current source");
    const core::Real dln_a = stable_log_scale_ratio(
        current_a, state.a_prev);
    const long double half_interval =
        0.5L * static_cast<long double>(dln_a);

    const core::Real updated = checked_scaled_linear_combination<4>(
        {
            state.integrated_source,
            state.integrated_source_compensation,
            state.source_prev,
            source,
        },
        {1.0L, -1.0L, half_interval, half_interval},
        "Layzer-Irvine integrated source");
    const core::Real updated_compensation =
        checked_scaled_linear_combination<5>(
            {
                updated,
                state.integrated_source,
                state.integrated_source_compensation,
                state.source_prev,
                source,
            },
            {1.0L, -1.0L, 1.0L, -half_interval, -half_interval},
            "Layzer-Irvine source compensation");
    const core::Real residual = checked_scaled_linear_combination<3>(
        {energy, state.initial_E, updated},
        {1.0L, -1.0L, 1.0L},
        "Layzer-Irvine residual");

    ConservationState candidate = state;
    candidate.a_prev = current_a;
    candidate.integrated_source = updated;
    candidate.integrated_source_compensation = updated_compensation;
    candidate.source_prev = source;
    candidate.last_kinetic = current_K;
    candidate.last_potential = current_W;
    candidate.last_energy = energy;
    candidate.last_source = source;
    candidate.last_residual = residual;
    state = candidate;
    return residual;
}

void ConservationChecks::reset_state(
    const core::ParticleStore& particles,
    core::Real current_a,
    core::Real current_W,
    ConservationState& state) const {
    const core::Real current_K =
        compute_kinetic_energy(particles, current_a);
    reset_layzer_irvine_state(
        current_a, current_K, current_W, state);
}

core::Real ConservationChecks::evaluate_layzer_irvine(
    const core::ParticleStore& particles,
    core::Real current_a,
    core::Real current_W,
    ConservationState& state) const {
    const core::Real current_K =
        compute_kinetic_energy(particles, current_a);
    return update_layzer_irvine_state(
        current_a, current_K, current_W, state);
}

} // namespace validation
} // namespace cosmo_nbody
