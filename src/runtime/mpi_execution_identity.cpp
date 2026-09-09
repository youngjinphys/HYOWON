#include "cosmo_nbody/runtime/mpi_execution_identity.hpp"

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::runtime {
namespace {

[[maybe_unused]] void append_u64(std::string& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
    }
}

[[maybe_unused]] void append_i32(std::string& out, int value) {
    const auto bits = static_cast<std::uint32_t>(static_cast<std::int32_t>(value));
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        out.push_back(static_cast<char>((bits >> shift) & 0xffU));
    }
}

[[maybe_unused]] void append_bool(std::string& out, bool value) {
    out.push_back(value ? '\x01' : '\x00');
}

[[maybe_unused]] void append_real(std::string& out, core::Real value) {
    static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
    append_u64(out, core::portable_bit_cast<std::uint64_t>(value));
}

[[maybe_unused]] void append_field(std::string& out, std::string_view value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

[[maybe_unused]] void append_real_vector(
    std::string& out,
    const std::vector<core::Real>& values) {
    append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const core::Real value : values) append_real(out, value);
}

// Canonical numerical-method identity shared across ranks. Execution-resource
// coordinates that may alter bitwise realization are recorded as provenance, not
// method identity; communicator size is checked separately.
[[maybe_unused]] std::string canonical_numerical_method_identity(
    const config::SimulationParameters& config) {
    std::string out;
    out.reserve(768);

    const auto& cosmology = config.get_cosmology();
    append_real(out, cosmology.h);
    append_real(out, cosmology.omega_m);
    append_real(out, cosmology.omega_lambda);
    append_real(out, cosmology.omega_b);
    append_real(out, cosmology.sigma8);
    append_real(out, cosmology.n_s);

    const auto& box = config.get_box();
    append_real(out, box.L);
    append_u64(out, box.N);
    append_u64(out, box.N_mesh);

    const auto& gravity = config.get_gravity();
    append_field(out, gravity.solver);
    append_real(out, gravity.eps);
    append_real(out, gravity.theta);
    append_real(out, gravity.split_scale_cells);
    append_real(out, gravity.cutoff_multiplier);
    append_bool(out, gravity.deconvolve_cic);

    const auto& time = config.get_time();
    append_real(out, time.z_start);
    append_real(out, time.z_final);
    append_real(out, time.delta_ln_a);
    append_field(out, time.step_policy);

    const auto& ic = config.get_ic();
    append_field(out, ic.mode);
    append_i32(out, ic.lpt_order);
    append_u64(out, config.ic_seed());
    append_real(out, ic.power_spectrum_redshift);
    append_field(out, ic.power_spectrum_fidelity);
    append_field(out, ic.amplitude_mode);
    append_field(out, ic.phase_pairing);
    append_u64(out, config.ic_mesh_per_dimension());
    append_bool(out, ic.max_mode_per_axis.has_value());
    if (ic.max_mode_per_axis) append_u64(out, *ic.max_mode_per_axis);
    append_field(out, ic.power_spectrum_sha256);
    append_field(out, ic.snapshot_sha256);

    // Snapshot targets are exact KDK integration boundaries and therefore are
    // numerical trajectory coordinates rather than mere output preferences.
    append_real_vector(out, config.get_output().snapshot_scale_factors);

    append_bool(out, config.get_runtime().mpi_enabled);

    // Guard the validated/derived values actually consumed by force/IC owners.
    append_i32(out, static_cast<int>(config.solver_kind()));
    append_real(out, config.d_mean());
    append_real(out, config.particle_mass());
    append_real(out, config.r_s());
    append_real(out, config.r_cut());
    append_real(out, config.k_Nyq());
    append_real(out, config.eps());
    append_u64(out, config.ic_mesh_per_dimension());
    append_u64(out, config.num_particles());
    return out;
}

[[maybe_unused]] bool locally_valid(
    const MpiExecutionIdentity& identity,
    int communicator_size) noexcept {
    return !identity.configuration.empty()
        && identity.real_bytes > 0
        && identity.communicator_size == communicator_size;
}

[[maybe_unused]] std::string encode(const MpiExecutionIdentity& identity) {
    std::string result;
    append_field(result, identity.configuration);
    append_i32(result, identity.real_bytes);
    append_i32(result, identity.communicator_size);
    return result;
}

#ifdef COSMO_NBODY_HAS_MPI
void require_world(int& rank, int& size, const char* context) {
    int initialized = 0;
    int finalized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0
        || MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
        throw std::runtime_error(std::string(context) + " requires active MPI");
    }
    int is_main_thread = 0;
    if (MPI_Is_thread_main(&is_main_thread) != MPI_SUCCESS
        || is_main_thread == 0) {
        throw std::runtime_error(
            std::string(context) + " requires the MPI main thread");
    }
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS
        || MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS
        || rank < 0 || size < 1 || rank >= size) {
        throw std::runtime_error(
            std::string(context) + " failed to query MPI_COMM_WORLD");
    }
}
#endif

} // namespace

MpiExecutionIdentityAgreement::MpiExecutionIdentityAgreement(
    const config::SimulationParameters& config,
    int rank,
    int size) {
    if (size < 1 || rank < 0 || rank >= size) {
        throw std::invalid_argument(
            "MPI numerical-consistency topology is invalid");
    }
    if (size == 1) return;

    MpiExecutionIdentity local_identity;
    std::exception_ptr stage_exception;
    try {
        local_identity = MpiExecutionIdentity{
            canonical_numerical_method_identity(config),
            static_cast<int>(sizeof(core::Real)),
            size};
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        stage_exception, size, "MPI numerical-method identity construction");
    require_mpi_execution_identity_agreement(local_identity);
}

void require_mpi_execution_identity_agreement(
    const MpiExecutionIdentity& local_identity) {
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_identity;
    throw std::runtime_error(
        "MPI numerical-method identity agreement requires an MPI-enabled build");
#else
    int rank = 0;
    int size = 0;
    require_world(rank, size, "MPI numerical-method identity agreement");

    const int local_invalid = locally_valid(local_identity, size) ? 0 : 1;
    int any_invalid = 0;
    if (MPI_Allreduce(
            &local_invalid, &any_invalid, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed for numerical-method identity validation");
    }
    if (any_invalid != 0) {
        throw std::invalid_argument(
            "MPI numerical-method identity requires a complete configuration and matching communicator size");
    }

    std::string local;
    std::exception_ptr stage_exception;
    try {
        local = encode(local_identity);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        stage_exception, size, "MPI numerical-method identity serialization");

    std::string reference_source;
    stage_exception = nullptr;
    try {
        if (rank == 0) reference_source = local;
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        stage_exception, size, "MPI numerical-method identity reference staging");

    const std::string reference = broadcast_string_collective(
        std::move(reference_source),
        0,
        rank,
        size,
        "MPI numerical-method identity");
    const int local_mismatch = local != reference ? 1 : 0;
    int any_mismatch = 0;
    if (MPI_Allreduce(
            &local_mismatch, &any_mismatch, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed for numerical-method identity agreement");
    }
    if (any_mismatch != 0) {
        throw std::runtime_error(
            "MPI ranks disagree on numerical-method configuration or scalar ABI");
    }
#endif
}

std::string agree_optional_sha256_collective(
    const std::string& local_value,
    const char* context) {
    if (context == nullptr || *context == '\0') {
        throw std::invalid_argument("SHA-256 agreement context must not be empty");
    }
#ifndef COSMO_NBODY_HAS_MPI
    if (!local_value.empty() && !io::is_canonical_sha256(local_value)) {
        throw std::invalid_argument(
            std::string(context) + " is not a canonical SHA-256");
    }
    return local_value;
#else
    int rank = 0;
    int size = 0;
    require_world(rank, size, context);
    const int local_invalid =
        !local_value.empty() && !io::is_canonical_sha256(local_value) ? 1 : 0;
    int any_invalid = 0;
    if (MPI_Allreduce(
            &local_invalid, &any_invalid, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context + " validation");
    }
    if (any_invalid != 0) {
        throw std::invalid_argument(
            std::string(context) + " contains a malformed SHA-256");
    }

    const int candidate = local_value.empty() ? size : rank;
    int source = size;
    if (MPI_Allreduce(
            &candidate, &source, 1,
            MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context + " source selection");
    }
    if (source == size) return {};

    std::string source_value;
    std::exception_ptr stage_exception;
    try {
        if (rank == source) source_value = local_value;
    } catch (...) {
        stage_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        stage_exception, size, "MPI SHA-256 source staging");

    const std::string agreed = broadcast_string_collective(
        std::move(source_value), source, rank, size, context);
    const int local_mismatch =
        !local_value.empty() && local_value != agreed ? 1 : 0;
    int any_mismatch = 0;
    if (MPI_Allreduce(
            &local_mismatch, &any_mismatch, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MPI_Allreduce failed for ") + context + " agreement");
    }
    if (any_mismatch != 0) {
        throw std::runtime_error(
            std::string(context) + " differs across non-empty MPI ranks");
    }
    return agreed;
#endif
}

} // namespace cosmo_nbody::runtime
