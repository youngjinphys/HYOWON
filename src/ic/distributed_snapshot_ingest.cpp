#include "cosmo_nbody/ic/distributed_snapshot_ingest.hpp"

#include "cosmo_nbody/domain/mpi_global_id_check.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/verified_snapshot_source.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/mpi_exact_uint64_sum.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/mpi_execution_identity.hpp"
#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::ic {
namespace {

#ifdef COSMO_NBODY_HAS_MPI
std::size_t checked_expected_population(
    const config::SimulationParameters& config) {
    const std::uint64_t count = config.num_particles();
    if (count == 0) {
        throw std::invalid_argument(
            "Distributed snapshot IC ingestion requires a non-empty population");
    }
    if (count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Distributed snapshot IC population does not fit size_t");
    }
    return static_cast<std::size_t>(count);
}

core::Real expected_total_mass(
    const config::SimulationParameters& config,
    std::size_t expected_particles) {
    math::ExactPositiveDoubleSum expected;
    expected.add_repeated(
        config.particle_mass(),
        static_cast<std::uint64_t>(expected_particles));
    const core::Real result = expected.value();
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::runtime_error(
            "Configured total mass is invalid for distributed snapshot IC validation");
    }
    return result;
}

void copy_owned_chunk(
    const core::ParticleStore& chunk,
    core::ParticleStore& destination,
    std::size_t destination_offset,
    std::optional<core::Real> uniform_mass) {
    const std::size_t count = chunk.num_owned_particles();
    if (destination_offset > destination.num_owned_particles()
        || count > destination.num_owned_particles() - destination_offset) {
        throw std::logic_error(
            "Distributed snapshot IC chunk exceeds destination storage");
    }
    if (chunk.num_ghost_particles() != 0) {
        throw std::logic_error(
            "Distributed snapshot IC assembly received an unexpected ghost suffix");
    }
    if (chunk.get_uniform_mass() != uniform_mass) {
        throw std::logic_error(
            "Distributed snapshot IC chunks disagree on mass representation");
    }

    auto dst_x = destination.get_positions_x();
    auto dst_y = destination.get_positions_y();
    auto dst_z = destination.get_positions_z();
    auto dst_px = destination.get_momenta_x();
    auto dst_py = destination.get_momenta_y();
    auto dst_pz = destination.get_momenta_z();
    auto dst_ids = destination.get_ids();
    auto dst_masses = uniform_mass.has_value()
        ? std::span<core::Real>{}
        : destination.get_masses();

    const auto src_x = chunk.get_positions_x().first(count);
    const auto src_y = chunk.get_positions_y().first(count);
    const auto src_z = chunk.get_positions_z().first(count);
    const auto src_px = chunk.get_momenta_x().first(count);
    const auto src_py = chunk.get_momenta_y().first(count);
    const auto src_pz = chunk.get_momenta_z().first(count);
    const auto src_ids = chunk.get_ids().first(count);
    const auto src_masses = uniform_mass.has_value()
        ? std::span<const core::Real>{}
        : chunk.get_masses().first(count);

    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t out = destination_offset + i;
        dst_x[out] = src_x[i];
        dst_y[out] = src_y[i];
        dst_z[out] = src_z[i];
        dst_px[out] = src_px[i];
        dst_py[out] = src_py[i];
        dst_pz[out] = src_pz[i];
        dst_ids[out] = src_ids[i];
        if (!uniform_mass.has_value()) dst_masses[out] = src_masses[i];
    }
}

std::vector<std::uint64_t> count_final_owner_population(
    const domain::DomainDecomposition& domain_decomposition,
    io::SnapshotIO& snapshot_io,
    const std::optional<io::VerifiedSnapshotSource>& source,
    const std::optional<io::SnapshotDescriptor>& descriptor,
    std::size_t expected_particles,
    std::size_t batch_particles,
    std::size_t batch_count,
    int rank,
    int size) {
    std::vector<std::uint64_t> owner_counts(
        static_cast<std::size_t>(size), 0);
    std::exception_ptr count_exception;
    if (rank == 0) {
        try {
            if (!source.has_value() || !descriptor.has_value()) {
                throw std::logic_error(
                    "Distributed snapshot IC admitted source is unavailable during owner counting");
            }
            for (std::size_t batch_index = 0;
                 batch_index < batch_count;
                 ++batch_index) {
                const std::size_t offset = batch_index * batch_particles;
                const std::size_t count = std::min(
                    batch_particles, expected_particles - offset);
                core::ParticleStore batch;
                // Reuse the exact admitted range reader rather than creating a
                // second snapshot interpretation path. This first pass is only
                // an allocation census; the second pass below remains the sole
                // phase-space publication path.
                snapshot_io.read_exact_initial_condition_range(
                    *source,
                    *descriptor,
                    offset,
                    count,
                    batch);
                const auto x = batch.get_positions_x().first(
                    batch.num_owned_particles());
                if (x.size() != count) {
                    throw std::logic_error(
                        "Distributed snapshot IC owner-count range size mismatch");
                }
                for (const core::Real position_x : x) {
                    const int owner = domain_decomposition.owner_rank(position_x);
                    if (owner < 0 || owner >= size) {
                        throw std::logic_error(
                            "Distributed snapshot IC owner count produced an invalid rank");
                    }
                    std::uint64_t& destination = owner_counts[
                        static_cast<std::size_t>(owner)];
                    if (destination == std::numeric_limits<std::uint64_t>::max()) {
                        throw std::overflow_error(
                            "Distributed snapshot IC owner population overflows uint64_t");
                    }
                    ++destination;
                }
            }
        } catch (...) {
            count_exception = std::current_exception();
        }
    }
    runtime::synchronize_mpi_exception(
        count_exception,
        size,
        "Distributed snapshot IC owner-population census");

    if (MPI_Bcast(
            owner_counts.data(),
            size,
            MPI_UINT64_T,
            0,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Distributed snapshot IC owner-population broadcast failed");
    }

    std::uint64_t counted_population = 0;
    for (const std::uint64_t count : owner_counts) {
        if (count > std::numeric_limits<std::uint64_t>::max()
                - counted_population) {
            throw std::overflow_error(
                "Distributed snapshot IC owner-population sum overflows uint64_t");
        }
        counted_population += count;
    }
    if (counted_population != static_cast<std::uint64_t>(expected_particles)) {
        throw std::runtime_error(
            "Distributed snapshot IC owner-population census changed the global population");
    }
    return owner_counts;
}
#endif

} // namespace

void load_snapshot_distributed(
    const config::SimulationParameters& config,
    domain::DomainDecomposition& domain_decomposition,
    core::ParticleStore& particles,
    int rank,
    int size) {
#if !defined(COSMO_NBODY_HAS_MPI)
    (void)config;
    (void)domain_decomposition;
    (void)particles;
    (void)rank;
    (void)size;
    throw std::runtime_error(
        "Distributed snapshot IC ingestion requires an MPI-enabled build");
#else
    runtime::require_active_mpi_main_thread(
        "Distributed snapshot IC ingestion");
    if (!config.get_runtime().mpi_enabled || size <= 1) {
        throw std::invalid_argument(
            "Distributed snapshot IC ingestion requires more than one active MPI rank");
    }
    if (rank < 0 || rank >= size
        || domain_decomposition.rank() != rank
        || domain_decomposition.size() != size) {
        throw std::invalid_argument(
            "Distributed snapshot IC ingestion rank topology is inconsistent");
    }
    if (config.get_ic().mode != "snapshot") {
        throw std::invalid_argument(
            "Distributed snapshot IC ingestion requires ic.mode=snapshot");
    }

    const std::size_t expected_particles =
        checked_expected_population(config);
    const std::size_t batch_particles = std::min(
        expected_particles, io::SNAPSHOT_READ_BATCH_PARTICLES);
    const std::size_t batch_count =
        1 + (expected_particles - 1) / batch_particles;

    std::optional<io::VerifiedSnapshotSource> source;
    std::optional<io::SnapshotDescriptor> descriptor;
    math::ExactPositiveDoubleSum observed_explicit_mass;
    std::exception_ptr admission_exception;
    if (rank == 0) {
        try {
            const auto& ic = config.get_ic();
            source.emplace(
                ic.snapshot_file,
                ic.expected_snapshot_sha256);
            descriptor.emplace(io::read_snapshot_descriptor(*source));
            io::require_snapshot_initial_condition_match(
                config, *descriptor, source->sha256());
        } catch (...) {
            admission_exception = std::current_exception();
        }
    }
    runtime::synchronize_mpi_exception(
        admission_exception,
        size,
        "Distributed snapshot IC exact-source admission");

    io::SnapshotIO snapshot_io(config);
    const std::vector<std::uint64_t> final_owner_counts =
        count_final_owner_population(
            domain_decomposition,
            snapshot_io,
            source,
            descriptor,
            expected_particles,
            batch_particles,
            batch_count,
            rank,
            size);
    const std::uint64_t local_expected_u64 = final_owner_counts[
        static_cast<std::size_t>(rank)];
    if (local_expected_u64 > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Distributed snapshot IC local owner population does not fit size_t");
    }
    const std::size_t local_expected =
        static_cast<std::size_t>(local_expected_u64);

    core::ParticleStore candidate;
    bool candidate_representation_initialized = false;
    std::size_t candidate_write_offset = 0;

    for (std::size_t batch_index = 0;
         batch_index < batch_count;
         ++batch_index) {
        const std::size_t offset = batch_index * batch_particles;
        const std::size_t count = std::min(
            batch_particles, expected_particles - offset);

        core::ParticleStore batch;
        std::exception_ptr read_exception;
        if (rank == 0) {
            try {
                if (!source.has_value() || !descriptor.has_value()) {
                    throw std::logic_error(
                        "Distributed snapshot IC admitted source is unavailable on rank 0");
                }
                snapshot_io.read_exact_initial_condition_range(
                    *source,
                    *descriptor,
                    offset,
                    count,
                    batch);
                if (!descriptor->uniform_mass) {
                    for (const core::Real mass : batch.get_masses()) {
                        observed_explicit_mass.add(mass);
                    }
                }
                // Temporary range routing must not publish source provenance.
                // Publication waits for the final re-hash of the same opened
                // object after every payload range has been consumed.
                batch.set_verified_snapshot_ic_sha256({});
            } catch (...) {
                read_exception = std::current_exception();
            }
        }
        runtime::synchronize_mpi_exception(
            read_exception,
            size,
            "Distributed snapshot IC bounded range read");

        std::optional<domain::DomainDecomposition> batch_decomposition;
        std::exception_ptr decomposition_exception;
        try {
            batch_decomposition.emplace(config, rank, size);
        } catch (...) {
            decomposition_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            decomposition_exception,
            size,
            "Distributed snapshot IC batch-domain construction");
        if (!batch_decomposition.has_value()) {
            throw std::logic_error(
                "Distributed snapshot IC batch-domain construction returned no object");
        }
        batch_decomposition->partition_domain(batch);

        std::exception_ptr append_exception;
        try {
            const auto batch_uniform_mass = batch.get_uniform_mass();
            if (!candidate_representation_initialized) {
                candidate.set_uniform_mass(batch_uniform_mass);
                candidate.resize(local_expected);
                candidate_representation_initialized = true;
            } else if (candidate.get_uniform_mass() != batch_uniform_mass) {
                throw std::logic_error(
                    "Distributed snapshot IC batches disagree on mass representation");
            }
            if (!batch.get_verified_snapshot_ic_sha256().empty()) {
                throw std::logic_error(
                    "Temporary distributed snapshot batch published provenance before source completion");
            }

            const std::size_t append_count = batch.num_owned_particles();
            if (candidate_write_offset > local_expected
                || append_count > local_expected - candidate_write_offset) {
                throw std::runtime_error(
                    "Distributed snapshot IC routed population exceeds the owner census");
            }
            copy_owned_chunk(
                batch,
                candidate,
                candidate_write_offset,
                batch_uniform_mass);
            candidate_write_offset += append_count;
        } catch (...) {
            append_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            append_exception,
            size,
            "Distributed snapshot IC routed-range accumulation");
    }

    std::exception_ptr assembly_completion_exception;
    try {
        if (!candidate_representation_initialized) {
            throw std::logic_error(
                "Distributed snapshot IC did not establish a local mass representation");
        }
        if (candidate_write_offset != local_expected) {
            throw std::runtime_error(
                "Distributed snapshot IC routed population disagrees with the owner census");
        }
    } catch (...) {
        assembly_completion_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        assembly_completion_exception,
        size,
        "Distributed snapshot IC routed-range completion");

    std::exception_ptr source_completion_exception;
    if (rank == 0) {
        try {
            if (!source.has_value() || !descriptor.has_value()) {
                throw std::logic_error(
                    "Distributed snapshot IC source disappeared before completion validation");
            }
            if (!descriptor->uniform_mass
                && observed_explicit_mass.value()
                    != expected_total_mass(config, expected_particles)) {
                throw std::runtime_error(
                    "Snapshot IC total mass does not match configured Omega_m and box volume");
            }
            source->verify_unchanged();
        } catch (...) {
            source_completion_exception = std::current_exception();
        }
    }
    runtime::synchronize_mpi_exception(
        source_completion_exception,
        size,
        "Distributed snapshot IC source completion validation");

    std::string local_verified_sha256;
    std::string local_source_run_metadata_json;
    std::exception_ptr provenance_preparation_exception;
    try {
        if (!candidate_representation_initialized) {
            throw std::logic_error(
                "Distributed snapshot IC did not establish a local mass representation");
        }
        if (rank == 0) {
            if (!source.has_value() || !descriptor.has_value()) {
                throw std::logic_error(
                    "Distributed snapshot IC source disappeared before provenance publication");
            }
            local_verified_sha256 = source->sha256();
            local_source_run_metadata_json = descriptor->run_metadata_json;
        }
    } catch (...) {
        provenance_preparation_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        provenance_preparation_exception,
        size,
        "Distributed snapshot IC provenance preparation");
    const std::string verified_sha256 =
        runtime::agree_optional_sha256_collective(
            local_verified_sha256,
            "distributed snapshot IC provenance");
    if (verified_sha256.empty()) {
        throw std::logic_error(
            "Distributed snapshot IC completion produced no exact-source identity");
    }
    const std::string source_run_metadata_json =
        runtime::broadcast_string_collective(
            local_source_run_metadata_json,
            0,
            rank,
            size,
            "distributed snapshot IC source RunMetadataJson");
    if (source_run_metadata_json.empty()) {
        throw std::logic_error(
            "Distributed snapshot IC completion produced no source RunMetadataJson");
    }

    std::exception_ptr provenance_publication_exception;
    try {
        config.set_verified_snapshot_ic_provenance(
            verified_sha256, source_run_metadata_json);
        candidate.set_verified_snapshot_ic_sha256(verified_sha256);
        candidate.set_acceleration_validity(core::FieldValidity::INVALID);
        candidate.set_ghost_validity(core::FieldValidity::INVALID);
    } catch (...) {
        provenance_publication_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        provenance_publication_exception,
        size,
        "Distributed snapshot IC local provenance publication");

    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (candidate.num_owned_particles()
            > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Distributed snapshot IC local population exceeds uint64_t");
        }
    }
    const std::uint64_t global_population = math::mpi_exact_uint64_sum(
        static_cast<std::uint64_t>(candidate.num_owned_particles()));
    if (global_population != static_cast<std::uint64_t>(expected_particles)) {
        throw std::runtime_error(
            "Distributed snapshot IC population changed during bounded routing");
    }
    domain::require_globally_unique_particle_ids(
        candidate.get_ids().first(candidate.num_owned_particles()));

    // Establish the runner-owned decomposition only once after the complete
    // exact source has been admitted. This publishes canonical stable-ID state
    // and snapshot provenance for all subsequent sparse migration.
    domain_decomposition.partition_domain(candidate);
    candidate.set_acceleration_validity(core::FieldValidity::INVALID);
    candidate.set_ghost_validity(core::FieldValidity::INVALID);

    particles = std::move(candidate);
    if (rank == 0) {
        std::cout
            << "Successfully ingested " << expected_particles
            << " particles from snapshot IC file '"
            << config.get_ic().snapshot_file
            << "' through bounded two-pass distributed routing with exact per-rank preallocation, immutable input identity, and mass normalization preserved.\n";
    }
#endif
}

void admit_snapshot_provenance_distributed(
    const config::SimulationParameters& config,
    int rank,
    int size) {
#if !defined(COSMO_NBODY_HAS_MPI)
    (void)config;
    (void)rank;
    (void)size;
    throw std::runtime_error(
        "Distributed snapshot provenance admission requires an MPI-enabled build");
#else
    runtime::require_active_mpi_main_thread(
        "Distributed snapshot provenance admission");
    if (!config.get_runtime().mpi_enabled || size <= 1
        || rank < 0 || rank >= size) {
        throw std::invalid_argument(
            "Distributed snapshot provenance admission has an invalid MPI topology");
    }
    if (config.get_ic().mode != "snapshot") return;

    std::optional<io::VerifiedSnapshotSource> source;
    std::optional<io::SnapshotDescriptor> descriptor;
    std::exception_ptr admission_exception;
    if (rank == 0) {
        try {
            const auto& ic = config.get_ic();
            source.emplace(
                ic.snapshot_file,
                ic.expected_snapshot_sha256);
            descriptor.emplace(io::read_snapshot_descriptor(*source));
            io::require_snapshot_initial_condition_match(
                config, *descriptor, source->sha256());
            source->verify_unchanged();
        } catch (...) {
            admission_exception = std::current_exception();
        }
    }
    runtime::synchronize_mpi_exception(
        admission_exception,
        size,
        "Distributed restart source-IC descriptor admission");

    const std::string local_sha256 = rank == 0
        ? source->sha256() : std::string{};
    const std::string verified_sha256 =
        runtime::agree_optional_sha256_collective(
            local_sha256,
            "distributed restart source-IC provenance");
    const std::string_view local_source_run_metadata_json = rank == 0
        ? std::string_view{descriptor->run_metadata_json}
        : std::string_view{};
    const std::string source_run_metadata_json =
        runtime::broadcast_string_collective(
            local_source_run_metadata_json,
            0,
            rank,
            size,
            "distributed restart source-IC RunMetadataJson");

    std::exception_ptr publication_exception;
    try {
        if (verified_sha256.empty() || source_run_metadata_json.empty()) {
            throw std::logic_error(
                "Distributed restart source-IC provenance is incomplete");
        }
        config.set_verified_snapshot_ic_provenance(
            verified_sha256, source_run_metadata_json);
    } catch (...) {
        publication_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        publication_exception,
        size,
        "Distributed restart source-IC provenance publication");
#endif
}

} // namespace cosmo_nbody::ic
