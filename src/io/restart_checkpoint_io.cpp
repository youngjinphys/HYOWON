#include "cosmo_nbody/io/restart_checkpoint_io.hpp"

#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::io {
namespace {

constexpr const char* kManifestFilename = "checkpoint.meta";
constexpr const char* kManifestHeader = "hyowon.restart.checkpoint";

struct CheckpointShardRecord {
    int rank{0};
    std::uint64_t particle_count{0};
    bool uniform_mass{false};
    std::string object_name;
    std::string restart_state_sha256;
};

struct CheckpointManifest {
    int ranks{0};
    std::uint64_t step{0};
    core::Real current_a{0.0};
    core::Real delta_ln_a{0.0};
    std::uint64_t global_particle_count{0};
    std::vector<CheckpointShardRecord> shards;
};

std::uint64_t checked_size_to_u64(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds uint64 range");
        }
    }
    return static_cast<std::uint64_t>(value);
}

std::string shard_name(int rank) {
    if (rank < 0) {
        throw std::invalid_argument("Checkpoint rank must be non-negative");
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "rank_" << std::setw(6) << std::setfill('0') << rank << ".hdf5";
    return out.str();
}

std::filesystem::path checkpoint_path(
    const std::filesystem::path& restart_base,
    std::uint64_t step) {
    if (restart_base.empty() || restart_base.filename().empty()) {
        throw std::invalid_argument("Restart checkpoint base path must not be empty");
    }
    const auto absolute = std::filesystem::absolute(restart_base).lexically_normal();
    return absolute.parent_path()
        / (absolute.filename().string() + "_step_" + std::to_string(step));
}

void require_directory(const std::filesystem::path& path, const char* role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(
            std::string(role) + " must be an existing non-symlink directory: "
            + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
}

void create_checkpoint_directory(const std::filesystem::path& path) {
    require_directory(path.parent_path(), "Checkpoint parent directory");
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (!error && std::filesystem::exists(status)) {
        throw std::runtime_error(
            "Checkpoint for this integration step already exists: " + path.string());
    }
    if (error && error != std::errc::no_such_file_or_directory) {
        throw std::runtime_error(
            "Could not inspect checkpoint destination: " + error.message());
    }
    error.clear();
    if (!std::filesystem::create_directory(path, error) || error) {
        throw std::runtime_error(
            "Could not create checkpoint directory: " + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
    require_directory(path, "Checkpoint directory");
}

void require_regular_file(const std::filesystem::path& path, const char* role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_regular_file(status)) {
        throw std::runtime_error(
            std::string(role) + " must be a regular non-symlink file: "
            + path.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
}

std::string render_manifest(const CheckpointManifest& manifest) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << kManifestHeader << '\n'
        << "ranks " << manifest.ranks << '\n'
        << "step " << manifest.step << '\n'
        << std::setprecision(std::numeric_limits<core::Real>::max_digits10)
        << "current_a " << manifest.current_a << '\n'
        << "delta_ln_a " << manifest.delta_ln_a << '\n'
        << "global_particle_count " << manifest.global_particle_count << '\n';
    for (const auto& shard : manifest.shards) {
        out << "shard " << shard.rank << ' '
            << shard.particle_count << ' '
            << (shard.uniform_mass ? 1 : 0) << ' '
            << shard.restart_state_sha256 << '\n';
    }
    return out.str();
}

void require_token(std::istream& input, const char* expected) {
    std::string token;
    if (!(input >> token) || token != expected) {
        throw std::runtime_error(
            std::string("Checkpoint manifest expected token: ") + expected);
    }
}

CheckpointManifest read_manifest(
    const std::filesystem::path& directory,
    int expected_ranks,
    std::uint64_t configured_particle_count) {
    require_directory(directory, "Restart checkpoint");
    const auto manifest_path = directory / kManifestFilename;
    require_regular_file(manifest_path, "Checkpoint manifest");

    std::ifstream input(manifest_path, std::ios::binary);
    input.imbue(std::locale::classic());
    if (!input) {
        throw std::runtime_error(
            "Could not open checkpoint manifest: " + manifest_path.string());
    }

    require_token(input, kManifestHeader);
    require_token(input, "ranks");
    int ranks = 0;
    if (!(input >> ranks) || ranks < 1 || ranks != expected_ranks) {
        throw std::runtime_error(
            "Checkpoint rank count does not match the active runtime");
    }

    CheckpointManifest manifest;
    manifest.ranks = ranks;
    require_token(input, "step");
    if (!(input >> manifest.step)) {
        throw std::runtime_error("Checkpoint manifest step is invalid");
    }
    require_token(input, "current_a");
    if (!(input >> manifest.current_a)
        || !std::isfinite(manifest.current_a) || manifest.current_a <= 0.0) {
        throw std::runtime_error("Checkpoint manifest scale factor is invalid");
    }
    require_token(input, "delta_ln_a");
    if (!(input >> manifest.delta_ln_a)
        || !std::isfinite(manifest.delta_ln_a) || manifest.delta_ln_a <= 0.0) {
        throw std::runtime_error("Checkpoint manifest delta_ln_a is invalid");
    }
    require_token(input, "global_particle_count");
    if (!(input >> manifest.global_particle_count)
        || manifest.global_particle_count != configured_particle_count) {
        throw std::runtime_error(
            "Checkpoint global particle count does not match configured N^3");
    }

    manifest.shards.reserve(static_cast<std::size_t>(ranks));
    std::uint64_t summed_particles = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        require_token(input, "shard");
        int parsed_rank = -1;
        std::uint64_t particle_count = 0;
        int uniform_flag = -1;
        std::string state_sha256;
        if (!(input >> parsed_rank >> particle_count >> uniform_flag >> state_sha256)
            || parsed_rank != rank
            || (uniform_flag != 0 && uniform_flag != 1)
            || !is_canonical_sha256(state_sha256)) {
            throw std::runtime_error("Checkpoint shard manifest record is invalid");
        }
        if (particle_count > configured_particle_count
            || summed_particles > configured_particle_count - particle_count) {
            throw std::runtime_error("Checkpoint shard particle counts are invalid");
        }
        summed_particles += particle_count;
        manifest.shards.push_back(CheckpointShardRecord{
            rank,
            particle_count,
            uniform_flag == 1,
            shard_name(rank),
            std::move(state_sha256)});
    }
    if (summed_particles != manifest.global_particle_count) {
        throw std::runtime_error(
            "Checkpoint shard particle counts do not reproduce global N^3");
    }

    std::string trailing;
    if (input >> trailing) {
        throw std::runtime_error(
            "Checkpoint manifest contains unexpected trailing content");
    }
    return manifest;
}

std::string manifest_sha256(const std::filesystem::path& directory) {
    return sha256_file(directory / kManifestFilename);
}

void require_local_decoded_state(
    const CheckpointManifest& manifest,
    const CheckpointShardRecord& shard,
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper) {
    if (checked_size_to_u64(
            particles.num_owned_particles(), "Restart local particle count")
            != shard.particle_count
        || particles.get_uniform_mass().has_value() != shard.uniform_mass) {
        throw std::runtime_error(
            "Decoded checkpoint shard disagrees with manifest particle layout");
    }
    if (checked_size_to_u64(stepper.current_step(), "Restart step")
            != manifest.step
        || stepper.current_a() != manifest.current_a
        || stepper.delta_ln_a() != manifest.delta_ln_a) {
        throw std::runtime_error(
            "Decoded checkpoint shard disagrees with manifest time state");
    }
}

RestartStateIdentityResult read_local_shard(
    const config::SimulationParameters& config,
    const CheckpointManifest& manifest,
    const CheckpointShardRecord& shard,
    const std::filesystem::path& shard_path,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out) {
    require_regular_file(shard_path, "Checkpoint shard");
    RestartIO restart(config);
    auto session = restart.open_restart_session(shard_path.string());
    auto identity = restart.read_restart_with_identity(
        session,
        particles_out,
        stepper_out,
        shard.restart_state_sha256);
    if (!identity.persisted || identity.sha256 != shard.restart_state_sha256) {
        throw std::runtime_error(
            "Checkpoint shard state identity disagrees with manifest");
    }
    require_local_decoded_state(manifest, shard, particles_out, stepper_out);
    return identity;
}

CheckpointShardRecord write_local_shard(
    const config::SimulationParameters& config,
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::filesystem::path& directory,
    int rank) {
    const auto path = directory / shard_name(rank);
    RestartIO restart(config);
    const auto identity = restart.write_restart_with_identity(
        particles, stepper, path.string());
    if (!identity.persisted || !is_canonical_sha256(identity.sha256)) {
        throw std::runtime_error(
            "Checkpoint shard writer did not persist a canonical state digest");
    }
    return CheckpointShardRecord{
        rank,
        checked_size_to_u64(
            particles.num_owned_particles(), "Checkpoint particle count"),
        particles.get_uniform_mass().has_value(),
        shard_name(rank),
        identity.sha256};
}

void write_serial_checkpoint(
    const config::SimulationParameters& config,
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::filesystem::path& restart_base) {
    const std::uint64_t step = checked_size_to_u64(
        stepper.current_step(), "Checkpoint step");
    const auto directory = checkpoint_path(restart_base, step);
    create_checkpoint_directory(directory);

    auto shard = write_local_shard(config, particles, stepper, directory, 0);
    if (shard.particle_count != config.num_particles()) {
        throw std::runtime_error(
            "Serial checkpoint particle count does not equal configured N^3");
    }
    CheckpointManifest manifest;
    manifest.ranks = 1;
    manifest.step = step;
    manifest.current_a = stepper.current_a();
    manifest.delta_ln_a = stepper.delta_ln_a();
    manifest.global_particle_count = shard.particle_count;
    manifest.shards.push_back(std::move(shard));
    write_text_durable_atomic(
        directory / kManifestFilename,
        render_manifest(manifest),
        "restart checkpoint manifest");
}

std::string read_serial_checkpoint(
    const config::SimulationParameters& config,
    const std::filesystem::path& checkpoint_directory,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out) {
    const auto directory =
        std::filesystem::absolute(checkpoint_directory).lexically_normal();
    const CheckpointManifest manifest = read_manifest(
        directory, 1, config.num_particles());
    const std::string manifest_hash = manifest_sha256(directory);
    const auto& shard = manifest.shards.front();
    (void)read_local_shard(
        config,
        manifest,
        shard,
        directory / shard.object_name,
        particles_out,
        stepper_out);
    return manifest_hash;
}

#ifdef COSMO_NBODY_HAS_MPI
void require_active_topology(int rank, int ranks) {
    if (ranks < 1 || rank < 0 || rank >= ranks) {
        throw std::invalid_argument("Checkpoint MPI topology is invalid");
    }
    runtime::require_active_mpi_main_thread("Checkpoint MPI topology");
    int actual_rank = -1;
    int actual_ranks = 0;
    const int rank_status = MPI_Comm_rank(MPI_COMM_WORLD, &actual_rank);
    const int size_status = MPI_Comm_size(MPI_COMM_WORLD, &actual_ranks);
    if (rank_status != MPI_SUCCESS || size_status != MPI_SUCCESS) {
        const int status = rank_status != MPI_SUCCESS ? rank_status : size_status;
        (void)MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
    runtime::synchronize_mpi_failure(
        actual_rank != rank || actual_ranks != ranks ? 1 : 0,
        ranks,
        "Checkpoint topology disagrees with MPI_COMM_WORLD");
}

void require_collective_time_state(
    const time::TimeStepper& stepper,
    int ranks) {
    const std::uint64_t local_step = checked_size_to_u64(
        stepper.current_step(), "Checkpoint step");
    const core::Real local_a = stepper.current_a();
    const core::Real local_delta = stepper.delta_ln_a();
    std::uint64_t min_step = 0, max_step = 0;
    core::Real min_a = 0.0, max_a = 0.0;
    core::Real min_delta = 0.0, max_delta = 0.0;
    const int s1 = MPI_Allreduce(
        &local_step, &min_step, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    const int s2 = MPI_Allreduce(
        &local_step, &max_step, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    const int s3 = MPI_Allreduce(
        &local_a, &min_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    const int s4 = MPI_Allreduce(
        &local_a, &max_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const int s5 = MPI_Allreduce(
        &local_delta, &min_delta, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    const int s6 = MPI_Allreduce(
        &local_delta, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (s1 != MPI_SUCCESS || s2 != MPI_SUCCESS || s3 != MPI_SUCCESS
        || s4 != MPI_SUCCESS || s5 != MPI_SUCCESS || s6 != MPI_SUCCESS) {
        (void)ranks;
        throw std::runtime_error("Checkpoint time-state reduction failed");
    }
    if (min_step != max_step || min_a != max_a || min_delta != max_delta) {
        throw std::runtime_error(
            "MPI ranks disagree on checkpoint step or scale-factor state");
    }
}

CheckpointManifest gather_manifest(
    const time::TimeStepper& stepper,
    const CheckpointShardRecord& local,
    int ranks,
    std::uint64_t configured_particle_count) {
    const std::uint64_t local_count = local.particle_count;
    std::uint64_t global_count = 0;
    if (MPI_Allreduce(
            &local_count, &global_count, 1,
            MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Checkpoint global particle-count reduction failed");
    }
    if (global_count != configured_particle_count) {
        throw std::runtime_error(
            "Checkpoint global particle count does not equal configured N^3");
    }

    std::vector<std::uint64_t> counts;
    std::vector<int> uniform_flags;
    std::vector<char> hashes;
    std::exception_ptr allocation_exception;
    try {
        counts.resize(static_cast<std::size_t>(ranks));
        uniform_flags.resize(static_cast<std::size_t>(ranks));
        hashes.resize(static_cast<std::size_t>(ranks) * SHA256_HEX_CHARACTER_COUNT);
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        allocation_exception, ranks, "Allocate checkpoint manifest gather storage");

    const int uniform = local.uniform_mass ? 1 : 0;
    const int s1 = MPI_Allgather(
        &local_count, 1, MPI_UINT64_T,
        counts.data(), 1, MPI_UINT64_T,
        MPI_COMM_WORLD);
    const int s2 = MPI_Allgather(
        &uniform, 1, MPI_INT,
        uniform_flags.data(), 1, MPI_INT,
        MPI_COMM_WORLD);
    const int s3 = MPI_Allgather(
        local.restart_state_sha256.data(),
        static_cast<int>(SHA256_HEX_CHARACTER_COUNT), MPI_CHAR,
        hashes.data(),
        static_cast<int>(SHA256_HEX_CHARACTER_COUNT), MPI_CHAR,
        MPI_COMM_WORLD);
    if (s1 != MPI_SUCCESS || s2 != MPI_SUCCESS || s3 != MPI_SUCCESS) {
        throw std::runtime_error("Checkpoint shard metadata all-gather failed");
    }

    CheckpointManifest manifest;
    std::exception_ptr materialization_exception;
    try {
        manifest.ranks = ranks;
        manifest.step = checked_size_to_u64(stepper.current_step(), "Checkpoint step");
        manifest.current_a = stepper.current_a();
        manifest.delta_ln_a = stepper.delta_ln_a();
        manifest.global_particle_count = global_count;
        manifest.shards.reserve(static_cast<std::size_t>(ranks));
        for (int source = 0; source < ranks; ++source) {
            const auto offset = static_cast<std::size_t>(source)
                * SHA256_HEX_CHARACTER_COUNT;
            std::string hash(
                hashes.data() + offset,
                hashes.data() + offset + SHA256_HEX_CHARACTER_COUNT);
            if (!is_canonical_sha256(hash)) {
                throw std::runtime_error(
                    "Gathered checkpoint shard state digest is malformed");
            }
            const int uniform_flag = uniform_flags[static_cast<std::size_t>(source)];
            if (uniform_flag != 0 && uniform_flag != 1) {
                throw std::runtime_error(
                    "Gathered checkpoint mass-representation flag is invalid");
            }
            manifest.shards.push_back(CheckpointShardRecord{
                source,
                counts[static_cast<std::size_t>(source)],
                uniform_flag == 1,
                shard_name(source),
                std::move(hash)});
        }
    } catch (...) {
        materialization_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        materialization_exception, ranks, "Materialize checkpoint manifest");
    return manifest;
}
#endif

} // namespace

void RestartCheckpointIO::write_restart(
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::string& restart_base) const {
    write_serial_checkpoint(config_, particles, stepper, restart_base);
}

std::string RestartCheckpointIO::read_restart(
    const std::string& checkpoint_directory,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out) const {
    return read_serial_checkpoint(
        config_, checkpoint_directory, particles_out, stepper_out);
}

void write_restart_checkpoint_collective(
    const config::SimulationParameters& config,
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::filesystem::path& restart_base,
    int rank,
    int ranks) {
    if (ranks <= 1 || !config.get_runtime().mpi_enabled) {
        throw std::invalid_argument(
            "Collective checkpoint writer requires an active multi-rank MPI run");
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)particles;
    (void)stepper;
    (void)restart_base;
    (void)rank;
    throw std::runtime_error(
        "Collective checkpoint writer requires an MPI-enabled build");
#else
    require_active_topology(rank, ranks);
    require_collective_time_state(stepper, ranks);
    const std::uint64_t step = checked_size_to_u64(
        stepper.current_step(), "Checkpoint step");
    const auto directory = checkpoint_path(restart_base, step);

    std::exception_ptr creation_exception;
    if (rank == 0) {
        try {
            create_checkpoint_directory(directory);
        } catch (...) {
            creation_exception = std::current_exception();
        }
    }
    runtime::synchronize_mpi_exception(
        creation_exception, ranks, "Create checkpoint directory");
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Checkpoint directory visibility barrier failed");
    }

    CheckpointShardRecord local_shard;
    std::exception_ptr local_exception;
    try {
        local_shard = write_local_shard(
            config, particles, stepper, directory, rank);
    } catch (...) {
        local_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        local_exception, ranks, "Write checkpoint rank shard");

    CheckpointManifest manifest = gather_manifest(
        stepper,
        local_shard,
        ranks,
        config.num_particles());

    std::string manifest_text;
    std::exception_ptr render_exception;
    try {
        manifest_text = render_manifest(manifest);
    } catch (...) {
        render_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        render_exception, ranks, "Render checkpoint manifest");

    std::exception_ptr publication_exception;
    if (rank == 0) {
        try {
            write_text_durable_atomic(
                directory / kManifestFilename,
                manifest_text,
                "restart checkpoint manifest");
        } catch (...) {
            publication_exception = std::current_exception();
        }
    }
    runtime::synchronize_mpi_exception(
        publication_exception, ranks, "Publish checkpoint manifest");
#endif
}

std::string read_restart_checkpoint_collective(
    const config::SimulationParameters& config,
    const std::filesystem::path& checkpoint_directory,
    int rank,
    int ranks,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out) {
    if (ranks <= 1 || rank < 0 || rank >= ranks) {
        throw std::invalid_argument(
            "Collective checkpoint reader requires a valid multi-rank topology");
    }
    if (!config.get_runtime().mpi_enabled) {
        throw std::invalid_argument(
            "Collective checkpoint reader requires configured MPI mode");
    }
#ifndef COSMO_NBODY_HAS_MPI
    (void)checkpoint_directory;
    (void)particles_out;
    (void)stepper_out;
    throw std::runtime_error(
        "Collective checkpoint reader requires an MPI-enabled build");
#else
    require_active_topology(rank, ranks);
    const auto directory =
        std::filesystem::absolute(checkpoint_directory).lexically_normal();

    std::optional<CheckpointManifest> parsed_manifest;
    std::string local_manifest_sha;
    std::exception_ptr manifest_exception;
    try {
        parsed_manifest.emplace(read_manifest(
            directory, ranks, config.num_particles()));
        local_manifest_sha = manifest_sha256(directory);
    } catch (...) {
        manifest_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        manifest_exception, ranks, "Read checkpoint manifest");

    std::string root_sha = rank == 0 ? local_manifest_sha : std::string{};
    root_sha = runtime::broadcast_string_collective(
        root_sha, 0, rank, ranks, "checkpoint manifest identity");
    runtime::synchronize_mpi_failure(
        local_manifest_sha == root_sha ? 0 : 1,
        ranks,
        "MPI ranks observed different checkpoint manifest bytes");

    const CheckpointManifest& manifest = *parsed_manifest;
    const auto& shard = manifest.shards.at(static_cast<std::size_t>(rank));
    std::exception_ptr read_exception;
    try {
        (void)read_local_shard(
            config,
            manifest,
            shard,
            directory / shard.object_name,
            particles_out,
            stepper_out);
    } catch (...) {
        read_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        read_exception, ranks, "Read checkpoint rank shard");
    return root_sha;
#endif
}

} // namespace cosmo_nbody::io
