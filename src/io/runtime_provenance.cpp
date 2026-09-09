#include "cosmo_nbody/io/runtime_provenance.hpp"

#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::io {

namespace {

void require_candidate(
    const std::string& value,
    const char* label) {
    if (!value.empty() && !is_canonical_sha256(value)) {
        throw std::invalid_argument(
            std::string(label)
            + " must contain exactly 64 lowercase hexadecimal characters");
    }
}

void require_agreement(
    const std::string& lhs,
    const std::string& rhs) {
    if (!lhs.empty() && !rhs.empty() && lhs != rhs) {
        throw std::invalid_argument(
            "Snapshot IC SHA-256 values disagree");
    }
}

struct GatheredRankDigests {
    std::uint64_t available_rank_count{0};
    std::string local_digest;
    std::vector<char> rank_ordered_digest_bytes;
};

GatheredRankDigests gather_rank_digests(
    std::string_view local_value,
    const runtime::RuntimeContext& context,
    const char* stage_prefix) {
    if (stage_prefix == nullptr || *stage_prefix == '\0') {
        throw std::invalid_argument(
            "Execution-provenance digest stage must not be empty");
    }
    if (context.size() < 1 || context.rank() < 0
        || context.rank() >= context.size()) {
        throw std::logic_error(
            "Execution provenance received an invalid runtime topology");
    }

    GatheredRankDigests result;
    std::exception_ptr stage_exception;
    try {
        result.local_digest = sha256_text(local_value);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    const std::string hashing_stage =
        std::string(stage_prefix) + " local identity hashing";
    runtime::synchronize_mpi_exception(
        stage_exception,
        context.size(),
        hashing_stage.c_str());
    if (!is_canonical_sha256(result.local_digest)) {
        throw std::logic_error(
            "Execution-provenance SHA-256 implementation returned a non-canonical digest");
    }

    const int local_available = local_value.empty() ? 0 : 1;
    int available_rank_count = local_available;
    if (context.size() > 1) {
#ifdef COSMO_NBODY_HAS_MPI
        runtime::require_active_mpi_main_thread(stage_prefix);
        if (MPI_Allreduce(
                &local_available,
                &available_rank_count,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                std::string(stage_prefix)
                + " availability reduction failed");
        }
#else
        throw std::runtime_error(
            "Multi-rank execution provenance requires an MPI-enabled build");
#endif
    }
    if (available_rank_count < 0 || available_rank_count > context.size()) {
        throw std::logic_error(
            "Execution-provenance availability count is outside the rank range");
    }
    result.available_rank_count =
        static_cast<std::uint64_t>(available_rank_count);

    const std::size_t rank_count = static_cast<std::size_t>(context.size());
    if (rank_count > std::numeric_limits<std::size_t>::max()
                         / SHA256_HEX_CHARACTER_COUNT) {
        throw std::overflow_error(
            "Execution-provenance rank digest storage overflows size_t");
    }

    stage_exception = nullptr;
    try {
        result.rank_ordered_digest_bytes.resize(
            rank_count * SHA256_HEX_CHARACTER_COUNT);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    const std::string allocation_stage =
        std::string(stage_prefix) + " rank digest allocation";
    runtime::synchronize_mpi_exception(
        stage_exception,
        context.size(),
        allocation_stage.c_str());

    if (context.size() == 1) {
        for (std::size_t index = 0;
             index < SHA256_HEX_CHARACTER_COUNT;
             ++index) {
            result.rank_ordered_digest_bytes[index] =
                result.local_digest[index];
        }
    } else {
#ifdef COSMO_NBODY_HAS_MPI
        runtime::require_active_mpi_main_thread(stage_prefix);
        const int digest_character_count = static_cast<int>(
            SHA256_HEX_CHARACTER_COUNT);
        if (MPI_Allgather(
                result.local_digest.data(),
                digest_character_count,
                MPI_CHAR,
                result.rank_ordered_digest_bytes.data(),
                digest_character_count,
                MPI_CHAR,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                std::string(stage_prefix) + " rank digest gather failed");
        }
#else
        throw std::runtime_error(
            "Multi-rank execution provenance requires an MPI-enabled build");
#endif
    }
    return result;
}

bool same_rank_digest(
    const std::vector<char>& bytes,
    std::size_t lhs_rank,
    std::size_t rhs_rank) noexcept {
    const std::size_t lhs = lhs_rank * SHA256_HEX_CHARACTER_COUNT;
    const std::size_t rhs = rhs_rank * SHA256_HEX_CHARACTER_COUNT;
    for (std::size_t index = 0;
         index < SHA256_HEX_CHARACTER_COUNT;
         ++index) {
        if (bytes[lhs + index] != bytes[rhs + index]) return false;
    }
    return true;
}

std::string rank_digest_string(
    const std::vector<char>& bytes,
    std::size_t rank) {
    const std::size_t offset = rank * SHA256_HEX_CHARACTER_COUNT;
    return std::string(
        bytes.data() + offset,
        SHA256_HEX_CHARACTER_COUNT);
}

std::string hash_domain_and_rank_digests(
    std::string_view domain,
    const std::vector<char>& digest_bytes) {
    if (domain.empty()) {
        throw std::invalid_argument(
            "Execution-provenance identity domain must not be empty");
    }
    std::string material;
    constexpr std::size_t separator_bytes = 1U;
    if (digest_bytes.size()
            > std::numeric_limits<std::size_t>::max() - separator_bytes
        || domain.size()
            > std::numeric_limits<std::size_t>::max()
                - separator_bytes - digest_bytes.size()) {
        throw std::overflow_error(
            "Execution-provenance identity material overflows size_t");
    }
    material.reserve(domain.size() + separator_bytes + digest_bytes.size());
    material.append(domain);
    material.push_back('\0');
    material.append(digest_bytes.data(), digest_bytes.size());
    return sha256_text(material);
}

RankOrderedRuntimeStringIdentity collect_rank_ordered_string_identity(
    std::string_view local_value,
    const runtime::RuntimeContext& context,
    std::string_view domain) {
    GatheredRankDigests gathered = gather_rank_digests(
        local_value,
        context,
        "Execution-provenance runtime string identity");

    const std::size_t rank_count = static_cast<std::size_t>(context.size());
    bool uniform = true;
    for (std::size_t rank = 1; rank < rank_count && uniform; ++rank) {
        uniform = same_rank_digest(
            gathered.rank_ordered_digest_bytes, 0, rank);
    }

    std::string aggregate_digest;
    std::exception_ptr stage_exception;
    try {
        aggregate_digest = hash_domain_and_rank_digests(
            domain,
            gathered.rank_ordered_digest_bytes);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception,
        context.size(),
        "Execution-provenance rank-ordered identity hashing");
    if (!is_canonical_sha256(aggregate_digest)) {
        throw std::logic_error(
            "Execution-provenance aggregate SHA-256 is non-canonical");
    }

    const bool complete_uniform_value =
        uniform
        && gathered.available_rank_count
            == static_cast<std::uint64_t>(context.size());
    return RankOrderedRuntimeStringIdentity{
        gathered.available_rank_count,
        uniform,
        complete_uniform_value ? gathered.local_digest : std::string{},
        std::move(aggregate_digest)};
}

RuntimeRankPartitionIdentity collect_rank_partition_identity(
    std::string_view local_value,
    const runtime::RuntimeContext& context,
    std::string_view domain) {
    GatheredRankDigests gathered = gather_rank_digests(
        local_value,
        context,
        "Execution-provenance processor partition");

    RuntimeRankPartitionIdentity result;
    result.available_rank_count = gathered.available_rank_count;
    if (gathered.available_rank_count
        != static_cast<std::uint64_t>(context.size())) {
        return result;
    }

    const std::size_t rank_count = static_cast<std::size_t>(context.size());
    std::vector<std::uint64_t> canonical_labels;
    std::unordered_map<std::string, std::uint64_t> label_by_digest;
    std::exception_ptr stage_exception;
    try {
        canonical_labels.resize(rank_count);
        label_by_digest.reserve(rank_count);
        std::uint64_t next_label = 0;
        for (std::size_t rank = 0; rank < rank_count; ++rank) {
            std::string digest = rank_digest_string(
                gathered.rank_ordered_digest_bytes, rank);
            const auto existing = label_by_digest.find(digest);
            if (existing != label_by_digest.end()) {
                canonical_labels[rank] = existing->second;
                continue;
            }
            if (next_label == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error(
                    "Execution-provenance processor partition label overflows uint64_t");
            }
            canonical_labels[rank] = next_label;
            label_by_digest.emplace(std::move(digest), next_label);
            ++next_label;
        }
        result.unique_value_count = next_label;

        if (domain.size() > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Execution-provenance partition domain exceeds uint64_t");
        }
        Sha256Accumulator accumulator;
        const std::array<std::uint64_t, 1> domain_size{
            static_cast<std::uint64_t>(domain.size())};
        accumulator.update_canonical_uint64(domain_size);
        accumulator.update(std::as_bytes(std::span{
            domain.data(), domain.size()}));
        accumulator.update_canonical_uint64(canonical_labels);
        result.rank_partition_sha256 = accumulator.finish_hex();
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception,
        context.size(),
        "Execution-provenance processor partition canonicalization");
    if (!is_canonical_sha256(result.rank_partition_sha256)) {
        throw std::logic_error(
            "Execution-provenance processor partition SHA-256 is non-canonical");
    }
    return result;
}

void append_nullable_sha256(
    std::ostream& out,
    std::string_view key,
    const std::string& value) {
    out << "  \"" << key << "\": ";
    if (value.empty()) {
        out << "null";
    } else {
        if (!is_canonical_sha256(value)) {
            throw std::logic_error(
                "Execution provenance contains a non-canonical SHA-256");
        }
        out << "\"" << value << "\"";
    }
}

} // namespace

ExecutionProvenance collect_execution_provenance(
    const runtime::RuntimeContext& context) {
    ExecutionProvenance result;
    result.mpi_active = context.mpi_active();
    result.topology = context.collect_topology_diagnostics();
    result.mpi_library_version = collect_rank_ordered_string_identity(
        context.mpi_library_version(),
        context,
        "HYOWON.execution_provenance.mpi_library_version.v1");
    result.processor_partition = collect_rank_partition_identity(
        context.processor_name(),
        context,
        "HYOWON.execution_provenance.processor_partition.v1");
    return result;
}

std::string execution_provenance_json(
    const ExecutionProvenance& provenance) {
    const auto& topology = provenance.topology;
    std::ostringstream out;
    out << "{\n"
        << "  \"product_kind\": \"execution_provenance\",\n"
        << "  \"semantics\": \"descriptive_execution_coordinates_not_numerical_method_identity\",\n"
        << "  \"observation_phase\": \"post_dynamics\",\n"
        << "  \"topology_scope\": \"terminal_observation_not_a_time_invariance_certificate\",\n"
        << "  \"mpi_active\": "
        << (provenance.mpi_active ? "true" : "false") << ",\n"
        << "  \"rank_count\": " << topology.rank_count << ",\n"
        << "  \"shared_memory_domain_count\": "
        << topology.shared_memory_domain_count << ",\n"
        << "  \"local_size_min\": " << topology.local_size_min << ",\n"
        << "  \"local_size_max\": " << topology.local_size_max << ",\n"
        << "  \"effective_threads_min\": "
        << topology.effective_threads_min << ",\n"
        << "  \"effective_threads_max\": "
        << topology.effective_threads_max << ",\n"
        << "  \"visible_cpu_count_min\": "
        << topology.visible_cpu_count_min << ",\n"
        << "  \"visible_cpu_count_max\": "
        << topology.visible_cpu_count_max << ",\n"
        << "  \"automatic_thread_ceiling_min\": "
        << topology.automatic_thread_ceiling_min << ",\n"
        << "  \"automatic_thread_ceiling_max\": "
        << topology.automatic_thread_ceiling_max << ",\n"
        << "  \"shared_unbound_affinity_rank_count\": "
        << topology.shared_unbound_affinity_rank_count << ",\n"
        << "  \"mpi_library_version_identity_scope\": "
           "\"sha256_of_exact_MPI_Get_library_version_bytes_when_uniform_plus_rank_ordered_aggregate\",\n"
        << "  \"mpi_library_version_available_rank_count\": "
        << provenance.mpi_library_version.available_rank_count << ",\n"
        << "  \"mpi_library_version_uniform_across_ranks\": "
        << (provenance.mpi_library_version.uniform_across_ranks
                ? "true" : "false") << ",\n";
    append_nullable_sha256(
        out,
        "mpi_library_version_uniform_value_sha256",
        provenance.mpi_library_version.uniform_value_sha256);
    out << ",\n"
        << "  \"mpi_library_version_rank_ordered_sha256\": \""
        << provenance.mpi_library_version.rank_ordered_sha256 << "\",\n"
        << "  \"processor_partition_identity_scope\": "
           "\"canonical_first_occurrence_rank_labels_from_MPI_Get_processor_name_equivalence_raw_names_and_name_hashes_omitted\",\n"
        << "  \"processor_name_available_rank_count\": "
        << provenance.processor_partition.available_rank_count << ",\n"
        << "  \"processor_partition_unique_count\": "
        << provenance.processor_partition.unique_value_count << ",\n";
    append_nullable_sha256(
        out,
        "processor_partition_sha256",
        provenance.processor_partition.rank_partition_sha256);
    out << ",\n"
        << "  \"processor_partition_matches_shared_memory_domain_count\": ";
    if (provenance.processor_partition.rank_partition_sha256.empty()) {
        out << "null";
    } else {
        out << (provenance.processor_partition.unique_value_count
                    == topology.shared_memory_domain_count
                ? "true" : "false");
    }
    out << "\n}\n";
    return out.str();
}

void write_execution_provenance(
    const std::filesystem::path& diagnostics_directory,
    const ExecutionProvenance& provenance) {
    const auto path = diagnostics_directory / "execution_provenance.json";
    const std::string document = execution_provenance_json(provenance);
    const std::string expected_sha256 = sha256_text(document);
    write_text_durable_atomic_validated(
        path,
        document,
        "execution provenance",
        [expected_sha256](const std::filesystem::path& published_path) {
            require_file_sha256(
                published_path,
                expected_sha256,
                "Execution provenance exact-byte validation");
        });
}

std::string resolve_verified_snapshot_ic_sha256(
    const config::SimulationParameters& config,
    const std::string& runtime_snapshot_ic_sha256) {
    const auto& ic = config.get_ic();
    const std::string& shared_runtime_sha256 =
        config.verified_snapshot_ic_sha256();
    require_candidate(
        runtime_snapshot_ic_sha256,
        "Explicit runtime snapshot IC SHA-256");
    require_candidate(
        shared_runtime_sha256,
        "Shared runtime snapshot IC SHA-256");
    require_candidate(
        ic.snapshot_sha256,
        "Immutable snapshot IC SHA-256");

    if (ic.mode == "generate") {
        if (!runtime_snapshot_ic_sha256.empty()
            || !shared_runtime_sha256.empty()) {
            throw std::invalid_argument(
                "Generated IC state cannot carry external snapshot provenance");
        }
        return {};
    }
    if (ic.mode != "snapshot") {
        throw std::logic_error(
            "Runtime provenance encountered an unsupported IC mode");
    }

    require_agreement(runtime_snapshot_ic_sha256, shared_runtime_sha256);
    require_agreement(runtime_snapshot_ic_sha256, ic.snapshot_sha256);
    require_agreement(shared_runtime_sha256, ic.snapshot_sha256);

    if (!runtime_snapshot_ic_sha256.empty()) {
        return runtime_snapshot_ic_sha256;
    }
    if (!shared_runtime_sha256.empty()) {
        return shared_runtime_sha256;
    }
    return ic.snapshot_sha256;
}

} // namespace cosmo_nbody::io
