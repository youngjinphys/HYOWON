#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace cosmo_nbody::io {

struct RankOrderedRuntimeStringIdentity {
    std::uint64_t available_rank_count{0};
    bool uniform_across_ranks{true};
    std::string uniform_value_sha256;
    std::string rank_ordered_sha256;
};

struct RuntimeRankPartitionIdentity {
    std::uint64_t available_rank_count{0};
    std::uint64_t unique_value_count{0};
    std::string rank_partition_sha256;
};

// Terminal observation of the execution environment. This is descriptive
// provenance, not a numerical-method identity and not a claim that topology or
// affinity remained unchanged for the entire run.
struct ExecutionProvenance {
    bool mpi_active{false};
    runtime::RuntimeTopologyDiagnostics topology;
    RankOrderedRuntimeStringIdentity mpi_library_version;
    RuntimeRankPartitionIdentity processor_partition;
};

// All MPI ranks must call this in the same order while RuntimeContext is live.
// The MPI library is represented by bounded SHA-256 identities. Processor names
// are used only to construct a canonical rank-equivalence partition; neither raw
// processor names nor hashes of those names are persisted.
ExecutionProvenance collect_execution_provenance(
    const runtime::RuntimeContext& context);

std::string execution_provenance_json(
    const ExecutionProvenance& provenance);

void write_execution_provenance(
    const std::filesystem::path& diagnostics_directory,
    const ExecutionProvenance& provenance);

// Resolve a verified runtime snapshot-IC digest against the immutable
// configuration. Generated ICs must not carry external snapshot provenance, and
// two independently supplied non-empty identities must agree exactly.
std::string resolve_verified_snapshot_ic_sha256(
    const config::SimulationParameters& config,
    const std::string& runtime_snapshot_ic_sha256 = {});

} // namespace cosmo_nbody::io
