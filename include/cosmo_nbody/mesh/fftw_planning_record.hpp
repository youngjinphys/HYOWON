#pragma once

#include <cstddef>
#include <string>

namespace cosmo_nbody::mesh {

// FFTW planning provenance for one serial or distributed transaction; it does
// not imply wisdom or printed-plan portability across environments.
struct FftwPlanningRecord {
    std::string backend;
    std::string planner_rigor;
    std::string planner_schema;
    std::string planner_flags;
    std::string wisdom_cache_policy;
    std::string plan_identity_scope;
    std::size_t grid_size{0};
    int thread_count{1};
    std::size_t mpi_rank_count{1};
    std::string fftw_version;
    std::string fftw_build_identity_sha256;
    // Hash of current provider-path contents resolved from callable FFTW symbols;
    // it does not identify a replaced inode that remains mapped in the process.
    std::string fftw_provider_path_content_sha256;
    std::string runtime_system_identity_sha256;

    // Empty when explicit wisdom caching was not requested.
    std::string wisdom_path;

    // If true, imported_wisdom_sha256 hashes the exact descriptor-bound bytes
    // imported before this transaction planned any transforms.
    bool wisdom_imported{false};
    std::string imported_wisdom_sha256;

    // Exact wisdom_path bytes after admission/publication. In publisher races this
    // records the winner, not the bytes that selected already-created local plans.
    std::string canonical_wisdom_sha256;

    // Hash of process-wide FFTW wisdom immediately after this transaction's plans;
    // not an individual plan identity.
    std::string planned_wisdom_corpus_sha256;

    // Hash of labeled printed plans for the serial backend. Empty for FFTW-MPI,
    // where only collective planner policy is recorded.
    std::string plan_representation_sha256;

    // Create-only cache publication outcome.
    std::string wisdom_publication_outcome;
    bool wisdom_published{false};
};

} // namespace cosmo_nbody::mesh
