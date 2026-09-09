#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/io/metadata.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cosmo_nbody {
namespace io {

struct SnapshotMetadataMemoryPlan {
    std::uint64_t metadata_dynamic_payload_bytes{0};
    std::uint64_t metadata_json_storage_bytes{0};
    std::uint64_t physics_fingerprint_storage_bytes{0};
    std::uint64_t metadata_serialization_phase_bytes{0};
    std::uint64_t retained_attribute_bytes{0};
    std::uint64_t reader_attribute_peak_bytes{0};
    std::uint64_t serial_metadata_peak_bytes{0};
    std::uint64_t parallel_rank_agreement_reference_bytes{0};
    std::uint64_t parallel_rank_agreement_phase_bytes{0};
    std::uint64_t parallel_metadata_peak_bytes{0};
};

// Pure checked byte-count accounting for native snapshot metadata. Execution
// passes actual bounded lengths; offline graphs pass the shared maxima. The
// rank-agreement reference is a vector<char> and therefore has no NUL byte.
SnapshotMetadataMemoryPlan snapshot_metadata_memory_plan(
    std::uint64_t metadata_dynamic_payload_bytes,
    std::uint64_t metadata_json_logical_bytes,
    std::uint64_t physics_fingerprint_logical_bytes);

// Deterministic fingerprint of the declared causal/numerical configuration
// carried by a native snapshot. It is an SPMD rank-agreement value, not a
// bitwise state digest and not proof that two trajectories are physically or
// numerically equivalent. Snapshot boundaries are included because TimeStepper
// materializes them as exact KDK integration boundaries. MPI mode is included
// because it selects a different execution path, while rank count, thread count,
// validation diagnostics, restart cadence, and filesystem locations are not
// represented here. Those execution/provenance coordinates belong in
// RunMetadata or runtime diagnostics and must be compared explicitly when needed.
std::string snapshot_physics_fingerprint(
    const config::SimulationParameters& config,
    const std::string& verified_snapshot_ic_sha256 = {});

} // namespace io
} // namespace cosmo_nbody
