#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cosmo_nbody {
namespace domain {

// Public because the restart allocation graph must charge the
// same bounded transport extent that execution uses. Changing this constant is
// an allocation-lifetime change and requires planner plus multi-round tests.
inline constexpr std::size_t global_id_exchange_chunk_records = 65536;

// Shared sizing rule for the reusable payload buffers. Execution uses
// the actual local send/receive populations while the fail-closed planner uses
// their admitted upper bounds; both must clamp through this function so the
// planned sizes and vector capacities cannot drift apart.
constexpr std::uint64_t global_id_exchange_chunk_capacity(
    std::uint64_t population) noexcept {
    return std::min(
        population,
        static_cast<std::uint64_t>(global_id_exchange_chunk_records));
}

// Exact local portion of the canonical global stable-ID set. Equal IDs are
// routed to the same deterministic hash owner, sorted, and checked for global
// uniqueness. The complete distributed set is the rank-ordered collection of
// these local buckets; no digest or commutative fingerprint decides uniqueness.
struct GlobalParticleIdSetSnapshot {
    std::vector<core::ParticleId> canonical_local_bucket;
    int communicator_size{1};
};

GlobalParticleIdSetSnapshot capture_global_particle_id_set(
    std::span<const core::ParticleId> local_ids);

// Canonicalizes the post-operation IDs with the same hash-owner routing and
// compares every sorted bucket element exactly. Any insertion, deletion, or
// substitution fails collectively even when population and uniqueness remain
// unchanged.
void require_global_particle_id_set_preserved(
    const GlobalParticleIdSetSnapshot& before,
    std::span<const core::ParticleId> local_after_ids);

// Exact distributed uniqueness check for stable ParticleIDs.
//
// In an MPI build with an initialized multi-rank MPI_COMM_WORLD, IDs are
// deterministically hash-partitioned so equal IDs arrive on the same checker
// rank. Any duplicate causes every rank to fail at the same collective point.
// Payload traffic uses fixed global_id_exchange_chunk_records Sendrecv rounds;
// one rank may still own the complete hashed receive population in the
// fail-closed memory bound. In serial/single-rank execution, the same validation
// is checked locally.
void require_globally_unique_particle_ids(
    std::span<const core::ParticleId> local_ids);

// Exact distributed uniqueness check for canonical domain-owned IDs.
//
// Every rank must already publish its local IDs in strict increasing order.
// The MPI implementation streams each other rank's sorted sequence through one
// fixed-capacity receive buffer and performs an exact merge comparison against
// the caller's span. It neither materializes an N_local send order nor retains
// a hash-owned result bucket. Equal IDs are therefore detected without a
// digest, while additional payload storage is bounded by
// global_id_exchange_chunk_records ParticleIDs independent of population.
//
// This specialized path is for post-partition runtime state. Arbitrary-order
// external IC/restart admission must continue to use
// require_globally_unique_particle_ids().
void require_globally_unique_sorted_particle_ids(
    std::span<const core::ParticleId> local_ids);

} // namespace domain
} // namespace cosmo_nbody
