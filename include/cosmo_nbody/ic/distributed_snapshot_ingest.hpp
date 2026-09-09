#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/domain/domain_decomposition.hpp"

namespace cosmo_nbody::ic {

// Collective bounded-memory ingestion for an already-generated native snapshot
// IC. Rank 0 retains the exact opened-file identity while reading bounded row
// ranges; each range is immediately routed through the canonical domain
// ownership rule. The caller-visible ParticleStore is published only after the
// complete source has been re-hashed, total mass/population/ID identity has been
// checked, and the final local stores have been canonicalized.
void load_snapshot_distributed(
    const config::SimulationParameters& config,
    domain::DomainDecomposition& domain_decomposition,
    core::ParticleStore& particles,
    int rank,
    int size);

// Collective descriptor-only source re-admission for restart resume. Rank 0
// verifies the configured immutable HDF5 object and broadcasts its already-
// bounded canonical RunMetadataJson; particle datasets are not read.
void admit_snapshot_provenance_distributed(
    const config::SimulationParameters& config,
    int rank,
    int size);

} // namespace cosmo_nbody::ic
