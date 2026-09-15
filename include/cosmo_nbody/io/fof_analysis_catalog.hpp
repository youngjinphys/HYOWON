// One HDF5 catalog stores stable-ID connectivity in /FoFMembership and
// snapshot-derived quantities in /HaloDerivedProperties. The property group
// binds the exact membership group path.
#pragma once

#include "cosmo_nbody/analysis/halo_derived_properties.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"
#include "cosmo_nbody/io/metadata.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace cosmo_nbody::io {

// Immutable context required to persist a derived FoF catalog.
// Snapshot analysis supplies these facts directly from its descriptor and
// observed execution metadata.
struct FoFAnalysisCatalogContext {
    core::Real box_size{0.0};
    std::string physics_fingerprint;
    RunMetadata metadata;
    std::optional<core::Real> expected_scale_factor{std::nullopt};
};

class FoFAnalysisCatalogIO {
public:
    explicit FoFAnalysisCatalogIO(FoFAnalysisCatalogContext context);

    // Persist stable IDs rather than transient ParticleStore indices. The two
    // spans must be one-to-one and ordered by identical halo IDs. The writer
    // independently verifies membership, mass sums, velocity convention,
    // periodic centers, provenance, and the exact cross-group binding. Public
    // publication is lock-owned and does not return until the final file and
    // parent directory have been durably synchronized.
    void write_durable(
        const std::string& filename,
        std::span<const halo::FoFMembership> memberships,
        std::span<const analysis::HaloDerivedProperties> properties,
        const core::ParticleStore& particles,
        core::Real linking_length_b,
        std::size_t min_particles,
        core::Real scale_factor,
        ProductLineage lineage) const;

private:
    // Private serialization step: validates before rename; write_durable owns
    // process coordination and post-rename durability.
    void write(
        const std::string& filename,
        std::span<const halo::FoFMembership> memberships,
        std::span<const analysis::HaloDerivedProperties> properties,
        const core::ParticleStore& particles,
        core::Real linking_length_b,
        std::size_t min_particles,
        core::Real scale_factor,
        ProductLineage lineage) const;

    FoFAnalysisCatalogContext context_;
};

} // namespace cosmo_nbody::io
