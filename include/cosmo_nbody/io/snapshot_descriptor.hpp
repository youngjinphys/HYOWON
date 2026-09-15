#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/io/output_schema.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace cosmo_nbody::config {
class SimulationParameters;
}

namespace cosmo_nbody::io {

class VerifiedSnapshotSource;

// Verified upstream generator coordinates; unavailable when immediate source
// identity survives but canonical generator metadata does not.
struct SnapshotGenerationProvenance {
    bool available{false};
    std::uint64_t seed{0};
    std::uint64_t ic_mesh_per_dimension{0};
    // Fourier support may be unavailable in descriptive provenance that is not
    // admissible as a generated-IC source for evolution.
    std::optional<std::uint64_t> ic_max_mode_per_axis;
    std::optional<std::uint64_t> ic_effective_max_mode_per_axis;
    int lpt_order{0};
    std::string power_spectrum_sha256;
    core::Real power_spectrum_redshift{0.0};
    std::string power_spectrum_fidelity;
    std::string ic_amplitude_mode;
    std::string ic_phase_pairing;
    std::string ic_lattice_convention;
};

// Immutable native-snapshot physics metadata read before particle allocation.
struct SnapshotDescriptor {
    core::Real scale_factor{0.0};
    core::Real redshift{0.0};
    core::Real box_size_Mpc_h{0.0};
    core::Real omega_m{0.0};
    core::Real omega_lambda{0.0};
    core::Real omega_b{0.0};
    core::Real hubble_param{0.0};

    std::uint64_t particle_count{0};
    bool uniform_mass{false};
    core::Real uniform_particle_mass{0.0};

    std::uint64_t particles_per_dimension{0};
    std::uint64_t pm_mesh_per_dimension{0};
    std::uint64_t ic_mesh_per_dimension{0};
    std::uint64_t ic_seed{0};
    int lpt_order{0};
    core::Real softening_comoving_Mpc_h{0.0};
    core::Real sigma8{0.0};
    core::Real spectral_index_ns{0.0};
    std::string ic_amplitude_mode;
    std::string ic_phase_pairing;
    std::string power_spectrum_fidelity;
    SnapshotGenerationProvenance generation_provenance;

    std::string physics_fingerprint;
    std::string run_metadata_json;

    // Exact current snapshot-object identity, distinct from upstream IC identity.
    std::string native_snapshot_object_sha256;

    // Size is bound to the schema's semantic-binding count.
    std::array<
        std::string,
        schema::NATIVE_SNAPSHOT_SEMANTIC_BINDING_COUNT>
        semantic_binding_values{};
};

SnapshotDescriptor read_snapshot_descriptor(const std::string& filename);
SnapshotDescriptor read_snapshot_descriptor(
    const VerifiedSnapshotSource& source);

// Parse canonical RunMetadataJson for republishing verified generator provenance.
SnapshotGenerationProvenance read_snapshot_generation_provenance(
    std::string_view run_metadata_json);

// Bind exact-IC descriptor fields to the admitted source and evolution coordinates.
void require_snapshot_initial_condition_match(
    const config::SimulationParameters& config,
    const SnapshotDescriptor& descriptor,
    std::string_view admitted_snapshot_sha256);

// Field comparison permits different physics/resolution but requires common
// periodic domain and phase-space/unit semantics.
void require_field_comparison_match(
    const SnapshotDescriptor& reference,
    const SnapshotDescriptor& candidate);

// Stable-ID pair links require one trajectory identity and particle-ID domain.
void require_pair_link_match(
    const SnapshotDescriptor& earlier,
    const SnapshotDescriptor& later);

} // namespace cosmo_nbody::io
