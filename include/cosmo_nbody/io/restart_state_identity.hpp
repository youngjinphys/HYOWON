#pragma once
#include "cosmo_nbody/core/types.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
namespace cosmo_nbody::io {

struct RestartStateFieldDigest {
    std::uint64_t element_bytes{0};
    std::uint64_t element_count{0};
    std::uint64_t payload_bytes{0};
    std::string sha256;
};

struct RestartStateDigestInput {
    int rank{0};
    int ranks{0};
    std::uint64_t step{0};
    core::Real current_a{0.0};
    core::Real delta_ln_a{0.0};
    std::string_view dynamics_sha256;
    std::uint64_t particle_count{0};
    std::optional<core::Real> uniform_mass;
    std::optional<RestartStateFieldDigest> masses;
    std::array<RestartStateFieldDigest, 7> particle_fields;
};

struct RestartStateIdentityInput {
    int rank{0};
    int ranks{0};
    std::uint64_t step{0};
    core::Real current_a{0.0};
    core::Real delta_ln_a{0.0};
    std::string_view dynamics_sha256;
    std::span<const core::Real> positions_x;
    std::span<const core::Real> positions_y;
    std::span<const core::Real> positions_z;
    std::span<const core::Real> momenta_x;
    std::span<const core::Real> momenta_y;
    std::span<const core::Real> momenta_z;
    std::span<const core::ParticleId> ids;
    std::optional<core::Real> uniform_mass;
    std::span<const core::Real> masses;
};

// Construct the persisted semantic state identity from already-computed field
// digests. Field order is positions x/y/z, momenta x/y/z, then particle IDs.
// This is shared by the in-memory writer and bounded HDF5 verifier.
std::string sha256_restart_state_from_field_digests(
    const RestartStateDigestInput& input);

std::string sha256_restart_state(const RestartStateIdentityInput& input);
} // namespace cosmo_nbody::io
