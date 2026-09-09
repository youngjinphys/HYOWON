#pragma once

#include "cosmo_nbody/core/particle_store.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace cosmo_nbody::analysis::detail {

enum class MembershipValidationPath {
    StrictParticleIdOrder,
    GeneralUnique
};

// Canonical FoF membership is strictly increasing in stable ParticleID and is
// validated allocation-free. Noncanonical callers fall back to a stable-ID
// uniqueness set; repeated indices are rejected by the same check.
inline MembershipValidationPath require_unique_owned_members(
    const core::ParticleStore& particles,
    std::span<const std::size_t> members,
    const char* out_of_range_message,
    const char* duplicate_message) {
    const std::size_t owned = particles.num_owned_particles();
    const auto ids = particles.get_ids();
    if (ids.size() < owned) {
        throw std::logic_error(
            "Membership validation ParticleID storage is shorter than the owned count");
    }

    bool strictly_increasing_particle_ids = true;
    bool have_previous = false;
    core::ParticleId previous_id{};
    for (const std::size_t index : members) {
        if (index >= owned) {
            throw std::out_of_range(out_of_range_message);
        }
        const core::ParticleId id = ids[index];
        if (have_previous && !(previous_id < id)) {
            strictly_increasing_particle_ids = false;
        }
        previous_id = id;
        have_previous = true;
    }
    if (strictly_increasing_particle_ids) {
        return MembershipValidationPath::StrictParticleIdOrder;
    }

    std::unordered_set<core::ParticleId> unique_ids;
    unique_ids.reserve(members.size());
    for (const std::size_t index : members) {
        if (!unique_ids.insert(ids[index]).second) {
            throw std::invalid_argument(duplicate_message);
        }
    }
    return MembershipValidationPath::GeneralUnique;
}

// Return one deterministic arithmetic schedule. Canonical membership aliases
// the caller span; only a valid noncanonical fallback is copied and sorted by
// stable ParticleID.
inline std::span<const std::size_t> canonical_unique_owned_members(
    const core::ParticleStore& particles,
    std::span<const std::size_t> members,
    std::vector<std::size_t>& canonical_scratch,
    const char* out_of_range_message,
    const char* duplicate_message) {
    const auto path = require_unique_owned_members(
        particles,
        members,
        out_of_range_message,
        duplicate_message);
    canonical_scratch.clear();
    if (path == MembershipValidationPath::StrictParticleIdOrder) {
        return members;
    }

    canonical_scratch.assign(members.begin(), members.end());
    const auto ids = particles.get_ids();
    std::sort(
        canonical_scratch.begin(),
        canonical_scratch.end(),
        [&](std::size_t lhs, std::size_t rhs) {
            if (ids[lhs] != ids[rhs]) return ids[lhs] < ids[rhs];
            return lhs < rhs;
        });
    return std::span<const std::size_t>(canonical_scratch);
}

} // namespace cosmo_nbody::analysis::detail
