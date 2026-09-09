#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmo_nbody {
namespace core {

// Exact local uniqueness with bounded contiguous storage: strictly increasing
// IDs need no workspace, dense IDs in [0,n) use an n-bit bitmap, and other
// layouts use one sorted uint64 copy. This avoids unordered_set allocation
// overhead in large snapshot/restart and repeated domain-store validation.
inline void require_unique_particle_ids_bounded(
    std::span<const ParticleId> ids,
    const char* context) {
    bool strictly_increasing = true;
    for (std::size_t index = 1; index < ids.size(); ++index) {
        if (ids[index - 1] >= ids[index]) {
            strictly_increasing = false;
            break;
        }
    }
    if (strictly_increasing) return;

    const std::size_t n = ids.size();
    bool dense_range = true;
    for (const ParticleId id : ids) {
        if (id >= static_cast<std::uint64_t>(n)) {
            dense_range = false;
            break;
        }
    }

    if (dense_range) {
        const std::size_t words = n / 64 + (n % 64 != 0 ? 1 : 0);
        std::vector<std::uint64_t> seen(words, 0);
        for (const ParticleId id : ids) {
            const std::size_t index = static_cast<std::size_t>(id);
            const std::size_t word = index / 64;
            const std::uint64_t mask = 1ULL << (index % 64);
            if ((seen[word] & mask) != 0) {
                throw std::invalid_argument(
                    std::string(context)
                    + " contains duplicate stable ParticleIDs");
            }
            seen[word] |= mask;
        }
        return;
    }

    std::vector<std::uint64_t> sorted(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument(
            std::string(context)
            + " contains duplicate stable ParticleIDs");
    }
}

} // namespace core
} // namespace cosmo_nbody
