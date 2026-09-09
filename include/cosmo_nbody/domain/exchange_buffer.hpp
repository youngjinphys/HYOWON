#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace cosmo_nbody {
namespace domain {

enum class ExchangePayloadKind : std::uint16_t {
    Migration = 1,
    Ghost = 2
};

// Full phase-space ownership-transfer record. Migration must preserve momentum,
// so no field is optional here.
struct ExchangeParticle {
    core::Real x{0.0};
    core::Real y{0.0};
    core::Real z{0.0};
    core::Real px{0.0};
    core::Real py{0.0};
    core::Real pz{0.0};
    core::Real mass{0.0};
    core::ParticleId id{0};
};

struct RoutedExchangeParticle {
    int destination_rank{0};
    ExchangeParticle particle;
};

// Read-only TreePM source record. Ghosts are not integrated on the receiving
// rank, so momentum is absent; mass remains explicit in transport and may use
// uniform storage after receipt.
struct GhostExchangeParticle {
    core::Real x{0.0};
    core::Real y{0.0};
    core::Real z{0.0};
    core::Real mass{0.0};
    core::ParticleId id{0};
};

struct RoutedGhostExchangeParticle {
    int destination_rank{0};
    GhostExchangeParticle particle;
};

class ExchangeBuffer {
public:
    // Deterministic order: destination rank, then stable particle ID. Duplicate
    // (destination,id) pairs are rejected because they make receive semantics
    // ambiguous and can silently double-count mass.
    static void sort_routed_in_place(
        std::span<RoutedExchangeParticle> routed);
    static std::vector<RoutedExchangeParticle> sort_routed(
        std::span<const RoutedExchangeParticle> routed);

    static void sort_routed_ghosts_in_place(
        std::span<RoutedGhostExchangeParticle> routed);
};

} // namespace domain
} // namespace cosmo_nbody
