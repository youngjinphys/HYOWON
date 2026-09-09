#pragma once

#include "cosmo_nbody/domain/exchange_buffer.hpp"

#include <span>
#include <vector>

namespace cosmo_nbody::domain {

// Full phase-space migration in destination/ID order; successful transfer releases
// routing storage before the caller merges incoming and retained owned records.
std::vector<ExchangeParticle> exchange_migration_bounded(
    std::vector<RoutedExchangeParticle>& sorted_routed,
    int rank,
    int size);

// Deterministic ghost transport carries force-source state only, excluding momentum.
std::vector<GhostExchangeParticle> exchange_routed_ghosts_bounded(
    std::span<const RoutedGhostExchangeParticle> sorted_routed,
    int rank,
    int size);

} // namespace cosmo_nbody::domain
