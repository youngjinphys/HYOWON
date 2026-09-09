#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/domain/exchange_buffer.hpp"

namespace cosmo_nbody::domain {

// Requires a fully finite phase-space record, positive finite mass, and
// canonical periodic coordinates in the half-open interval [0, box_size).
void validate_received_exchange_particle(
    const ExchangeParticle& particle,
    core::Real box_size);

// Ghost records intentionally omit momentum because they are force sources, not
// locally integrated particles. Position/mass semantics remain identical to the
// corresponding fields in a migration record.
void validate_received_ghost_exchange_particle(
    const GhostExchangeParticle& particle,
    core::Real box_size);

} // namespace cosmo_nbody::domain
