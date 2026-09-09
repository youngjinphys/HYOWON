#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstdint>

namespace cosmo_nbody::math {

// Compares the exact positive sum of particle masses with the exact positive sum
// of the already-rounded CIC mesh cells. The tolerance accounts for binary64
// weight construction and cell accumulation. A separate transport invariant
// below isolates inter-rank plane transfer from this broader deposition error.
void require_cic_mass_conservation(
    core::Real particle_mass,
    core::Real deposited_mass,
    std::uint64_t particle_count);

// Compares the exact global sum of the pre-exchange representation
// (local cells plus outgoing planes) with the exact global sum of final local
// cells. Only non-negative boundary-cell additions separate these values. The
// machine-epsilon bound rejects transport discrepancies larger than its admitted
// floating-point envelope; it does not prove detection of an arbitrarily small
// omitted or duplicated contribution.
void require_cic_transport_conservation(
    core::Real pre_exchange_mass,
    core::Real post_exchange_mass);

} // namespace cosmo_nbody::math
