#pragma once

#include <cstddef>

namespace cosmo_nbody::io {

// Provenance-text resource ceiling, not a simulation/accuracy threshold.
inline constexpr std::size_t MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES =
    1024U * 1024U;

} // namespace cosmo_nbody::io
