#pragma once

#include "cosmo_nbody/core/explicit_value.hpp"

namespace cosmo_nbody::core {

// Semantic alias for required Boolean coordinates. Omission is represented by
// the same generic ExplicitValue<T> mechanism used by other scalar choices.
using ExplicitBoolean = ExplicitValue<bool>;

} // namespace cosmo_nbody::core
