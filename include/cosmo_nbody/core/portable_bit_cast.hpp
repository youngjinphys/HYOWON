// memcpy-based object-representation cast for supported C++20 libraries that
// do not expose std::bit_cast. Constraints require equal-size trivially
// copyable source and destination types.
#pragma once

#include <cstring>
#include <type_traits>

namespace cosmo_nbody::core {

template <typename To, typename From>
requires (
    sizeof(To) == sizeof(From)
    && std::is_trivially_copyable_v<To>
    && std::is_trivially_copyable_v<From>
    && std::is_trivially_default_constructible_v<To>)
inline To portable_bit_cast(const From& source) noexcept {
    To destination;
    std::memcpy(&destination, &source, sizeof(To));
    return destination;
}

} // namespace cosmo_nbody::core
