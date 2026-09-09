#pragma once

#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::math {

static_assert(
    std::numeric_limits<core::Real>::is_iec559,
    "Numerical floating-environment checks require IEC 60559 / IEEE-754 core::Real semantics");
static_assert(
    std::numeric_limits<core::Real>::has_denorm == std::denorm_present,
    "Numerical floating-environment checks require core::Real subnormal support");
static_assert(sizeof(core::Real) == sizeof(std::uint64_t));

inline bool round_to_nearest_active() noexcept {
    return std::fegetround() == FE_TONEAREST;
}

inline bool gradual_underflow_active() noexcept {
    // The type may advertise subnormals while per-thread FTZ/DAZ controls alter
    // arithmetic. Compare object representations rather than floating values:
    // under DAZ, a subnormal operand can compare equal to zero and otherwise
    // make a flushed FMA result falsely appear to equal denorm_min().
    volatile core::Real one = core::Real{1.0};
    volatile core::Real tiny = std::numeric_limits<core::Real>::denorm_min();
    const core::Real result = std::fma(
        static_cast<core::Real>(one),
        static_cast<core::Real>(tiny),
        core::Real{0.0});
    const core::Real expected = std::numeric_limits<core::Real>::denorm_min();
    return core::portable_bit_cast<std::uint64_t>(result)
        == core::portable_bit_cast<std::uint64_t>(expected);
}

inline bool strict_floating_environment_active() noexcept {
    return round_to_nearest_active() && gradual_underflow_active();
}

inline void require_round_to_nearest(std::string_view role) {
    if (!round_to_nearest_active()) {
        throw std::runtime_error(
            std::string(role)
            + " requires round-to-nearest floating-point semantics");
    }
}

inline void require_strict_floating_environment(std::string_view role) {
    if (!strict_floating_environment_active()) {
        throw std::runtime_error(
            std::string(role)
            + " requires round-to-nearest and gradual-underflow floating-point semantics");
    }
}

} // namespace cosmo_nbody::math
