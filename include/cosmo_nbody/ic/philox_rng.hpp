#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

namespace cosmo_nbody::ic::rng {

inline constexpr std::string_view PHILOX_METHOD =
    "philox4x32_10_mode_counter_open53_box_muller";

// Stream IDs are part of the RNG mapping; stochastic modules must use explicit,
// non-overlapping identifiers.
inline constexpr std::uint32_t STREAM_GAUSSIAN_FOURIER = 0U;
inline constexpr std::uint32_t STREAM_FIXED_PHASE = 1U;

using Counter4x32 = std::array<std::uint32_t, 4>;
using Key2x32 = std::array<std::uint32_t, 2>;

// Random123 Philox4x32 with ten rounds; the 128-bit counter and 64-bit key remain
// structurally separate.
Counter4x32 philox4x32_10(Counter4x32 counter, Key2x32 key) noexcept;

// Map one Philox block to two binary64 uniforms in (0,1) on a 53-bit interior
// grid, excluding both endpoints.
std::pair<core::Real, core::Real> uniform_pair_open01(
    std::uint64_t key,
    std::int32_t kx,
    std::int32_t ky,
    std::int32_t kz,
    std::uint32_t stream) noexcept;

// Box-Muller pair from open-interval uniforms; u1 is never zero.
std::pair<core::Real, core::Real> normal_pair(
    std::uint64_t key,
    std::int32_t kx,
    std::int32_t ky,
    std::int32_t kz,
    std::uint32_t stream = STREAM_GAUSSIAN_FOURIER) noexcept;

} // namespace cosmo_nbody::ic::rng
