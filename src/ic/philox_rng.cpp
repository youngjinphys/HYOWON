#include "cosmo_nbody/ic/philox_rng.hpp"

#include <cmath>
#include <numbers>

namespace cosmo_nbody::ic::rng {

namespace {

constexpr std::uint32_t PHILOX_M0 = 0xD2511F53U;
constexpr std::uint32_t PHILOX_M1 = 0xCD9E8D57U;
constexpr std::uint32_t PHILOX_W0 = 0x9E3779B9U;
constexpr std::uint32_t PHILOX_W1 = 0xBB67AE85U;
constexpr core::Real TWO_POW_MINUS_53 = 0x1.0p-53;
constexpr core::Real HALF_FIRST_CELL = 0x1.0p-54;

std::pair<std::uint32_t, std::uint32_t> multiply_high_low(
    std::uint32_t lhs,
    std::uint32_t rhs) noexcept {
    const std::uint64_t product =
        static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs);
    return {
        static_cast<std::uint32_t>(product >> 32),
        static_cast<std::uint32_t>(product)};
}

core::Real open_uniform_from_words(
    std::uint32_t high,
    std::uint32_t low) noexcept {
    // Map 53 addressed bits into (0,1): j>0 maps to j*2^-53 and zero to 2^-54.
    // This avoids a top midpoint that can round to 1.0 in binary64.
    const std::uint64_t bits =
        (static_cast<std::uint64_t>(high) << 21)
        | (static_cast<std::uint64_t>(low) >> 11);
    if (bits == 0) return HALF_FIRST_CELL;
    return static_cast<core::Real>(bits) * TWO_POW_MINUS_53;
}

} // namespace

Counter4x32 philox4x32_10(Counter4x32 counter, Key2x32 key) noexcept {
    for (unsigned int round = 0; round < 10; ++round) {
        const auto [hi0, lo0] = multiply_high_low(PHILOX_M0, counter[0]);
        const auto [hi1, lo1] = multiply_high_low(PHILOX_M1, counter[2]);
        counter = {
            static_cast<std::uint32_t>(hi1 ^ counter[1] ^ key[0]),
            lo1,
            static_cast<std::uint32_t>(hi0 ^ counter[3] ^ key[1]),
            lo0};
        if (round != 9) {
            key[0] += PHILOX_W0;
            key[1] += PHILOX_W1;
        }
    }
    return counter;
}

std::pair<core::Real, core::Real> uniform_pair_open01(
    std::uint64_t key,
    std::int32_t kx,
    std::int32_t ky,
    std::int32_t kz,
    std::uint32_t stream) noexcept {
    const Counter4x32 counter{
        static_cast<std::uint32_t>(kx),
        static_cast<std::uint32_t>(ky),
        static_cast<std::uint32_t>(kz),
        stream};
    const Key2x32 philox_key{
        static_cast<std::uint32_t>(key),
        static_cast<std::uint32_t>(key >> 32)};
    const auto words = philox4x32_10(counter, philox_key);
    return {
        open_uniform_from_words(words[0], words[1]),
        open_uniform_from_words(words[2], words[3])};
}

std::pair<core::Real, core::Real> normal_pair(
    std::uint64_t key,
    std::int32_t kx,
    std::int32_t ky,
    std::int32_t kz,
    std::uint32_t stream) noexcept {
    const auto [u1, u2] = uniform_pair_open01(key, kx, ky, kz, stream);
    const core::Real radius = std::sqrt(-2.0 * std::log(u1));
    const core::Real theta = 2.0 * std::numbers::pi * u2;
    return {radius * std::cos(theta), radius * std::sin(theta)};
}

} // namespace cosmo_nbody::ic::rng
