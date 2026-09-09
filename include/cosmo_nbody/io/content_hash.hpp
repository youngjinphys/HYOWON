#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace cosmo_nbody::io {

inline constexpr std::size_t SHA256_HEX_CHARACTER_COUNT = 64U;

// Incremental SHA-256 equivalent to hashing concatenated update() inputs.
class Sha256Accumulator {
public:
    Sha256Accumulator();
    ~Sha256Accumulator();

    Sha256Accumulator(Sha256Accumulator&&) noexcept;
    Sha256Accumulator& operator=(Sha256Accumulator&&) noexcept;

    Sha256Accumulator(const Sha256Accumulator&) = delete;
    Sha256Accumulator& operator=(const Sha256Accumulator&) = delete;

    void update(std::span<const std::byte> bytes);

    // Canonical little-endian identity encoding; Real is finite binary64 with
    // signed zero normalized to +0. Streaming does not scale allocation with input.
    void update_canonical_uint64(std::span<const std::uint64_t> values);
    void update_canonical_real(std::span<const core::Real> values);
    std::string finish_hex();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Canonical digest text: exactly 64 lowercase hexadecimal characters.
inline bool is_canonical_sha256(std::string_view value) noexcept {
    if (value.size() != SHA256_HEX_CHARACTER_COUNT) return false;
    for (const char character : value) {
        const bool decimal = character >= '0' && character <= '9';
        const bool lower_hex = character >= 'a' && character <= 'f';
        if (!decimal && !lower_hex) return false;
    }
    return true;
}

std::string sha256_bytes(std::span<const std::byte> bytes);

// Hash exactly the string_view bytes; no terminator or metadata is added.
inline std::string sha256_text(std::string_view text) {
    return sha256_bytes(std::as_bytes(std::span{text.data(), text.size()}));
}

// Hash exact file contents; POSIX reads fail closed on object replacement/mutation.
std::string sha256_file(const std::filesystem::path& path);

// Order-independent logical particle-state hash sorted by unique ParticleId;
// coordinates use canonical finite binary64 and duplicate IDs are rejected.
std::string sha256_particle_state(
    std::span<const core::ParticleId> particle_ids,
    std::span<const core::Real> positions_x,
    std::span<const core::Real> positions_y,
    std::span<const core::Real> positions_z);

// Re-hash a file and require the earlier immutable identity.
void require_file_sha256(
    const std::filesystem::path& path,
    const std::string& expected_sha256,
    const char* context);

} // namespace cosmo_nbody::io
