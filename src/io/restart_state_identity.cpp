#include "cosmo_nbody/io/restart_state_identity.hpp"

#include "cosmo_nbody/io/content_hash.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace cosmo_nbody::io {
namespace {

constexpr std::string_view completed_global_step_phase =
    "completed_global_step";
constexpr std::string_view restart_state_product_kind = "restart_state";
constexpr std::size_t restart_state_root_fixed_bytes = 2048U;
constexpr std::uint64_t uniform_mass_tag = 0U;
constexpr std::uint64_t per_particle_mass_tag = 1U;

static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<core::Real>::is_iec559);
static_assert(std::numeric_limits<core::Real>::radix == 2);
static_assert(std::numeric_limits<core::Real>::digits == 53);
static_assert(std::numeric_limits<core::Real>::max_exponent == 1024);
static_assert(std::numeric_limits<core::Real>::min_exponent == -1021);
static_assert(std::is_trivially_copyable_v<core::Real>);
static_assert(std::is_trivially_copyable_v<core::ParticleId>);
static_assert(sizeof(core::ParticleId) == sizeof(std::uint64_t));

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
    std::array<std::byte, sizeof(value)> bytes{};
    for (std::size_t shift = 0; shift < 64U; shift += 8U) {
        bytes[shift / 8U] =
            static_cast<std::byte>((value >> shift) & 0xffU);
    }
    output.insert(output.end(), bytes.begin(), bytes.end());
}

void append_blob(std::vector<std::byte>& output, std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "Restart-state identity blob length exceeds uint64 range");
    }
    append_u64(output, static_cast<std::uint64_t>(value.size()));
    for (const unsigned char character : value) {
        output.push_back(static_cast<std::byte>(character));
    }
}

void append_real_bits(std::vector<std::byte>& output, core::Real value) {
    std::uint64_t u64_val;
    std::memcpy(&u64_val, &value, sizeof(u64_val));
    append_u64(output, u64_val);
}

std::uint64_t checked_count(std::size_t value, const char* context) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(std::string(context) + " exceeds uint64 range");
    }
    return static_cast<std::uint64_t>(value);
}

template <typename Value>
std::uint64_t checked_byte_count(
    std::span<const Value> values,
    const char* context) {
    static_assert(std::is_trivially_copyable_v<Value>);
    if (values.size()
        > std::numeric_limits<std::uint64_t>::max() / sizeof(Value)) {
        throw std::overflow_error(
            std::string(context) + " byte count exceeds uint64 range");
    }
    return static_cast<std::uint64_t>(values.size()) * sizeof(Value);
}

void require_finite_values(
    std::span<const core::Real> values,
    const char* context) {
    for (const core::Real value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                std::string("Restart-state identity requires finite ") + context);
        }
    }
}

void require_positive_finite_values(
    std::span<const core::Real> values,
    const char* context) {
    for (const core::Real value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(
                std::string("Restart-state identity requires positive finite ")
                + context);
        }
    }
}

void append_field_descriptor(
    std::vector<std::byte>& root,
    std::string_view tag,
    const RestartStateFieldDigest& field) {
    append_blob(root, tag);
    append_u64(root, field.element_bytes);
    append_u64(root, field.element_count);
    append_u64(root, field.payload_bytes);
    append_blob(root, field.sha256);
}

template <typename Value>
RestartStateFieldDigest make_field_digest(
    std::span<const Value> values) {
    static_assert(std::is_trivially_copyable_v<Value>);
    return RestartStateFieldDigest{
        sizeof(Value),
        checked_count(values.size(), "Restart-state field count"),
        checked_byte_count(values, "Restart-state field"),
        sha256_bytes(std::as_bytes(values))};
}

void require_matching_size(
    std::size_t observed,
    std::size_t expected,
    const char* field) {
    if (observed != expected) {
        throw std::invalid_argument(
            std::string("Restart-state identity ") + field
            + " count does not match particle IDs");
    }
}

} // namespace

std::string sha256_restart_state_from_field_digests(
    const RestartStateDigestInput& input) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Restart-state identity requires a qualified little-endian host");
    }
    if (input.ranks <= 0 || input.rank < 0 || input.rank >= input.ranks) {
        throw std::invalid_argument(
            "Restart-state identity requires rank in [0, ranks)");
    }
    if (!std::isfinite(input.current_a) || input.current_a <= 0.0) {
        throw std::invalid_argument(
            "Restart-state identity requires a positive finite scale factor");
    }
    if (!std::isfinite(input.delta_ln_a) || input.delta_ln_a <= 0.0) {
        throw std::invalid_argument(
            "Restart-state identity requires positive finite delta_ln_a");
    }
    if (!is_canonical_sha256(input.dynamics_sha256)) {
        throw std::invalid_argument(
            "Restart-state identity requires a canonical dynamics SHA-256");
    }

    const auto require_field = [&](const RestartStateFieldDigest& field) {
        if (field.element_bytes != sizeof(std::uint64_t)
            || field.element_count != input.particle_count
            || field.element_count > std::numeric_limits<std::uint64_t>::max()
                    / field.element_bytes
            || field.payload_bytes
                != field.element_count * field.element_bytes
            || !is_canonical_sha256(field.sha256)) {
            throw std::invalid_argument(
                "Restart-state field digest is inconsistent");
        }
    };
    for (const auto& field : input.particle_fields) require_field(field);
    if (input.uniform_mass.has_value()) {
        if (input.masses.has_value()
            || !std::isfinite(*input.uniform_mass)
            || *input.uniform_mass <= 0.0) {
            throw std::invalid_argument(
                "Restart-state uniform mass digest input is invalid");
        }
    } else {
        if (!input.masses.has_value()) {
            throw std::invalid_argument(
                "Restart-state per-particle mass digest is absent");
        }
        require_field(*input.masses);
    }

    const std::size_t reserve_bytes =
        SHA256_HEX_CHARACTER_COUNT + restart_state_root_fixed_bytes;
    std::vector<std::byte> root;
    root.reserve(reserve_bytes);

    append_blob(root, restart_state_product_kind);
    append_u64(root, static_cast<std::uint64_t>(input.rank));
    append_u64(root, static_cast<std::uint64_t>(input.ranks));
    append_blob(root, completed_global_step_phase);
    append_u64(root, input.step);
    append_real_bits(root, input.current_a);
    append_real_bits(root, input.delta_ln_a);
    append_blob(root, input.dynamics_sha256);
    append_u64(root, input.particle_count);

    if (input.uniform_mass.has_value()) {
        append_u64(root, uniform_mass_tag);
        append_real_bits(root, *input.uniform_mass);
    } else {
        append_u64(root, per_particle_mass_tag);
        append_field_descriptor(root, "masses", *input.masses);
    }

    static constexpr std::array<std::string_view, 7> tags{
        "positions_x", "positions_y", "positions_z",
        "momenta_x", "momenta_y", "momenta_z", "ids"};
    for (std::size_t index = 0; index < tags.size(); ++index) {
        append_field_descriptor(root, tags[index], input.particle_fields[index]);
    }
    return sha256_bytes(root);
}

std::string sha256_restart_state(const RestartStateIdentityInput& input) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Restart-state identity requires a qualified little-endian host");
    }

    if (input.ranks <= 0 || input.rank < 0 || input.rank >= input.ranks) {
        throw std::invalid_argument(
            "Restart-state identity requires rank in [0, ranks)");
    }
    if (!std::isfinite(input.current_a) || input.current_a <= 0.0) {
        throw std::invalid_argument(
            "Restart-state identity requires a positive finite scale factor");
    }
    if (!std::isfinite(input.delta_ln_a) || input.delta_ln_a <= 0.0) {
        throw std::invalid_argument(
            "Restart-state identity requires positive finite delta_ln_a");
    }
    if (!is_canonical_sha256(input.dynamics_sha256)) {
        throw std::invalid_argument(
            "Restart-state identity requires a canonical dynamics SHA-256");
    }

    const std::size_t particle_count = input.ids.size();
    require_matching_size(input.positions_x.size(), particle_count, "positions_x");
    require_matching_size(input.positions_y.size(), particle_count, "positions_y");
    require_matching_size(input.positions_z.size(), particle_count, "positions_z");
    require_matching_size(input.momenta_x.size(), particle_count, "momenta_x");
    require_matching_size(input.momenta_y.size(), particle_count, "momenta_y");
    require_matching_size(input.momenta_z.size(), particle_count, "momenta_z");

    require_finite_values(input.positions_x, "positions_x");
    require_finite_values(input.positions_y, "positions_y");
    require_finite_values(input.positions_z, "positions_z");
    require_finite_values(input.momenta_x, "momenta_x");
    require_finite_values(input.momenta_y, "momenta_y");
    require_finite_values(input.momenta_z, "momenta_z");

    if (input.uniform_mass.has_value()) {
        if (!input.masses.empty()) {
            throw std::invalid_argument(
                "Restart-state identity uniform mass cannot retain per-particle masses");
        }
        if (!std::isfinite(*input.uniform_mass) || *input.uniform_mass <= 0.0) {
            throw std::invalid_argument(
                "Restart-state identity requires positive finite uniform mass");
        }
    } else {
        require_matching_size(input.masses.size(), particle_count, "masses");
        require_positive_finite_values(input.masses, "per-particle masses");
    }

    RestartStateDigestInput digest_input;
    digest_input.rank = input.rank;
    digest_input.ranks = input.ranks;
    digest_input.step = input.step;
    digest_input.current_a = input.current_a;
    digest_input.delta_ln_a = input.delta_ln_a;
    digest_input.dynamics_sha256 = input.dynamics_sha256;
    digest_input.particle_count = checked_count(
        particle_count, "Restart-state particle count");
    digest_input.uniform_mass = input.uniform_mass;
    if (!input.uniform_mass.has_value()) {
        digest_input.masses = make_field_digest(input.masses);
    }
    digest_input.particle_fields = {
        make_field_digest(input.positions_x),
        make_field_digest(input.positions_y),
        make_field_digest(input.positions_z),
        make_field_digest(input.momenta_x),
        make_field_digest(input.momenta_y),
        make_field_digest(input.momenta_z),
        make_field_digest(input.ids)};
    return sha256_restart_state_from_field_digests(digest_input);
}

} // namespace cosmo_nbody::io
