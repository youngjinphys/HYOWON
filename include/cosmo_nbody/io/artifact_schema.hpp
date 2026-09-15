#pragma once

#include <cstddef>
#include <string_view>

namespace cosmo_nbody::io::artifact_schema {

// Canonical identities for current HYOWON-native artifacts. Readers accept only
// these identities and fail closed when an artifact does not match the current
// native contract.
inline constexpr std::string_view SNAPSHOT_ATTRIBUTE = "SnapshotSchema";
inline constexpr std::string_view RESTART_ATTRIBUTE = "RestartSchema";
inline constexpr std::string_view SNAPSHOT = "hyowon.snapshot";
inline constexpr std::string_view RESTART = "hyowon.restart";

inline constexpr std::size_t MAXIMUM_RESTART_SCHEMA_BYTES = RESTART.size();

} // namespace cosmo_nbody::io::artifact_schema
