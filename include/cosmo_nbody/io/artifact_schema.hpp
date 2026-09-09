#pragma once

#include <cstddef>
#include <string_view>

namespace cosmo_nbody::io::artifact_schema {

// Persisted compatibility/provenance bytes, independent of HYOWON release names.
// Refactoring call sites must not rename these identities or their semantics.
struct SnapshotSchema {
    std::string_view identity;
};

struct RestartSchema {
    std::string_view identity;
    bool includes_input_paths;
    bool includes_ic_support;
};

inline constexpr std::string_view SNAPSHOT_ATTRIBUTE = "SnapshotSchema";
inline constexpr std::string_view RESTART_ATTRIBUTE = "RestartSchema";

inline constexpr SnapshotSchema CURRENT_SNAPSHOT{"hyowon.snapshot.v1"};

// v1/v2 lack generated-IC Fourier support and cannot attest those runs.
// Snapshot-based legacy restarts remain readable through their source SHA-256.
inline constexpr RestartSchema PATH_BOUND_RESTART{"hyowon.restart.v1", true, false};
inline constexpr RestartSchema CONTENT_BOUND_RESTART{"hyowon.restart.v2", false, false};
inline constexpr RestartSchema CURRENT_RESTART{"hyowon.restart.v3", false, true};

inline constexpr std::size_t MAXIMUM_RESTART_SCHEMA_BYTES =
    PATH_BOUND_RESTART.identity.size() > CURRENT_RESTART.identity.size()
        ? PATH_BOUND_RESTART.identity.size()
        : CURRENT_RESTART.identity.size();

[[nodiscard]] constexpr const RestartSchema* lookup_restart(
    std::string_view identity) noexcept {
    if (identity == CURRENT_RESTART.identity) return &CURRENT_RESTART;
    if (identity == CONTENT_BOUND_RESTART.identity) return &CONTENT_BOUND_RESTART;
    if (identity == PATH_BOUND_RESTART.identity) return &PATH_BOUND_RESTART;
    return nullptr;
}

// Additional bytes already persisted outside the snapshot/restart HDF5 schema.
// They belong at the same compatibility boundary even though they do not select
// an HDF5 decoder. Keep the historical bytes unchanged.
inline constexpr std::string_view RESTART_CHECKPOINT_MANIFEST =
    "hyowon_restart_checkpoint_v1";
inline constexpr std::string_view SNAPSHOT_TARGET_ENCODING = "indexed_key_v1";
inline constexpr std::string_view SERIAL_FFTW_PLANNER =
    "serial_fftw_r2c_c2r_3d_out_of_place_aligned_and_unaligned_v2";
inline constexpr std::string_view MPI_FFTW_PLANNER =
    "fftw_mpi_r2c_c2r_3d_out_of_place_v2";

} // namespace cosmo_nbody::io::artifact_schema
