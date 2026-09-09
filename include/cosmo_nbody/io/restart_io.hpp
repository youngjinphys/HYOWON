#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/time/time_stepper.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/io/exact_hdf5_file_identity.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/restart_hdf5_storage.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>

namespace cosmo_nbody {
namespace io {

// Checkpoints contain particle and exact step state, input/dynamics identity,
// and a persisted-state digest. Native HDF5 types require supported little-endian
// binary64 hosts; reads reject mismatches before publishing state.

// Preserve the strong exception guarantee of read_restart() state publication.
static_assert(std::is_nothrow_move_assignable_v<core::ParticleStore>);
static_assert(std::is_nothrow_move_assignable_v<time::TimeStepper>);

struct RestartLayout {
    std::uint64_t particle_count{0};
    bool uniform_mass{false};
};

struct RestartStateIdentityResult {
    std::string sha256;
    bool persisted{false};
};

// Move-only session binding layout inspection and payload reads to one HDF5
// object. The requested path guards CLI argument reuse; reads continue through
// the opened object even if that pathname is later replaced.
class RestartReadSession {
public:
    RestartReadSession(RestartReadSession&&) noexcept = default;
    RestartReadSession& operator=(RestartReadSession&&) noexcept = default;

    RestartReadSession(const RestartReadSession&) = delete;
    RestartReadSession& operator=(const RestartReadSession&) = delete;

    bool matches_path(const std::string& filename) const;
    bool exact_object_identity_available() const noexcept;
    std::uint64_t exact_object_size_bytes() const;
    void finish_exact_object_identity() const;
    std::string finish_exact_object_sha256() const;

private:
    friend class RestartIO;

    explicit RestartReadSession(
        ExactHdf5FileIdentity identity) noexcept;

    hid_t file_id() const noexcept { return identity_.file_id(); }

    ExactHdf5FileIdentity identity_;
};

class RestartIO {
public:
    explicit RestartIO(const config::SimulationParameters& config);

    // Low-level single-object codec. SimulationRunner uses RestartCheckpointIO
    // so serial and MPI restarts share one step-addressed checkpoint layout.
    void write_restart(const core::ParticleStore& particles,
                       const time::TimeStepper& stepper,
                       const std::string& filename) const;
    RestartStateIdentityResult write_restart_with_identity(
        const core::ParticleStore& particles,
        const time::TimeStepper& stepper,
        const std::string& filename) const;

    // Read only allocation-shaping particle count and mass representation.
    // Full dynamics, dataset, payload, and step-boundary validation remains in
    // read_restart() before state publication.
    RestartLayout inspect_restart_layout(const std::string& filename) const;
    RestartLayout inspect_restart_layout(
        const RestartReadSession& session) const;

    RestartReadSession open_restart_session(
        const std::string& filename) const;

    void read_restart(const std::string& filename,
                      core::ParticleStore& particles_out,
                      time::TimeStepper& stepper_out) const;
    void read_restart(RestartReadSession& session,
                      core::ParticleStore& particles_out,
                      time::TimeStepper& stepper_out) const;
    RestartStateIdentityResult read_restart_with_identity(
        const std::string& filename,
        core::ParticleStore& particles_out,
        time::TimeStepper& stepper_out,
        std::string_view expected_state_sha256 = {}) const;
    RestartStateIdentityResult read_restart_with_identity(
        RestartReadSession& session,
        core::ParticleStore& particles_out,
        time::TimeStepper& stepper_out,
        std::string_view expected_state_sha256 = {}) const;

    // Expose only the cardinality needed by low-level layout guards.
    std::uint64_t configured_particle_count() const noexcept {
        return config_.num_particles();
    }

private:
    config::SimulationParameters config_;
};

} // namespace io
} // namespace cosmo_nbody
