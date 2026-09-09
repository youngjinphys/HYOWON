#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/io/restart_io.hpp"
#include "cosmo_nbody/time/time_stepper.hpp"

#include <filesystem>
#include <string>

namespace cosmo_nbody::io {

// Canonical restart storage: one step-addressed checkpoint directory containing
// one shard per MPI rank and a manifest published after all shards are durable.
// Manifest/shard digests are exact integrity measurements of persisted state.
class RestartCheckpointIO final : public RestartIO {
public:
    explicit RestartCheckpointIO(
        const config::SimulationParameters& config)
        : RestartIO(config), config_(config) {}

    void write_restart(
        const core::ParticleStore& particles,
        const time::TimeStepper& stepper,
        const std::string& restart_base) const;

    // Reads exactly the checkpoint directory named by checkpoint_directory and
    // returns the SHA-256 of its manifest for output provenance.
    std::string read_restart(
        const std::string& checkpoint_directory,
        core::ParticleStore& particles_out,
        time::TimeStepper& stepper_out) const;

private:
    config::SimulationParameters config_;
};

void write_restart_checkpoint_collective(
    const config::SimulationParameters& config,
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::filesystem::path& restart_base,
    int rank,
    int ranks);

// Reads exactly one explicitly named checkpoint directory. Every rank verifies
// the same manifest bytes and reads only its own shard. Returns the common
// manifest SHA-256 for trajectory provenance.
std::string read_restart_checkpoint_collective(
    const config::SimulationParameters& config,
    const std::filesystem::path& checkpoint_directory,
    int rank,
    int ranks,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out);

} // namespace cosmo_nbody::io
