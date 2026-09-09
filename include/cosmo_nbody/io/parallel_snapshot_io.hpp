#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <filesystem>

namespace cosmo_nbody {
namespace io {

class ParallelSnapshotIO {
public:
    explicit ParallelSnapshotIO(
        const config::SimulationParameters& config);

    ParallelSnapshotIO(
        const config::SimulationParameters& config,
        std::filesystem::path output_directory);

    // Collective over MPI_COMM_WORLD; rank 0 alone performs staged serial-HDF5
    // publication while all ranks contribute bounded rank-contiguous batches.
    void write_snapshot(
        const core::ParticleStore& local_particles,
        core::Real current_a,
        int snapshot_index) const;

private:
    // No non-root rank performs output-path filesystem operations.
    void write_snapshot_payload(
        const core::ParticleStore& local_particles,
        core::Real current_a,
        int snapshot_index) const;

    config::SimulationParameters config_;
    std::filesystem::path output_directory_;
};

} // namespace io
} // namespace cosmo_nbody
