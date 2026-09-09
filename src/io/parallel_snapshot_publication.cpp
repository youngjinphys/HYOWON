#include "cosmo_nbody/io/parallel_snapshot_io.hpp"

#include <stdexcept>

namespace cosmo_nbody::io {

void ParallelSnapshotIO::write_snapshot(
    const core::ParticleStore& local_particles,
    core::Real current_a,
    int snapshot_index) const {
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_particles;
    (void)current_a;
    (void)snapshot_index;
    throw std::runtime_error(
        "Distributed snapshot output requires an MPI-enabled build");
#else
    write_snapshot_payload(local_particles, current_a, snapshot_index);
#endif
}

} // namespace cosmo_nbody::io
