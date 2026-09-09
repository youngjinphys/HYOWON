#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include <string>

namespace cosmo_nbody::io {

// Analysis payload reader bound to the snapshot's immutable descriptor.
class AnalysisSnapshotReader {
public:
    static void read_complete(
        const std::string& filename,
        const SnapshotDescriptor& descriptor,
        core::ParticleStore& particles,
        core::Real& scale_factor);
};

} // namespace cosmo_nbody::io
