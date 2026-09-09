#include "cosmo_nbody/io/analysis_snapshot_reader.hpp"

#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/verified_snapshot_source.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::io {

void AnalysisSnapshotReader::read_complete(
    const std::string& filename,
    const SnapshotDescriptor& descriptor,
    core::ParticleStore& particles,
    core::Real& scale_factor) {
    if (descriptor.particle_count == 0
        || descriptor.particles_per_dimension == 0
        || descriptor.pm_mesh_per_dimension == 0
        || !is_canonical_sha256(
            descriptor.native_snapshot_object_sha256)) {
        throw std::invalid_argument(
            "Analysis snapshot descriptor is structurally incomplete");
    }
    if (descriptor.particle_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Analysis snapshot descriptor particle count does not fit size_t");
    }
    const std::size_t expected =
        static_cast<std::size_t>(descriptor.particle_count);
    const SnapshotReadContext context{
        descriptor.box_size_Mpc_h,
        descriptor.omega_m,
        descriptor.omega_lambda,
        descriptor.hubble_param,
        expected,
    };

    // Re-open the operator pathname only to admit an object whose complete
    // content identity matches the descriptor-time object. All HDF5 payload
    // reads then use the retained descriptor alias, so a later pathname rename
    // cannot redirect ingestion to another snapshot.
    VerifiedSnapshotSource source(
        filename, descriptor.native_snapshot_object_sha256);
    core::ParticleStore candidate;
    core::Real candidate_scale_factor = 0.0;
    read_snapshot_with_context(
        context,
        source.hdf5_read_path(),
        candidate,
        candidate_scale_factor,
        SnapshotReadPolicy::AnalysisSubset,
        SnapshotPopulationAdmission::exact(expected));

    // Re-hash that same opened object before publishing anything. This rejects
    // in-place mutation during HDF5 ingestion as well as pathname ABA, while
    // keeping caller-visible state transactional on every failure.
    source.verify_unchanged();

    // Descriptor and payload epochs are the same stored binary64 attribute,
    // not independently recomputed physical estimates. Any bitwise numerical
    // disagreement means the payload is not the immutable descriptor object
    // admitted by this reader and must fail closed.
    if (candidate_scale_factor != descriptor.scale_factor) {
        throw std::runtime_error(
            "Analysis snapshot payload epoch disagrees with its immutable descriptor");
    }
    if (candidate.num_owned_particles() != expected) {
        throw std::runtime_error(
            "Analysis snapshot payload particle count disagrees with its immutable descriptor");
    }

    particles = std::move(candidate);
    scale_factor = candidate_scale_factor;
}

} // namespace cosmo_nbody::io
