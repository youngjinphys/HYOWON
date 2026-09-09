#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_file_publication.hpp"

#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody {
namespace io {
namespace {

std::string exception_text(const std::exception_ptr& exception) {
    try {
        std::rethrow_exception(exception);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "unknown non-standard exception";
    }
}

} // namespace

void SnapshotIO::write_snapshot(
    const core::ParticleStore& particles,
    core::Real current_a,
    int snapshot_index) const {
    std::ostringstream path;
    path << "snapshot_" << snapshot_index << ".hdf5";
    write_snapshot_to_path(
        particles,
        current_a,
        path.str(),
        SnapshotWritePolicy::RequireAbsent);
}

void SnapshotIO::write_snapshot_to_path(
    const core::ParticleStore& particles,
    core::Real current_a,
    const std::string& filename,
    SnapshotWritePolicy policy,
    std::string* published_object_sha256_out) const {
    DurableFilePublicationPolicy publication_policy;
    switch (policy) {
    case SnapshotWritePolicy::RequireAbsent:
        publication_policy = DurableFilePublicationPolicy::RequireAbsent;
        break;
    case SnapshotWritePolicy::ReplaceExisting:
        publication_policy = DurableFilePublicationPolicy::ReplaceExisting;
        break;
    default:
        throw std::invalid_argument("Unknown snapshot write policy");
    }

    DurableFilePublication publication(
        filename,
        "native snapshot publication",
        publication_policy);
    try {
        write_snapshot_payload_to_path(
            particles,
            current_a,
            publication.staging_path().string());

        std::string staged_sha256;
        if (published_object_sha256_out != nullptr) {
            staged_sha256 = sha256_file(publication.staging_path());
        }

        publication.publish_nonempty();
        if (published_object_sha256_out != nullptr) {
            *published_object_sha256_out = std::move(staged_sha256);
        }
    } catch (...) {
        const std::exception_ptr primary = std::current_exception();
        if (!publication.canonical_visible()) {
            try {
                publication.discard_unpublished();
            } catch (...) {
                const std::exception_ptr cleanup = std::current_exception();
                throw std::runtime_error(
                    "Native snapshot publication failed before visibility ("
                    + exception_text(primary)
                    + "); explicit unpublished-stage cleanup also failed ("
                    + exception_text(cleanup)
                    + "). The unpublished staging object may remain for inspection.");
            }
        }
        std::rethrow_exception(primary);
    }
}

} // namespace io
} // namespace cosmo_nbody
