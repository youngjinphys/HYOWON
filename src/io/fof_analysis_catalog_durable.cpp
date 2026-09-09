#include "cosmo_nbody/io/fof_analysis_catalog.hpp"

#include "cosmo_nbody/io/durable_file_publication.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::io {

void FoFAnalysisCatalogIO::write_durable(
    const std::string& filename,
    std::span<const halo::FoFMembership> memberships,
    std::span<const analysis::HaloDerivedProperties> properties,
    const core::ParticleStore& particles,
    core::Real linking_length_b,
    std::size_t min_particles,
    core::Real scale_factor,
    ProductLineage lineage) const {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Durable split FoF analysis catalog filename must not be empty");
    }

    DurableFilePublication publication(
        std::filesystem::path(filename),
        "split FoF analysis catalog",
        DurableFilePublicationPolicy::RequireAbsent);
    write(
        publication.staging_path().string(),
        memberships,
        properties,
        particles,
        linking_length_b,
        min_particles,
        scale_factor,
        lineage);
    publication.publish_nonempty();
}

} // namespace cosmo_nbody::io
