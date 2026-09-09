// HDF5 persistence for analyzer-produced non-FoF membership views. Canonical FoF
// membership is referenced, not duplicated; derived groups own flat member-ID lists.
#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/halo_definition.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

namespace cosmo_nbody::analysis {

class HaloMembershipCatalogWriter {
public:
    HaloMembershipCatalogWriter(
        const std::filesystem::path& path,
        const std::filesystem::path& source_snapshot,
        std::string_view source_native_snapshot_object_sha256,
        core::Real scale_factor,
        core::Real fof_linking_length_b,
        std::size_t fof_min_particles,
        std::size_t peak_density_k_neighbors,
        std::size_t deblended_min_particles,
        core::Real peak_saddle_merge_ratio,
        std::string_view fof_membership_product);
    ~HaloMembershipCatalogWriter();

    HaloMembershipCatalogWriter(const HaloMembershipCatalogWriter&) = delete;
    HaloMembershipCatalogWriter& operator=(
        const HaloMembershipCatalogWriter&) = delete;
    HaloMembershipCatalogWriter(HaloMembershipCatalogWriter&&) = delete;
    HaloMembershipCatalogWriter& operator=(
        HaloMembershipCatalogWriter&&) = delete;

    void append_deblended_density_basin(
        std::size_t candidate_id,
        std::size_t deblended_seed_id,
        core::ParticleId peak_particle_id,
        std::span<const core::ParticleId> member_ids);

    // Unresolved SO crossings use available=false and an empty member span.
    void append_geometric_so(
        std::size_t candidate_id,
        std::size_t deblended_seed_id,
        core::ParticleId peak_particle_id,
        halo::StandardSOMassDefinition definition,
        bool available,
        std::span<const core::ParticleId> member_ids);

    void finalize();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cosmo_nbody::analysis
