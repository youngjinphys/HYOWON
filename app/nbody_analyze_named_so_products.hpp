#pragma once

#include "cosmo_nbody/analysis/halo_membership_catalog.hpp"
#include "cosmo_nbody/analysis/halo_shape.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/exact_periodic_aperture.hpp"
#include "cosmo_nbody/halo/standard_so_evaluator.hpp"

#include <filesystem>
#include <string_view>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {

void write_named_so_derived_products(
    const std::filesystem::path& path,
    const core::ParticleStore& particles,
    const halo::ExactPeriodicApertureQuery* aperture_query,
    analysis::HaloMembershipCatalogWriter& membership_output,
    const std::vector<halo::StandardSOSeedEvaluation>& evaluations,
    core::Real box_size,
    core::Real scale_factor,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles,
    std::size_t peak_density_k_neighbors,
    std::size_t deblended_min_particles,
    core::Real peak_saddle_merge_ratio,
    const analysis::HaloShapeOptions& shape_options);

} // namespace cosmo_nbody::app::nbody_analyze
