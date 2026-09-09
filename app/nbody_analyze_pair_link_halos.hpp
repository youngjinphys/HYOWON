#pragma once

#include "nbody_analyze_parallel.hpp"

#include "cosmo_nbody/analysis/halo_derived_properties.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"
#include "cosmo_nbody/halo/pair_links.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {

inline std::vector<halo::PairLinkHalo> materialize_pair_link_halos(
    const std::vector<halo::FoFMembership>& memberships,
    const std::vector<analysis::HaloDerivedProperties>& properties,
    const core::ParticleStore& particles) {
    if (memberships.size() != properties.size()) {
        throw std::invalid_argument(
            "Pair-link membership and property counts differ");
    }

    std::vector<halo::PairLinkHalo> result(memberships.size());
    const auto ids = particles.get_ids();
    const std::size_t owned = particles.num_owned_particles();
    parallel_for_indices(
        memberships.size(),
        [&](std::size_t group_index) {
            const auto& membership = memberships[group_index];
            const auto& property = properties[group_index];
            if (membership.id != property.halo_id
                || membership.particle_indices.size()
                    != property.particle_count) {
                throw std::invalid_argument(
                    "Pair-link membership and property rows are misaligned");
            }

            auto& pair_link_halo = result[group_index];
            pair_link_halo.id = membership.id;
            pair_link_halo.mass = property.mass;
            pair_link_halo.particle_ids.reserve(
                membership.particle_indices.size());
            for (const std::size_t index : membership.particle_indices) {
                if (index >= owned) {
                    throw std::out_of_range(
                        "FoF member index exceeds owned particles while building pair links");
                }
                pair_link_halo.particle_ids.push_back(ids[index]);
            }
            std::sort(
                pair_link_halo.particle_ids.begin(),
                pair_link_halo.particle_ids.end());
        });
    return result;
}

} // namespace cosmo_nbody::app::nbody_analyze
