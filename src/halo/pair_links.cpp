#include "cosmo_nbody/halo/pair_links.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cosmo_nbody::halo {
namespace {

std::size_t checked_add(
    std::size_t lhs,
    std::size_t rhs,
    const char* context) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(context);
    }
    return lhs + rhs;
}

void validate_local_membership(const PairLinkHalo& halo) {
    if (!std::isfinite(halo.mass) || halo.mass < 0.0) {
        throw std::invalid_argument(
            "Pair-link candidate mass must be finite and non-negative");
    }
    if (std::adjacent_find(
            halo.particle_ids.begin(), halo.particle_ids.end())
        != halo.particle_ids.end()) {
        throw std::invalid_argument(
            "Pair-link candidate contains adjacent duplicate ParticleIDs");
    }
    if (!std::is_sorted(halo.particle_ids.begin(), halo.particle_ids.end())) {
        std::vector<core::ParticleId> sorted = halo.particle_ids;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument(
                "Pair-link candidate contains duplicate ParticleIDs");
        }
    }
}

std::size_t validate_catalog(
    const std::vector<PairLinkHalo>& halos,
    const char* epoch_label) {
    std::unordered_set<std::size_t> candidate_ids;
    candidate_ids.reserve(halos.size());
    std::unordered_set<core::ParticleId> particle_ids;

    std::size_t total_particles = 0;
    for (const auto& halo : halos) {
        validate_local_membership(halo);
        if (!candidate_ids.insert(halo.id).second) {
            throw std::invalid_argument(
                std::string("Pair-link duplicate candidate ID in ")
                + epoch_label + " snapshot");
        }
        total_particles = checked_add(
            total_particles,
            halo.particle_ids.size(),
            "Pair-link catalog particle count overflows size_t");
    }
    particle_ids.reserve(total_particles);
    for (const auto& halo : halos) {
        for (const core::ParticleId id : halo.particle_ids) {
            if (!particle_ids.insert(id).second) {
                throw std::invalid_argument(
                    std::string("Pair-link ParticleID appears in multiple ")
                    + epoch_label + " candidates");
            }
        }
    }
    return total_particles;
}

bool better_primary(
    const PairLink& challenger,
    const PairLink& incumbent,
    const std::unordered_map<std::size_t, const PairLinkHalo*>& earlier_by_id) {
    if (challenger.shared_particle_count != incumbent.shared_particle_count) {
        return challenger.shared_particle_count > incumbent.shared_particle_count;
    }
    const PairLinkHalo& challenger_halo =
        *earlier_by_id.at(challenger.earlier_candidate_id);
    const PairLinkHalo& incumbent_halo =
        *earlier_by_id.at(incumbent.earlier_candidate_id);
    if (challenger_halo.mass != incumbent_halo.mass) {
        return challenger_halo.mass > incumbent_halo.mass;
    }
    if (challenger_halo.particle_ids.size()
        != incumbent_halo.particle_ids.size()) {
        return challenger_halo.particle_ids.size()
            > incumbent_halo.particle_ids.size();
    }
    return challenger.earlier_candidate_id < incumbent.earlier_candidate_id;
}

} // namespace

PairLinkBuilder::PairLinkBuilder(
    std::size_t min_shared_particles,
    core::Real min_earlier_shared_fraction)
    : min_shared_particles_(min_shared_particles),
      min_earlier_shared_fraction_(min_earlier_shared_fraction) {
    if (min_shared_particles_ == 0) {
        throw std::invalid_argument(
            "Pair-link minimum shared-particle count must be positive");
    }
    if (!std::isfinite(min_earlier_shared_fraction_)
        || min_earlier_shared_fraction_ < 0.0
        || min_earlier_shared_fraction_ > 1.0) {
        throw std::invalid_argument(
            "Pair-link minimum earlier shared fraction must lie in [0,1]");
    }
}

PairLinkSet PairLinkBuilder::link_pair(
    const std::vector<PairLinkHalo>& earlier,
    const std::vector<PairLinkHalo>& later) const {
    (void)validate_catalog(earlier, "earlier");
    const std::size_t later_particle_count =
        validate_catalog(later, "later");

    std::unordered_map<core::ParticleId, std::size_t> later_by_particle;
    later_by_particle.reserve(later_particle_count);
    for (std::size_t later_index = 0;
         later_index < later.size();
         ++later_index) {
        for (const core::ParticleId id : later[later_index].particle_ids) {
            later_by_particle.emplace(id, later_index);
        }
    }

    std::unordered_map<std::size_t, const PairLinkHalo*> earlier_by_id;
    earlier_by_id.reserve(earlier.size());
    for (const auto& halo : earlier) {
        earlier_by_id.emplace(halo.id, &halo);
    }

    PairLinkSet result;
    std::vector<bool> later_has_link(later.size(), false);
    std::vector<std::size_t> matched_later_indices;

    for (const auto& earlier_halo : earlier) {
        matched_later_indices.clear();
        matched_later_indices.reserve(earlier_halo.particle_ids.size());
        for (const core::ParticleId id : earlier_halo.particle_ids) {
            const auto found = later_by_particle.find(id);
            if (found != later_by_particle.end()) {
                matched_later_indices.push_back(found->second);
            }
        }
        if (matched_later_indices.empty()) {
            result.unmatched_earlier_candidate_ids.push_back(earlier_halo.id);
            continue;
        }

        std::sort(matched_later_indices.begin(), matched_later_indices.end());
        std::vector<std::pair<std::size_t, std::size_t>> ranked;
        for (std::size_t begin = 0;
             begin < matched_later_indices.size();) {
            std::size_t end = begin + 1;
            while (end < matched_later_indices.size()
                   && matched_later_indices[end]
                       == matched_later_indices[begin]) {
                ++end;
            }
            ranked.push_back({matched_later_indices[begin], end - begin});
            begin = end;
        }
        std::sort(
            ranked.begin(),
            ranked.end(),
            [&](const auto& lhs, const auto& rhs) {
                if (lhs.second != rhs.second) return lhs.second > rhs.second;
                return later[lhs.first].id < later[rhs.first].id;
            });

        const std::size_t best_shared = ranked.front().second;
        const std::size_t equal_best_count = static_cast<std::size_t>(
            std::count_if(
                ranked.begin(),
                ranked.end(),
                [best_shared](const auto& row) {
                    return row.second == best_shared;
                }));
        const std::size_t links_before = result.links.size();

        for (const auto& [later_index, shared] : ranked) {
            const core::Real earlier_fraction =
                earlier_halo.particle_ids.empty()
                ? core::Real{0.0}
                : static_cast<core::Real>(shared)
                    / static_cast<core::Real>(
                        earlier_halo.particle_ids.size());
            if (shared < min_shared_particles_
                || earlier_fraction < min_earlier_shared_fraction_) {
                continue;
            }
            const auto& later_halo = later[later_index];
            const core::Real later_fraction =
                later_halo.particle_ids.empty()
                ? core::Real{0.0}
                : static_cast<core::Real>(shared)
                    / static_cast<core::Real>(later_halo.particle_ids.size());
            result.links.push_back({
                earlier_halo.id,
                later_halo.id,
                shared,
                earlier_fraction,
                later_fraction,
                shared == best_shared && equal_best_count == 1,
                false,
                shared == best_shared && equal_best_count > 1,
            });
            later_has_link[later_index] = true;
        }
        if (result.links.size() == links_before) {
            result.unmatched_earlier_candidate_ids.push_back(earlier_halo.id);
        }
    }

    for (std::size_t index = 0; index < later.size(); ++index) {
        if (!later_has_link[index]) {
            result.new_later_candidate_ids.push_back(later[index].id);
        }
    }

    std::unordered_map<std::size_t, std::vector<std::size_t>>
        link_indices_by_later;
    link_indices_by_later.reserve(later.size());
    for (std::size_t index = 0; index < result.links.size(); ++index) {
        link_indices_by_later[result.links[index].later_candidate_id]
            .push_back(index);
    }
    for (auto& [later_id, indices] : link_indices_by_later) {
        (void)later_id;
        std::size_t best = indices.front();
        for (const std::size_t candidate : indices) {
            if (candidate != best
                && better_primary(
                    result.links[candidate],
                    result.links[best],
                    earlier_by_id)) {
                best = candidate;
            }
        }
        result.links[best].is_primary_earlier_candidate_for_later = true;
    }

    std::sort(
        result.links.begin(), result.links.end(),
        [](const PairLink& lhs, const PairLink& rhs) {
            if (lhs.earlier_candidate_id != rhs.earlier_candidate_id) {
                return lhs.earlier_candidate_id < rhs.earlier_candidate_id;
            }
            return lhs.later_candidate_id < rhs.later_candidate_id;
        });
    std::sort(
        result.unmatched_earlier_candidate_ids.begin(),
        result.unmatched_earlier_candidate_ids.end());
    std::sort(
        result.new_later_candidate_ids.begin(),
        result.new_later_candidate_ids.end());
    return result;
}

} // namespace cosmo_nbody::halo
