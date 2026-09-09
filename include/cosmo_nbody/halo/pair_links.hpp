#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <vector>

namespace cosmo_nbody::halo {

// Candidate membership at one epoch; this type has no merger-tree semantics.
struct PairLinkHalo {
    std::size_t id{0};
    core::Real mass{0.0};
    std::vector<core::ParticleId> particle_ids;
};

struct PairLink {
    std::size_t earlier_candidate_id{0};
    std::size_t later_candidate_id{0};
    std::size_t shared_particle_count{0};
    core::Real earlier_shared_fraction{0.0};
    core::Real later_shared_fraction{0.0};

    // Unique largest-overlap later candidate for this earlier candidate.
    bool is_unambiguous_forward_match{false};

    // Deterministic primary incoming link for this later candidate.
    bool is_primary_earlier_candidate_for_later{false};

    // Equal-largest overlap from one earlier candidate to multiple later candidates.
    bool is_ambiguous{false};
};

struct PairLinkSet {
    std::vector<PairLink> links;
    std::vector<std::size_t> unmatched_earlier_candidate_ids;
    std::vector<std::size_t> new_later_candidate_ids;
};

// Deterministic stable-ParticleID overlap for one snapshot pair, not a merger tree.
class PairLinkBuilder {
public:
    explicit PairLinkBuilder(
        std::size_t min_shared_particles = 1,
        core::Real min_earlier_shared_fraction = 0.0);

    PairLinkSet link_pair(
        const std::vector<PairLinkHalo>& earlier,
        const std::vector<PairLinkHalo>& later) const;

private:
    std::size_t min_shared_particles_{1};
    core::Real min_earlier_shared_fraction_{0.0};
};

} // namespace cosmo_nbody::halo
