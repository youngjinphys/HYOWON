#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/halo/halo_definition.hpp"
#include "cosmo_nbody/halo/spherical_overdensity.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace cosmo_nbody::halo {

struct NamedSOSeed {
    std::size_t candidate_id{0};
    std::size_t deblended_seed_id{0};
    core::ParticleId peak_particle_id{0};
    core::Vec3 center{};
    HaloCenterSelection center_selection{};
};

struct StandardSOSeedEvaluation {
    NamedSOSeed seed;
    std::array<SOMassMeasurement, 3> measurements;
};

// Evaluate M200m, M200c, and BN98 for each ordered seed; unresolved crossings
// remain explicit measurements with crossing_resolved=false.
class StandardSOEvaluator {
public:
    explicit StandardSOEvaluator(SOContext context);

    std::vector<StandardSOSeedEvaluation> evaluate(
        const core::ParticleStore& particles,
        const std::vector<NamedSOSeed>& seeds,
        core::Real scale_factor) const;

private:
    SOContext context_;
};

} // namespace cosmo_nbody::halo
