#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/cosmology/cosmology_model.hpp"

namespace cosmo_nbody {
namespace cosmology {

// For p=a^2 dx/dt and gravity acceleration g, KDK uses
// D=integral da/(a^3 H) and K=integral da/(a^2 H), so x+=pD and p+=gK.
// Force solvers supply g without additional scale-factor factors.
// The integrals are resolved to binary64 representation scale; this is not a
// scientific accuracy criterion.
class DriftKickIntegrator {
public:
    explicit DriftKickIntegrator(const CosmologyModel& cosmo_model);

    core::Real drift_factor(core::Real a1, core::Real a2);

    core::Real kick_factor(core::Real a1, core::Real a2);

private:
    const CosmologyModel& cosmo_model_;

    core::Real compute_drift(core::Real a1, core::Real a2) const;
    core::Real compute_kick(core::Real a1, core::Real a2) const;
};

} // namespace cosmology
} // namespace cosmo_nbody
