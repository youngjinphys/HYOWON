#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/cosmology/drift_kick_factors.hpp"
#include "cosmo_nbody/time/time_stepper.hpp"

#include <functional>
#include <optional>
#include <span>

namespace cosmo_nbody {
namespace time {

class LeapfrogIntegrator {
public:
    // Callback scale factor matches the particle state; the integrator validates
    // the returned acceleration.
    using ForceSolverFunc = std::function<void(
        std::span<const core::Real>,
        std::span<const core::Real>,
        std::span<const core::Real>,
        std::span<const core::Real>,
        std::span<core::Real>,
        std::span<core::Real>,
        std::span<core::Real>,
        core::Real)>;

    LeapfrogIntegrator(
        core::ParticleStore& particles,
        cosmology::DriftKickIntegrator& factors,
        ForceSolverFunc force_solver,
        core::Real box_size,
        int collective_size = 1);

    void step(const TimeStep& step);

    // Reuse a VALID force only at exactly the requested epoch; callers that
    // mutate positions outside step() must invalidate acceleration themselves.
    void ensure_force_at(core::Real scale_factor);

    // Maximum ||D p_mid|| from the last completed drift; telemetry only, not an
    // error estimate or timestep acceptance criterion.
    std::optional<core::Real> last_step_max_drift_displacement() const noexcept {
        return last_step_max_drift_displacement_;
    }

private:
    core::ParticleStore& particles_;
    cosmology::DriftKickIntegrator& factors_;
    ForceSolverFunc force_solver_;
    core::Real box_size_;
    int collective_size_;
    std::optional<core::Real> last_step_max_drift_displacement_;
    // Epoch of the VALID acceleration field, cleared when that field is retired.
    std::optional<core::Real> prepared_force_scale_factor_;

    void trigger_force_solve(core::Real scale_factor);
};

} // namespace time
} // namespace cosmo_nbody
