#include "cosmo_nbody/time/leapfrog_integrator.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/time/particle_update.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace time {
LeapfrogIntegrator::LeapfrogIntegrator(
    core::ParticleStore& particles,
    cosmology::DriftKickIntegrator& factors,
    ForceSolverFunc force_solver,
    core::Real box_size,
    int collective_size)
    : particles_(particles),
      factors_(factors),
      force_solver_(std::move(force_solver)),
      box_size_(box_size),
      collective_size_(collective_size) {
    if (collective_size_ < 1) {
        throw std::invalid_argument("LeapfrogIntegrator collective size must be positive");
    }
    if (!force_solver_) {
        throw std::invalid_argument(
            "LeapfrogIntegrator requires a force solver callback");
    }
    if (!std::isfinite(box_size_) || box_size_ <= 0.0) {
        throw std::invalid_argument(
            "LeapfrogIntegrator box size must be finite and positive");
    }
}

void LeapfrogIntegrator::trigger_force_solve(core::Real scale_factor) {
    if (!std::isfinite(scale_factor)
        || scale_factor <= 0.0
        || scale_factor > 1.0) {
        throw std::invalid_argument(
            "LeapfrogIntegrator force scale factor must lie in (0,1]");
    }

    // Invalidate the stored acceleration before the callback can clear or
    // partially replace it; successful publication restores validity and epoch.
    particles_.set_acceleration_validity(core::FieldValidity::INVALID);
    prepared_force_scale_factor_.reset();

    auto acc_x = particles_.mutable_accelerations_x();
    auto acc_y = particles_.mutable_accelerations_y();
    auto acc_z = particles_.mutable_accelerations_z();
    std::size_t count = acc_x.size();

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t i = 0; i < count; ++i) {
        acc_x[i] = 0.0;
        acc_y[i] = 0.0;
        acc_z[i] = 0.0;
    }

    const std::span<const core::Real> masses =
        particles_.get_uniform_mass().has_value()
        ? std::span<const core::Real>{}
        : particles_.get_masses();

    // Callbacks that perform collectives must synchronize their own internal
    // stages. Synchronize their return here as well, so a local exception does
    // not leave peers advancing into the acceleration publication collective.
    std::exception_ptr force_error;
    try {
        force_solver_(
            particles_.get_positions_x(),
            particles_.get_positions_y(),
            particles_.get_positions_z(),
            masses,
            acc_x,
            acc_y,
            acc_z,
            scale_factor);
    } catch (...) {
        force_error = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        force_error, collective_size_, "Leapfrog force callback");

    // The callback may migrate particles and reallocate storage. Reacquire the
    // acceleration spans before their canonical finite check and publication.
    acc_x = particles_.mutable_accelerations_x();
    acc_y = particles_.mutable_accelerations_y();
    acc_z = particles_.mutable_accelerations_z();
    count = acc_x.size();

    int invalid_force = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for reduction(|:invalid_force) schedule(static) \
        if(runtime::should_use_host_parallel_team(count))
#endif
    for (std::size_t i = 0; i < count; ++i) {
        invalid_force |= !std::isfinite(acc_x[i])
            || !std::isfinite(acc_y[i])
            || !std::isfinite(acc_z[i]);
    }
    if (collective_size_ > 1) {
        runtime::synchronize_mpi_failure(invalid_force, collective_size_,
            "Leapfrog force callback produced a non-finite acceleration");
    }
    if (invalid_force != 0) {
        throw std::overflow_error(
            "Leapfrog force callback produced a non-finite acceleration");
    }

    particles_.set_acceleration_validity(core::FieldValidity::VALID);
    prepared_force_scale_factor_ = scale_factor;
}

void LeapfrogIntegrator::ensure_force_at(core::Real scale_factor) {
    if (particles_.get_acceleration_validity() == core::FieldValidity::VALID
        && prepared_force_scale_factor_
        && *prepared_force_scale_factor_ == scale_factor) {
        return;
    }
    trigger_force_solve(scale_factor);
}

void LeapfrogIntegrator::step(const TimeStep& step) {
    last_step_max_drift_displacement_.reset();
    ensure_force_at(step.a_begin());

    core::Real D = 0.0, K1 = 0.0, K2 = 0.0;
    std::exception_ptr factor_error;
    try {
        D = factors_.drift_factor(step.a_begin(), step.a_end());
        K1 = factors_.kick_factor(step.a_begin(), step.a_half());
        K2 = factors_.kick_factor(step.a_half(), step.a_end());
    } catch (...) {
        factor_error = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        factor_error, collective_size_, "Leapfrog drift/kick factors");

    std::size_t N = particles_.num_owned_particles();
    auto px = particles_.get_positions_x();
    auto py = particles_.get_positions_y();
    auto pz = particles_.get_positions_z();
    auto momx = particles_.get_momenta_x();
    auto momy = particles_.get_momenta_y();
    auto momz = particles_.get_momenta_z();
    auto accx = particles_.get_accelerations_x();
    auto accy = particles_.get_accelerations_y();
    auto accz = particles_.get_accelerations_z();

    int invalid_kick_drift = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for reduction(|:invalid_kick_drift) schedule(static) \
        if(runtime::should_use_host_parallel_team(N))
#endif
    for (std::size_t i = 0; i < N; ++i) {
        KickDriftCandidate candidate;
        invalid_kick_drift |= !finite_kick_drift_candidate(
            px[i], py[i], pz[i],
            momx[i], momy[i], momz[i],
            accx[i], accy[i], accz[i],
            K1, D, box_size_, candidate);
    }
    if (collective_size_ > 1) {
        runtime::synchronize_mpi_failure(invalid_kick_drift, collective_size_,
            "Leapfrog K1/drift update is not finite and representable");
    }
    if (invalid_kick_drift != 0) {
        throw std::overflow_error(
            "Leapfrog K1/drift update is not finite and representable");
    }

    // Publish only after preflight succeeds. Accumulate the exact published
    // drift in the same pass to avoid another O(N) diagnostic scan.
    core::Real maximum_drift = 0.0;
    int invalid_drift = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for reduction(max:maximum_drift) \
        reduction(|:invalid_drift) schedule(static) \
        if(runtime::should_use_host_parallel_team(N))
#endif
    for (std::size_t i = 0; i < N; ++i) {
        KickDriftCandidate candidate;
        (void)finite_kick_drift_candidate(
            px[i], py[i], pz[i],
            momx[i], momy[i], momz[i],
            accx[i], accy[i], accz[i],
            K1, D, box_size_, candidate);
        momx[i] = candidate.momentum_x;
        momy[i] = candidate.momentum_y;
        momz[i] = candidate.momentum_z;
        px[i] = candidate.position_x;
        py[i] = candidate.position_y;
        pz[i] = candidate.position_z;
        const core::Real displacement = core::scale_safe_norm3(
            D * candidate.momentum_x,
            D * candidate.momentum_y,
            D * candidate.momentum_z);
        invalid_drift |= !std::isfinite(displacement);
        if (std::isfinite(displacement)) {
            maximum_drift = std::max(maximum_drift, displacement);
        }
    }
    last_step_max_drift_displacement_ = invalid_drift == 0
        ? std::optional<core::Real>{maximum_drift}
        : std::nullopt;

    particles_.set_acceleration_validity(core::FieldValidity::STALE);
    prepared_force_scale_factor_.reset();
    trigger_force_solve(step.a_end());

    // The force callback may migrate particles and reallocate storage.
    N = particles_.num_owned_particles();
    momx = particles_.get_momenta_x();
    momy = particles_.get_momenta_y();
    momz = particles_.get_momenta_z();
    accx = particles_.get_accelerations_x();
    accy = particles_.get_accelerations_y();
    accz = particles_.get_accelerations_z();

    int invalid_kick = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for reduction(|:invalid_kick) schedule(static) \
        if(runtime::should_use_host_parallel_team(N))
#endif
    for (std::size_t i = 0; i < N; ++i) {
        core::Real next_x = 0.0;
        core::Real next_y = 0.0;
        core::Real next_z = 0.0;
        invalid_kick |= !finite_kick_candidate(
            momx[i], momy[i], momz[i],
            accx[i], accy[i], accz[i],
            K2, next_x, next_y, next_z);
    }
    if (collective_size_ > 1) {
        runtime::synchronize_mpi_failure(invalid_kick, collective_size_,
            "Leapfrog K2 update is not finite and representable");
    }
    if (invalid_kick != 0) {
        throw std::overflow_error(
            "Leapfrog K2 update is not finite and representable");
    }

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        if(runtime::should_use_host_parallel_team(N))
#endif
    for (std::size_t i = 0; i < N; ++i) {
        core::Real next_x = 0.0;
        core::Real next_y = 0.0;
        core::Real next_z = 0.0;
        (void)finite_kick_candidate(
            momx[i], momy[i], momz[i],
            accx[i], accy[i], accz[i],
            K2, next_x, next_y, next_z);
        momx[i] = next_x;
        momy[i] = next_y;
        momz[i] = next_z;
    }
}

} // namespace time
} // namespace cosmo_nbody
