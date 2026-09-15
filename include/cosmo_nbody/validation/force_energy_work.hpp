#pragma once

#include "cosmo_nbody/core/types.hpp"

namespace cosmo_nbody::validation {

struct ForceEnergyWorkPoint {
    core::Real scale_factor{0.0};
    core::Real hubble_rate{0.0};
    core::Real momentum_force_contraction{0.0};
    core::Real directional_potential_energy_comoving{0.0};
    core::Real directional_cic_self_energy_comoving{0.0};
    core::Real power_defect{0.0};
    core::Real power_defect_over_hubble{0.0};
};

struct ForceEnergyWorkState {
    bool initialized{false};
    ForceEnergyWorkPoint previous{};
    core::Real integrated_work{0.0};
    core::Real integrated_work_compensation{0.0};
};

// For fixed-comoving-kernel pure PM under p=a^2 xdot and pdot=g/a:
//   d(K+W)/dt + H(2K+W) = P_def,
//   P_def = [sum m p.g + D_p U0,raw - D_p U0,self] / a^3.
// This measures structural force-energy work; it is not a continuum-accuracy
// certificate and it is intentionally unavailable for TreePM here.
ForceEnergyWorkPoint make_force_energy_work_point(
    core::Real scale_factor,
    core::Real hubble_rate,
    core::Real momentum_force_contraction,
    core::Real directional_potential_energy_comoving,
    core::Real directional_cic_self_energy_comoving);

void reset_force_energy_work_state(
    const ForceEnergyWorkPoint& initial,
    ForceEnergyWorkState& state);

// Trapezoidal integral of P_def/H over an already-validated d ln a interval.
core::Real update_force_energy_work_state(
    const ForceEnergyWorkPoint& current,
    core::Real delta_ln_a,
    ForceEnergyWorkState& state);

core::Real force_energy_integrated_work(
    const ForceEnergyWorkState& state);

// LI residual minus the measured structural work. This remainder still includes
// time integration, quadrature, CIC-boundary convention, and floating-point
// effects; callers must not relabel it as a pure timestep error.
core::Real force_energy_closure_residual(
    core::Real layzer_irvine_residual,
    const ForceEnergyWorkState& state);

} // namespace cosmo_nbody::validation
