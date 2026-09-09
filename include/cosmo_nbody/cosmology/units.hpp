// Internal units: length Mpc/h, mass 1e10 M_sun/h, velocity km/s.
// In these units H0=100 and the h factors cancel from G.
#pragma once

#include "cosmo_nbody/core/types.hpp"
#include <numbers>

namespace cosmo_nbody {
namespace cosmology {
namespace units {

    constexpr core::Real G_SI = 6.67430e-11; // CODATA 2018, m^3/(kg s^2)
    constexpr core::Real Mpc_in_m = 3.085677581491367e22;
    constexpr core::Real Msun_in_kg = 1.98847e30;

    // SI realization of the internal base units; h factors cancel.
    constexpr core::Real MassUnit_in_kg = 1e10 * Msun_in_kg;
    constexpr core::Real LengthUnit_in_m = Mpc_in_m;
    constexpr core::Real VelocityUnit_in_m_s = 1000.0;

    // G in (Mpc/h)(km/s)^2/(1e10 M_sun/h).
    constexpr core::Real G = G_SI * MassUnit_in_kg / (LengthUnit_in_m * VelocityUnit_in_m_s * VelocityUnit_in_m_s);

    constexpr core::Real H0 = 100.0;

    constexpr core::Real rho_crit0 = 3.0 * H0 * H0 / (8.0 * std::numbers::pi * G);

    // CGS conversion constants written to snapshot metadata.
    constexpr core::Real UnitLength_in_cm = LengthUnit_in_m * 100.0;
    constexpr core::Real UnitMass_in_g = MassUnit_in_kg * 1000.0;
    constexpr core::Real UnitVelocity_in_cm_s = VelocityUnit_in_m_s * 100.0;
    constexpr core::Real UnitTime_in_s = LengthUnit_in_m / VelocityUnit_in_m_s;

} // namespace units
} // namespace cosmology
} // namespace cosmo_nbody
