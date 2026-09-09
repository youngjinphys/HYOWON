// Gaussian TreePM split:
//   phi_L(k) = phi_N(k) exp(-k^2 r_s^2)
//   M_S(q) = erfc(q/2) + q/sqrt(pi) exp(-q^2/4), q=r/r_s.
// For a Plummer target the tree force is f_Plummer - f_Newtonian,long, not
// f_Plummer*M_S. The matching potential correction is shifted to zero at r_cut;
// the potential is continuous there, while the force is generally discontinuous.
#pragma once

#include "cosmo_nbody/core/types.hpp"

namespace cosmo_nbody {
namespace gravity {

class ForceSplitKernel {
public:
    ForceSplitKernel(
        core::Real split_scale,
        core::Real cutoff_multiplier);

    core::Real split_scale() const noexcept { return split_scale_; }
    core::Real cutoff_radius() const noexcept { return cutoff_radius_; }

    core::Real short_range_potential_multiplier(core::Real r) const;
    core::Real short_range_force_multiplier(core::Real r) const;

    // Plummer-total minus Gaussian-long potential correction, shifted to zero at
    // r_cut. Multiply by -G*m_i*m_j/a for peculiar pair potential energy.
    core::Real short_range_potential_correction(
        core::Real r,
        core::Real eps) const;

    // Uses the exact dyadic squared-radius gap near r_cut when rounded hypot would
    // otherwise misclassify an in-cutoff binary64 displacement.
    core::Real short_range_potential_correction(
        const core::Vec3& displacement,
        core::Real eps) const;

    // Newtonian Gaussian long-range vector coefficient; the r=0 limit is
    // 1/(6 sqrt(pi) r_s^3).
    core::Real long_range_newtonian_force_factor(core::Real r) const;

    // Computes G*m*dx*(f_Plummer-f_long) through a dimensionless ratio so the
    // residual is retained when the dimensional factors nearly cancel or exceed
    // the binary64 range individually.
    core::Vec3 scale_safe_short_range_acceleration(
        const core::Vec3& displacement,
        core::Real source_mass,
        core::Real eps) const;

    core::Real long_range_filter(core::Real k_squared) const;

    // Compare physical radii directly; squaring a finite cutoff can overflow.
    bool inside_cutoff_radius(core::Real radius) const;

private:
    // Signed complementary force fraction 1-f_long/f_Plummer, evaluated without
    // subtracting dimensional coefficients.
    core::Real short_range_force_fraction_of_plummer(
        core::Real r,
        core::Real eps) const;

    // Stable local quadrature for the near-cutoff potential difference.
    core::Real near_cutoff_shifted_potential(
        core::Real r,
        core::Real eps) const;

    core::Real split_scale_;
    core::Real cutoff_radius_;
};

} // namespace gravity
} // namespace cosmo_nbody
