#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace cosmo_nbody {
namespace cosmology {

// Maximum component-wise relative discrepancies at common nodes of two nested
// growth grids; these are observations, not error bounds or thresholds.
struct GrowthNestedGridDiscrepancy {
    core::Real D1{0.0};
    core::Real f1{0.0};
    core::Real D2{0.0};
    core::Real f2{0.0};
};

struct GrowthIntegrationDiagnostics {
    bool used_eds_shortcut{false};
    core::Real start_scale_factor{1.0};
    std::size_t coarse_step_count{0};
    std::size_t nominal_step_count{0};
    std::size_t refined_step_count{0};
    GrowthNestedGridDiscrepancy coarse_to_nominal;
    GrowthNestedGridDiscrepancy nominal_to_refined;
};

class CosmologyModel {
public:
    // Growth integration is internal: its boundary and ln(a) resolution derive
    // from binary64 precision and RK4 order, not caller-supplied coordinates.
    explicit CosmologyModel(const config::CosmologyParams& params);

    // Expansion history is supported for 0 < a <= 1; configuration rejects
    // future-time evolution because the growth table ends at a=1.
    core::Real H(core::Real a) const;
    core::Real E(core::Real a) const; // E(a) = H(a) / H0

    core::Real Omega_m(core::Real a) const;
    core::Real Omega_lambda(core::Real a) const;

    // Linear growth D1 and logarithmic derivative f1=d ln D1/d ln a.
    core::Real D1(core::Real a) const;
    core::Real f1(core::Real a) const;

    // Second-order Lagrangian growth D2 and
    // f2=d ln |D2|/d ln a. D2 is negative for the growing EdS convention.
    core::Real D2(core::Real a) const;
    core::Real f2(core::Real a) const;

    const GrowthIntegrationDiagnostics& growth_integration_diagnostics()
        const noexcept {
        return growth_diagnostics_;
    }

private:
    config::CosmologyParams params_;

    struct GrowthPoint {
        core::Real a;
        core::Real D1;
        core::Real f1;
        core::Real D2;
        core::Real f2;
    };
    std::vector<GrowthPoint> growth_table_;
    GrowthIntegrationDiagnostics growth_diagnostics_;

    void precompute_growth_table();

    // Coupled growth ODE in x=ln(a):
    // y = [D1, dD1/dx, D2, dD2/dx].
    // D2 uses the standard fastest-growing 2LPT convention, whose EdS limit is
    // D2=-(3/7)D1^2.
    void growth_ode_rhs(
        core::Real lna,
        const core::Real y[4],
        core::Real dydx[4]) const;

    core::Real interpolate_component(core::Real a, int component) const;
};

} // namespace cosmology
} // namespace cosmo_nbody
