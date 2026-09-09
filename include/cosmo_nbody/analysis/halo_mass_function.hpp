#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

struct HaloMassFunctionOptions {
    // Caller-declared fixed logarithmic estimator; no sample-dependent range/bins.
    core::Real min_mass{0.0};
    core::Real max_mass{0.0};
    int num_bins{0};

    // Exact non-empty mass-definition label, e.g. FoF, M200m, or M200c.
    std::string mass_definition;
};

struct HaloMassFunctionBin {
    core::Real mass_low{0.0};
    core::Real mass_high{0.0};
    core::Real mass_geometric_mean{0.0};
    std::size_t count{0};

    // dn/dlnM in inverse simulation-volume units.
    core::Real number_density_per_ln_mass{0.0};
    core::Real poisson_error_per_ln_mass{0.0};
};

struct HaloMassFunctionResult {
    std::string mass_definition;
    core::Real volume{0.0};
    core::Real delta_ln_mass{0.0};
    std::size_t input_halo_count{0};
    std::size_t underflow_count{0};
    std::size_t overflow_count{0};
    std::vector<HaloMassFunctionBin> bins;
};

class HaloMassFunction {
public:
    // Fixed [min,max] logarithmic bins; final bin includes max. Uncertainty is
    // sqrt(N)/(V*dlnM) only and does not model cosmic variance.
    static HaloMassFunctionResult compute(
        std::span<const core::Real> halo_masses,
        core::Real box_size,
        const HaloMassFunctionOptions& options);
};

} // namespace analysis
} // namespace cosmo_nbody
