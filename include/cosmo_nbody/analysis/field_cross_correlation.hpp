#pragma once

#include "cosmo_nbody/analysis/periodic_domain.hpp"
#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/explicit_boolean.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

struct FieldCrossBin {
    std::size_t shell_index{0};
    core::Real k_low{0.0};
    core::Real k_mean{0.0};
    core::Real k_high{0.0};
    core::Real p_a{0.0};
    core::Real p_b{0.0};
    core::Real p_cross{0.0};

    // Direct shell mean of |delta_B-delta_A|^2, not reconstructed from powers.
    core::Real p_delta{0.0};

    // Relative metrics require non-zero reference power; use the explicit flag.
    core::Real delta_p_fraction{0.0};
    core::Real transfer_amplitude_ratio{0.0};
    core::Real e_delta{0.0};
    bool relative_metrics_defined{false};

    // Raw estimator is not clamped to [-1,1]; undefined zero-denominator cases
    // are distinguished from represented r=0 and any Cauchy excess is published.
    core::Real r{0.0};
    bool correlation_defined{false};
    core::Real absolute_cauchy_excess{0.0};
    std::size_t mode_count{0};
};

struct FieldComparisonOptions {
    int num_bins{0};

    // Interlacing is an explicit estimator coordinate; false is valid.
    core::ExplicitBoolean interlaced{};

    // Absolute h/Mpc support, independent of comparison-mesh round trips.
    core::Real max_k_h_Mpc{0.0};
};

struct FieldComparisonSummary {
    std::vector<FieldCrossBin> bins;

    // Residual/reference RMS is defined only for positive reference RMS.
    core::Real normalized_residual{0.0};
    bool normalized_residual_defined{false};
    core::Real residual_rms{0.0};
    core::Real reference_rms{0.0};
    core::Real k_fundamental{0.0};
    core::Real k_nyquist{0.0};
    core::Real evaluated_k_max{0.0};
};

class FieldCrossCorrelation {
public:
    FieldCrossCorrelation(
        PeriodicDomain domain,
        int mesh_size,
        config::MemoryPolicyParams memory_policy = {});

    FieldComparisonSummary compare(
        const core::ParticleStore& reference,
        const core::ParticleStore& candidate,
        const FieldComparisonOptions& options) const;

private:
    PeriodicDomain domain_;
    int mesh_size_;
    config::MemoryPolicyParams memory_policy_;
};

} // namespace analysis
} // namespace cosmo_nbody
