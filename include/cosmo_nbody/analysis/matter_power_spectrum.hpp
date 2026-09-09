#pragma once

#include "cosmo_nbody/analysis/fourier_density_field.hpp"
#include "cosmo_nbody/analysis/periodic_domain.hpp"
#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/explicit_boolean.hpp"
#include "cosmo_nbody/core/particle_store.hpp"

#include <cstddef>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

struct PowerSpectrumBin {
    core::Real k_low{0.0};
    core::Real k_mean{0.0};
    core::Real k_high{0.0};
    core::Real p_raw{0.0};
    // Power after requested shot-noise subtraction and CIC-window handling;
    // this is estimator processing, not an accuracy correction.
    core::Real p_corrected{0.0};
    core::Real mean_cic_power_deconvolution_factor{0.0};
    core::Real shot_noise_reference{0.0};
    std::size_t mode_count{0};
};

struct PowerSpectrumOptions {
    int num_bins{0};

    // Scientific estimator coordinates; false is valid, so omission is distinct.
    core::ExplicitBoolean subtract_shot_noise{};
    core::ExplicitBoolean interlaced{};

    // Measurement support, not a trust criterion; caller must choose (0,1].
    core::Real max_k_fraction_nyquist{0.0};
};

class MatterPowerSpectrum {
public:
    MatterPowerSpectrum(
        PeriodicDomain domain,
        int mesh_size,
        config::MemoryPolicyParams memory_policy = {});

    std::vector<PowerSpectrumBin> compute_pk(
        const core::ParticleStore& particles,
        const PowerSpectrumOptions& options) const;

private:
    PeriodicDomain domain_;
    int mesh_size_;
    config::MemoryPolicyParams memory_policy_;
};

} // namespace analysis
} // namespace cosmo_nbody
