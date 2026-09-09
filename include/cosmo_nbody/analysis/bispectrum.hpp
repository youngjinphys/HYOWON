// Equilateral periodic-box bispectrum using logarithmic k shells and inverse-FFT
// closed-triad contractions. B_raw is the CIC-grid diagnostic. For uniform mass,
// B_corrected deconvolves primary CIC windows and removes the uniform-Poisson term
// using ordered-triad participation and the native P(k) correction. Signal aliases
// remain, so mesh/k convergence is still required.
#pragma once

#include "cosmo_nbody/analysis/fourier_density_field.hpp"
#include "cosmo_nbody/analysis/periodic_domain.hpp"
#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <vector>

namespace cosmo_nbody {
namespace analysis {

struct BispectrumBin {
    core::Real k_mean{0.0};
    core::Real B_raw{0.0};
    core::Real B_corrected{0.0};

    // Store the FFTW floating estimate of the exact integer ordered-triad count
    // and its nearest-integer residual.
    core::Real closed_triad_count_estimate{0.0};
    core::Real closed_triad_integer_residual{0.0};
};

struct BispectrumOptions {
    int num_k_bins{0};
    core::Real max_k_fraction_nyquist{0.0};
};

class BispectrumEstimator {
public:
    BispectrumEstimator(
        PeriodicDomain domain,
        int mesh_size,
        config::MemoryPolicyParams memory_policy = {});

    std::vector<BispectrumBin> compute_equilateral(
        const core::ParticleStore& particles,
        const BispectrumOptions& options) const;

private:
    PeriodicDomain domain_;
    int mesh_size_;
    config::MemoryPolicyParams memory_policy_;
};

} // namespace analysis
} // namespace cosmo_nbody
