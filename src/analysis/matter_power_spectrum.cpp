#include "cosmo_nbody/analysis/matter_power_spectrum.hpp"

#include "cosmo_nbody/analysis/cic_spectral_window.hpp"
#include "cosmo_nbody/analysis/spectral_numeric.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace analysis {
namespace {

struct PowerSpectrumAccumulator {
    long double k_sum{0.0L};
    long double p_raw_sum{0.0L};
    long double p_corrected_sum{0.0L};
    long double cic_power_deconvolution_factor_sum{0.0L};
    std::size_t mode_count{0};
};

void checked_add_count(std::size_t& destination, std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - destination) {
        throw std::overflow_error("MatterPowerSpectrum mode count overflow");
    }
    destination += value;
}

} // namespace

MatterPowerSpectrum::MatterPowerSpectrum(
    PeriodicDomain domain,
    int mesh_size,
    config::MemoryPolicyParams memory_policy)
    : domain_(std::move(domain)),
      mesh_size_(mesh_size),
      memory_policy_(std::move(memory_policy)) {
    if (mesh_size_ < 2) {
        throw std::invalid_argument(
            "MatterPowerSpectrum mesh_size must be >= 2");
    }
}

std::vector<PowerSpectrumBin> MatterPowerSpectrum::compute_pk(
    const core::ParticleStore& particles,
    const PowerSpectrumOptions& options) const {
    if (options.num_bins < 1) {
        throw std::invalid_argument(
            "MatterPowerSpectrum num_bins must be positive");
    }

    const bool subtract_shot_noise = options.subtract_shot_noise.value();
    const bool interlaced = options.interlaced.value();
    const core::Real max_fraction = options.max_k_fraction_nyquist;
    if (!std::isfinite(max_fraction)
        || max_fraction <= 0.0 || max_fraction > 1.0) {
        throw std::invalid_argument(
            "MatterPowerSpectrum max_k_fraction_nyquist must be in (0,1]");
    }
    if (interlaced && subtract_shot_noise) {
        throw std::invalid_argument(
            "Interlaced shot-noise subtraction is disabled until the estimator uses an interlacing-matched alias model");
    }

    const FourierDensityField field = FourierDensityBuilder::build(
        domain_, particles, mesh_size_, interlaced, memory_policy_);
    const core::Real L = domain_.box_size();
    const mesh::MeshGeometry geometry(
        L, static_cast<std::size_t>(mesh_size_));

    const core::Real k_nyquist =
        std::numbers::pi * static_cast<core::Real>(mesh_size_) / L;
    const core::Real k_max = max_fraction * k_nyquist;
    const core::Real k_fundamental = 2.0 * std::numbers::pi / L;
    if (!(k_max > k_fundamental)) {
        throw std::invalid_argument(
            "MatterPowerSpectrum requested range contains no nonzero shell");
    }
    const detail::PositiveLogBinGrid bin_grid(
        k_fundamental, k_max, options.num_bins);

    const core::Real shot_noise = detail::scaled_box_volume_times(
        L,
        field.mass_square_fraction,
        "MatterPowerSpectrum shot noise");
    const core::Real dx = L / static_cast<core::Real>(mesh_size_);
    const bool even_mesh = (mesh_size_ % 2) == 0;
    const std::size_t grid_size = geometry.grid_size();
    const std::size_t nz_complex = grid_size / 2 + 1;
    const std::size_t bin_count = bin_grid.bin_count();
    if (grid_size > std::numeric_limits<std::size_t>::max() / bin_count) {
        throw std::overflow_error(
            "MatterPowerSpectrum slab accumulator size overflows size_t");
    }

    std::vector<PowerSpectrumAccumulator> slab_bins(grid_size * bin_count);
    std::vector<std::exception_ptr> slab_exceptions(grid_size);

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t ix = 0; ix < grid_size; ++ix) {
        try {
            PowerSpectrumAccumulator* local = slab_bins.data() + ix * bin_count;
            const core::Real kx = geometry.k_component(ix);
            const core::Real sx_arg = 0.5 * kx * dx;
            const long double cic_x = detail::cic_window_axis_wide(kx, dx);

            for (std::size_t iy = 0; iy < grid_size; ++iy) {
                const core::Real ky = geometry.k_component(iy);
                const core::Real sy_arg = 0.5 * ky * dx;
                const long double cic_y = detail::cic_window_axis_wide(ky, dx);

                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const core::Real kz = geometry.k_component(iz);
                    const core::Real sz_arg = 0.5 * kz * dx;
                    const long double cic_z = detail::cic_window_axis_wide(kz, dx);

                    const core::Real k_mag = detail::spectral_norm3(kx, ky, kz);
                    if (!(k_mag > 0.0) || k_mag > k_max) continue;

                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    const core::Real p_raw =
                        detail::scaled_box_volume_times_complex_norm_squared(
                            L,
                            field.modes[index],
                            "MatterPowerSpectrum raw power");

                    const long double window = cic_x * cic_y * cic_z;
                    if (!std::isfinite(window) || window <= 0.0L
                        || window > 1.0L) {
                        throw std::overflow_error(
                            "MatterPowerSpectrum CIC window escaped its exact range (0,1]");
                    }
                    const long double window_power = window * window;
                    const long double deconvolution_factor_wide =
                        1.0L / window_power;
                    if (!std::isfinite(deconvolution_factor_wide)
                        || deconvolution_factor_wide < 1.0L) {
                        throw std::overflow_error(
                            "MatterPowerSpectrum CIC power deconvolution factor is invalid");
                    }
                    const core::Real deconvolution_factor =
                        detail::checked_real_result(
                            deconvolution_factor_wide,
                            "MatterPowerSpectrum CIC power deconvolution factor");

                    const core::Real sin2_x = std::sin(sx_arg) * std::sin(sx_arg);
                    const core::Real sin2_y = std::sin(sy_arg) * std::sin(sy_arg);
                    const core::Real sin2_z = std::sin(sz_arg) * std::sin(sz_arg);
                    const core::Real aliased_shot_noise = shot_noise
                        * (1.0 - 2.0 / 3.0 * sin2_x)
                        * (1.0 - 2.0 / 3.0 * sin2_y)
                        * (1.0 - 2.0 / 3.0 * sin2_z);

                    core::Real p_corrected = p_raw;
                    if (subtract_shot_noise) {
                        p_corrected =
                            detail::scaled_box_volume_times_complex_norm_squared_minus(
                                L,
                                field.modes[index],
                                aliased_shot_noise,
                                "MatterPowerSpectrum shot-noise-subtracted power");
                    }
                    p_corrected *= deconvolution_factor;
                    if (!std::isfinite(p_corrected)) {
                        throw std::overflow_error(
                            "MatterPowerSpectrum corrected power is not finite");
                    }

                    const std::size_t multiplicity =
                        (iz == 0 || (even_mesh && iz == grid_size / 2)) ? 1U : 2U;
                    const int bin_index = bin_grid.inclusive_bin_index(k_mag);
                    if (bin_index < 0) continue;

                    auto& bin = local[static_cast<std::size_t>(bin_index)];
                    const long double mult = static_cast<long double>(multiplicity);
                    bin.k_sum += static_cast<long double>(k_mag) * mult;
                    bin.p_raw_sum += static_cast<long double>(p_raw) * mult;
                    bin.p_corrected_sum +=
                        static_cast<long double>(p_corrected) * mult;
                    bin.cic_power_deconvolution_factor_sum +=
                        static_cast<long double>(deconvolution_factor) * mult;
                    bin.mode_count += multiplicity;
                }
            }
        } catch (...) {
            slab_exceptions[ix] = std::current_exception();
        }
    }

    for (const auto& exception : slab_exceptions) {
        if (exception) std::rethrow_exception(exception);
    }

    std::vector<PowerSpectrumAccumulator> totals(bin_count);
    for (std::size_t ix = 0; ix < grid_size; ++ix) {
        const PowerSpectrumAccumulator* local =
            slab_bins.data() + ix * bin_count;
        for (std::size_t bin_index = 0; bin_index < bin_count; ++bin_index) {
            totals[bin_index].k_sum += local[bin_index].k_sum;
            totals[bin_index].p_raw_sum += local[bin_index].p_raw_sum;
            totals[bin_index].p_corrected_sum +=
                local[bin_index].p_corrected_sum;
            totals[bin_index].cic_power_deconvolution_factor_sum +=
                local[bin_index].cic_power_deconvolution_factor_sum;
            checked_add_count(
                totals[bin_index].mode_count, local[bin_index].mode_count);
        }
    }

    std::vector<PowerSpectrumBin> bins(bin_count);
    for (std::size_t bin_index = 0; bin_index < bin_count; ++bin_index) {
        bins[bin_index].k_low = bin_grid.edge(bin_index);
        bins[bin_index].k_high = bin_grid.edge(bin_index + 1U);
        bins[bin_index].shot_noise_reference = shot_noise;

        const auto& total = totals[bin_index];
        if (total.mode_count == 0) continue;
        const long double count = static_cast<long double>(total.mode_count);
        bins[bin_index].k_mean = detail::checked_real_result(
            total.k_sum / count, "MatterPowerSpectrum mean wavenumber");
        bins[bin_index].p_raw = detail::checked_real_result(
            total.p_raw_sum / count, "MatterPowerSpectrum raw power");
        bins[bin_index].p_corrected = detail::checked_real_result(
            total.p_corrected_sum / count, "MatterPowerSpectrum corrected power");
        bins[bin_index].mean_cic_power_deconvolution_factor =
            detail::checked_real_result(
                total.cic_power_deconvolution_factor_sum / count,
                "MatterPowerSpectrum mean CIC power deconvolution factor");
        bins[bin_index].mode_count = total.mode_count;
    }
    return bins;
}

} // namespace analysis
} // namespace cosmo_nbody
