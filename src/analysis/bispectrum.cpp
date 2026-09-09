#include "cosmo_nbody/analysis/bispectrum.hpp"

#include "cosmo_nbody/analysis/spectral_numeric.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
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

struct CICModeCorrection {
    core::Real window{0.0};
    core::Real aliased_shot_noise{0.0};
};

CICModeCorrection cic_mode_correction(
    core::Real kx,
    core::Real ky,
    core::Real kz,
    core::Real dx,
    core::Real shot_noise) {
    const core::Real ax = 0.5 * kx * dx;
    const core::Real ay = 0.5 * ky * dx;
    const core::Real az = 0.5 * kz * dx;
    const auto sinc = [](core::Real value) {
        return value == 0.0 ? core::Real{1.0} : std::sin(value) / value;
    };
    const core::Real sx = sinc(ax);
    const core::Real sy = sinc(ay);
    const core::Real sz = sinc(az);
    return {
        (sx * sx) * (sy * sy) * (sz * sz),
        shot_noise
            * (1.0 - 2.0 / 3.0 * std::sin(ax) * std::sin(ax))
            * (1.0 - 2.0 / 3.0 * std::sin(ay) * std::sin(ay))
            * (1.0 - 2.0 / 3.0 * std::sin(az) * std::sin(az))};
}

long double checked_mesh_cells(std::size_t n) {
    const long double q = static_cast<long double>(n)
        * static_cast<long double>(n)
        * static_cast<long double>(n);
    if (!std::isfinite(q) || q <= 0.0L) {
        throw std::overflow_error("Bispectrum mesh cell count is invalid");
    }
    return q;
}

std::size_t checked_add_modes(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error("Bispectrum shell mode count overflow");
    }
    return lhs + rhs;
}

bool any_invalid(const std::vector<int>& flags) {
    return std::any_of(
        flags.begin(), flags.end(), [](int value) { return value != 0; });
}

} // namespace

BispectrumEstimator::BispectrumEstimator(
    PeriodicDomain domain,
    int mesh_size,
    config::MemoryPolicyParams memory_policy)
    : domain_(std::move(domain)),
      mesh_size_(mesh_size),
      memory_policy_(std::move(memory_policy)) {
    if (mesh_size_ < 4) {
        throw std::invalid_argument(
            "BispectrumEstimator mesh_size must be >= 4");
    }
}

std::vector<BispectrumBin> BispectrumEstimator::compute_equilateral(
    const core::ParticleStore& particles,
    const BispectrumOptions& options) const {
    if (options.num_k_bins < 1) {
        throw std::invalid_argument(
            "Bispectrum num_k_bins must be positive");
    }
    // Cubing an unpadded inverse FFT enforces closure modulo the mesh size.
    // Requiring 3*k_max < 2*k_Nyquist excludes nonzero wrapped sums on every
    // axis, so the contraction counts actual zero-sum Fourier triads.
    if (!std::isfinite(options.max_k_fraction_nyquist)
        || options.max_k_fraction_nyquist <= 0.0
        || options.max_k_fraction_nyquist >= 2.0 / 3.0) {
        throw std::invalid_argument(
            "Bispectrum max_k_fraction_nyquist must be in (0,2/3) to exclude wrapped FFT triads");
    }
    if (!particles.get_uniform_mass().has_value()) {
        throw std::invalid_argument(
            "Bispectrum Poisson correction requires uniform particle mass");
    }

    const FourierDensityField field = FourierDensityBuilder::build(
        domain_, particles, mesh_size_, false, memory_policy_);
    const core::Real L = domain_.box_size();
    const mesh::MeshGeometry geometry(
        L,
        static_cast<std::size_t>(mesh_size_));
    mesh::FFTBackend fft(geometry);

    const core::Real k_fund = 2.0 * std::numbers::pi / L;
    const core::Real k_nyq =
        std::numbers::pi * static_cast<core::Real>(mesh_size_) / L;
    const core::Real k_max = options.max_k_fraction_nyquist * k_nyq;
    if (!std::isfinite(k_fund) || k_fund <= 0.0
        || !std::isfinite(k_max) || k_max <= k_fund) {
        throw std::invalid_argument(
            "Bispectrum configured k range is too narrow");
    }

    const core::Real dlogk = detail::stable_positive_log_ratio(
        k_max, k_fund)
        / static_cast<core::Real>(options.num_k_bins);
    if (!std::isfinite(dlogk) || dlogk <= 0.0) {
        throw std::overflow_error(
            "Bispectrum logarithmic bin width is invalid");
    }

    const std::size_t N = geometry.grid_size();
    const std::size_t nzc = N / 2 + 1;
    // Floating conversion of a strictly sub-2/3 support can round onto the
    // forbidden boundary. Enforce 3*abs(mode) < N in integer lattice space as
    // well; division avoids overflow and makes every triad's axis sum < N.
    const std::size_t maximum_axis_mode = (N - 1U) / 3U;
    const auto axis_mode_is_supported = [&](std::size_t index) {
        const std::size_t magnitude = index <= N / 2U ? index : N - index;
        return magnitude <= maximum_axis_mode;
    };
    const long double q = checked_mesh_cells(N);
    const long double q2 = q * q;
    if (!std::isfinite(q2)) {
        throw std::overflow_error(
            "Bispectrum mesh-cell square is invalid");
    }
    const core::Real dx = geometry.cell_size();
    const core::Real shot_noise = detail::scaled_box_volume_times(
        L,
        field.mass_square_fraction,
        "Bispectrum shot noise");
    if (!std::isfinite(shot_noise) || shot_noise <= 0.0) {
        throw std::overflow_error(
            "Bispectrum shot-noise term is invalid");
    }

    std::vector<BispectrumBin> bins(
        static_cast<std::size_t>(options.num_k_bins));

    // Each filtered inverse transform is consumed before the next one is built.
    // The indicator field must survive for the k- and P(k)-weighted contractions;
    // every other transform reuses one complex and one real mesh. This exchanges
    // repeated O(N^3) shell construction for a bounded peak without changing any
    // Fourier coefficient, inverse transform, or ordered slab reduction.
    mesh::ComplexField shell_modes(geometry.complex_size());
    mesh::RealField shell_real(geometry.real_size());
    mesh::RealField indicator_real(geometry.real_size());

    for (int bin_index = 0;
         bin_index < options.num_k_bins;
         ++bin_index) {
        const auto logarithmic_edge = [&](int edge_index) {
            if (edge_index == 0) return k_fund;
            if (edge_index == options.num_k_bins) return k_max;
            const core::Real exponent =
                static_cast<core::Real>(edge_index) * dlogk;
            const core::Real relative_increment = std::expm1(exponent);
            return std::fma(k_fund, relative_increment, k_fund);
        };
        const core::Real k_lo = logarithmic_edge(bin_index);
        const core::Real k_hi = logarithmic_edge(bin_index + 1);
        if (!std::isfinite(k_lo) || !std::isfinite(k_hi)
            || k_lo <= 0.0 || k_hi <= k_lo) {
            throw std::overflow_error(
                "Bispectrum shell edges are invalid");
        }
        const auto in_shell = [&](core::Real k) {
            return k >= k_lo
                && (bin_index + 1 == options.num_k_bins
                    ? k <= k_hi
                    : k < k_hi);
        };

        const auto fill_shell = [&](auto&& make_value, const char* error_message) {
            std::fill(
                shell_modes.begin(), shell_modes.end(),
                std::complex<core::Real>{0.0, 0.0});
            std::vector<std::size_t> stored_modes_by_slab(N, 0);
            std::vector<int> invalid_by_slab(N, 0);
            std::vector<std::exception_ptr> exceptions_by_slab(N);
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (std::size_t ix = 0; ix < N; ++ix) {
                try {
                    if (!axis_mode_is_supported(ix)) continue;
                    std::size_t local_modes = 0;
                    int local_invalid = 0;
                    const core::Real kx = geometry.k_component(ix);
                    for (std::size_t iy = 0; iy < N; ++iy) {
                        if (!axis_mode_is_supported(iy)) continue;
                        const core::Real ky = geometry.k_component(iy);
                        for (std::size_t iz = 0; iz < nzc; ++iz) {
                            if (!axis_mode_is_supported(iz)) continue;
                            if (ix == 0 && iy == 0 && iz == 0) continue;
                            const core::Real kz = geometry.k_component(iz);
                            const core::Real k = detail::spectral_norm3(kx, ky, kz);
                            const auto correction = cic_mode_correction(
                                kx, ky, kz, dx, shot_noise);
                            if (!in_shell(k) || !(correction.window > 0.0)) continue;
                            const std::size_t index =
                                geometry.complex_index(ix, iy, iz);
                            const std::complex<core::Real> value = make_value(
                                index, k, correction);
                            if (!std::isfinite(value.real())
                                || !std::isfinite(value.imag())) {
                                local_invalid = 1;
                                continue;
                            }
                            shell_modes[index] = value;
                            ++local_modes;
                        }
                    }
                    stored_modes_by_slab[ix] = local_modes;
                    invalid_by_slab[ix] = local_invalid;
                } catch (...) {
                    exceptions_by_slab[ix] = std::current_exception();
                }
            }
            for (const auto& exception : exceptions_by_slab) {
                if (exception) std::rethrow_exception(exception);
            }
            if (any_invalid(invalid_by_slab)) {
                throw std::overflow_error(error_message);
            }
            std::size_t stored_modes = 0;
            for (const std::size_t local_modes : stored_modes_by_slab) {
                stored_modes = checked_add_modes(stored_modes, local_modes);
            }
            return stored_modes;
        };

        const std::size_t stored_modes_in_shell = fill_shell(
            [&](std::size_t index, core::Real, const CICModeCorrection&) {
                return field.modes[index];
            },
            "Bispectrum raw shell mode is non-finite");
        if (stored_modes_in_shell == 0) continue;
        fft.inverse(shell_modes, shell_real);
        std::vector<long double> delta3_by_slab(N, 0.0L);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            long double sum = 0.0L;
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const long double value = shell_real[
                        geometry.real_index(ix, iy, iz)];
                    sum += value * value * value;
                }
            }
            delta3_by_slab[ix] = sum;
        }
        long double sum_delta3 = 0.0L;
        for (const long double partial : delta3_by_slab) {
            sum_delta3 += partial;
        }

        const std::size_t deconvolved_modes = fill_shell(
            [&](std::size_t index, core::Real, const CICModeCorrection& correction) {
                return field.modes[index] / correction.window;
            },
            "Bispectrum CIC-deconvolved mode is non-finite");
        if (deconvolved_modes != stored_modes_in_shell) {
            throw std::logic_error(
                "Bispectrum deconvolved shell changed mode cardinality");
        }
        fft.inverse(shell_modes, shell_real);
        std::vector<long double> delta3_deconv_by_slab(N, 0.0L);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            long double sum = 0.0L;
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const long double value = shell_real[
                        geometry.real_index(ix, iy, iz)];
                    sum += value * value * value;
                }
            }
            delta3_deconv_by_slab[ix] = sum;
        }
        long double sum_delta3_deconv = 0.0L;
        for (const long double partial : delta3_deconv_by_slab) {
            sum_delta3_deconv += partial;
        }

        const std::size_t indicator_modes = fill_shell(
            [](std::size_t, core::Real, const CICModeCorrection&) {
                return std::complex<core::Real>{1.0, 0.0};
            },
            "Bispectrum indicator shell mode is non-finite");
        if (indicator_modes != stored_modes_in_shell) {
            throw std::logic_error(
                "Bispectrum indicator shell changed mode cardinality");
        }
        fft.inverse(shell_modes, indicator_real);
        std::vector<long double> indicator3_by_slab(N, 0.0L);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            long double sum = 0.0L;
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const long double value = indicator_real[
                        geometry.real_index(ix, iy, iz)];
                    sum += value * value * value;
                }
            }
            indicator3_by_slab[ix] = sum;
        }
        long double sum_indicator3 = 0.0L;
        for (const long double partial : indicator3_by_slab) {
            sum_indicator3 += partial;
        }

        if (!std::isfinite(sum_delta3)
            || !std::isfinite(sum_delta3_deconv)
            || !std::isfinite(sum_indicator3)) {
            throw std::runtime_error(
                "Bispectrum shell reduction is non-finite");
        }
        const long double triad_estimate_wide = q2 * sum_indicator3;
        if (!std::isfinite(triad_estimate_wide)) {
            throw std::overflow_error(
                "Bispectrum FFT indicator contraction is not representable");
        }
        const long double nearest_triad_count = std::round(triad_estimate_wide);
        if (nearest_triad_count <= 0.0L) {
            // Exact arithmetic would produce a non-negative integer count. A
            // nearest integer of zero means this represented shell provides no
            // usable closed-triad normalization. No empirical tolerance is used.
            continue;
        }
        if (sum_indicator3 <= 0.0L) {
            throw std::runtime_error(
                "Bispectrum closed-triad normalization is not positive");
        }
        const long double triad_integer_residual_wide =
            std::abs(triad_estimate_wide - nearest_triad_count);

        const std::size_t weighted_modes = fill_shell(
            [](std::size_t, core::Real k, const CICModeCorrection&) {
                return std::complex<core::Real>{k, 0.0};
            },
            "Bispectrum weighted shell mode is non-finite");
        if (weighted_modes != stored_modes_in_shell) {
            throw std::logic_error(
                "Bispectrum weighted shell changed mode cardinality");
        }
        fft.inverse(shell_modes, shell_real);
        std::vector<long double> k_indicator2_by_slab(N, 0.0L);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            long double sum = 0.0L;
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const std::size_t index = geometry.real_index(ix, iy, iz);
                    const long double mask = indicator_real[index];
                    sum += static_cast<long double>(shell_real[index])
                        * mask * mask;
                }
            }
            k_indicator2_by_slab[ix] = sum;
        }
        long double sum_k_indicator2 = 0.0L;
        for (const long double partial : k_indicator2_by_slab) {
            sum_k_indicator2 += partial;
        }

        const long double mean_k = sum_k_indicator2 / sum_indicator3;
        const long double b_value = detail::scaled_box_volume_squared_times_wide(
            L,
            sum_delta3 / sum_indicator3,
            "Bispectrum raw shell statistic");
        const long double b_deconvolved =
            detail::scaled_box_volume_squared_times_wide(
            L,
            sum_delta3_deconv / sum_indicator3,
            "Bispectrum deconvolved shell statistic");
        if (!std::isfinite(mean_k) || mean_k <= 0.0L
            || !std::isfinite(b_value) || !std::isfinite(b_deconvolved)) {
            throw std::runtime_error(
                "Bispectrum shell statistic is non-finite");
        }

        const std::size_t power_modes = fill_shell(
            [&](std::size_t index, core::Real, const CICModeCorrection& correction) {
                const long double w2 =
                    static_cast<long double>(correction.window)
                    * static_cast<long double>(correction.window);
                const core::Real p_mode =
                    detail::scaled_box_volume_times_complex_norm_squared_minus(
                        L,
                        field.modes[index],
                        correction.aliased_shot_noise,
                        "Bispectrum per-mode shot-noise-corrected power");
                const long double corrected =
                    static_cast<long double>(p_mode) / w2;
                if (!std::isfinite(corrected)
                    || std::abs(corrected) > static_cast<long double>(
                        std::numeric_limits<core::Real>::max())) {
                    return std::complex<core::Real>{
                        std::numeric_limits<core::Real>::quiet_NaN(), 0.0};
                }
                return std::complex<core::Real>{
                    static_cast<core::Real>(corrected), 0.0};
            },
            "Bispectrum per-mode power correction is not representable");
        if (power_modes != stored_modes_in_shell) {
            throw std::logic_error(
                "Bispectrum corrected-power shell changed mode cardinality");
        }
        fft.inverse(shell_modes, shell_real);
        std::vector<long double> p_indicator2_by_slab(N, 0.0L);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            long double sum = 0.0L;
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const std::size_t index = geometry.real_index(ix, iy, iz);
                    const long double mask = indicator_real[index];
                    sum += static_cast<long double>(shell_real[index])
                        * mask * mask;
                }
            }
            p_indicator2_by_slab[ix] = sum;
        }
        long double sum_p_indicator2 = 0.0L;
        for (const long double partial : p_indicator2_by_slab) {
            sum_p_indicator2 += partial;
        }
        const long double triad_weighted_power =
            sum_p_indicator2 / sum_indicator3;
        if (!std::isfinite(triad_weighted_power)) {
            throw std::runtime_error(
                "Bispectrum triad-weighted power correction is non-finite");
        }

        const long double shot_noise_ld =
            static_cast<long double>(shot_noise);
        const long double b_shot =
            3.0L * shot_noise_ld * triad_weighted_power
            + shot_noise_ld * shot_noise_ld;
        const long double b_corrected = b_deconvolved - b_shot;
        if (!std::isfinite(b_corrected)) {
            throw std::runtime_error(
                "Bispectrum corrected shell statistic is non-finite");
        }

        auto& bin = bins[static_cast<std::size_t>(bin_index)];
        bin.k_mean = detail::checked_real_result(
            mean_k, "Bispectrum mean shell wavenumber");
        bin.B_raw = detail::checked_real_result(
            b_value, "Bispectrum raw shell statistic");
        bin.B_corrected = detail::checked_real_result(
            b_corrected, "Bispectrum corrected shell statistic");
        bin.closed_triad_count_estimate = detail::checked_real_result(
            triad_estimate_wide,
            "Bispectrum closed-triad count estimate");
        bin.closed_triad_integer_residual = detail::checked_real_result(
            triad_integer_residual_wide,
            "Bispectrum closed-triad integer residual");
        if (!std::isfinite(bin.k_mean) || !std::isfinite(bin.B_raw)
            || !std::isfinite(bin.B_corrected)
            || !std::isfinite(bin.closed_triad_count_estimate)
            || !std::isfinite(bin.closed_triad_integer_residual)) {
            throw std::overflow_error(
                "Bispectrum output is not representable in core::Real");
        }
    }

    return bins;
}

} // namespace analysis
} // namespace cosmo_nbody
