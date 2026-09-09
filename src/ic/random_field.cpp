#include "cosmo_nbody/ic/random_field.hpp"

#include "cosmo_nbody/ic/particle_lattice_bandlimit.hpp"
#include "cosmo_nbody/ic/philox_rng.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace ic {

namespace {

#ifdef COSMO_NBODY_HAS_OPENMP
std::size_t square_work_items(std::size_t n) noexcept {
    if (n == 0) return 0;
    if (n > std::numeric_limits<std::size_t>::max() / n) {
        return std::numeric_limits<std::size_t>::max();
    }
    return n * n;
}
#endif

long long mode_number_from_index(int index, int n) {
    const long long mode = index;
    const long long half = static_cast<long long>(n / 2);
    return mode <= half ? mode : mode - static_cast<long long>(n);
}

std::int32_t mode_counter_word(long long mode) {
    if (mode < static_cast<long long>(INT32_MIN)
        || mode > static_cast<long long>(INT32_MAX)) {
        throw std::overflow_error(
            "RandomField integer mode does not fit Philox 32-bit counter word");
    }
    return static_cast<std::int32_t>(mode);
}

template <bool ConstructOutput>
void store_mode(
    std::complex<core::Real>* data,
    std::size_t index,
    std::complex<core::Real> value) noexcept {
    if constexpr (ConstructOutput) {
        std::construct_at(data + index, value);
    } else {
        data[index] = value;
    }
}

template <bool ConstructOutput>
void generate_impl(
    const config::SimulationParameters& config,
    const LinearPowerSpectrum& pk,
    std::complex<core::Real>* data,
    std::size_t output_size,
    std::size_t mesh_per_dimension) {
    static_assert(
        std::is_nothrow_copy_constructible_v<std::complex<core::Real>>,
        "Parallel external random-field construction must not throw");
    if (mesh_per_dimension == 0
        || mesh_per_dimension > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error(
            "RandomField mesh dimension must be in [1, INT_MAX]");
    }
    const int N = static_cast<int>(mesh_per_dimension);
    const int nx = N;
    const int ny = N;
    const int nz = N / 2 + 1;
    const bool has_nyquist_plane = (N % 2) == 0;
    const core::Real L = config.get_box().L;

    const mesh::MeshGeometry geometry(L, mesh_per_dimension);
    if (output_size != geometry.complex_size()) {
        throw std::invalid_argument(
            "RandomField output size does not match requested r2c IC mesh");
    }
    if (output_size > 0 && data == nullptr) {
        throw std::invalid_argument(
            "RandomField output storage is null for a non-empty IC mesh");
    }

    const std::uint64_t seed = config.get_ic().seed;
    const bool fixed_amplitude =
        config.get_ic().amplitude_mode == "fixed";
    const bool reverse_pair =
        config.get_ic().phase_pairing == "pair_b";
    const std::size_t max_mode =
        static_cast<std::size_t>(config.ic_effective_max_mode_per_axis());
    const core::Real k_factor = 2.0 * std::numbers::pi / L;
    const core::Real n_real = static_cast<core::Real>(N);

    const auto max_mode_signed = static_cast<std::int64_t>(max_mode);
    const core::Real represented_max_k = represented_mode_wavenumber(
        max_mode_signed, max_mode_signed, max_mode_signed, k_factor);
    if (!std::isfinite(represented_max_k) || represented_max_k <= 0.0) {
        throw std::overflow_error(
            "RandomField represented Fourier support maximum is not finite and positive");
    }
    (void)pk.evaluate(represented_max_k);

    // Keep N^6/L^3 in scaled form because the normalization may exceed Real
    // even when multiplication by a small valid P(k) yields finite variance.
    const std::array<core::Real, 6> variance_numerators{
        n_real, n_real, n_real, n_real, n_real, n_real};
    const std::array<core::Real, 3> variance_denominators{L, L, L};
    const math::ScaledPositiveProduct variance_normalization =
        math::ScaledPositiveProduct::from_quotient(
            variance_numerators,
            variance_denominators,
            "random-field invariant variance normalization");

    int invalid_power = 0;
    int invalid_variance = 0;
    std::vector<std::exception_ptr> slab_exceptions(
        static_cast<std::size_t>(nx));
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for \
        reduction(|:invalid_power,invalid_variance) schedule(static) \
        if(runtime::should_use_host_parallel_team( \
            square_work_items(mesh_per_dimension)))
#endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const long long kx = mode_number_from_index(i, N);
            const long long ky = mode_number_from_index(j, N);
            for (int k = 0; k < nz; ++k) {
                const long long kz = k;
                const core::Real k_mag = represented_mode_wavenumber(
                    static_cast<std::int64_t>(kx),
                    static_cast<std::int64_t>(ky),
                    static_cast<std::int64_t>(kz),
                    k_factor);
                const std::size_t index = geometry.complex_index(
                    static_cast<std::size_t>(i),
                    static_cast<std::size_t>(j),
                    static_cast<std::size_t>(k));

                if (kx == 0 && ky == 0 && kz == 0) {
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }

                if (!mode_within_axis_limit(kx, max_mode)
                    || !mode_within_axis_limit(ky, max_mode)
                    || !mode_within_axis_limit(kz, max_mode)) {
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }

                bool conjugate = false;
                int canonical_i = i;
                int canonical_j = j;
                const bool stored_conjugate_plane =
                    k == 0 || (has_nyquist_plane && k == N / 2);
                bool is_self_conjugate = false;

                if (stored_conjugate_plane) {
                    const int conj_i = (N - i) % N;
                    const int conj_j = (N - j) % N;
                    is_self_conjugate =
                        conj_i == i && conj_j == j;
                    if (!is_self_conjugate
                        && (conj_i < i
                            || (conj_i == i && conj_j < j))) {
                        canonical_i = conj_i;
                        canonical_j = conj_j;
                        conjugate = true;
                    }
                }

                std::int32_t counter_x = 0;
                std::int32_t counter_y = 0;
                std::int32_t counter_z = 0;
                core::Real p_k = 0.0;
                try {
                    const long long cx =
                        mode_number_from_index(canonical_i, N);
                    const long long cy =
                        mode_number_from_index(canonical_j, N);
                    counter_x = mode_counter_word(cx);
                    counter_y = mode_counter_word(cy);
                    counter_z = mode_counter_word(kz);
                    p_k = pk.evaluate(k_mag);
                } catch (...) {
                    if (!slab_exceptions[static_cast<std::size_t>(i)]) {
                        slab_exceptions[static_cast<std::size_t>(i)] =
                            std::current_exception();
                    }
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }

                if (!std::isfinite(p_k) || p_k <= 0.0) {
                    invalid_power = 1;
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }

                // Preserve the ordinary binary64 path bit-for-bit. Only when the
                // variance itself cannot be materialized, or variance/2 rounds
                // to zero, take the square root in mantissa/exponent form so a
                // representable Fourier amplitude is not lost with its square.
                std::optional<core::Real> variance;
                try {
                    variance = variance_normalization.multiplied_by(
                        p_k,
                        "random-field mode variance");
                } catch (const std::underflow_error&) {
                    variance.reset();
                } catch (const std::overflow_error&) {
                    variance.reset();
                } catch (const std::exception&) {
                    invalid_variance = 1;
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }

                const auto standard_deviation = [&](core::Real divisor,
                                                     const char* role) {
                    if (variance.has_value()) {
                        if (divisor == core::Real{1.0}) {
                            return std::sqrt(*variance);
                        }
                        const core::Real reduced = *variance / divisor;
                        if (reduced > 0.0) return std::sqrt(reduced);
                    }
                    return variance_normalization
                        .square_root_multiplied_by_quotient(
                            p_k, divisor, role);
                };

                std::complex<core::Real> generated{0.0, 0.0};
                try {
                    if (fixed_amplitude) {
                        const core::Real amplitude = standard_deviation(
                            core::Real{1.0},
                            "random-field fixed mode amplitude");
                        const auto [phase_uniform, unused] =
                            rng::uniform_pair_open01(
                                seed,
                                counter_x,
                                counter_y,
                                counter_z,
                                rng::STREAM_FIXED_PHASE);
                        (void)unused;
                        if (is_self_conjugate) {
                            const core::Real sign = phase_uniform < 0.5
                                ? core::Real{-1.0} : core::Real{1.0};
                            generated = {sign * amplitude, 0.0};
                        } else {
                            const core::Real phase =
                                2.0 * std::numbers::pi * phase_uniform;
                            core::Real imag = amplitude * std::sin(phase);
                            if (conjugate) imag = -imag;
                            generated = {
                                amplitude * std::cos(phase), imag};
                        }
                    } else {
                        const auto [n1, n2] = rng::normal_pair(
                            seed,
                            counter_x,
                            counter_y,
                            counter_z,
                            rng::STREAM_GAUSSIAN_FOURIER);
                        if (is_self_conjugate) {
                            const core::Real amplitude = standard_deviation(
                                core::Real{1.0},
                                "random-field Gaussian self-conjugate amplitude");
                            generated = {amplitude * n1, 0.0};
                        } else {
                            const core::Real stddev = standard_deviation(
                                core::Real{2.0},
                                "random-field Gaussian component standard deviation");
                            core::Real imag = stddev * n2;
                            if (conjugate) imag = -imag;
                            generated = {stddev * n1, imag};
                        }
                    }
                } catch (const std::underflow_error&) {
                    invalid_variance = 1;
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                } catch (const std::overflow_error&) {
                    invalid_variance = 1;
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                } catch (const std::exception&) {
                    if (!slab_exceptions[static_cast<std::size_t>(i)]) {
                        slab_exceptions[static_cast<std::size_t>(i)] =
                            std::current_exception();
                    }
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }

                if (!std::isfinite(generated.real())
                    || !std::isfinite(generated.imag())) {
                    invalid_variance = 1;
                    store_mode<ConstructOutput>(data, index, {0.0, 0.0});
                    continue;
                }
                store_mode<ConstructOutput>(data, index, generated);
                if (reverse_pair) {
                    data[index] = -data[index];
                }
            }
        }
    }

    for (const auto& exception : slab_exceptions) {
        if (exception) std::rethrow_exception(exception);
    }
    if (invalid_power) {
        throw std::runtime_error(
            "RandomField power spectrum returned invalid P(k)");
    }
    if (invalid_variance) {
        throw std::overflow_error(
            "RandomField mode amplitude is not representable as finite core::Real");
    }
}

} // namespace

void RandomField::generate(
    const config::SimulationParameters& config,
    const LinearPowerSpectrum& pk,
    mesh::ComplexField& out_field,
    std::size_t mesh_per_dimension) {
    generate_impl<false>(
        config,
        pk,
        out_field.data(),
        out_field.size(),
        mesh_per_dimension);
}

void RandomField::generate_into_uninitialized_storage(
    const config::SimulationParameters& config,
    const LinearPowerSpectrum& pk,
    std::complex<core::Real>* external_data,
    std::size_t external_size,
    std::size_t mesh_per_dimension) {
    generate_impl<true>(
        config,
        pk,
        external_data,
        external_size,
        mesh_per_dimension);
}

} // namespace ic
} // namespace cosmo_nbody
