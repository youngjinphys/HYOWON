#include "cosmo_nbody/analysis/filament_statistics.hpp"

#include "cosmo_nbody/analysis/fourier_density_field.hpp"
#include "cosmo_nbody/analysis/symmetric_eigensystem.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/fourier_hessian.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

namespace {

using core::Real;

WebEnvironment environment_from_eigenvalues(
    Real l1, Real l2, Real l3, Real threshold) {
    int n_above = 0;
    if (l1 > threshold) ++n_above;
    if (l2 > threshold) ++n_above;
    if (l3 > threshold) ++n_above;
    switch (n_above) {
        case 0: return WebEnvironment::Void;
        case 1: return WebEnvironment::Sheet;
        case 2: return WebEnvironment::Filament;
        default: return WebEnvironment::Node;
    }
}

std::size_t checked_cell_count(std::size_t n_mesh) {
    if (n_mesh > std::numeric_limits<std::size_t>::max() / n_mesh) {
        throw std::overflow_error("T-web mesh_size^2 overflows size_t");
    }
    const std::size_t n2 = n_mesh * n_mesh;
    if (n_mesh > std::numeric_limits<std::size_t>::max() / n2) {
        throw std::overflow_error("T-web mesh_size^3 overflows size_t");
    }
    return n2 * n_mesh;
}

Real gaussian_smoothing_weight(
    Real k_scale,
    Real normalized_k2,
    Real radius) {
    if (radius == 0.0) return 1.0;

    const long double scaled_k = static_cast<long double>(k_scale)
        * static_cast<long double>(radius);
    if (!std::isfinite(scaled_k)) {
        // The exact Gaussian tends to zero for |k|R -> infinity. Treat an
        // exponent-range overflow in this dimensionless product as that limit,
        // rather than turning a physically negligible high-k mode into an error.
        return 0.0;
    }
    const long double kr2 = scaled_k * scaled_k
        * static_cast<long double>(normalized_k2);
    if (!std::isfinite(kr2)) return 0.0;

    const long double weight = std::exp(-0.5L * kr2);
    if (!std::isfinite(weight) || weight < 0.0L || weight > 1.0L) {
        throw std::runtime_error(
            "T-web Gaussian smoothing kernel is non-finite");
    }
    // Underflow to zero is the correct asymptotic Gaussian limit and is safe.
    return static_cast<Real>(weight);
}

} // namespace

TidalWebField FilamentStatistics::build_tidal_web_field(
    const PeriodicDomain& domain,
    const core::ParticleStore& particles,
    const TidalWebOptions& options) {
    if (options.mesh_size < 4) {
        throw std::invalid_argument("T-web mesh_size must be >= 4");
    }
    if (!std::isfinite(options.gaussian_smoothing_radius)
        || options.gaussian_smoothing_radius < 0.0) {
        throw std::invalid_argument(
            "T-web gaussian_smoothing_radius must be finite and non-negative");
    }
    if (!std::isfinite(options.lambda_threshold)) {
        throw std::invalid_argument("lambda_threshold must be finite");
    }
    const int N = options.mesh_size;
    const std::size_t n_mesh = static_cast<std::size_t>(N);
    const std::size_t ncell = checked_cell_count(n_mesh);
    const Real fft_rescale = static_cast<Real>(ncell);
    if (!std::isfinite(fft_rescale) || fft_rescale <= 0.0) {
        throw std::overflow_error(
            "T-web mesh cell count is not representable");
    }

    const FourierDensityField field = FourierDensityBuilder::build(
        domain, particles, N, false, options.memory_policy);
    const mesh::MeshGeometry geometry(
        domain.box_size(), n_mesh);
    const std::size_t nzc = n_mesh / 2 + 1;

    std::vector<Real> txx(ncell, 0.0);
    std::vector<Real> tyy(ncell, 0.0);
    std::vector<Real> tzz(ncell, 0.0);
    std::vector<Real> txy(ncell, 0.0);
    std::vector<Real> txz(ncell, 0.0);
    std::vector<Real> tyz(ncell, 0.0);

    mesh::FFTBackend fft(geometry);
    mesh::ComplexField modes(geometry.complex_size());
    // FFTBackend::inverse requires the logical N^3 real field, not the padded
    // in-place FFTW storage extent.
    mesh::RealField real(geometry.real_size());

    auto fill_and_ifft = [&](
        int axis_a, int axis_b,
        std::vector<Real>& output) {
        std::fill(
            modes.begin(), modes.end(),
            std::complex<Real>{0.0, 0.0});
        std::vector<std::uint8_t> spectral_failure(n_mesh, std::uint8_t{0});
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < n_mesh; ++ix) {
            const Real kx = geometry.k_component(ix);
            for (std::size_t iy = 0; iy < n_mesh; ++iy) {
                const Real ky = geometry.k_component(iy);
                for (std::size_t iz = 0; iz < nzc; ++iz) {
                    const Real kz = geometry.k_component(iz);
                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    const Real k_scale = std::max({
                        std::abs(kx), std::abs(ky), std::abs(kz)});
                    if (!std::isfinite(k_scale)) {
                        spectral_failure[ix] = std::uint8_t{1};
                        modes[index] = {0.0, 0.0};
                        continue;
                    }
                    if (k_scale == 0.0) {
                        modes[index] = {0.0, 0.0};
                        continue;
                    }
                    const Real nx = kx / k_scale;
                    const Real ny = ky / k_scale;
                    const Real nz = kz / k_scale;
                    const Real k2 = nx * nx + ny * ny + nz * nz;
                    if (!std::isfinite(k2) || k2 <= 0.0) {
                        spectral_failure[ix] = std::uint8_t{1};
                        modes[index] = {0.0, 0.0};
                        continue;
                    }
                    const Real components[3]{nx, ny, nz};
                    const bool nyquist[3]{
                        n_mesh % 2 == 0 && ix == n_mesh / 2,
                        n_mesh % 2 == 0 && iy == n_mesh / 2,
                        n_mesh % 2 == 0 && iz == n_mesh / 2};
                    const Real kernel_value =
                        mesh::real_fourier_hessian_numerator(
                            axis_a, axis_b, components, nyquist) / k2;
                    if (!std::isfinite(kernel_value)) {
                        spectral_failure[ix] = std::uint8_t{1};
                        modes[index] = {0.0, 0.0};
                        continue;
                    }
                    const Real smoothing = gaussian_smoothing_weight(
                        k_scale,
                        k2,
                        options.gaussian_smoothing_radius);
                    const Real filtered_kernel = kernel_value * smoothing;
                    if (!std::isfinite(filtered_kernel)) {
                        spectral_failure[ix] = std::uint8_t{1};
                        modes[index] = {0.0, 0.0};
                        continue;
                    }

                    // FourierDensityBuilder stores DFT[delta]/N^3, while
                    // FFTBackend::inverse applies another 1/N^3. Multiply by
                    // N^3 here so the real tensor is
                    //   T_ij(x)=sum_k (ki kj/k^2) delta_k W(kR) exp(ikx),
                    // not T_ij/N^3. This is essential for nonzero lambda_th.
                    modes[index] = field.modes[index]
                        * (fft_rescale * filtered_kernel);
                }
            }
        }
        if (std::any_of(
                spectral_failure.begin(), spectral_failure.end(),
                [](std::uint8_t failure) { return failure != 0; })) {
            throw std::runtime_error(
                "T-web spectral kernel is non-finite");
        }

        fft.inverse(modes, real);
        std::vector<std::uint8_t> real_failure(n_mesh, std::uint8_t{0});
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < n_mesh; ++ix) {
            for (std::size_t iy = 0; iy < n_mesh; ++iy) {
                for (std::size_t iz = 0; iz < n_mesh; ++iz) {
                    const std::size_t output_index =
                        (ix * n_mesh + iy) * n_mesh + iz;
                    const Real value = real[
                        geometry.real_index(ix, iy, iz)];
                    if (!std::isfinite(value)) {
                        real_failure[ix] = std::uint8_t{1};
                        continue;
                    }
                    output[output_index] = value;
                }
            }
        }
        if (std::any_of(
                real_failure.begin(), real_failure.end(),
                [](std::uint8_t failure) { return failure != 0; })) {
            throw std::runtime_error(
                "T-web real-space tensor is non-finite");
        }
    };

    fill_and_ifft(0, 0, txx);
    fill_and_ifft(1, 1, tyy);
    fill_and_ifft(2, 2, tzz);
    fill_and_ifft(0, 1, txy);
    fill_and_ifft(0, 2, txz);
    fill_and_ifft(1, 2, tyz);

    TidalWebField result;
    result.mesh_size = N;
    result.gaussian_smoothing_radius = options.gaussian_smoothing_radius;
    result.lambda_threshold = options.lambda_threshold;
    result.environment.assign(ncell, WebEnvironment::Void);
    if (options.materialize_filament_directions) {
        result.filament_direction.assign(ncell, core::Vec3{});
        result.filament_direction_valid.assign(ncell, std::uint8_t{0});
    }
    bool invalid_eigensystem = false;

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(||:invalid_eigensystem)
#endif
    for (std::size_t i = 0; i < ncell; ++i) {
        SymmetricEigenvalues3 eigenvalues;
        try {
            eigenvalues = symmetric_eigenvalues_3x3(
                txx[i], txy[i], txz[i],
                tyy[i], tyz[i], tzz[i]);
        } catch (...) {
            invalid_eigensystem = true;
            continue;
        }
        if (!std::isfinite(eigenvalues.largest)
            || !std::isfinite(eigenvalues.middle)
            || !std::isfinite(eigenvalues.smallest)) {
            invalid_eigensystem = true;
            continue;
        }

        const WebEnvironment environment = environment_from_eigenvalues(
            eigenvalues.largest,
            eigenvalues.middle,
            eigenvalues.smallest,
            options.lambda_threshold);
        result.environment[i] = environment;
        if (!options.materialize_filament_directions
            || environment != WebEnvironment::Filament) {
            continue;
        }

        try {
            const auto direction = symmetric_eigenvector_3x3(
                txx[i], txy[i], txz[i],
                tyy[i], tyz[i], tzz[i],
                eigenvalues.smallest);
            if (!direction.has_value()) continue;
            result.filament_direction[i] = *direction;
            result.filament_direction_valid[i] = std::uint8_t{1};
        } catch (...) {
            invalid_eigensystem = true;
        }
    }
    if (invalid_eigensystem) {
        throw std::runtime_error(
            "T-web eigensystem is non-finite or unrepresentable");
    }
    return result;
}

TidalWebSummary FilamentStatistics::classify_tidal_web(
    const PeriodicDomain& domain,
    const core::ParticleStore& particles,
    const TidalWebOptions& options) {
    TidalWebOptions summary_options = options;
    summary_options.materialize_filament_directions = false;
    const TidalWebField field = build_tidal_web_field(
        domain, particles, summary_options);

    TidalWebSummary summary;
    summary.gaussian_smoothing_radius = field.gaussian_smoothing_radius;
    summary.lambda_threshold = field.lambda_threshold;
    std::size_t void_cells = 0;
    std::size_t sheet_cells = 0;
    std::size_t filament_cells = 0;
    std::size_t node_cells = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:void_cells,sheet_cells,filament_cells,node_cells)
#endif
    for (std::size_t index = 0; index < field.environment.size(); ++index) {
        switch (field.environment[index]) {
            case WebEnvironment::Void: ++void_cells; break;
            case WebEnvironment::Sheet: ++sheet_cells; break;
            case WebEnvironment::Filament: ++filament_cells; break;
            case WebEnvironment::Node: ++node_cells; break;
        }
    }
    summary.void_cells = void_cells;
    summary.sheet_cells = sheet_cells;
    summary.filament_cells = filament_cells;
    summary.node_cells = node_cells;

    const Real inverse_cells = field.environment.empty()
        ? 0.0
        : 1.0 / static_cast<Real>(field.environment.size());
    summary.void_volume_fraction =
        summary.void_cells * inverse_cells;
    summary.sheet_volume_fraction =
        summary.sheet_cells * inverse_cells;
    summary.filament_volume_fraction =
        summary.filament_cells * inverse_cells;
    summary.node_volume_fraction =
        summary.node_cells * inverse_cells;
    return summary;
}

TidalWebCellSample FilamentStatistics::sample_cell(
    const TidalWebField& field,
    Real box_size,
    core::Vec3 position) {
    if (field.mesh_size < 1) {
        throw std::invalid_argument(
            "Cannot sample a T-web field with non-positive mesh_size");
    }
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "T-web sample box_size must be finite and positive");
    }
    if (!std::isfinite(position.x) || !std::isfinite(position.y)
        || !std::isfinite(position.z)) {
        throw std::invalid_argument(
            "T-web sample position must be finite");
    }

    const std::size_t n_mesh = static_cast<std::size_t>(field.mesh_size);
    const std::size_t expected_size = checked_cell_count(n_mesh);
    if (field.environment.size() != expected_size) {
        throw std::invalid_argument(
            "T-web environment array does not match mesh_size^3");
    }
    if (field.filament_direction.size() != expected_size
        || field.filament_direction_valid.size() != expected_size) {
        throw std::invalid_argument(
            "T-web cell sampling requires a materialized filament-direction field");
    }

    auto cell_index = [&](Real coordinate) -> std::size_t {
        const Real wrapped = math::wrap(coordinate, box_size);
        const Real scaled = wrapped / box_size * static_cast<Real>(n_mesh);
        return std::min(
            static_cast<std::size_t>(scaled), n_mesh - 1);
    };

    const std::size_t ix = cell_index(position.x);
    const std::size_t iy = cell_index(position.y);
    const std::size_t iz = cell_index(position.z);
    const std::size_t index = (ix * n_mesh + iy) * n_mesh + iz;

    TidalWebCellSample result;
    result.environment = field.environment[index];
    result.filament_direction = field.filament_direction[index];
    result.filament_direction_valid =
        field.filament_direction_valid[index] != std::uint8_t{0};
    return result;
}

} // namespace analysis
} // namespace cosmo_nbody
