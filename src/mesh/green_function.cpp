#include "cosmo_nbody/mesh/green_function.hpp"
#include "cosmo_nbody/cosmology/units.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody {
namespace mesh {

GreenFunction::GreenFunction(
    const MeshGeometry& geom,
    PMForceMethod method)
    : geom_(geom),
      method_(method),
      laplacian_symbol_1d_(geom.grid_size()),
      assignment_window_1d_(geom.grid_size())
{
    const bool spectral_gradient = method_.uses_spectral_gradient();
    const core::Real n = static_cast<core::Real>(geom_.grid_size());
    core::Real inverse_cell_squared = 0.0;
    core::Real largest_undeconvolved_kernel = 0.0;
    if (!spectral_gradient) {
        const core::Real inverse_cell = 1.0 / geom_.cell_size();
        inverse_cell_squared = inverse_cell * inverse_cell;
        if (!std::isfinite(inverse_cell_squared)
            || inverse_cell_squared <= 0.0) {
            throw std::overflow_error(
                "Discrete Green-function scale is not representable");
        }
    } else if (geom_.grid_size() > 1) {
        const core::Real fundamental =
            2.0 * std::numbers::pi / geom_.box_size();
        const core::Real fundamental_squared = fundamental * fundamental;
        const core::Real maximum_component =
            std::numbers::pi / geom_.cell_size();
        const core::Real maximum_k_squared =
            3.0 * maximum_component * maximum_component;
        if (!std::isfinite(fundamental_squared)
            || fundamental_squared <= 0.0
            || !std::isfinite(maximum_k_squared)
            || maximum_k_squared <= 0.0) {
            throw std::overflow_error(
                "Continuum Green-function wave-number range is not representable");
        }
        largest_undeconvolved_kernel =
            4.0 * std::numbers::pi * cosmology::units::G
            / fundamental_squared;
        if (!std::isfinite(largest_undeconvolved_kernel)) {
            throw std::overflow_error(
                "Continuum Green-function amplitude is not representable");
        }
    }

    core::Real minimum_nonzero_laplacian =
        std::numeric_limits<core::Real>::infinity();
    for (std::size_t index = 0; index < geom_.grid_size(); ++index) {
        const core::Real k_dx_half = std::numbers::pi
            * static_cast<core::Real>(geom_.mode_number(index)) / n;
        const core::Real sine_half = std::sin(k_dx_half);
        if (!spectral_gradient) {
            const core::Real symbol =
                -4.0 * inverse_cell_squared * sine_half * sine_half;
            if (!std::isfinite(symbol)
                || (index != 0 && symbol == 0.0)) {
                throw std::overflow_error(
                    "Discrete Green-function mode is not representable");
            }
            laplacian_symbol_1d_[index] = symbol;
            if (index != 0) {
                minimum_nonzero_laplacian = std::min(
                    minimum_nonzero_laplacian,
                    std::abs(symbol));
            }
        }
        assignment_window_1d_[index] = W_k_1D(index);
        if (!std::isfinite(assignment_window_1d_[index])
            || assignment_window_1d_[index] <= 0.0
            || assignment_window_1d_[index] > 1.0) {
            throw std::overflow_error(
                "CIC assignment window escaped its exact range (0,1]");
        }
    }

    if (!spectral_gradient && geom_.grid_size() > 1) {
        largest_undeconvolved_kernel =
            4.0 * std::numbers::pi * cosmology::units::G
            / minimum_nonzero_laplacian;
        if (!std::isfinite(largest_undeconvolved_kernel)) {
            throw std::overflow_error(
                "Discrete Green-function amplitude is not representable");
        }
    }

    if (method_.deconvolves_cic() && geom_.grid_size() > 1) {
        const core::Real minimum_window = *std::min_element(
            assignment_window_1d_.begin(),
            assignment_window_1d_.end());
        const core::Real minimum_window_3 =
            minimum_window * minimum_window * minimum_window;
        const core::Real minimum_window_6 =
            minimum_window_3 * minimum_window_3;
        const long double conservative_bound =
            static_cast<long double>(largest_undeconvolved_kernel)
            / static_cast<long double>(minimum_window_6);
        const bool bound_proves_representability =
            std::isfinite(minimum_window_6)
            && minimum_window_6 > 0.0
            && std::isfinite(conservative_bound)
            && conservative_bound
                <= static_cast<long double>(
                    std::numeric_limits<core::Real>::max());

        // Use the O(N) conservative bound first; scan modes only when
        // inconclusive, avoiding false rejection and O(N^3) normal-path cost.
        if (!bound_proves_representability) {
            const std::size_t N = geom_.grid_size();
            const std::size_t nz_complex = N / 2 + 1;
            for (std::size_t local_ix = 0;
                 local_ix < geom_.local_n0();
                 ++local_ix) {
                const std::size_t ix = geom_.local_0_start() + local_ix;
                for (std::size_t iy = 0; iy < N; ++iy) {
                    for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                        (void)evaluate_kernel(ix, iy, iz);
                    }
                }
            }
        }
    }
}

core::Real GreenFunction::W_k_1D(std::size_t i) const {
    const long double argument = std::numbers::pi_v<long double>
        * static_cast<long double>(geom_.mode_number(i))
        / static_cast<long double>(geom_.grid_size());
    if (!std::isfinite(argument)) {
        throw std::overflow_error(
            "CIC assignment-window argument is not representable");
    }
    if (argument == 0.0L) return 1.0;

    const long double raw_sinc = std::sin(argument) / argument;
    if (!std::isfinite(raw_sinc)) {
        throw std::overflow_error(
            "CIC assignment-window sinc is not representable");
    }
    // Clamp only representational overshoot to the exact |sinc(x)| <= 1 bound.
    const long double sinc_magnitude = std::min(
        1.0L, std::abs(raw_sinc));
    const long double window = sinc_magnitude * sinc_magnitude;
    if (!std::isfinite(window) || window <= 0.0L || window > 1.0L
        || window > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(
            "CIC assignment window is not representable");
    }
    return static_cast<core::Real>(window);
}

core::Real GreenFunction::evaluate_kernel(
    std::size_t ix,
    std::size_t iy,
    std::size_t iz) const {
    if (ix == 0 && iy == 0 && iz == 0) {
        return 0.0;
    }

    core::Real D_k;
    if (method_.uses_spectral_gradient()) {
        // Continuum Green function paired with the spectral -ik derivative.
        D_k = -geom_.k_mag2(ix, iy, iz);
    } else {
        // Use -2*sin(theta/2)^2 instead of cos(theta)-1 to avoid cancellation
        // while preserving the declared discrete Laplacian.
        D_k = laplacian_symbol_1d_[ix]
            + laplacian_symbol_1d_[iy]
            + laplacian_symbol_1d_[iz];
    }

    core::Real phi_k_factor =
        4.0 * std::numbers::pi * cosmology::units::G / D_k;

    if (method_.is_treepm_long_range()) {
        const core::Real k2 = geom_.k_mag2(ix, iy, iz);
        const core::Real r_s = method_.split_scale();
        phi_k_factor *= std::exp(-k2 * r_s * r_s);
    }

    if (method_.deconvolves_cic()) {
        const core::Real wx = assignment_window_1d_[ix];
        const core::Real wy = assignment_window_1d_[iy];
        const core::Real wz = assignment_window_1d_[iz];
        const core::Real w_total = wx * wy * wz;
        phi_k_factor /= (w_total * w_total);
    }

    if (!std::isfinite(phi_k_factor)) {
        throw std::overflow_error(
            "PM Green-function kernel is not representable");
    }
    return phi_k_factor;
}

void GreenFunction::apply(ComplexField& field_k) const {
    if (field_k.size() != geom_.complex_size()) {
        throw std::invalid_argument(
            "Green-function field size does not match mesh geometry");
    }
    const std::size_t N = geom_.grid_size();
    const std::size_t nz_complex = N / 2 + 1;
    std::vector<std::exception_ptr> slab_exceptions(geom_.local_n0());

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t local_ix = 0; local_ix < geom_.local_n0(); ++local_ix) {
        try {
            const std::size_t ix = geom_.local_0_start() + local_ix;
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const std::size_t idx =
                        geom_.complex_index(local_ix, iy, iz);
                    const core::Real kernel_val =
                        evaluate_kernel(ix, iy, iz);
                    field_k[idx] *= kernel_val;
                }
            }
        } catch (...) {
            slab_exceptions[local_ix] = std::current_exception();
        }
    }

    for (const auto& exception : slab_exceptions) {
        if (exception) std::rethrow_exception(exception);
    }
}

} // namespace mesh
} // namespace cosmo_nbody