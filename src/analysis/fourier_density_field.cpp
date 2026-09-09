#include "cosmo_nbody/analysis/fourier_density_field.hpp"

#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/mass_assignment.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

namespace {

using WideReal = long double;

struct CompensatedNonnegativeSum {
    WideReal sum{0.0L};
    WideReal correction{0.0L};

    void add(WideReal value, const char* label) {
        if (!std::isfinite(value) || value < 0.0L) {
            throw std::invalid_argument(
                std::string(label) + " contribution must be finite and non-negative");
        }
        if (value == 0.0L) return;
        const WideReal updated = sum + value;
        if (!std::isfinite(updated)) {
            throw std::overflow_error(
                std::string(label) + " accumulation overflowed");
        }
        correction += std::abs(sum) >= std::abs(value)
            ? (sum - updated) + value
            : (value - updated) + sum;
        if (!std::isfinite(correction)) {
            throw std::overflow_error(
                std::string(label) + " compensation overflowed");
        }
        sum = updated;
    }

    WideReal positive_value(const char* label) const {
        const WideReal result = sum + correction;
        if (!std::isfinite(result) || result <= 0.0L) {
            throw std::overflow_error(
                std::string(label) + " compensated sum is not finite and positive");
        }
        return result;
    }
};

core::Real checked_positive_real(WideReal value, const char* label) {
    if (!std::isfinite(value) || value <= 0.0L
        || value > static_cast<WideReal>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(
            std::string(label) + " is not representable in core::Real");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::overflow_error(
            std::string(label) + " rounded outside the finite positive range");
    }
    return result;
}

core::Real checked_mass_square_fraction(WideReal value) {
    const core::Real result = checked_positive_real(
        value, "FourierDensityBuilder mass-square fraction");
    const core::Real tolerance =
        64.0 * std::numeric_limits<core::Real>::epsilon();
    if (result > 1.0 + tolerance) {
        throw std::logic_error(
            "FourierDensityBuilder mass-square fraction exceeds unity");
    }
    return std::min(core::Real{1.0}, result);
}

std::size_t checked_cube(std::size_t value) {
    if (value != 0
        && value > std::numeric_limits<std::size_t>::max() / value) {
        throw std::overflow_error(
            "FourierDensityBuilder mesh_size^2 overflows size_t");
    }
    const std::size_t square = value * value;
    if (square != 0
        && value > std::numeric_limits<std::size_t>::max() / square) {
        throw std::overflow_error(
            "FourierDensityBuilder mesh_size^3 overflows size_t");
    }
    return square * value;
}

bool any_failure(const std::vector<std::uint8_t>& failures) {
    return std::any_of(
        failures.begin(), failures.end(),
        [](std::uint8_t failure) { return failure != 0; });
}

} // namespace

FourierDensityField FourierDensityBuilder::build(
    const PeriodicDomain& domain,
    const core::ParticleStore& particles,
    int mesh_size,
    bool interlaced,
    config::MemoryPolicyParams memory_policy) {
    if (mesh_size < 2) {
        throw std::invalid_argument(
            "FourierDensityBuilder mesh_size must be >= 2");
    }

    const std::size_t mesh_n = static_cast<std::size_t>(mesh_size);
    const std::size_t mesh_cells = checked_cube(mesh_n);
    const core::Real mesh_cells_real = static_cast<core::Real>(mesh_cells);
    if (!std::isfinite(mesh_cells_real) || mesh_cells_real <= 0.0) {
        throw std::overflow_error(
            "FourierDensityBuilder mesh cell count is not representable");
    }

    const std::size_t n = particles.num_owned_particles();
    if (n == 0) {
        throw std::invalid_argument(
            "FourierDensityBuilder requires owned particles");
    }

    const core::Real L = domain.box_size();
    mesh::MeshGeometry geometry(L, mesh_n);
    mesh::FFTBackend fft(geometry);

    const auto original_x = particles.get_positions_x().first(n);
    const auto original_y = particles.get_positions_y().first(n);
    const auto original_z = particles.get_positions_z().first(n);
    bool invalid_position = false;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(||:invalid_position)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        invalid_position = invalid_position
            || !std::isfinite(original_x[i])
            || !std::isfinite(original_y[i])
            || !std::isfinite(original_z[i]);
    }
    if (invalid_position) {
        throw std::invalid_argument(
            "FourierDensityBuilder particle positions must be finite");
    }

    // Density contrast and weighted Poisson shot noise are invariant under a
    // common rescaling of all particle masses. Choose one positive mass unit,
    // accumulate moments from the exact core::Real ratios that CIC will deposit,
    // and let CIC perform the division at use time. This avoids both absolute
    // mass overflow and an additional O(N_particle) normalized-mass array.
    core::Real normalized_mass_sum = 0.0;
    core::Real mass_square_fraction = 0.0;
    core::Real deposition_mass_unit = 1.0;
    std::span<const core::Real> deposit_masses;
    const std::optional<core::Real> deposit_uniform_mass =
        particles.get_uniform_mass();
    if (deposit_uniform_mass.has_value()) {
        if (!std::isfinite(*deposit_uniform_mass)
            || *deposit_uniform_mass <= 0.0) {
            throw std::invalid_argument(
                "FourierDensityBuilder uniform mass must be finite and positive");
        }
        deposition_mass_unit = *deposit_uniform_mass;
        const WideReal count = static_cast<WideReal>(n);
        normalized_mass_sum = checked_positive_real(
            count, "FourierDensityBuilder normalized mass sum");
        mass_square_fraction = checked_mass_square_fraction(1.0L / count);
    } else {
        deposit_masses = particles.get_masses().first(n);
        deposition_mass_unit = 0.0;
        for (const core::Real mass : deposit_masses) {
            if (!std::isfinite(mass) || mass <= 0.0) {
                throw std::invalid_argument(
                    "FourierDensityBuilder masses must be finite and positive");
            }
            deposition_mass_unit = std::max(deposition_mass_unit, mass);
        }
        if (!std::isfinite(deposition_mass_unit)
            || deposition_mass_unit <= 0.0) {
            throw std::logic_error(
                "FourierDensityBuilder failed to establish a positive mass unit");
        }

        CompensatedNonnegativeSum weight_sum;
        CompensatedNonnegativeSum squared_weight_sum;
        for (const core::Real mass : deposit_masses) {
            const core::Real deposited_weight = mass / deposition_mass_unit;
            if (!std::isfinite(deposited_weight)
                || deposited_weight <= 0.0
                || deposited_weight > 1.0) {
                throw std::overflow_error(
                    "FourierDensityBuilder positive normalized mass weight is not representable");
            }
            const WideReal represented_weight =
                static_cast<WideReal>(deposited_weight);
            weight_sum.add(
                represented_weight,
                "FourierDensityBuilder normalized mass sum");
            squared_weight_sum.add(
                represented_weight * represented_weight,
                "FourierDensityBuilder normalized squared-mass sum");
        }

        const WideReal weight_sum_wide = weight_sum.positive_value(
            "FourierDensityBuilder normalized mass sum");
        const WideReal squared_weight_sum_wide =
            squared_weight_sum.positive_value(
                "FourierDensityBuilder normalized squared-mass sum");
        const WideReal ratio = squared_weight_sum_wide
            / (weight_sum_wide * weight_sum_wide);
        normalized_mass_sum = checked_positive_real(
            weight_sum_wide,
            "FourierDensityBuilder normalized mass sum");
        mass_square_fraction = checked_mass_square_fraction(ratio);
    }

    mesh::CICMassAssignment cic(
        geometry, std::move(memory_policy), deposition_mass_unit);

    const core::Real mean_mass_per_cell =
        normalized_mass_sum / mesh_cells_real;
    const core::Real fft_norm = 1.0 / mesh_cells_real;
    if (!std::isfinite(mean_mass_per_cell) || mean_mass_per_cell <= 0.0
        || !std::isfinite(fft_norm) || fft_norm <= 0.0) {
        throw std::overflow_error(
            "FourierDensityBuilder derived normalization is invalid");
    }

    auto build_one = [&](std::span<const core::Real> x,
                         std::span<const core::Real> y,
                         std::span<const core::Real> z) {
        mesh::RealField density(geometry.real_size());
        cic.deposit(
            x, y, z,
            deposit_masses, deposit_uniform_mass,
            density);
        std::vector<std::uint8_t> contrast_failure(
            geometry.local_n0(), std::uint8_t{0});
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < geometry.local_n0(); ++ix) {
            for (std::size_t iy = 0; iy < geometry.grid_size(); ++iy) {
                for (std::size_t iz = 0; iz < geometry.grid_size(); ++iz) {
                    const std::size_t index = geometry.real_index(ix, iy, iz);
                    const core::Real contrast =
                        density[index] / mean_mass_per_cell - 1.0;
                    if (!std::isfinite(contrast)) {
                        contrast_failure[ix] = std::uint8_t{1};
                        continue;
                    }
                    density[index] = contrast;
                }
            }
        }
        if (any_failure(contrast_failure)) {
            throw std::runtime_error(
                "FourierDensityBuilder density contrast is non-finite");
        }

        std::vector<std::complex<core::Real>> modes(
            geometry.complex_size());
        fft.forward_external_output(
            density, modes.data(), modes.size());
        const std::size_t nz_complex = geometry.grid_size() / 2 + 1;
        std::vector<std::uint8_t> mode_failure(
            geometry.local_n0(), std::uint8_t{0});
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < geometry.local_n0(); ++ix) {
            for (std::size_t iy = 0; iy < geometry.grid_size(); ++iy) {
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    modes[index] *= fft_norm;
                    if (!std::isfinite(modes[index].real())
                        || !std::isfinite(modes[index].imag())) {
                        mode_failure[ix] = std::uint8_t{1};
                    }
                }
            }
        }
        if (any_failure(mode_failure)) {
            throw std::runtime_error(
                "FourierDensityBuilder Fourier mode is non-finite");
        }
        return modes;
    };

    std::vector<std::complex<core::Real>> modes = build_one(
        original_x, original_y, original_z);

    if (interlaced) {
        const core::Real shift = 0.5 * geometry.cell_size();
        if (!std::isfinite(shift) || shift <= 0.0) {
            throw std::overflow_error(
                "FourierDensityBuilder interlacing shift is invalid");
        }
        std::vector<core::Real> shifted_x(n);
        std::vector<core::Real> shifted_y(n);
        std::vector<core::Real> shifted_z(n);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t i = 0; i < n; ++i) {
            // Depositing x-shift on the standard grid is equivalent to
            // depositing x on a grid whose origin is shifted by +shift.
            shifted_x[i] = math::wrap(original_x[i] - shift, L);
            shifted_y[i] = math::wrap(original_y[i] - shift, L);
            shifted_z[i] = math::wrap(original_z[i] - shift, L);
        }
        const auto shifted_modes = build_one(
            shifted_x, shifted_y, shifted_z);

        const std::size_t nz_complex = geometry.grid_size() / 2 + 1;
        std::vector<std::uint8_t> interlace_failure(
            geometry.grid_size(), std::uint8_t{0});
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ix = 0; ix < geometry.grid_size(); ++ix) {
            const core::Real kx = geometry.k_component(ix);
            for (std::size_t iy = 0; iy < geometry.grid_size(); ++iy) {
                const core::Real ky = geometry.k_component(iy);
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const core::Real kz = geometry.k_component(iz);
                    const std::size_t index = geometry.complex_index(ix, iy, iz);
                    const core::Real phase = -(kx + ky + kz) * shift;
                    const std::complex<core::Real> correction{
                        std::cos(phase), std::sin(phase)};
                    modes[index] = 0.5 * (
                        modes[index] + shifted_modes[index] * correction);
                    if (!std::isfinite(modes[index].real())
                        || !std::isfinite(modes[index].imag())) {
                        interlace_failure[ix] = std::uint8_t{1};
                    }
                }
            }
        }
        if (any_failure(interlace_failure)) {
            throw std::runtime_error(
                "FourierDensityBuilder interlaced mode is non-finite");
        }
    }

    // Mean-density subtraction defines the k=0 mode to be exactly zero. Set it
    // explicitly to prevent a tiny roundoff residual from contaminating field
    // comparison diagnostics.
    modes[geometry.complex_index(0, 0, 0)] = {0.0, 0.0};

    FourierDensityField field;
    field.mesh_size = mesh_size;
    field.box_size = L;
    field.interlaced = interlaced;
    field.particle_count = n;
    field.mass_square_fraction = mass_square_fraction;
    field.modes = std::move(modes);
    return field;
}

} // namespace analysis
} // namespace cosmo_nbody
