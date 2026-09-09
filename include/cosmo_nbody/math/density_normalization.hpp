#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::math {

// Prepare the PM/conservation density transform
//
//   rho_i - rho_bar = (m_i - M / N_cell) / dx^3.
//
// Evaluate the cancellation-sensitive numerator
//
//   N_cell * m_i - M
//
// with a fused multiply-add. When 1/(N_cell*dx^3) is directly representable
// and M/N_cell is itself an exact binary quotient, the common path is one FMA
// plus one multiplication. Otherwise the same factor remains in
// ScaledPositiveProduct mantissa/exponent form until it is combined with the
// numerator. Restricting direct_available() to an exact quotient is important:
// PM callers may use mean_mass_per_cell() and inverse_cell_volume() directly,
// and subtraction against a rounded M/N_cell would otherwise define a different
// zero point from normalize().
class DensityNormalization {
public:
    DensityNormalization(
        core::Real total_mass,
        std::size_t cell_count,
        core::Real cell_size,
        std::string_view role)
        : total_mass_(total_mass),
          cell_count_real_(cell_count_as_real(cell_count, role)),
          inverse_cell_count_(1.0 / cell_count_real_),
          scaled_inverse_volume_(make_scaled_inverse_volume(cell_size, role)),
          scaled_centered_density_(make_scaled_centered_density(
              cell_count_real_, cell_size, role)),
          scaled_density_(make_scaled_density(total_mass, cell_size, role)) {
        if (!std::isfinite(total_mass_) || total_mass_ <= 0.0) {
            throw std::invalid_argument(
                std::string(role) + " total mass must be finite and positive");
        }
        if (!std::isfinite(cell_size) || cell_size <= 0.0) {
            throw std::invalid_argument(
                std::string(role) + " cell size must be finite and positive");
        }

        mean_mass_per_cell_ = total_mass_ / cell_count_real_;
        const core::Real cell_area = cell_size * cell_size;
        const core::Real cell_volume = cell_area * cell_size;
        const core::Real inverse_volume = 1.0 / cell_volume;
        const bool exact_mean_quotient =
            std::isfinite(mean_mass_per_cell_)
            && mean_mass_per_cell_ > 0.0
            && std::fma(
                mean_mass_per_cell_, cell_count_real_, -total_mass_) == 0.0;
        if (exact_mean_quotient
            && std::isfinite(inverse_volume)
            && inverse_volume > 0.0) {
            direct_available_ = true;
            inverse_cell_volume_ = inverse_volume;
            direct_centered_scale_ = inverse_volume * inverse_cell_count_;
        }
    }

    bool direct_available() const noexcept {
        return direct_available_;
    }

    core::Real mean_mass_per_cell() const noexcept {
        return mean_mass_per_cell_;
    }

    core::Real inverse_cell_volume() const noexcept {
        return inverse_cell_volume_;
    }

    core::Real normalize(
        core::Real cell_mass,
        std::string_view role) const {
        if (!std::isfinite(cell_mass) || cell_mass < 0.0) {
            throw std::invalid_argument(
                std::string(role)
                + " cell mass must be finite and non-negative");
        }

        // Algebraically exact centering before the volume scale:
        //   (N*m-M)/(N*dx^3) == (m-M/N)/dx^3.
        // FMA avoids separately rounding N*m and its subtraction from M.
        const core::Real centered_numerator = std::fma(
            cell_mass, cell_count_real_, -total_mass_);

        if (direct_available_) {
            if (std::isfinite(centered_numerator)) {
                if (centered_numerator == 0.0) return 0.0;
                if (std::isfinite(direct_centered_scale_)
                    && direct_centered_scale_ > 0.0) {
                    const core::Real direct =
                        centered_numerator * direct_centered_scale_;
                    if (std::isfinite(direct) && direct != 0.0) {
                        return direct;
                    }
                }
                const core::Real magnitude =
                    scaled_centered_density_.multiplied_by(
                        std::abs(centered_numerator), role);
                return std::copysign(magnitude, centered_numerator);
            }

            // A very large off-mean cell can make N*m overflow although
            // m-M/N is still representable. Cancellation is not the limiting
            // operation in that regime, so use the mass-space expression and
            // retain the scaled inverse-volume fallback if its final multiply
            // leaves binary64 range.
            const core::Real mass_difference =
                cell_mass - mean_mass_per_cell_;
            if (mass_difference == 0.0) return 0.0;
            const core::Real direct =
                mass_difference * inverse_cell_volume_;
            if (std::isfinite(direct) && direct != 0.0) {
                return direct;
            }
            const core::Real magnitude =
                scaled_inverse_volume_.multiplied_by(
                    std::abs(mass_difference), role);
            return std::copysign(magnitude, mass_difference);
        }

        if (std::isfinite(centered_numerator)) {
            if (centered_numerator == 0.0) return 0.0;
            const core::Real magnitude =
                scaled_centered_density_.multiplied_by(
                    std::abs(centered_numerator), role);
            return std::copysign(magnitude, centered_numerator);
        }

        // A very large off-mean cell can make N_cell*m_i overflow even though
        // m_i-M/N_cell remains representable. This branch is not cancellation
        // sensitive, so retain the mass-space fallback rather than rejecting a
        // representable final density solely because the fused numerator grew.
        if (mean_mass_per_cell_ > 0.0) {
            const core::Real mass_difference =
                cell_mass - mean_mass_per_cell_;
            if (mass_difference == 0.0) return 0.0;
            const core::Real magnitude =
                scaled_inverse_volume_.multiplied_by(
                    std::abs(mass_difference), role);
            return std::copysign(magnitude, mass_difference);
        }

        // If M/N_cell itself rounds to zero and the fused numerator could not
        // be represented, use a bounded dimensionless fraction before applying
        // the scaled total-density normalization.
        const core::Real cell_fraction = cell_mass / total_mass_;
        if (!std::isfinite(cell_fraction) || cell_fraction < 0.0) {
            throw std::overflow_error(
                std::string(role)
                + " normalized cell-mass fraction is not representable");
        }
        const core::Real density_fraction =
            cell_fraction - inverse_cell_count_;
        if (density_fraction == 0.0) return 0.0;

        const core::Real magnitude = scaled_density_.multiplied_by(
            std::abs(density_fraction), role);
        return std::copysign(magnitude, density_fraction);
    }

private:
    static core::Real cell_count_as_real(
        std::size_t cell_count,
        std::string_view role) {
        if (cell_count == 0) {
            throw std::invalid_argument(
                std::string(role) + " cell count must be positive");
        }
        const core::Real count = static_cast<core::Real>(cell_count);
        const core::Real inverse = 1.0 / count;
        if (!std::isfinite(count) || count <= 0.0
            || !std::isfinite(inverse) || inverse <= 0.0) {
            throw std::overflow_error(
                std::string(role) + " cell-count normalization is not representable");
        }
        return count;
    }

    static ScaledPositiveProduct make_scaled_inverse_volume(
        core::Real cell_size,
        std::string_view role) {
        const std::array<core::Real, 1> numerator{1.0};
        const std::array<core::Real, 3> denominator{
            cell_size, cell_size, cell_size};
        return ScaledPositiveProduct::from_quotient(
            numerator, denominator, role);
    }

    static ScaledPositiveProduct make_scaled_centered_density(
        core::Real cell_count,
        core::Real cell_size,
        std::string_view role) {
        const std::array<core::Real, 1> numerator{1.0};
        const std::array<core::Real, 4> denominator{
            cell_count, cell_size, cell_size, cell_size};
        return ScaledPositiveProduct::from_quotient(
            numerator, denominator, role);
    }

    static ScaledPositiveProduct make_scaled_density(
        core::Real total_mass,
        core::Real cell_size,
        std::string_view role) {
        const std::array<core::Real, 1> numerator{total_mass};
        const std::array<core::Real, 3> denominator{
            cell_size, cell_size, cell_size};
        return ScaledPositiveProduct::from_quotient(
            numerator, denominator, role);
    }

    core::Real total_mass_{0.0};
    core::Real cell_count_real_{0.0};
    core::Real inverse_cell_count_{0.0};
    ScaledPositiveProduct scaled_inverse_volume_;
    ScaledPositiveProduct scaled_centered_density_;
    ScaledPositiveProduct scaled_density_;
    core::Real mean_mass_per_cell_{0.0};
    core::Real inverse_cell_volume_{0.0};
    core::Real direct_centered_scale_{0.0};
    bool direct_available_{false};
};

} // namespace cosmo_nbody::math