#include "cosmo_nbody/analysis/halo_mass_function.hpp"

#include "cosmo_nbody/analysis/spectral_numeric.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

HaloMassFunctionResult HaloMassFunction::compute(
    std::span<const core::Real> halo_masses,
    core::Real box_size,
    const HaloMassFunctionOptions& options) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "HaloMassFunction box_size must be finite and positive");
    }
    if (!std::isfinite(options.min_mass)
        || !std::isfinite(options.max_mass)
        || options.min_mass <= 0.0
        || options.max_mass <= options.min_mass) {
        throw std::invalid_argument(
            "HaloMassFunction requires 0 < min_mass < max_mass");
    }
    if (options.num_bins <= 0) {
        throw std::invalid_argument(
            "HaloMassFunction num_bins must be positive");
    }
    if (options.mass_definition.empty()
        || options.mass_definition == "unknown") {
        throw std::invalid_argument(
            "HaloMassFunction requires an explicit mass definition");
    }

    const core::Real volume = box_size * box_size * box_size;
    if (!std::isfinite(volume) || volume <= 0.0) {
        throw std::overflow_error(
            "HaloMassFunction box volume is invalid");
    }

    const core::Real log_span = detail::stable_positive_log_ratio(
        options.max_mass, options.min_mass);
    const core::Real delta_ln_mass =
        log_span / static_cast<core::Real>(options.num_bins);
    if (!std::isfinite(delta_ln_mass) || delta_ln_mass <= 0.0) {
        throw std::overflow_error(
            "HaloMassFunction logarithmic binning is invalid");
    }

    HaloMassFunctionResult result;
    result.mass_definition = options.mass_definition;
    result.volume = volume;
    result.delta_ln_mass = delta_ln_mass;
    result.input_halo_count = halo_masses.size();
    result.bins.resize(static_cast<std::size_t>(options.num_bins));

    std::vector<core::Real> edges(
        static_cast<std::size_t>(options.num_bins) + 1U);
    for (int edge = 0; edge <= options.num_bins; ++edge) {
        const core::Real fraction =
            static_cast<core::Real>(edge)
            / static_cast<core::Real>(options.num_bins);
        edges[static_cast<std::size_t>(edge)] =
            detail::stable_positive_log_interpolate(
                options.min_mass, options.max_mass, fraction);
    }

    for (int bin = 0; bin < options.num_bins; ++bin) {
        const core::Real lo = edges[static_cast<std::size_t>(bin)];
        const core::Real hi = edges[static_cast<std::size_t>(bin + 1)];
        const core::Real center = std::sqrt(lo) * std::sqrt(hi);
        if (!std::isfinite(lo) || !std::isfinite(hi)
            || !std::isfinite(center)
            || lo <= 0.0 || hi <= lo || center <= 0.0
            || center < lo || center > hi) {
            throw std::overflow_error(
                "HaloMassFunction bin edge is invalid");
        }
        auto& output = result.bins[static_cast<std::size_t>(bin)];
        output.mass_low = lo;
        output.mass_high = hi;
        output.mass_geometric_mean = center;
    }

    for (const core::Real mass : halo_masses) {
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "HaloMassFunction halo masses must be finite and positive");
        }
        if (mass < options.min_mass) {
            if (result.underflow_count == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error(
                    "HaloMassFunction underflow count overflows size_t");
            }
            ++result.underflow_count;
            continue;
        }
        if (mass > options.max_mass) {
            if (result.overflow_count == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error(
                    "HaloMassFunction overflow count overflows size_t");
            }
            ++result.overflow_count;
            continue;
        }

        int bin = options.num_bins - 1;
        if (mass < options.max_mass) {
            const auto upper = std::upper_bound(edges.begin(), edges.end(), mass);
            const auto edge_index = static_cast<std::size_t>(
                std::distance(edges.begin(), upper));
            if (edge_index == 0U || edge_index > edges.size()) {
                throw std::logic_error(
                    "HaloMassFunction edge search produced an invalid bin index");
            }
            bin = static_cast<int>(edge_index - 1U);
            bin = std::clamp(bin, 0, options.num_bins - 1);
        }
        auto& output = result.bins[static_cast<std::size_t>(bin)];
        if (output.count == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "HaloMassFunction bin count overflows size_t");
        }
        ++output.count;
    }

    const std::array<core::Real, 2> normalization_factors{
        volume, delta_ln_mass};
    for (auto& bin : result.bins) {
        if (bin.count == 0U) {
            bin.number_density_per_ln_mass = 0.0;
            bin.poisson_error_per_ln_mass = 0.0;
            continue;
        }

        const core::Real count = static_cast<core::Real>(bin.count);
        const std::array<core::Real, 1> count_factor{count};
        const std::array<core::Real, 1> poisson_factor{std::sqrt(count)};
        bin.number_density_per_ln_mass = math::scaled_positive_product_quotient(
            count_factor,
            normalization_factors,
            "HaloMassFunction number density");
        bin.poisson_error_per_ln_mass = math::scaled_positive_product_quotient(
            poisson_factor,
            normalization_factors,
            "HaloMassFunction Poisson error");
        if (!std::isfinite(bin.number_density_per_ln_mass)
            || !std::isfinite(bin.poisson_error_per_ln_mass)) {
            throw std::overflow_error(
                "HaloMassFunction bin statistic is non-finite");
        }
    }

    return result;
}

} // namespace analysis
} // namespace cosmo_nbody
