#include "cosmo_nbody/analysis/halo_vmax.hpp"

#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace cosmo_nbody::analysis {

VmaxResult compute_vmax(
    const core::ParticleStore& particles,
    std::vector<halo::PeriodicNeighbor>& members,
    core::Real scale_factor) {
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "Named SO Vmax requires a finite positive scale factor");
    }
    const std::size_t owned = particles.num_owned_particles();
    std::unordered_set<std::size_t> seen_indices;
    seen_indices.reserve(members.size());
    for (const auto& member : members) {
        if (member.particle_index >= owned) {
            throw std::out_of_range(
                "Named SO Vmax member index exceeds owned particles");
        }
        if (!seen_indices.insert(member.particle_index).second) {
            throw std::invalid_argument(
                "Named SO Vmax member indices must be unique");
        }
        if (!std::isfinite(member.radius) || member.radius < 0.0) {
            throw std::invalid_argument(
                "Named SO Vmax requires finite non-negative member radii");
        }
        const core::Real mass_value = particles.mass_at(
            member.particle_index);
        if (!std::isfinite(mass_value) || mass_value <= 0.0) {
            throw std::invalid_argument(
                "Named SO Vmax requires finite positive member masses");
        }
    }
    std::sort(
        members.begin(), members.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.radius != rhs.radius) return lhs.radius < rhs.radius;
            return lhs.particle_index < rhs.particle_index;
        });

    math::ExactPositiveDoubleSum enclosed_mass_accumulator;
    long double best_log_v = -std::numeric_limits<long double>::infinity();
    core::Real best_radius = 0.0;
    const long double log_g = std::log(
        static_cast<long double>(cosmology::units::G));
    const long double log_a = std::log(
        static_cast<long double>(scale_factor));
    for (const auto& member : members) {
        const core::Real mass_value = particles.mass_at(member.particle_index);
        enclosed_mass_accumulator.add(mass_value);
        const core::Real enclosed_mass_value =
            enclosed_mass_accumulator.value();
        const long double enclosed_mass =
            static_cast<long double>(enclosed_mass_value);
        if (!std::isfinite(enclosed_mass) || enclosed_mass <= 0.0L) {
            throw std::overflow_error(
                "Named SO Vmax enclosed mass is invalid");
        }
        if (member.radius == 0.0) continue;

        // Compare logarithms so physical-radius and squared-velocity
        // intermediates need not fit the working floating-point format.
        const long double log_v = 0.5L * (
            log_g
            + std::log(enclosed_mass)
            - log_a
            - std::log(static_cast<long double>(member.radius)));
        if (!std::isfinite(log_v)) {
            throw std::overflow_error(
                "Named SO Vmax circular-velocity logarithm is invalid");
        }
        if (log_v > best_log_v) {
            best_log_v = log_v;
            best_radius = member.radius;
        }
    }
    if (!std::isfinite(best_log_v)) {
        return {};
    }

    const long double log_real_max = std::log(static_cast<long double>(
        std::numeric_limits<core::Real>::max()));
    if (best_log_v > log_real_max
        || best_log_v < core::real_round_to_zero_log_threshold()) {
        return {};
    }
    const long double vmax = std::exp(best_log_v);
    if (!std::isfinite(vmax)
        || vmax < 0.0L
        || vmax > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error("Named SO Vmax is not representable");
    }
    const core::Real narrowed_vmax = static_cast<core::Real>(vmax);
    if (!std::isfinite(narrowed_vmax) || narrowed_vmax <= 0.0) {
        return {};
    }
    return {true, narrowed_vmax, best_radius};
}

} // namespace cosmo_nbody::analysis
