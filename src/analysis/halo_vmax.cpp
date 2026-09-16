#include "cosmo_nbody/analysis/halo_vmax.hpp"

#include "cosmo_nbody/core/exact_binary64_norm.hpp"
#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace cosmo_nbody::analysis {
namespace {

core::Real canonical_periodic_radius(
    const core::Vec3& center,
    const core::Vec3& position,
    core::Real box_size) {
    core::detail::ExactBinary64SquareAccumulator positive;
    core::detail::ExactBinary64SquareAccumulator negative;
    math::detail::add_exact_wrapped_distance_squared(
        center, position, box_size, positive, negative);

    const int relation = positive.compare(negative);
    if (relation < 0) {
        throw std::logic_error(
            "Exact periodic squared distance became negative");
    }
    if (relation == 0) return 0.0;

    // Form a range-safe square-root seed from the exact dyadic d^2. The seed
    // may differ by one final rounding step from RN(sqrt(d^2)); it is not used
    // as the scientific decision. Exact comparisons below find the adjacent
    // binary64 bracket and then resolve round-to-nearest, ties-to-even exactly.
    const auto squared = positive.positive_difference(negative);
    core::Real mantissa = squared.mantissa;
    int exponent = squared.exponent;
    if (exponent % 2 != 0) {
        mantissa *= 2.0;
        --exponent;
    }
    core::Real estimate = std::scalbn(
        std::sqrt(mantissa), exponent / 2);
    if (!std::isfinite(estimate) || estimate <= 0.0) {
        throw std::overflow_error(
            "Named SO Vmax periodic shell radius is not representable");
    }

    const auto compare_radius = [&](core::Real radius) {
        auto lhs = positive;
        auto rhs = negative;
        rhs.add_square(radius);
        return lhs.compare(rhs);
    };

    const int estimate_relation = compare_radius(estimate);
    if (estimate_relation == 0) return estimate;

    core::Real lower = 0.0;
    core::Real upper = 0.0;
    if (estimate_relation < 0) {
        // exact d < estimate: descend until lower < d < upper (or equality).
        upper = estimate;
        lower = std::nextafter(upper, core::Real{0.0});
        int lower_relation = compare_radius(lower);
        while (lower_relation < 0) {
            upper = lower;
            lower = std::nextafter(lower, core::Real{0.0});
            lower_relation = compare_radius(lower);
        }
        if (lower_relation == 0) return lower;
    } else {
        // exact d > estimate: ascend until lower < d < upper (or equality).
        lower = estimate;
        upper = std::nextafter(
            lower, std::numeric_limits<core::Real>::infinity());
        if (!std::isfinite(upper)) {
            throw std::overflow_error(
                "Named SO Vmax periodic shell radius exceeds binary64");
        }
        int upper_relation = compare_radius(upper);
        while (upper_relation > 0) {
            lower = upper;
            upper = std::nextafter(
                upper, std::numeric_limits<core::Real>::infinity());
            if (!std::isfinite(upper)) {
                throw std::overflow_error(
                    "Named SO Vmax periodic shell radius exceeds binary64");
            }
            upper_relation = compare_radius(upper);
        }
        if (upper_relation == 0) return upper;
    }

    // lower < d < upper and the two are adjacent binary64 values. For positive
    // distances, d <= midpoint iff 4*d^2 <= (lower+upper)^2. Build both sides
    // from exact dyadic operations so the rounding boundary itself is not rounded.
    auto midpoint_positive = positive;
    auto midpoint_negative = negative;
    for (int repeat = 1; repeat < 4; ++repeat) {
        midpoint_positive.add_accumulator(positive);
        midpoint_negative.add_accumulator(negative);
    }
    midpoint_negative.add_square(lower);
    midpoint_negative.add_square(upper);
    midpoint_negative.add_product(lower, upper);
    midpoint_negative.add_product(lower, upper);
    const int midpoint_relation =
        midpoint_positive.compare(midpoint_negative);
    if (midpoint_relation < 0) return lower;
    if (midpoint_relation > 0) return upper;

    // At an exact midpoint IEEE-754 round-to-nearest selects the adjacent value
    // with an even trailing significand bit. Positive finite binary64 values are
    // monotonically ordered by their raw bit patterns, so bit 0 identifies it.
    const std::uint64_t lower_bits =
        core::portable_bit_cast<std::uint64_t>(lower);
    return (lower_bits & 1U) == 0U ? lower : upper;
}

} // namespace

VmaxResult compute_vmax(
    const core::ParticleStore& particles,
    core::Vec3 center,
    core::Real box_size,
    std::vector<halo::PeriodicNeighbor>& members,
    core::Real scale_factor) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "Named SO Vmax requires a finite positive box size");
    }
    if (!std::isfinite(center.x)
        || !std::isfinite(center.y)
        || !std::isfinite(center.z)) {
        throw std::invalid_argument("Named SO Vmax center must be finite");
    }
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "Named SO Vmax requires a finite positive scale factor");
    }
    center = math::wrap(center, box_size);

    const std::size_t owned = particles.num_owned_particles();
    const auto positions_x = particles.get_positions_x();
    const auto positions_y = particles.get_positions_y();
    const auto positions_z = particles.get_positions_z();
    const auto wrapped_position = [&](std::size_t particle_index) {
        return math::wrap(core::Vec3{
            positions_x[particle_index],
            positions_y[particle_index],
            positions_z[particle_index]}, box_size);
    };

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
        const core::Vec3 position{
            positions_x[member.particle_index],
            positions_y[member.particle_index],
            positions_z[member.particle_index]};
        if (!std::isfinite(position.x)
            || !std::isfinite(position.y)
            || !std::isfinite(position.z)) {
            throw std::invalid_argument(
                "Named SO Vmax member positions must be finite");
        }
        const core::Real mass_value = particles.mass_at(
            member.particle_index);
        if (!std::isfinite(mass_value) || mass_value <= 0.0) {
            throw std::invalid_argument(
                "Named SO Vmax requires finite positive member masses");
        }
    }

    struct CachedMember {
        halo::PeriodicNeighbor member;
        core::Vec3 wrapped_position;
        core::detail::ExactBinary64PositiveDyadic squared_distance_key;
    };
    // One key and wrapped position per member replace O(M log M) repeated
    // geometric construction. Equal rounded keys still require exact ordering.
    std::vector<CachedMember> cached;
    cached.reserve(members.size());
    for (const auto& member : members) {
        const auto position = wrapped_position(member.particle_index);
        cached.push_back({member, position,
            math::minimum_image_squared_distance_key_wrapped(center, position, box_size)});
    }
    const auto exact_distance_compare = [&](const auto& lhs, const auto& rhs) {
        const int key_relation = math::minimum_image_squared_distance_keys_compare(
            lhs.squared_distance_key, rhs.squared_distance_key);
        if (key_relation != 0) return key_relation < 0;
        const int relation = math::minimum_image_distances_compare_wrapped(
            center,
            lhs.wrapped_position,
            rhs.wrapped_position,
            box_size);
        if (relation != 0) return relation < 0;
        return lhs.member.particle_index < rhs.member.particle_index;
    };
    std::sort(cached.begin(), cached.end(), exact_distance_compare);
    for (std::size_t index = 0; index < members.size(); ++index)
        members[index] = cached[index].member;

    const auto same_exact_shell = [&](const auto& lhs, const auto& rhs) {
        if (math::minimum_image_squared_distance_keys_compare(
                lhs.squared_distance_key, rhs.squared_distance_key) != 0) return false;
        return math::minimum_image_distances_compare_wrapped(
            center,
            lhs.wrapped_position,
            rhs.wrapped_position,
            box_size) == 0;
    };

    math::ExactPositiveDoubleSum enclosed_mass_accumulator;
    long double best_log_v = -std::numeric_limits<long double>::infinity();
    core::Real best_radius = 0.0;
    const long double log_g = std::log(
        static_cast<long double>(cosmology::units::G));
    const long double log_a = std::log(
        static_cast<long double>(scale_factor));

    std::size_t shell_begin = 0;
    while (shell_begin < members.size()) {
        std::size_t shell_end = shell_begin + 1;
        while (shell_end < members.size()
               && same_exact_shell(cached[shell_begin], cached[shell_end])) {
            ++shell_end;
        }
        for (std::size_t index = shell_begin; index < shell_end; ++index) {
            enclosed_mass_accumulator.add(
                particles.mass_at(members[index].particle_index));
        }

        const core::Real shell_radius = canonical_periodic_radius(
            center,
            cached[shell_begin].wrapped_position,
            box_size);
        if (shell_radius == 0.0) {
            shell_begin = shell_end;
            continue;
        }

        // M(<r) for a closed spherical aperture includes the complete exact
        // shell. Evaluate Vc only after all equidistant members have entered the
        // exact positive mass accumulator. The scalar radius is the correctly
        // rounded binary64 projection of the same exact torus geometry used to
        // form that shell, rather than the descriptive cached norm of an
        // arbitrary member.
        const long double log_v = 0.5L * (
            log_g
            + enclosed_mass_accumulator.log_value()
            - log_a
            - std::log(static_cast<long double>(shell_radius)));
        if (!std::isfinite(log_v)) {
            throw std::overflow_error(
                "Named SO Vmax circular-velocity logarithm is invalid");
        }
        if (log_v > best_log_v) {
            best_log_v = log_v;
            best_radius = shell_radius;
        }
        shell_begin = shell_end;
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
