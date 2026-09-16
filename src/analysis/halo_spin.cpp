#include "cosmo_nbody/analysis/halo_spin.hpp"

#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cosmo_nbody::analysis {
namespace {

using WideReal = long double;

struct CompensatedWideSum {
    WideReal sum{0.0L};
    WideReal correction{0.0L};

    void add(WideReal value, const char* label) {
        if (!std::isfinite(value)) throw std::overflow_error(label);
        const WideReal updated = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            correction += (sum - updated) + value;
        } else {
            correction += (value - updated) + sum;
        }
        if (!std::isfinite(updated) || !std::isfinite(correction)) {
            throw std::overflow_error(label);
        }
        sum = updated;
    }

    [[nodiscard]] WideReal value(const char* label) const {
        const WideReal result = sum + correction;
        if (!std::isfinite(result)) throw std::overflow_error(label);
        return result;
    }
};

struct ScaledValue {
    WideReal mantissa{0.0L};
    int exponent{0};
};

ScaledValue normalized_product(WideReal lhs, WideReal rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        throw std::invalid_argument("Halo-spin scaled product requires finite factors");
    }
    if (lhs == 0.0L || rhs == 0.0L) return {};
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const WideReal lhs_mantissa = std::frexp(lhs, &lhs_exponent);
    const WideReal rhs_mantissa = std::frexp(rhs, &rhs_exponent);
    int product_shift = 0;
    const WideReal product_mantissa = std::frexp(
        lhs_mantissa * rhs_mantissa, &product_shift);
    return {
        product_mantissa,
        lhs_exponent + rhs_exponent + product_shift,
    };
}

ScaledValue normalized_product_difference(
    WideReal lhs_a,
    WideReal lhs_b,
    WideReal rhs_a,
    WideReal rhs_b) {
    if (!std::isfinite(lhs_a) || !std::isfinite(lhs_b)
        || !std::isfinite(rhs_a) || !std::isfinite(rhs_b)) {
        throw std::invalid_argument(
            "Halo-spin scaled product difference requires finite factors");
    }
    const bool lhs_nonzero = lhs_a != 0.0L && lhs_b != 0.0L;
    const bool rhs_nonzero = rhs_a != 0.0L && rhs_b != 0.0L;
    if (!lhs_nonzero && !rhs_nonzero) return {};
    if (!lhs_nonzero) {
        ScaledValue rhs = normalized_product(rhs_a, rhs_b);
        rhs.mantissa = -rhs.mantissa;
        return rhs;
    }
    if (!rhs_nonzero) return normalized_product(lhs_a, lhs_b);

    int lhs_a_exponent = 0;
    int lhs_b_exponent = 0;
    int rhs_a_exponent = 0;
    int rhs_b_exponent = 0;
    const WideReal lhs_a_mantissa = std::frexp(lhs_a, &lhs_a_exponent);
    const WideReal lhs_b_mantissa = std::frexp(lhs_b, &lhs_b_exponent);
    const WideReal rhs_a_mantissa = std::frexp(rhs_a, &rhs_a_exponent);
    const WideReal rhs_b_mantissa = std::frexp(rhs_b, &rhs_b_exponent);
    const int lhs_exponent = lhs_a_exponent + lhs_b_exponent;
    const int rhs_exponent = rhs_a_exponent + rhs_b_exponent;
    const int common_exponent = std::max(lhs_exponent, rhs_exponent);
    const WideReal scaled_lhs_a = std::scalbn(
        lhs_a_mantissa, lhs_exponent - common_exponent);
    const WideReal scaled_rhs_a = std::scalbn(
        rhs_a_mantissa, rhs_exponent - common_exponent);

    const WideReal rounded_rhs = scaled_rhs_a * rhs_b_mantissa;
    const WideReal rhs_error = std::fma(
        -scaled_rhs_a, rhs_b_mantissa, rounded_rhs);
    const WideReal difference = std::fma(
            scaled_lhs_a, lhs_b_mantissa, -rounded_rhs)
        + rhs_error;
    if (difference == 0.0L) return {};
    int difference_shift = 0;
    const WideReal normalized = std::frexp(difference, &difference_shift);
    return {normalized, common_exponent + difference_shift};
}

struct ScaledSignedSum {
    WideReal sum{0.0L};
    WideReal correction{0.0L};
    int exponent{0};
    bool initialized{false};

    void normalize() {
        const WideReal total = sum + correction;
        if (total == 0.0L) {
            sum = 0.0L;
            correction = 0.0L;
            exponent = 0;
            initialized = false;
            return;
        }
        if (!std::isfinite(total)) {
            throw std::overflow_error("Halo-spin scaled signed sum became non-finite");
        }
        int shift = 0;
        (void)std::frexp(total, &shift);
        sum = std::scalbn(sum, -shift);
        correction = std::scalbn(correction, -shift);
        exponent += shift;
        if (!std::isfinite(sum) || !std::isfinite(correction)) {
            throw std::overflow_error("Halo-spin scaled signed normalization failed");
        }
    }

    void add(ScaledValue value) {
        if (value.mantissa == 0.0L) return;
        if (!std::isfinite(value.mantissa)) {
            throw std::invalid_argument("Halo-spin scaled contribution is non-finite");
        }
        if (!initialized) {
            sum = value.mantissa;
            correction = 0.0L;
            exponent = value.exponent;
            initialized = true;
            normalize();
            return;
        }
        if (value.exponent > exponent) {
            const int shift = exponent - value.exponent;
            const WideReal previous_sum = sum;
            const WideReal previous_correction = correction;
            sum = std::scalbn(sum, shift);
            correction = std::scalbn(correction, shift);
            if ((previous_sum != 0.0L && sum == 0.0L)
                || (previous_correction != 0.0L && correction == 0.0L)) {
                throw std::overflow_error(
                    "Halo-spin accumulation lost an exponent-separated term");
            }
            exponent = value.exponent;
        }
        const WideReal term = std::scalbn(
            value.mantissa, value.exponent - exponent);
        if (value.mantissa != 0.0L && term == 0.0L) {
            throw std::overflow_error(
                "Halo-spin accumulation lost an exponent-separated term");
        }
        const WideReal updated = sum + term;
        const WideReal compensation = std::abs(sum) >= std::abs(term)
            ? (sum - updated) + term
            : (term - updated) + sum;
        correction += compensation;
        if (!std::isfinite(updated) || !std::isfinite(correction)) {
            throw std::overflow_error("Halo-spin scaled accumulation failed");
        }
        sum = updated;
        normalize();
    }

    void add_product(WideReal lhs, WideReal rhs) {
        add(normalized_product(lhs, rhs));
    }

    void add_weighted_product_difference(
        WideReal weight,
        WideReal lhs_a,
        WideReal lhs_b,
        WideReal rhs_a,
        WideReal rhs_b) {
        ScaledValue determinant = normalized_product_difference(
            lhs_a, lhs_b, rhs_a, rhs_b);
        if (determinant.mantissa == 0.0L) return;
        ScaledValue weighted = normalized_product(weight, determinant.mantissa);
        weighted.exponent += determinant.exponent;
        add(weighted);
    }

    [[nodiscard]] WideReal normalized_value() const {
        if (!initialized) return 0.0L;
        const WideReal result = sum + correction;
        if (!std::isfinite(result)) {
            throw std::overflow_error("Halo-spin scaled signed sum is non-finite");
        }
        return result;
    }

    [[nodiscard]] WideReal value_divided_by_positive(WideReal denominator) const {
        if (!std::isfinite(denominator) || denominator <= 0.0L) {
            throw std::invalid_argument(
                "Halo-spin scaled quotient requires a finite positive denominator");
        }
        if (!initialized) return 0.0L;
        const WideReal normalized = normalized_value();
        int denominator_exponent = 0;
        const WideReal denominator_mantissa = std::frexp(
            denominator, &denominator_exponent);
        int quotient_shift = 0;
        const WideReal quotient_mantissa = std::frexp(
            normalized / denominator_mantissa, &quotient_shift);
        const long long quotient_exponent =
            static_cast<long long>(exponent)
            - static_cast<long long>(denominator_exponent)
            + static_cast<long long>(quotient_shift);
        if (quotient_exponent < static_cast<long long>(
                std::numeric_limits<int>::min())
            || quotient_exponent > static_cast<long long>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Halo-spin scaled quotient exponent is not representable");
        }
        const WideReal result = std::scalbn(
            quotient_mantissa, static_cast<int>(quotient_exponent));
        if (!std::isfinite(result)
            || (normalized != 0.0L && result == 0.0L)) {
            throw std::overflow_error("Halo-spin scaled quotient is not representable");
        }
        return result;
    }

    [[nodiscard]] WideReal value() const {
        if (!initialized) return 0.0L;
        const WideReal normalized = normalized_value();
        const WideReal result = std::scalbn(normalized, exponent);
        if (!std::isfinite(result)
            || (normalized != 0.0L && result == 0.0L)) {
            throw std::overflow_error("Halo-spin scaled sum is not representable");
        }
        return result;
    }
};

core::Real particle_mass(
    const core::ParticleStore& particles,
    std::size_t index) {
    if (particles.get_uniform_mass().has_value()) {
        return *particles.get_uniform_mass();
    }
    return particles.get_masses()[index];
}

bool representable_positive(WideReal value) {
    if (!std::isfinite(value)
        || value <= 0.0L
        || value > static_cast<WideReal>(
            std::numeric_limits<core::Real>::max())) {
        return false;
    }
    const core::Real narrowed = static_cast<core::Real>(value);
    return std::isfinite(narrowed) && narrowed > 0.0;
}

bool representable_nonnegative(WideReal value) {
    if (!std::isfinite(value)
        || value < 0.0L
        || value > static_cast<WideReal>(
            std::numeric_limits<core::Real>::max())) {
        return false;
    }
    const core::Real narrowed = static_cast<core::Real>(value);
    return std::isfinite(narrowed)
        && narrowed >= 0.0
        && (value == 0.0L || narrowed != 0.0);
}

core::Real checked_nonnegative_real(WideReal value, const char* message) {
    if (!representable_nonnegative(value)) throw std::overflow_error(message);
    return static_cast<core::Real>(value);
}

} // namespace

SpinResult HaloSpin::compute_members(
    const core::ParticleStore& particles,
    const core::Vec3& center,
    core::Real r_delta,
    core::Real mass_delta,
    core::Real box_size,
    core::Real scale_factor,
    std::span<halo::PeriodicNeighbor> members) {
    if (!std::isfinite(center.x) || !std::isfinite(center.y)
        || !std::isfinite(center.z)
        || !std::isfinite(r_delta) || r_delta <= 0.0
        || !std::isfinite(mass_delta) || mass_delta <= 0.0
        || !std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(scale_factor) || scale_factor <= 0.0
        || members.empty()) {
        throw std::invalid_argument(
            "Exact-member halo spin requires finite physical inputs and non-empty membership");
    }

    SpinResult result;
    result.r_delta = r_delta;
    result.mass_delta = mass_delta;
    result.particle_count = members.size();

    const std::size_t owned = particles.num_owned_particles();
    const auto x = particles.get_positions_x().first(owned);
    const auto y = particles.get_positions_y().first(owned);
    const auto z = particles.get_positions_z().first(owned);
    const auto px = particles.get_momenta_x().first(owned);
    const auto py = particles.get_momenta_y().first(owned);
    const auto pz = particles.get_momenta_z().first(owned);
    const auto ids = particles.get_ids().first(owned);
    const core::Vec3 wrapped_center = math::wrap(center, box_size);
    if (!std::isfinite(wrapped_center.x)
        || !std::isfinite(wrapped_center.y)
        || !std::isfinite(wrapped_center.z)) {
        throw std::invalid_argument(
            "Exact-member halo-spin center cannot be represented periodically");
    }

    for (const halo::PeriodicNeighbor& member : members) {
        if (member.particle_index >= owned) {
            throw std::out_of_range(
                "Exact-member halo-spin index exceeds owned particles");
        }
    }

    std::sort(
        members.begin(),
        members.end(),
        [&](const halo::PeriodicNeighbor& lhs,
            const halo::PeriodicNeighbor& rhs) {
            const core::ParticleId lhs_id = ids[lhs.particle_index];
            const core::ParticleId rhs_id = ids[rhs.particle_index];
            if (lhs_id != rhs_id) return lhs_id < rhs_id;
            return lhs.particle_index < rhs.particle_index;
        });
    for (std::size_t member_index = 1;
         member_index < members.size();
         ++member_index) {
        const std::size_t previous_index =
            members[member_index - 1].particle_index;
        const std::size_t current_index =
            members[member_index].particle_index;
        if (previous_index == current_index) {
            throw std::invalid_argument(
                "Exact-member halo-spin membership contains a duplicate particle index");
        }
        if (ids[previous_index] == ids[current_index]) {
            throw std::invalid_argument(
                "Exact-member halo-spin membership contains duplicate stable ParticleIDs");
        }
    }

    const WideReal log_physical_radius =
        std::log(static_cast<WideReal>(scale_factor))
        + std::log(static_cast<WideReal>(r_delta));
    if (!std::isfinite(log_physical_radius)) {
        throw std::overflow_error(
            "Exact-member halo-spin physical-radius logarithm is invalid");
    }

    math::ExactPositiveDoubleSum exact_mass_accumulator;
    CompensatedWideSum mass_sum_accumulator;
    ScaledSignedSum bulk_px_accumulator;
    ScaledSignedSum bulk_py_accumulator;
    ScaledSignedSum bulk_pz_accumulator;
    bool spin_direction_is_unique = true;

    for (const halo::PeriodicNeighbor& member : members) {
        const std::size_t index = member.particle_index;
        if (!std::isfinite(member.radius) || member.radius < 0.0) {
            throw std::invalid_argument(
                "Exact-member halo-spin radius must be finite and non-negative");
        }
        if (!std::isfinite(x[index]) || !std::isfinite(y[index])
            || !std::isfinite(z[index]) || !std::isfinite(px[index])
            || !std::isfinite(py[index]) || !std::isfinite(pz[index])) {
            throw std::invalid_argument(
                "Exact-member halo-spin phase-space coordinates must be finite");
        }

        const core::Vec3 wrapped_position = math::wrap(
            core::Vec3{x[index], y[index], z[index]}, box_size);
        if (!std::isfinite(wrapped_position.x)
            || !std::isfinite(wrapped_position.y)
            || !std::isfinite(wrapped_position.z)) {
            throw std::invalid_argument(
                "Exact-member halo-spin position cannot be represented periodically");
        }
        if (!math::minimum_image_distance_leq_wrapped(
                wrapped_center, wrapped_position, box_size, r_delta)) {
            throw std::invalid_argument(
                "Exact-member halo-spin particle lies outside the SO aperture");
        }
        const core::Vec3 displacement = math::minimum_image_displacement(
            wrapped_center, wrapped_position, box_size);
        const core::Real actual_radius = core::scale_safe_norm3(
            displacement.x, displacement.y, displacement.z);
        if (!std::isfinite(actual_radius)) {
            throw std::invalid_argument(
                "Exact-member halo-spin particle radius is non-finite");
        }
        if (actual_radius != member.radius) {
            throw std::invalid_argument(
                "Exact-member halo-spin cached radius disagrees with particle coordinates");
        }
        if (!math::minimum_image_displacement_is_directionally_unique(
                wrapped_center, wrapped_position, box_size)) {
            spin_direction_is_unique = false;
        }

        const core::Real mass_value = particle_mass(particles, index);
        if (!std::isfinite(mass_value) || mass_value <= 0.0) {
            throw std::invalid_argument(
                "Exact-member halo-spin masses must be finite and positive");
        }
        const WideReal mass = static_cast<WideReal>(mass_value);
        exact_mass_accumulator.add(mass_value);
        mass_sum_accumulator.add(
            mass, "Exact-member halo-spin mass accumulation overflowed");
        bulk_px_accumulator.add_product(mass, static_cast<WideReal>(px[index]));
        bulk_py_accumulator.add_product(mass, static_cast<WideReal>(py[index]));
        bulk_pz_accumulator.add_product(mass, static_cast<WideReal>(pz[index]));
    }

    const core::Real enclosed_mass = exact_mass_accumulator.value();
    const WideReal mass_sum = mass_sum_accumulator.value(
        "Exact-member halo-spin mass sum is non-finite");
    if (!representable_positive(mass_sum)) {
        throw std::overflow_error(
            "Exact-member halo-spin enclosed mass is not representable");
    }
    if (enclosed_mass != mass_delta) {
        throw std::logic_error(
            "Exact-member halo-spin mass does not equal the published SO mass");
    }

    if (!spin_direction_is_unique) {
        result.status = "ambiguous_periodic_cut_locus";
        return result;
    }

    const WideReal bulk_px =
        bulk_px_accumulator.value_divided_by_positive(mass_sum);
    const WideReal bulk_py =
        bulk_py_accumulator.value_divided_by_positive(mass_sum);
    const WideReal bulk_pz =
        bulk_pz_accumulator.value_divided_by_positive(mass_sum);
    if (!std::isfinite(bulk_px) || !std::isfinite(bulk_py)
        || !std::isfinite(bulk_pz)) {
        result.status = "nonfinite_bulk_velocity";
        return result;
    }

    // p=a*v_pec and r_phys=a*r_com, so a cancels in
    // J=sum m r_phys x (v_pec-v_bulk)=sum m r_com x (p-p_bulk).
    ScaledSignedSum jx_accumulator;
    ScaledSignedSum jy_accumulator;
    ScaledSignedSum jz_accumulator;
    for (const halo::PeriodicNeighbor& member : members) {
        const std::size_t index = member.particle_index;
        const core::Vec3 wrapped_position = math::wrap(
            core::Vec3{x[index], y[index], z[index]}, box_size);
        const core::Vec3 displacement = math::minimum_image_displacement(
            wrapped_center, wrapped_position, box_size);
        const WideReal mass = static_cast<WideReal>(
            particle_mass(particles, index));
        const WideReal rx = static_cast<WideReal>(displacement.x);
        const WideReal ry = static_cast<WideReal>(displacement.y);
        const WideReal rz = static_cast<WideReal>(displacement.z);
        const WideReal dpx = static_cast<WideReal>(px[index]) - bulk_px;
        const WideReal dpy = static_cast<WideReal>(py[index]) - bulk_py;
        const WideReal dpz = static_cast<WideReal>(pz[index]) - bulk_pz;
        jx_accumulator.add_weighted_product_difference(
            mass, ry, dpz, rz, dpy);
        jy_accumulator.add_weighted_product_difference(
            mass, rz, dpx, rx, dpz);
        jz_accumulator.add_weighted_product_difference(
            mass, rx, dpy, ry, dpx);
    }

    const WideReal jx = jx_accumulator.value();
    const WideReal jy = jy_accumulator.value();
    const WideReal jz = jz_accumulator.value();
    const WideReal angular_momentum = std::hypot(std::hypot(jx, jy), jz);
    const WideReal log_circular_velocity = 0.5L * (
        std::log(static_cast<WideReal>(cosmology::units::G))
        + std::log(static_cast<WideReal>(mass_delta))
        - log_physical_radius);
    const WideReal circular_velocity = std::exp(log_circular_velocity);
    if (!representable_nonnegative(angular_momentum)
        || !representable_positive(circular_velocity)) {
        result.status = "degenerate_spin_inputs";
        return result;
    }

    result.angular_momentum = checked_nonnegative_real(
        angular_momentum,
        "Exact-member halo angular momentum is not representable");
    result.circular_velocity = static_cast<core::Real>(circular_velocity);
    if (angular_momentum == 0.0L) {
        result.status = "computed";
        result.bullock_lambda = 0.0;
        return result;
    }

    const WideReal log_lambda =
        std::log(angular_momentum)
        - 0.5L * std::log(2.0L)
        - std::log(static_cast<WideReal>(mass_delta))
        - log_circular_velocity
        - log_physical_radius;
    const WideReal log_representable_max = std::log(static_cast<WideReal>(
        std::numeric_limits<core::Real>::max()));
    const WideReal log_min_lambda = core::real_round_to_zero_log_threshold();
    if (!std::isfinite(log_lambda)
        || log_lambda > log_representable_max
        || log_lambda < log_min_lambda) {
        result.status = "nonfinite_spin_parameter";
        return result;
    }
    const WideReal lambda = std::exp(log_lambda);
    if (!representable_nonnegative(lambda)) {
        result.status = "nonfinite_spin_parameter";
        return result;
    }

    result.status = "computed";
    result.bullock_lambda = static_cast<core::Real>(lambda);
    return result;
}

} // namespace cosmo_nbody::analysis
