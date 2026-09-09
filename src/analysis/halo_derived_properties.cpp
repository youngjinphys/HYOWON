#include "cosmo_nbody/analysis/halo_derived_properties.hpp"

#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "membership_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::analysis {
namespace {

struct CompensatedSum {
    long double sum{0.0L};
    long double correction{0.0L};

    void add(long double value) {
        if (!std::isfinite(value)) {
            throw std::overflow_error(
                "Halo derived-property accumulation term is non-finite");
        }
        const long double updated = sum + value;
        if (!std::isfinite(updated)) {
            throw std::overflow_error(
                "Halo derived-property accumulation overflowed");
        }
        const long double compensation = std::abs(sum) >= std::abs(value)
            ? (sum - updated) + value
            : (value - updated) + sum;
        correction += compensation;
        if (!std::isfinite(correction)) {
            throw std::overflow_error(
                "Halo derived-property compensation overflowed");
        }
        sum = updated;
    }

    long double value() const {
        const long double result = sum + correction;
        if (!std::isfinite(result)) {
            throw std::overflow_error(
                "Halo derived-property compensated sum is non-finite");
        }
        return result;
    }
};

struct ScaledValue {
    long double mantissa{0.0L};
    int exponent{0};
};

ScaledValue normalized_product(long double lhs, long double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        throw std::invalid_argument(
            "Halo scaled product requires finite factors");
    }
    if (lhs == 0.0L || rhs == 0.0L) return {};
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const long double lhs_mantissa = std::frexp(lhs, &lhs_exponent);
    const long double rhs_mantissa = std::frexp(rhs, &rhs_exponent);
    int product_shift = 0;
    const long double product_mantissa = std::frexp(
        lhs_mantissa * rhs_mantissa,
        &product_shift);
    return {
        product_mantissa,
        lhs_exponent + rhs_exponent + product_shift};
}

ScaledValue normalized_product_difference(
    long double lhs_a,
    long double lhs_b,
    long double rhs_a,
    long double rhs_b) {
    if (!std::isfinite(lhs_a) || !std::isfinite(lhs_b)
        || !std::isfinite(rhs_a) || !std::isfinite(rhs_b)) {
        throw std::invalid_argument(
            "Halo scaled product difference requires finite factors");
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
    const long double lhs_a_mantissa = std::frexp(
        lhs_a, &lhs_a_exponent);
    const long double lhs_b_mantissa = std::frexp(
        lhs_b, &lhs_b_exponent);
    const long double rhs_a_mantissa = std::frexp(
        rhs_a, &rhs_a_exponent);
    const long double rhs_b_mantissa = std::frexp(
        rhs_b, &rhs_b_exponent);
    const int lhs_exponent = lhs_a_exponent + lhs_b_exponent;
    const int rhs_exponent = rhs_a_exponent + rhs_b_exponent;
    const int common_exponent = std::max(lhs_exponent, rhs_exponent);
    const long double scaled_lhs_a = std::scalbn(
        lhs_a_mantissa,
        lhs_exponent - common_exponent);
    const long double scaled_rhs_a = std::scalbn(
        rhs_a_mantissa,
        rhs_exponent - common_exponent);

    // Recover the rounded second product's error with FMA, then add it to the
    // first fused product difference. This is the cancellation-preserving
    // difference-of-products algorithm, evaluated only on bounded mantissas.
    const long double rounded_rhs = scaled_rhs_a * rhs_b_mantissa;
    const long double rhs_error = std::fma(
        -scaled_rhs_a,
        rhs_b_mantissa,
        rounded_rhs);
    const long double difference = std::fma(
            scaled_lhs_a,
            lhs_b_mantissa,
            -rounded_rhs)
        + rhs_error;
    if (difference == 0.0L) return {};
    int difference_shift = 0;
    const long double normalized = std::frexp(
        difference,
        &difference_shift);
    return {normalized, common_exponent + difference_shift};
}

struct ScaledSignedSum {
    long double sum{0.0L};
    long double correction{0.0L};
    int exponent{0};
    bool initialized{false};

    void normalize() {
        const long double total = sum + correction;
        if (total == 0.0L) {
            sum = 0.0L;
            correction = 0.0L;
            exponent = 0;
            initialized = false;
            return;
        }
        if (!std::isfinite(total)) {
            throw std::overflow_error(
                "Halo scaled signed sum became non-finite");
        }
        int shift = 0;
        (void)std::frexp(total, &shift);
        sum = std::scalbn(sum, -shift);
        correction = std::scalbn(correction, -shift);
        exponent += shift;
        if (!std::isfinite(sum) || !std::isfinite(correction)) {
            throw std::overflow_error(
                "Halo scaled signed sum normalization failed");
        }
    }

    void add(ScaledValue value) {
        if (value.mantissa == 0.0L) return;
        if (!std::isfinite(value.mantissa)) {
            throw std::invalid_argument(
                "Halo scaled signed contribution is non-finite");
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
            const long double previous_sum = sum;
            const long double previous_correction = correction;
            sum = std::scalbn(sum, shift);
            correction = std::scalbn(correction, shift);
            if ((previous_sum != 0.0L && sum == 0.0L)
                || (previous_correction != 0.0L && correction == 0.0L)) {
                throw std::overflow_error(
                    "Halo scaled signed accumulation lost an exponent-separated term");
            }
            exponent = value.exponent;
        }
        const long double term = std::scalbn(
            value.mantissa,
            value.exponent - exponent);
        if (value.mantissa != 0.0L && term == 0.0L) {
            throw std::overflow_error(
                "Halo scaled signed accumulation lost an exponent-separated term");
        }
        const long double updated = sum + term;
        const long double compensation = std::abs(sum) >= std::abs(term)
            ? (sum - updated) + term
            : (term - updated) + sum;
        correction += compensation;
        if (!std::isfinite(updated) || !std::isfinite(correction)) {
            throw std::overflow_error(
                "Halo scaled signed accumulation failed");
        }
        sum = updated;
        normalize();
    }

    void add_product(long double lhs, long double rhs) {
        add(normalized_product(lhs, rhs));
    }

    void add_weighted_product_difference(
        long double weight,
        long double lhs_a,
        long double lhs_b,
        long double rhs_a,
        long double rhs_b) {
        ScaledValue determinant = normalized_product_difference(
            lhs_a,
            lhs_b,
            rhs_a,
            rhs_b);
        if (determinant.mantissa == 0.0L) return;
        ScaledValue weighted = normalized_product(
            weight,
            determinant.mantissa);
        weighted.exponent += determinant.exponent;
        add(weighted);
    }

    long double normalized_value() const {
        if (!initialized) return 0.0L;
        const long double result = sum + correction;
        if (!std::isfinite(result)) {
            throw std::overflow_error(
                "Halo scaled signed sum is non-finite");
        }
        return result;
    }

    long double value_divided_by_positive(long double denominator) const {
        if (!std::isfinite(denominator) || denominator <= 0.0L) {
            throw std::invalid_argument(
                "Halo scaled signed quotient requires a finite positive denominator");
        }
        if (!initialized) return 0.0L;

        const long double normalized = normalized_value();
        int denominator_exponent = 0;
        const long double denominator_mantissa = std::frexp(
            denominator, &denominator_exponent);
        int quotient_shift = 0;
        const long double quotient_mantissa = std::frexp(
            normalized / denominator_mantissa,
            &quotient_shift);
        const long long quotient_exponent =
            static_cast<long long>(exponent)
            - static_cast<long long>(denominator_exponent)
            + static_cast<long long>(quotient_shift);
        if (quotient_exponent < static_cast<long long>(
                std::numeric_limits<int>::min())
            || quotient_exponent > static_cast<long long>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Halo scaled signed quotient exponent is not representable");
        }
        const long double result = std::scalbn(
            quotient_mantissa,
            static_cast<int>(quotient_exponent));
        if (!std::isfinite(result)
            || (normalized != 0.0L && result == 0.0L)) {
            throw std::overflow_error(
                "Halo scaled signed quotient is not representable");
        }
        return result;
    }

    long double value() const {
        if (!initialized) return 0.0L;
        const long double normalized = normalized_value();
        const long double result = std::scalbn(normalized, exponent);
        if (!std::isfinite(result)
            || (normalized != 0.0L && result == 0.0L)) {
            throw std::overflow_error(
                "Halo scaled signed sum is not representable");
        }
        return result;
    }
};

long double divide_by_positive(
    long double numerator,
    long double denominator,
    const char* label) {
    if (!std::isfinite(numerator)
        || !std::isfinite(denominator)
        || denominator <= 0.0L) {
        throw std::invalid_argument(label);
    }
    const long double result = numerator / denominator;
    if (!std::isfinite(result)
        || (numerator != 0.0L && result == 0.0L)) {
        throw std::overflow_error(label);
    }
    return result;
}

struct WeightedScaledNorm {
    long double scale{0.0L};
    long double sumsq{0.0L};
    long double correction{0.0L};

    static long double sqrt_weight_ratio(
        long double weight,
        long double total_weight) {
        if (!std::isfinite(weight) || weight <= 0.0L
            || !std::isfinite(total_weight) || total_weight <= 0.0L
            || weight > total_weight) {
            throw std::invalid_argument(
                "Halo weighted norm requires finite positive bounded mass");
        }
        int weight_exponent = 0;
        int total_exponent = 0;
        const long double weight_mantissa = std::frexp(
            weight, &weight_exponent);
        const long double total_mantissa = std::frexp(
            total_weight, &total_exponent);
        int exponent_value = weight_exponent - total_exponent;
        int half_exponent = exponent_value / 2;
        int remainder = exponent_value - 2 * half_exponent;
        if (remainder < 0) {
            remainder += 2;
            --half_exponent;
        }
        long double ratio = std::sqrt(
            weight_mantissa / total_mantissa);
        if (remainder == 1) {
            ratio *= std::numbers::sqrt2_v<long double>;
        }
        ratio = std::scalbn(ratio, half_exponent);
        if (!std::isfinite(ratio) || ratio <= 0.0L
            || ratio > 1.0L + 64.0L * std::numeric_limits<long double>::epsilon()) {
            throw std::overflow_error(
                "Halo normalized mass weight is not representable");
        }
        return std::min(ratio, 1.0L);
    }

    void compensated_add(long double term) {
        if (!std::isfinite(term) || term < 0.0L) {
            throw std::overflow_error(
                "Halo weighted norm scaled square is invalid");
        }
        const long double updated = sumsq + term;
        const long double compensation = std::abs(sumsq) >= std::abs(term)
            ? (sumsq - updated) + term
            : (term - updated) + sumsq;
        correction += compensation;
        if (!std::isfinite(updated) || !std::isfinite(correction)) {
            throw std::overflow_error(
                "Halo weighted norm sum of squares overflowed");
        }
        sumsq = updated;
    }

    void add(
        long double magnitude,
        long double mass,
        long double total_mass) {
        if (!std::isfinite(magnitude) || magnitude < 0.0L) {
            throw std::invalid_argument(
                "Halo weighted norm requires a finite non-negative magnitude");
        }
        if (magnitude == 0.0L) return;
        const long double weighted_magnitude = magnitude
            * sqrt_weight_ratio(mass, total_mass);
        if (!std::isfinite(weighted_magnitude)
            || weighted_magnitude <= 0.0L) {
            throw std::overflow_error(
                "Halo mass-weighted magnitude is not representable");
        }
        if (scale == 0.0L) {
            scale = weighted_magnitude;
            sumsq = 1.0L;
            correction = 0.0L;
            return;
        }
        if (weighted_magnitude > scale) {
            const long double ratio = scale / weighted_magnitude;
            const long double previous =
                (sumsq + correction) * ratio * ratio;
            scale = weighted_magnitude;
            sumsq = 1.0L;
            correction = 0.0L;
            compensated_add(previous);
            return;
        }
        const long double ratio = weighted_magnitude / scale;
        compensated_add(ratio * ratio);
    }

    core::Real value(const char* label) const {
        if (scale == 0.0L) return 0.0;
        const long double normalized = sumsq + correction;
        if (!std::isfinite(normalized) || normalized <= 0.0L) {
            throw std::overflow_error(
                std::string("Halo ") + label
                + " normalized sum is invalid");
        }
        const long double result = scale * std::sqrt(normalized);
        if (!std::isfinite(result)
            || result > static_cast<long double>(
                std::numeric_limits<core::Real>::max())) {
            throw std::overflow_error(
                std::string("Halo ") + label
                + " is not representable in core::Real");
        }
        const core::Real rounded = static_cast<core::Real>(result);
        if (!std::isfinite(rounded) || rounded <= 0.0) {
            throw std::overflow_error(
                std::string("Halo ") + label
                + " rounded outside core::Real");
        }
        return rounded;
    }
};

core::Real representable_real(long double value, const char* label) {
    if (!std::isfinite(value)
        || std::abs(value) > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(
            std::string("Halo derived-property ") + label
            + " is not representable in core::Real");
    }
    const core::Real rounded = static_cast<core::Real>(value);
    if (!std::isfinite(rounded)
        || (value != 0.0L && rounded == 0.0)) {
        throw std::overflow_error(
            std::string("Halo derived-property ") + label
            + " underflows core::Real");
    }
    return rounded;
}

struct PeriodicCenterAccumulator {
    CompensatedSum normalized_weight_sum;
    CompensatedSum weighted_sin[3];
    CompensatedSum weighted_cos[3];

    void add(
        const core::Vec3& position,
        long double normalized_weight,
        core::Real box_size) {
        if (!std::isfinite(normalized_weight) || normalized_weight < 0.0L) {
            throw std::invalid_argument(
                "Halo periodic-center normalized weight must be finite and non-negative");
        }
        if (normalized_weight == 0.0L) return;
        normalized_weight_sum.add(normalized_weight);
        const core::Real coordinates[3]{position.x, position.y, position.z};
        for (int axis = 0; axis < 3; ++axis) {
            const long double coordinate = static_cast<long double>(
                math::wrap(coordinates[axis], box_size));
            const long double angle = 2.0L * std::numbers::pi_v<long double>
                * coordinate / static_cast<long double>(box_size);
            weighted_sin[axis].add(normalized_weight * std::sin(angle));
            weighted_cos[axis].add(normalized_weight * std::cos(angle));
        }
    }

    core::Real axis(
        int axis_index,
        core::Real box_size) const {
        if (axis_index < 0 || axis_index >= 3) {
            throw std::invalid_argument(
                "Halo derived properties require a valid periodic-center accumulator");
        }
        const long double total_weight = normalized_weight_sum.value();
        if (!(total_weight > 0.0L)) {
            throw std::invalid_argument(
                "Halo periodic center requires a positive normalized weight sum");
        }
        const long double sine = weighted_sin[axis_index].value();
        const long double cosine = weighted_cos[axis_index].value();
        const long double resultant = std::hypot(sine, cosine) / total_weight;
        const long double directional_resolution =
            128.0L * static_cast<long double>(
                std::numeric_limits<core::Real>::epsilon());
        if (!std::isfinite(resultant)
            || resultant <= directional_resolution) {
            throw std::invalid_argument(
                "Halo periodic center is not uniquely defined on this axis");
        }

        long double angle_value = std::atan2(sine, cosine);
        if (angle_value < 0.0L) {
            angle_value += 2.0L * std::numbers::pi_v<long double>;
        }
        return math::wrap(
            representable_real(
                angle_value * static_cast<long double>(box_size)
                    / (2.0L * std::numbers::pi_v<long double>),
                "periodic center"),
            box_size);
    }

    core::Vec3 center(core::Real box_size) const {
        return {
            axis(0, box_size),
            axis(1, box_size),
            axis(2, box_size),
        };
    }
};

void require_finite_vector(const core::Vec3& value, const char* label) {
    if (!std::isfinite(value.x)
        || !std::isfinite(value.y)
        || !std::isfinite(value.z)) {
        throw std::invalid_argument(std::string(label) + " must be finite");
    }
}

HaloDerivedProperties compute_from_membership(
    const core::ParticleStore& particles,
    std::size_t halo_id,
    std::span<const std::size_t> members,
    core::Real box_size,
    core::Real scale_factor) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "Halo derived-property box size must be finite and positive");
    }
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "Halo derived-property scale factor must be finite and positive");
    }
    const long double scale_factor_wide = static_cast<long double>(scale_factor);
    if (members.empty()) {
        throw std::invalid_argument(
            "Halo derived properties require non-empty membership");
    }

    const std::size_t owned = particles.num_owned_particles();
    const auto positions_x = particles.get_positions_x();
    const auto positions_y = particles.get_positions_y();
    const auto positions_z = particles.get_positions_z();
    const auto momenta_x = particles.get_momenta_x();
    const auto momenta_y = particles.get_momenta_y();
    const auto momenta_z = particles.get_momenta_z();
    if (positions_x.size() < owned
        || positions_y.size() < owned
        || positions_z.size() < owned
        || momenta_x.size() < owned
        || momenta_y.size() < owned
        || momenta_z.size() < owned) {
        throw std::logic_error(
            "Halo derived-property primary arrays are shorter than the owned count");
    }

    (void)detail::require_unique_owned_members(
        particles,
        members,
        "Halo membership index exceeds the owned snapshot particle count",
        "Halo membership contains a duplicate particle index");

    math::ExactPositiveDoubleSum total_mass_accumulator;
    ScaledSignedSum weighted_momentum[3];
    long double maximum_mass_wide = 0.0L;
    for (const std::size_t index : members) {
        const core::Real mass = particles.mass_at(index);
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "Halo member mass must be finite and positive");
        }
        const core::Vec3 position{
            positions_x[index], positions_y[index], positions_z[index]};
        const core::Vec3 momentum{
            momenta_x[index], momenta_y[index], momenta_z[index]};
        require_finite_vector(position, "Halo member position");
        require_finite_vector(momentum, "Halo member momentum");
        const long double wide_mass = static_cast<long double>(mass);
        maximum_mass_wide = std::max(maximum_mass_wide, wide_mass);
        total_mass_accumulator.add(mass);
        weighted_momentum[0].add_product(
            wide_mass, static_cast<long double>(momentum.x));
        weighted_momentum[1].add_product(
            wide_mass, static_cast<long double>(momentum.y));
        weighted_momentum[2].add_product(
            wide_mass, static_cast<long double>(momentum.z));
    }

    const core::Real total_mass = total_mass_accumulator.value();
    if (!std::isfinite(total_mass) || total_mass <= 0.0
        || !(maximum_mass_wide > 0.0L)) {
        throw std::overflow_error(
            "Halo derived-property total mass is not finite and positive");
    }
    const long double total_mass_wide = static_cast<long double>(total_mass);
    const long double mean_momentum_wide[3]{
        weighted_momentum[0].value_divided_by_positive(total_mass_wide),
        weighted_momentum[1].value_divided_by_positive(total_mass_wide),
        weighted_momentum[2].value_divided_by_positive(total_mass_wide)};
    const long double mean_velocity_wide[3]{
        divide_by_positive(
            mean_momentum_wide[0], scale_factor_wide,
            "Halo mean velocity x is not representable"),
        divide_by_positive(
            mean_momentum_wide[1], scale_factor_wide,
            "Halo mean velocity y is not representable"),
        divide_by_positive(
            mean_momentum_wide[2], scale_factor_wide,
            "Halo mean velocity z is not representable")};

    // Circular means depend only on relative positive weights. Normalize by the
    // largest member mass before multiplying by sin/cos so a common dimensional
    // mass scale cannot push otherwise meaningful direction terms into
    // underflow. A ratio that itself rounds to zero is too small to influence a
    // center that passes the much coarser binary64 directional-resolution gate.
    PeriodicCenterAccumulator periodic_center;
    for (const std::size_t index : members) {
        const long double normalized_weight =
            static_cast<long double>(particles.mass_at(index))
            / maximum_mass_wide;
        const core::Vec3 position{
            positions_x[index], positions_y[index], positions_z[index]};
        periodic_center.add(position, normalized_weight, box_size);
    }

    HaloDerivedProperties result;
    result.halo_id = halo_id;
    result.particle_count = members.size();
    result.mass = total_mass;
    result.center_of_mass = periodic_center.center(box_size);
    result.mean_momentum = {
        representable_real(mean_momentum_wide[0], "mean momentum x"),
        representable_real(mean_momentum_wide[1], "mean momentum y"),
        representable_real(mean_momentum_wide[2], "mean momentum z")};
    result.mean_velocity = {
        representable_real(mean_velocity_wide[0], "mean velocity x"),
        representable_real(mean_velocity_wide[1], "mean velocity y"),
        representable_real(mean_velocity_wide[2], "mean velocity z")};
    require_finite_vector(result.center_of_mass, "Halo center of mass");
    require_finite_vector(result.mean_momentum, "Halo mean momentum");
    require_finite_vector(result.mean_velocity, "Halo mean velocity");

    WeightedScaledNorm velocity_dispersion[3];
    ScaledSignedSum angular_momentum[3];
    WeightedScaledNorm weighted_radius;
    core::Real max_radius = 0.0;
    for (const std::size_t index : members) {
        const core::Real mass = particles.mass_at(index);
        const core::Vec3 position{
            positions_x[index], positions_y[index], positions_z[index]};
        const long double velocity[3]{
            divide_by_positive(
                static_cast<long double>(momenta_x[index]),
                scale_factor_wide,
                "Halo member velocity x is not representable"),
            divide_by_positive(
                static_cast<long double>(momenta_y[index]),
                scale_factor_wide,
                "Halo member velocity y is not representable"),
            divide_by_positive(
                static_cast<long double>(momenta_z[index]),
                scale_factor_wide,
                "Halo member velocity z is not representable")};
        const long double velocity_offset[3]{
            velocity[0] - mean_velocity_wide[0],
            velocity[1] - mean_velocity_wide[1],
            velocity[2] - mean_velocity_wide[2]};
        if (!math::minimum_image_displacement_is_directionally_unique(
                result.center_of_mass, position, box_size)) {
            throw std::invalid_argument(
                "Halo derived properties cannot unwrap a member on the exact periodic cut locus");
        }
        const core::Vec3 displacement = math::minimum_image_displacement(
            result.center_of_mass,
            position,
            box_size);
        const long double displacement_wide[3]{
            static_cast<long double>(displacement.x),
            static_cast<long double>(displacement.y),
            static_cast<long double>(displacement.z)};
        const long double wide_mass = static_cast<long double>(mass);

        for (int axis = 0; axis < 3; ++axis) {
            velocity_dispersion[axis].add(
                std::abs(velocity_offset[axis]),
                wide_mass,
                total_mass_wide);
        }
        angular_momentum[0].add_weighted_product_difference(
            wide_mass,
            displacement_wide[1], velocity_offset[2],
            displacement_wide[2], velocity_offset[1]);
        angular_momentum[1].add_weighted_product_difference(
            wide_mass,
            displacement_wide[2], velocity_offset[0],
            displacement_wide[0], velocity_offset[2]);
        angular_momentum[2].add_weighted_product_difference(
            wide_mass,
            displacement_wide[0], velocity_offset[1],
            displacement_wide[1], velocity_offset[0]);

        const core::Real radius = core::scale_safe_norm3(
            displacement.x,
            displacement.y,
            displacement.z);
        if (!std::isfinite(radius) || radius < 0.0) {
            throw std::overflow_error("Halo member radius is not finite");
        }
        weighted_radius.add(
            static_cast<long double>(radius),
            wide_mass,
            total_mass_wide);
        max_radius = std::max(max_radius, radius);
    }

    result.velocity_dispersion = {
        velocity_dispersion[0].value("velocity dispersion x"),
        velocity_dispersion[1].value("velocity dispersion y"),
        velocity_dispersion[2].value("velocity dispersion z")};
    result.velocity_dispersion_3d = core::scale_safe_norm3(
        result.velocity_dispersion.x,
        result.velocity_dispersion.y,
        result.velocity_dispersion.z);
    const long double angular_momentum_wide[3]{
        angular_momentum[0].value(),
        angular_momentum[1].value(),
        angular_momentum[2].value()};
    result.angular_momentum_comoving = {
        representable_real(angular_momentum_wide[0], "angular momentum x"),
        representable_real(angular_momentum_wide[1], "angular momentum y"),
        representable_real(angular_momentum_wide[2], "angular momentum z")};
    result.specific_angular_momentum_comoving = {
        representable_real(
            angular_momentum[0].value_divided_by_positive(total_mass_wide),
            "specific angular momentum x"),
        representable_real(
            angular_momentum[1].value_divided_by_positive(total_mass_wide),
            "specific angular momentum y"),
        representable_real(
            angular_momentum[2].value_divided_by_positive(total_mass_wide),
            "specific angular momentum z")};
    result.rms_radius_comoving = weighted_radius.value("RMS radius");
    result.max_radius_comoving = max_radius;

    require_finite_vector(
        result.velocity_dispersion, "Halo velocity dispersion");
    require_finite_vector(
        result.angular_momentum_comoving, "Halo angular momentum");
    require_finite_vector(
        result.specific_angular_momentum_comoving,
        "Halo specific angular momentum");
    if (!std::isfinite(result.velocity_dispersion_3d)
        || !std::isfinite(result.rms_radius_comoving)
        || !std::isfinite(result.max_radius_comoving)) {
        throw std::overflow_error(
            "Halo scalar derived property is not finite");
    }
    return result;
}

} // namespace

HaloDerivedProperties HaloDerivedPropertyAnalyzer::compute(
    const core::ParticleStore& particles,
    const halo::FoFMembership& membership,
    core::Real box_size,
    core::Real scale_factor) {
    return compute_from_membership(
        particles,
        membership.id,
        membership.particle_indices,
        box_size,
        scale_factor);
}

} // namespace cosmo_nbody::analysis
