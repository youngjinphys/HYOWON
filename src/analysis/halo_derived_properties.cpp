#include "cosmo_nbody/analysis/halo_derived_properties.hpp"

#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "membership_validation.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmo_nbody::analysis {
namespace {

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
        lhs_a_mantissa, lhs_exponent - common_exponent);
    const long double scaled_rhs_a = std::scalbn(
        rhs_a_mantissa, rhs_exponent - common_exponent);

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

// Integer units avoid both cancellation in Q-2*c*S+c*c*W and rounded
// comparisons between equal minima. Common powers of two are removed before
// arithmetic, so ordinary equal-mass samples use small integers. No precision
// cutoff or empirical ambiguity threshold is part of the center definition.
using CenterInteger = boost::multiprecision::cpp_int;

int binary64_integer_low_bit(core::Real value) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent = static_cast<int>((bits >> 52U) & 0x7ffU);
    auto significand = bits & ((std::uint64_t{1} << 52U) - 1U);
    if (exponent != 0) significand |= std::uint64_t{1} << 52U;
    if (significand == 0) return std::numeric_limits<int>::max();
    return (exponent == 0 ? 0 : exponent - 1)
        + static_cast<int>(std::countr_zero(significand));
}

CenterInteger binary64_integer_units(core::Real value, int removed_bits) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent = static_cast<int>((bits >> 52U) & 0x7ffU);
    auto significand = bits & ((std::uint64_t{1} << 52U) - 1U);
    if (exponent != 0) significand |= std::uint64_t{1} << 52U;
    CenterInteger result = significand;
    const int shift = (exponent == 0 ? 0 : exponent - 1) - removed_bits;
    if (shift >= 0) result <<= shift;
    else result >>= -shift; // Only common, exactly zero low bits are removed.
    return result;
}

// Round (numerator/denominator)*2^quantum to nearest binary64, ties to even.
// Integer division supplies the exact remainder, including subnormal and zero
// results. The final scalbn merely assembles an already representable value.
core::Real round_center_rational(
    const CenterInteger& numerator,
    const CenterInteger& denominator,
    int quantum) {
    if (numerator == 0) return 0.0;
    int ratio_exponent = static_cast<int>(boost::multiprecision::msb(numerator))
        - static_cast<int>(boost::multiprecision::msb(denominator));
    if (ratio_exponent >= 0) {
        if (numerator < (denominator << ratio_exponent)) --ratio_exponent;
    } else if ((numerator << -ratio_exponent) < denominator) {
        --ratio_exponent;
    }
    const int spacing_exponent = std::max(ratio_exponent + quantum - 52, -1074);
    CenterInteger scaled_numerator = numerator;
    CenterInteger scaled_denominator = denominator;
    const int shift = quantum - spacing_exponent;
    if (shift >= 0) scaled_numerator <<= shift;
    else scaled_denominator <<= -shift;
    CenterInteger significand = scaled_numerator / scaled_denominator;
    const CenterInteger remainder = scaled_numerator % scaled_denominator;
    const CenterInteger twice_remainder = remainder << 1;
    if (twice_remainder > scaled_denominator
        || (twice_remainder == scaled_denominator
            && static_cast<bool>(significand & 1))) {
        ++significand;
    }
    const auto integer_significand = significand.convert_to<std::uint64_t>();
    const core::Real result = std::scalbn(
        static_cast<core::Real>(integer_significand), spacing_exponent);
    if (!std::isfinite(result)) {
        throw std::overflow_error("Halo periodic intrinsic mass center is not representable");
    }
    return result;
}

struct PeriodicIntrinsicCenterAccumulator {
    struct Sample {
        core::Vec3 position{};
        core::Real weight{0.0};
    };
    struct Event {
        CenterInteger breakpoint;
        CenterInteger weight;
        CenterInteger initial_lift;
    };
    std::vector<Sample> samples;

    void add(const core::Vec3& position, core::Real weight, core::Real box_size) {
        if (!std::isfinite(weight) || weight <= 0.0) {
            throw std::invalid_argument(
                "Halo periodic intrinsic center requires finite positive weights");
        }
        const core::Vec3 wrapped = math::wrap(position, box_size);
        if (!std::isfinite(wrapped.x) || !std::isfinite(wrapped.y)
            || !std::isfinite(wrapped.z)) {
            throw std::invalid_argument(
                "Halo periodic intrinsic center requires finite wrapped positions");
        }
        samples.push_back({wrapped, weight});
    }

    core::Real axis(int axis_index, core::Real box_size) const {
        if (axis_index < 0 || axis_index >= 3 || samples.empty()) {
            throw std::invalid_argument(
                "Halo periodic intrinsic center requires a populated valid axis");
        }
        if (!std::isfinite(box_size) || !(box_size > 0.0)) {
            throw std::invalid_argument(
                "Halo periodic intrinsic center requires a finite positive box");
        }
        const auto coordinate_of = [axis_index](const Sample& sample) {
            return axis_index == 0 ? sample.position.x
                : axis_index == 1 ? sample.position.y : sample.position.z;
        };
        int weight_shift = std::numeric_limits<int>::max();
        // Doubled coordinates start in 2^-1075 physical units. Keep at least
        // one factor of two in the integer box so every antipode is integral.
        int coordinate_shift = binary64_integer_low_bit(box_size);
        for (const Sample& sample : samples) {
            weight_shift = std::min(weight_shift, binary64_integer_low_bit(sample.weight));
            const core::Real coordinate = coordinate_of(sample);
            if (coordinate != 0.0) {
                coordinate_shift = std::min(
                    coordinate_shift, binary64_integer_low_bit(coordinate) + 1);
            }
        }
        const CenterInteger box = binary64_integer_units(box_size, coordinate_shift - 1);
        const CenterInteger half_box = box >> 1;
        CenterInteger total_weight = 0;
        CenterInteger weighted_coordinate = 0;
        CenterInteger weighted_square = 0;
        std::vector<Event> events;
        events.reserve(samples.size());
        for (const Sample& sample : samples) {
            CenterInteger weight = binary64_integer_units(sample.weight, weight_shift);
            const CenterInteger coordinate = binary64_integer_units(
                coordinate_of(sample), coordinate_shift - 1);
            CenterInteger lift = coordinate;
            if (coordinate > half_box) lift -= box;
            total_weight += weight;
            weighted_coordinate += weight * lift;
            weighted_square += weight * lift * lift;
            CenterInteger breakpoint = coordinate + half_box;
            if (breakpoint >= box) breakpoint -= box;
            // An antipode at zero already has the right-hand +half_box lift.
            if (breakpoint != 0) {
                events.push_back({std::move(breakpoint), std::move(weight), std::move(lift)});
            }
        }
        std::sort(events.begin(), events.end(), [](const Event& lhs, const Event& rhs) {
            return lhs.breakpoint < rhs.breakpoint;
        });

        bool have_best = false;
        bool non_unique = false;
        CenterInteger best_objective;
        CenterInteger best_numerator;
        const auto consider_interval = [&](const CenterInteger& left, const CenterInteger& right) {
            // At each antipode the derivative has a strictly negative jump;
            // such a corner cannot be a local minimum. Every global minimum
            // is therefore a stationary quadratic mean S/W in its interval.
            // Closed endpoints also include the identical zero/box candidate.
            if (weighted_coordinate < total_weight * left
                || weighted_coordinate > total_weight * right) return;
            CenterInteger objective = total_weight * weighted_square
                - weighted_coordinate * weighted_coordinate;
            if (objective < 0) {
                throw std::logic_error("Exact intrinsic-center objective violated nonnegativity");
            }
            CenterInteger canonical_numerator = weighted_coordinate;
            if (canonical_numerator == total_weight * box) canonical_numerator = 0;
            if (!have_best || objective < best_objective) {
                have_best = true;
                non_unique = false;
                best_objective = std::move(objective);
                best_numerator = std::move(canonical_numerator);
            } else if (objective == best_objective && canonical_numerator != best_numerator) {
                non_unique = true;
            }
        };
        CenterInteger interval_left = 0;
        std::size_t event_index = 0;
        while (event_index < events.size()) {
            const CenterInteger& breakpoint = events[event_index].breakpoint;
            consider_interval(interval_left, breakpoint);
            std::size_t next_event = event_index;
            do {
                const Event& event = events[next_event];
                weighted_coordinate += event.weight * box;
                weighted_square += event.weight * box * (2 * event.initial_lift + box);
                ++next_event;
            } while (next_event < events.size() && events[next_event].breakpoint == breakpoint);
            interval_left = breakpoint;
            event_index = next_event;
        }
        consider_interval(interval_left, box);
        if (!have_best) {
            throw std::logic_error("Halo periodic intrinsic-center scan produced no candidate");
        }
        if (non_unique) {
            throw std::invalid_argument("Halo periodic intrinsic mass center is non-unique on this axis");
        }
        return math::wrap(round_center_rational(
            best_numerator, total_weight, coordinate_shift - 1075), box_size);
    }

    core::Vec3 center(core::Real box_size) const {
        return {axis(0, box_size), axis(1, box_size), axis(2, box_size)};
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
        total_mass_accumulator.add(mass);
        weighted_momentum[0].add_product(
            wide_mass, static_cast<long double>(momentum.x));
        weighted_momentum[1].add_product(
            wide_mass, static_cast<long double>(momentum.y));
        weighted_momentum[2].add_product(
            wide_mass, static_cast<long double>(momentum.z));
    }

    const core::Real total_mass = total_mass_accumulator.value();
    if (!std::isfinite(total_mass) || total_mass <= 0.0) {
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

    // The periodic mass center is the coordinate-wise intrinsic (Fréchet) mean
    // on the cubic torus: it minimizes the mass-weighted sum of squared periodic
    // geodesic distances. Each axis is piecewise quadratic between antipodes, so
    // sorting those breakpoints gives the global minimum without an empirical
    // resultant-length threshold or the chordal bias of the circular mean.
    PeriodicIntrinsicCenterAccumulator periodic_center;
    for (const std::size_t index : members) {
        const core::Vec3 position{
            positions_x[index], positions_y[index], positions_z[index]};
        periodic_center.add(
            position,
            particles.mass_at(index),
            box_size);
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
