#include "cosmo_nbody/validation/force_energy_work.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::validation {
namespace {

core::Real checked_real(long double value, const char* context) {
    if (!std::isfinite(value)
        || value > static_cast<long double>(std::numeric_limits<core::Real>::max())
        || value < -static_cast<long double>(std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(std::string(context) + " is not representable");
    }
    const core::Real result = static_cast<core::Real>(value);
    if (value != 0.0L && result == core::Real{0.0}) {
        throw std::underflow_error(std::string(context) + " underflowed core::Real");
    }
    return result;
}

// Retain the low part through signed cancellation instead of assuming that
// long double alone can resolve every separation between binary64 inputs.
long double compensated_sum_three(
    long double first, long double second, long double third) {
    long double sum = first;
    long double correction = 0.0L;
    for (const long double term : {second, third}) {
        const long double next = sum + term;
        correction += std::abs(sum) >= std::abs(term)
            ? (sum - next) + term
            : (term - next) + sum;
        sum = next;
    }
    return sum + correction;
}

void validate_point(const ForceEnergyWorkPoint& point) {
    if (!std::isfinite(point.scale_factor) || point.scale_factor <= 0.0
        || !std::isfinite(point.hubble_rate) || point.hubble_rate <= 0.0) {
        throw std::invalid_argument(
            "Force-energy work point requires finite positive scale factor and Hubble rate");
    }
    for (const core::Real value : {
             point.momentum_force_contraction,
             point.directional_potential_energy_comoving,
             point.directional_cic_self_energy_comoving,
             point.power_defect,
             point.power_defect_over_hubble}) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Force-energy work point contains a non-finite measurement");
        }
    }
}

} // namespace

ForceEnergyWorkPoint make_force_energy_work_point(
    core::Real scale_factor,
    core::Real hubble_rate,
    core::Real momentum_force_contraction,
    core::Real directional_potential_energy_comoving,
    core::Real directional_cic_self_energy_comoving) {
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0
        || !std::isfinite(hubble_rate) || hubble_rate <= 0.0) {
        throw std::invalid_argument(
            "Force-energy work measurement requires finite positive scale factor and Hubble rate");
    }
    for (const core::Real value : {
             momentum_force_contraction,
             directional_potential_energy_comoving,
             directional_cic_self_energy_comoving}) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Force-energy work measurement inputs must be finite");
        }
    }

    const long double numerator = compensated_sum_three(
        static_cast<long double>(momentum_force_contraction),
        static_cast<long double>(directional_potential_energy_comoving),
        -static_cast<long double>(directional_cic_self_energy_comoving));
    const long double a = static_cast<long double>(scale_factor);
    const long double power = numerator / (a * a * a);
    const core::Real power_defect = checked_real(
        power, "Force-energy power defect");
    const core::Real power_over_hubble = checked_real(
        power / static_cast<long double>(hubble_rate),
        "Force-energy power defect divided by H");

    ForceEnergyWorkPoint point{
        scale_factor,
        hubble_rate,
        momentum_force_contraction,
        directional_potential_energy_comoving,
        directional_cic_self_energy_comoving,
        power_defect,
        power_over_hubble};
    validate_point(point);
    return point;
}

void reset_force_energy_work_state(
    const ForceEnergyWorkPoint& initial,
    ForceEnergyWorkState& state) {
    validate_point(initial);
    state = ForceEnergyWorkState{};
    state.initialized = true;
    state.previous = initial;
}

core::Real force_energy_integrated_work(
    const ForceEnergyWorkState& state) {
    if (!state.initialized
        || !std::isfinite(state.integrated_work)
        || !std::isfinite(state.integrated_work_compensation)) {
        throw std::logic_error(
            "Force-energy work state is not initialized with finite accumulators");
    }
    return checked_real(
        static_cast<long double>(state.integrated_work)
            - static_cast<long double>(state.integrated_work_compensation),
        "Force-energy integrated work");
}

core::Real update_force_energy_work_state(
    const ForceEnergyWorkPoint& current,
    core::Real delta_ln_a,
    ForceEnergyWorkState& state) {
    validate_point(current);
    if (!state.initialized) {
        throw std::logic_error(
            "Force-energy work state must be reset before update");
    }
    validate_point(state.previous);
    if (!std::isfinite(delta_ln_a) || delta_ln_a <= 0.0
        || !(current.scale_factor > state.previous.scale_factor)) {
        throw std::invalid_argument(
            "Force-energy work update requires a finite positive increasing ln(a) interval");
    }

    const long double expected_delta =
        std::log(static_cast<long double>(current.scale_factor))
        - std::log(static_cast<long double>(state.previous.scale_factor));
    const long double supplied_delta = static_cast<long double>(delta_ln_a);
    const long double scale = std::max({
        1.0L, std::abs(expected_delta), std::abs(supplied_delta)});
    if (std::abs(expected_delta - supplied_delta)
        > 512.0L
            * static_cast<long double>(std::numeric_limits<core::Real>::epsilon())
            * scale) {
        throw std::invalid_argument(
            "Force-energy work delta_ln_a disagrees with endpoint scale factors");
    }

    (void)force_energy_integrated_work(state);
    const long double previous_sum =
        static_cast<long double>(state.integrated_work);
    const long double increment = 0.5L * supplied_delta
        * (static_cast<long double>(state.previous.power_defect_over_hubble)
           + static_cast<long double>(current.power_defect_over_hubble));
    const core::Real updated = checked_real(
        previous_sum + increment,
        "Force-energy integrated work update");
    // The represented total is sum - compensation. Never fold the old low
    // component into the large sum before adding: that can erase it even in
    // extended precision. Accumulate the new rounding error separately.
    const long double rounded_sum = static_cast<long double>(updated);
    const long double rounding_error =
        std::abs(previous_sum) >= std::abs(increment)
        ? (previous_sum - rounded_sum) + increment
        : (increment - rounded_sum) + previous_sum;
    const core::Real compensation = checked_real(
        static_cast<long double>(state.integrated_work_compensation)
            - rounding_error,
        "Force-energy integrated work compensation");

    ForceEnergyWorkState candidate = state;
    candidate.previous = current;
    candidate.integrated_work = updated;
    candidate.integrated_work_compensation = compensation;
    const core::Real result = force_energy_integrated_work(candidate);
    state = candidate;
    return result;
}

core::Real force_energy_closure_residual(
    core::Real layzer_irvine_residual,
    const ForceEnergyWorkState& state) {
    if (!std::isfinite(layzer_irvine_residual)) {
        throw std::invalid_argument(
            "Force-energy closure requires a finite Layzer-Irvine residual");
    }
    (void)force_energy_integrated_work(state);
    // Subtract the two-component work before narrowing the residual. Rounding
    // the integrated work first can turn a resolved closure remainder into 0.
    return checked_real(
        compensated_sum_three(
            static_cast<long double>(layzer_irvine_residual),
            -static_cast<long double>(state.integrated_work),
            static_cast<long double>(state.integrated_work_compensation)),
        "Force-energy closure residual");
}

} // namespace cosmo_nbody::validation
