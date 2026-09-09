#include "cosmo_nbody/time/time_stepper.hpp"

#include "cosmo_nbody/math/exact_geometric_mean.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cosmo_nbody {
namespace time {

namespace {

core::Real stable_positive_log_ratio(core::Real upper, core::Real lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)
        || !(upper > lower) || lower <= 0.0) {
        throw std::invalid_argument(
            "TimeStepper logarithmic interval must be finite and positive");
    }
    const core::Real relative = (upper - lower) / lower;
    const core::Real value = std::isfinite(relative)
        ? std::log1p(relative)
        : std::log(upper) - std::log(lower);
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::overflow_error(
            "TimeStepper logarithmic interval is not representable");
    }
    return value;
}

core::Real conservative_step_count_real(
    core::Real span,
    core::Real delta_ln_a) {
    const core::Real ratio = span / delta_ln_a;
    if (!std::isfinite(ratio) || ratio <= 0.0) {
        throw std::overflow_error(
            "TimeStepper logarithmic step count is not representable");
    }
    // A true ratio just above an integer can round down to that integer.
    // Bias the candidate-count upper bound by one representable value; exact
    // cadence boundaries are decided during materialization below.
    const core::Real upper_ratio = std::nextafter(
        ratio, std::numeric_limits<core::Real>::infinity());
    const core::Real count = std::ceil(upper_ratio);
    if (!std::isfinite(count)) {
        throw std::overflow_error(
            "TimeStepper logarithmic step count is not representable");
    }
    return std::max(core::Real{1.0}, count);
}

core::Real advance_logarithmically(core::Real current, core::Real delta_ln_a) {
    const core::Real relative_increment = std::expm1(delta_ln_a);
    const core::Real next = std::isfinite(relative_increment)
        ? std::fma(current, relative_increment, current)
        : std::exp(std::log(current) + delta_ln_a);
    if (!std::isfinite(next)) {
        throw std::overflow_error(
            "TimeStepper scale-factor boundary is not representable");
    }
    return next;
}

core::Real stable_geometric_mean(core::Real lhs, core::Real rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)
        || lhs <= 0.0 || rhs <= 0.0 || lhs == rhs) {
        throw std::invalid_argument(
            "TimeStep endpoints must be finite, positive, and distinct");
    }
    const core::Real lower = std::min(lhs, rhs);
    const core::Real upper = std::max(lhs, rhs);
    const core::Real result = math::exact_geometric_mean(lower, upper);
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::overflow_error(
            "TimeStep half-step scale factor is not representable");
    }
    // KDK needs two non-empty kick intervals. Reject endpoint-rounded geometric
    // means as a representation limit, not as a timestep-accuracy threshold.
    if (!(result > lower) || !(result < upper)) {
        throw std::underflow_error(
            "TimeStep has no distinct representable geometric half-step");
    }
    return result;
}

std::vector<core::Real> normalized_stops(
    core::Real a_start,
    core::Real a_final,
    std::vector<core::Real> stop_scale_factors) {
    core::Real previous_stop = a_start;
    for (const core::Real stop : stop_scale_factors) {
        if (!std::isfinite(stop) || stop <= a_start || stop > a_final) {
            throw std::invalid_argument(
                "TimeStepper stop scale factors must be in (a_start, a_final]");
        }
        if (!(stop > previous_stop)) {
            throw std::invalid_argument(
                "TimeStepper stop scale factors must be strictly increasing");
        }
        previous_stop = stop;
    }
    if (stop_scale_factors.empty()
        || stop_scale_factors.back() != a_final) {
        stop_scale_factors.push_back(a_final);
    }
    return stop_scale_factors;
}

void require_schedule_inputs(
    core::Real a_start,
    core::Real a_final,
    core::Real delta_ln_a) {
    if (!std::isfinite(a_start) || !std::isfinite(a_final)
        || !std::isfinite(delta_ln_a)) {
        throw std::invalid_argument("TimeStepper inputs must be finite");
    }
    if (a_start <= 0.0 || a_final <= 0.0) {
        throw std::invalid_argument("Scale factors must be positive");
    }
    if (delta_ln_a <= 0.0) {
        throw std::invalid_argument("delta_ln_a must be positive");
    }
    if (a_start > a_final) {
        throw std::invalid_argument(
            "TimeStepper production schedule must be non-descending");
    }
}

std::uint64_t segment_step_ceiling(
    core::Real stop,
    core::Real anchor,
    core::Real delta_ln_a) {
    const core::Real span = stable_positive_log_ratio(stop, anchor);
    const core::Real count_real = conservative_step_count_real(
        span, delta_ln_a);
    if (count_real >= static_cast<core::Real>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::length_error(
            "TimeStepper logarithmic step count exceeds boundary capacity");
    }
    return static_cast<std::uint64_t>(count_real);
}

} // namespace

TimeStep::TimeStep(core::Real a_begin, core::Real a_end)
    : a_begin_(a_begin),
      a_half_(stable_geometric_mean(a_begin, a_end)),
      a_end_(a_end) {}

std::uint64_t TimeStepper::planned_boundary_count(
    core::Real a_start,
    core::Real a_final,
    core::Real delta_ln_a,
    std::span<const core::Real> stop_scale_factors) {
    require_schedule_inputs(a_start, a_final, delta_ln_a);
    if (a_start == a_final) return 1;
    const std::vector<core::Real> stops = normalized_stops(
        a_start,
        a_final,
        std::vector<core::Real>(
            stop_scale_factors.begin(), stop_scale_factors.end()));
    std::uint64_t total = 1;
    core::Real anchor = a_start;
    for (const core::Real stop : stops) {
        const std::uint64_t count = segment_step_ceiling(
            stop, anchor, delta_ln_a);
        if (count > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::length_error(
                "TimeStepper logarithmic step count exceeds boundary capacity");
        }
        total += count;
        anchor = stop;
    }
    return total;
}

std::uint64_t TimeStepper::planned_boundary_storage_bytes(
    core::Real a_start,
    core::Real a_final,
    core::Real delta_ln_a,
    std::span<const core::Real> stop_scale_factors) {
    const std::uint64_t count = planned_boundary_count(
        a_start, a_final, delta_ln_a, stop_scale_factors);
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(core::Real)) {
        throw std::length_error(
            "TimeStepper boundary table exceeds addressable bytes");
    }
    return count * sizeof(core::Real);
}

TimeStepper::TimeStepper(core::Real a_start,
                         core::Real a_final,
                         core::Real delta_ln_a,
    std::vector<core::Real> stop_scale_factors)
    : current_a_(a_start),
      delta_ln_a_(std::abs(delta_ln_a)) {
    require_schedule_inputs(a_start, a_final, delta_ln_a);
    if (a_start == a_final) {
        finished_ = true;
        boundaries_ = {a_start};
        return;
    }

    const std::uint64_t planned = planned_boundary_count(
        a_start, a_final, delta_ln_a_, stop_scale_factors);
    if (planned > boundaries_.max_size()) {
        throw std::length_error(
            "TimeStepper logarithmic step count exceeds boundary capacity");
    }
    boundaries_.reserve(static_cast<std::size_t>(planned));
    stop_scale_factors = normalized_stops(
        a_start, a_final, std::move(stop_scale_factors));

    boundaries_.push_back(a_start);
    const auto append_executable_boundary = [this](core::Real next) {
        if (!(next > boundaries_.back()) || !std::isfinite(next)) {
            throw std::underflow_error(
                "TimeStepper boundary does not advance monotonically");
        }
        // Reject schedules whose stored boundary cannot form a KDK half-step,
        // before any trajectory state is mutated.
        (void)stable_geometric_mean(boundaries_.back(), next);
        boundaries_.push_back(next);
    };

    for (const core::Real stop : stop_scale_factors) {
        const core::Real anchor = boundaries_.back();
        const core::Real span = stable_positive_log_ratio(stop, anchor);
        // Candidate count is conservative; exact-stop placement is decided from
        // cadence boundaries generated directly from the segment anchor, not
        // from ratios of independently rounded boundaries.
        const core::Real count_real = conservative_step_count_real(
            span, delta_ln_a_);
        const std::size_t remaining_capacity =
            boundaries_.max_size() - boundaries_.size();
        if (count_real >= static_cast<core::Real>(remaining_capacity)) {
            throw std::length_error(
                "TimeStepper logarithmic step count exceeds boundary capacity");
        }
        const auto count = static_cast<std::size_t>(count_real);
        bool stop_appended = false;
        for (std::size_t k = 1; k < count; ++k) {
            const core::Real offset = delta_ln_a_
                * static_cast<core::Real>(k);
            if (!std::isfinite(offset)) {
                throw std::overflow_error(
                    "TimeStepper logarithmic boundary offset is not representable");
            }
            const core::Real next = advance_logarithmically(anchor, offset);
            if (next >= stop) {
                append_executable_boundary(stop);
                stop_appended = true;
                break;
            }
            // If rounding leaves an ordinary cadence immediately below a stop
            // with no representable midpoint, prefer the stop. Resolvable tails
            // remain distinct; adjacent requested stops are still rejected.
            const core::Real tail_half = math::exact_geometric_mean(next, stop);
            if (!(tail_half > next && tail_half < stop)) {
                append_executable_boundary(stop);
                stop_appended = true;
                break;
            }
            append_executable_boundary(next);
        }
        if (!stop_appended) {
            append_executable_boundary(stop);
        }
    }

    num_steps_ = boundaries_.size() - 1;
    finished_ = num_steps_ == 0;
}

std::optional<TimeStep> TimeStepper::next_step() {
    if (finished_) return std::nullopt;

    if (direction_ == 1) {
        if (current_step_idx_ >= num_steps_) {
            finished_ = true;
            return std::nullopt;
        }

        TimeStep step(
            boundaries_[current_step_idx_],
            boundaries_[current_step_idx_ + 1]);

        ++current_step_idx_;
        current_a_ = step.a_end();
        if (current_step_idx_ == num_steps_) finished_ = true;
        return step;
    }

    if (current_step_idx_ == 0) {
        finished_ = true;
        return std::nullopt;
    }

    TimeStep step(
        boundaries_[current_step_idx_],
        boundaries_[current_step_idx_ - 1]);

    --current_step_idx_;
    current_a_ = step.a_end();
    if (current_step_idx_ == 0) finished_ = true;
    return step;
}

void TimeStepper::reverse_direction() {
    direction_ = -direction_;
    finished_ = false;
}

void TimeStepper::restore_state(std::size_t step_index) {
    if (step_index > num_steps_) {
        throw std::invalid_argument("TimeStepper restart step exceeds boundary table");
    }
    current_step_idx_ = step_index;
    current_a_ = boundaries_[step_index];
    finished_ = direction_ == 1
        ? current_step_idx_ == num_steps_
        : current_step_idx_ == 0;
}

void TimeStepper::restore_state(std::size_t step_index, core::Real scale_factor) {
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "TimeStepper restart scale factor must be finite and positive");
    }
    if (step_index > num_steps_) {
        throw std::invalid_argument("TimeStepper restart step exceeds boundary table");
    }
    const core::Real expected = boundaries_[step_index];
    if (scale_factor != expected) {
        throw std::invalid_argument(
            "TimeStepper restart scale factor does not match the deterministic boundary");
    }
    restore_state(step_index);
}

} // namespace time
} // namespace cosmo_nbody
