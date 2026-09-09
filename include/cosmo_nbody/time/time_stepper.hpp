#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace cosmo_nbody {
namespace time {

class TimeStep final {
public:
    // A timestep is defined only by two distinct, finite, positive endpoints.
    // The canonical log-scale midpoint is derived here so callers cannot inject
    // a midpoint inconsistent with the KDK cadence.
    TimeStep(core::Real a_begin, core::Real a_end);

    core::Real a_begin() const noexcept { return a_begin_; }
    core::Real a_half() const noexcept { return a_half_; }
    core::Real a_end() const noexcept { return a_end_; }

private:
    core::Real a_begin_;
    core::Real a_half_;
    core::Real a_end_;
};

class TimeStepper {
public:
    // Builds an exact step-boundary sequence. Each requested stop scale factor
    // and a_final are exact boundaries; ordinary cadence uses delta_ln_a in
    // log(a), with the last step before each stop shortened as necessary.
    // If rounding leaves an ordinary boundary immediately before a stop with
    // no distinct KDK midpoint between them, use the stop instead of that
    // ordinary boundary. Explicit stops are never coalesced. This permits a
    // representability-sized excess over the nominal cadence, not a physical
    // timestep tolerance or a change to the requested output epoch.
    TimeStepper(core::Real a_start,
                core::Real a_final,
                core::Real delta_ln_a,
                std::vector<core::Real> stop_scale_factors = {});

    // Conservative number of stored scale-factor boundaries for the same
    // construction rules, including a_start, exact stops, and a_final. Used by
    // resource admission so the O(N_step) table is charged before allocation.
    static std::uint64_t planned_boundary_count(
        core::Real a_start,
        core::Real a_final,
        core::Real delta_ln_a,
        std::span<const core::Real> stop_scale_factors = {});

    static std::uint64_t planned_boundary_storage_bytes(
        core::Real a_start,
        core::Real a_final,
        core::Real delta_ln_a,
        std::span<const core::Real> stop_scale_factors = {});

    // No invalid sentinel TimeStep is needed: exhaustion is represented by an
    // empty optional and every returned object already satisfies its invariant.
    std::optional<TimeStep> next_step();
    void reverse_direction();

    // Restore a step-boundary state. step_idx is the number of completed
    // forward steps and must index the precomputed boundary sequence.
    void restore_state(std::size_t step_idx);
    void restore_state(std::size_t step_idx, core::Real current_a);

    core::Real current_a() const { return current_a_; }
    core::Real delta_ln_a() const { return delta_ln_a_; }
    std::size_t current_step() const { return current_step_idx_; }
    std::size_t num_steps() const { return num_steps_; }
    bool is_finished() const { return finished_; }

    const std::vector<core::Real>& boundaries() const { return boundaries_; }

private:
    core::Real current_a_;
    core::Real delta_ln_a_;

    std::size_t current_step_idx_{0};
    std::size_t num_steps_{0};
    bool finished_{false};
    int direction_{1};

    std::vector<core::Real> boundaries_;
};

} // namespace time
} // namespace cosmo_nbody
