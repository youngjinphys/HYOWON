// Two-point estimator for a complete uniform periodic cube with r<=L/2.
// The reference is the analytic Euclidean shell measure, not a random catalog.
#pragma once

#include "cosmo_nbody/core/explicit_value.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace cosmo_nbody::analysis {

enum class TwoPointBinning {
    Linear,
    Logarithmic,
};

using ExplicitTwoPointBinning = core::ExplicitValue<TwoPointBinning>;

struct TwoPointOptions {
    core::Real min_radius{0.0};
    core::Real max_radius{0.0};
    int num_bins{0};
    // Both binning modes are valid; omission is a distinct invalid state.
    ExplicitTwoPointBinning binning{};
    std::string tracer_label;
};

struct TwoPointBin {
    core::Real radius_low{0.0};
    core::Real radius_high{0.0};
    core::Real radius_midpoint{0.0};

    std::size_t data_data_pair_count{0};
    // Retained output fields: no random pairs are measured by this estimator.
    std::size_t data_random_pair_count{0};
    std::size_t random_random_pair_count{0};

    core::Real data_data_normalized{0.0};
    core::Real data_random_normalized{0.0};
    core::Real random_random_normalized{0.0};

    core::Real expected_pair_probability{0.0};
    core::Real xi{0.0};
};

struct TwoPointResult {
    core::Real box_size{0.0};
    std::size_t input_point_count{0};
    std::size_t used_point_count{0};
    std::size_t random_point_count{0};

    std::size_t data_data_pair_count_in_range{0};
    std::size_t data_random_pair_count_in_range{0};
    std::size_t random_random_pair_count_in_range{0};
    std::size_t pair_count_in_range{0};

    std::string random_catalog{"none"};
    std::string tracer_label;
    TwoPointBinning binning{TwoPointBinning::Linear};
    std::string estimator{"periodic_analytic_shell"};
    std::string reference_measure{
        "exact_uniform_periodic_shell_rmax_le_half_box"};
    std::string bin_convention{"lower_inclusive_upper_exclusive"};
    std::vector<TwoPointBin> bins;
};

class TwoPointCorrelation {
public:
    // Uniform periodic cube: p_shell=4*pi*(r_hi^3-r_lo^3)/(3L^3), xi=DD/p_shell-1.
    static TwoPointResult compute(
        std::span<const core::Vec3> points,
        core::Real box_size,
        const TwoPointOptions& options);
};

} // namespace cosmo_nbody::analysis
