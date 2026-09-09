#pragma once

#include "cosmo_nbody/core/explicit_boolean.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

struct HaloShapeOptions {
    // Tensor choice is explicit; false is valid, while zero iteration/tolerance is invalid.
    core::ExplicitBoolean use_reduced_tensor{};
    int max_iterations{0};
    core::Real convergence_tolerance{0.0};
};

struct HaloShapeResult {
    core::Real lambda_a{0.0};
    core::Real lambda_b{0.0};
    core::Real lambda_c{0.0};

    core::Vec3 major_axis{0.0, 0.0, 0.0};
    core::Vec3 intermediate_axis{0.0, 0.0, 0.0};
    core::Vec3 minor_axis{0.0, 0.0, 0.0};

    core::Real axis_ratio_b_over_a{0.0};
    core::Real axis_ratio_c_over_a{0.0};
    // Triaxiality is undefined for exactly spherical represented tensors; use the flag.
    core::Real triaxiality{0.0};
    std::size_t particle_count{0};
    std::size_t directional_particle_count{0};
    bool converged{false};
    bool rank_deficient_reduced_metric{false};
    bool solver_indeterminate{false};
    bool periodic_cut_locus_ambiguous{false};
    bool zero_spatial_extent{false};
    int iterations{0};

    bool computed() const noexcept {
        return iterations > 0
            && !solver_indeterminate
            && !periodic_cut_locus_ambiguous
            && !zero_spatial_extent;
    }
    bool triaxiality_defined() const noexcept {
        return computed() && axis_ratio_c_over_a != 1.0;
    }
};

class HaloShapeAnalyzer {
public:
    static HaloShapeResult compute(
        const core::ParticleStore& particles,
        core::Vec3 center,
        core::Real box_size,
        std::span<const std::size_t> member_indices,
        const HaloShapeOptions& options);

private:
    static HaloShapeResult compute_once(
        const core::ParticleStore& particles,
        core::Vec3 center,
        core::Real box_size,
        std::span<const std::size_t> member_indices,
        core::Vec3 major_axis,
        core::Vec3 intermediate_axis,
        core::Vec3 minor_axis,
        core::Real q,
        core::Real s,
        const HaloShapeOptions& options);
};

} // namespace analysis
} // namespace cosmo_nbody
