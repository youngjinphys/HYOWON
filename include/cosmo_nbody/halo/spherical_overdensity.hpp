#pragma once

#include "cosmo_nbody/cosmology/flat_matter_lambda.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace halo {

enum class SOReferenceDensity {
    Unknown,
    BoxMeanMatter,
    MeanMatter,
    Critical,
    VirialCritical
};

std::string_view so_reference_density_name(SOReferenceDensity reference) noexcept;

struct SOResourcePolicy {
    core::Real memory_budget_gib{0.0};
};

struct SOContext {
    core::Real box_size{0.0};
    core::Real omega_m{0.0};
    core::Real omega_lambda{0.0};
    std::uint64_t requested_threads{1};
    SOResourcePolicy memory_policy{};
};

struct SOHalo {
    std::size_t id{0};
    core::Vec3 center{0.0, 0.0, 0.0};
    core::Real radius{0.0};
    core::Real mass{0.0};
    std::size_t particle_count{0};
    core::Real overdensity_threshold{0.0};
    core::Real reference_density{0.0};
    SOReferenceDensity reference_density_kind{SOReferenceDensity::Unknown};
};

struct SOExecutionPlan {
    std::size_t particle_count{0};
    std::size_t center_count{0};
    std::size_t requested_workers{0};
    std::size_t worker_count{0};
    std::size_t aggregate_scratch_cap_bytes{0};
    std::size_t center_staging_bytes{0};
    std::size_t output_staging_bytes{0};
    std::size_t candidate_bytes_per_worker{0};
    std::size_t query_axis_bytes_per_worker{0};
    std::size_t worker_scratch_bytes_per_worker{0};
    std::size_t candidate_scratch_cap_bytes{0};
    std::size_t estimated_candidate_peak_bytes{0};
    std::size_t estimated_query_axis_peak_bytes{0};
    std::size_t spatial_index_cap_bytes{0};
    std::size_t spatial_index_bytes{0};
    std::size_t estimated_worker_phase_peak_bytes{0};
    std::size_t estimated_output_phase_peak_bytes{0};
    std::size_t estimated_total_peak_bytes{0};
    std::size_t spatial_cells_per_dimension{0};
};

class SphericalOverdensityFinder {
public:
    // Fixed-threshold SO requires explicit Delta and reference density.
    explicit SphericalOverdensityFinder(
        SOContext context,
        core::Real overdensity_threshold,
        SOReferenceDensity reference_density);

    // Dynamic Bryan-Norman threshold is derived from Omega_m(a) at each epoch.
    explicit SphericalOverdensityFinder(
        SOContext context,
        SOReferenceDensity dynamic_reference_density)
        : context_(std::move(context)),
          overdensity_threshold_(0.0),
          reference_density_(dynamic_reference_density) {
        if (dynamic_reference_density != SOReferenceDensity::VirialCritical) {
            throw std::invalid_argument(
                "Dynamic SO constructor is reserved for VirialCritical");
        }
        if (!std::isfinite(context_.box_size) || context_.box_size <= 0.0
            || !std::isfinite(context_.omega_m)
            || !std::isfinite(context_.omega_lambda)
            || context_.omega_m <= 0.0
            || context_.omega_lambda < 0.0
            || !cosmology::has_flat_matter_lambda_closure(
                context_.omega_m, context_.omega_lambda)) {
            throw std::invalid_argument(
                "SO context requires a finite positive box and exact flat matter-plus-Lambda background");
        }
    }

    SOExecutionPlan execution_plan(
        std::size_t particle_count,
        std::size_t center_count) const;

    core::Real compute_reference_density(
        const core::ParticleStore& particles) const;
    core::Real compute_reference_density(
        const core::ParticleStore& particles,
        core::Real scale_factor) const;

    core::Real effective_overdensity_threshold(
        core::Real scale_factor) const;

    std::vector<SOHalo> find_halos(
        const core::ParticleStore& particles,
        const std::vector<core::Vec3>& centers) const;

    std::vector<SOHalo> find_halos(
        const core::ParticleStore& particles,
        const std::vector<core::Vec3>& centers,
        core::Real scale_factor) const;

private:
    SOContext context_;
    // Zero is reserved for the dynamic VirialCritical constructor.
    core::Real overdensity_threshold_{0.0};
    SOReferenceDensity reference_density_{SOReferenceDensity::Unknown};

    std::vector<SOHalo> find_halos_with_resolved_density(
        const core::ParticleStore& particles,
        const std::vector<core::Vec3>& centers,
        core::Real reference_density,
        core::Real effective_overdensity) const;
};

} // namespace halo
} // namespace cosmo_nbody
