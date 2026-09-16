#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace cosmo_nbody::halo {

struct DensityPeakDeblendOptions {
    // Scientific topology coordinates are explicit; sentinel values are invalid.
    std::size_t k_neighbors{0};
    // Historical "host" vocabulary in this deblender means a retained local
    // density basin only. It is not a cosmological host/subhalo classification.
    std::size_t minimum_host_particles{0};
    core::Real saddle_to_lower_peak_merge_ratio{-1.0};
};

struct DensityPeakDeblendExecutionPlan {
    std::size_t particle_count{0};
    std::size_t effective_k_neighbors{0};
    std::size_t maximum_directed_knn_edges{0};
    std::size_t conservative_payload_peak_bytes{0};
    bool retains_directed_knn_edges{false};
    bool includes_allocator_overhead{false};
};

struct DensityPeakRecord {
    std::size_t candidate_id{0};
    core::ParticleId peak_particle_id{0};
    std::size_t peak_particle_index{0};
    core::Vec3 position{};
    core::Real density{0.0};
    // True when the local peak survives the configured saddle merge and seeds a
    // deblended density basin; this does not assert distinct-host status.
    bool retained_as_host{false};
    std::optional<core::ParticleId> merged_into_peak_particle_id;
};

// Historical internal type name. Semantically this is a retained density-basin
// seed; no cosmological host/subhalo classification is implied.
struct DeblendedHostSeed {
    std::size_t candidate_id{0};
    std::size_t seed_id{0};
    core::ParticleId peak_particle_id{0};
    std::size_t peak_particle_index{0};
    core::Vec3 peak_position{};
    core::Real peak_density{0.0};
    bool meets_minimum_particle_count{false};
    std::vector<std::size_t> particle_indices;
};

struct DensityPeakDeblendResult {
    std::size_t candidate_id{0};
    std::size_t effective_k_neighbors{0};
    std::vector<DensityPeakRecord> peaks;
    std::vector<DeblendedHostSeed> hosts;
};

class DensityPeakDeblender {
public:
    explicit DensityPeakDeblender(DensityPeakDeblendOptions options);

    DensityPeakDeblendExecutionPlan execution_plan(
        std::size_t candidate_particle_count) const;

    DensityPeakDeblendResult deblend(
        const core::ParticleStore& particles,
        const FoFMembership& candidate,
        core::Real box_size) const;

private:
    DensityPeakDeblendOptions options_;
};

} // namespace cosmo_nbody::halo
