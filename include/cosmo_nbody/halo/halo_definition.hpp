#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/spherical_overdensity.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace cosmo_nbody::halo {

enum class HaloCenterKind {
    Unknown,
    PeriodicMassWeightedCircularCOM,
    DensityPeakParticle,
};

std::string_view halo_center_kind_name(HaloCenterKind kind) noexcept;

struct HaloCenterCandidate {
    HaloCenterKind kind{HaloCenterKind::Unknown};
    core::Vec3 position{};
    bool available{false};
};

struct HaloCenterSelection {
    HaloCenterKind selected_kind{HaloCenterKind::Unknown};
};

enum class StandardSOMassDefinition {
    Unknown,
    M200m,
    M200c,
    MvirBN98,
};

std::string_view standard_so_mass_definition_name(
    StandardSOMassDefinition definition) noexcept;

// M200m and M200c use fixed 200 thresholds; MvirBN98 derives Delta_vir(a)
// from the background cosmology.
struct SOMassRequest {
    StandardSOMassDefinition definition;
    SOReferenceDensity reference_density_kind;
    std::optional<core::Real> fixed_overdensity_threshold;
};

std::vector<SOMassRequest> standard_so_mass_requests();

struct SOMassMeasurement {
    StandardSOMassDefinition definition{StandardSOMassDefinition::Unknown};
    std::size_t candidate_id{0};
    std::size_t deblended_seed_id{0};
    core::ParticleId peak_particle_id{0};
    HaloCenterSelection center_selection{};
    core::Real scale_factor{0.0};
    SOReferenceDensity reference_density_kind{SOReferenceDensity::Unknown};
    core::Real reference_density{0.0};
    core::Real overdensity_threshold{0.0};
    core::Real radius{0.0};
    core::Real mass{0.0};
    std::size_t geometric_particle_count{0};
    bool crossing_resolved{false};
};

enum class HaloMembershipView {
    Unknown,
    FoFCandidateAll,
    DeblendedDensityBasinAll,
    GeometricSOAll,
    BoundHostExcludingSubstructure,
};

std::string_view halo_membership_view_name(HaloMembershipView view) noexcept;

struct HaloMembership {
    HaloMembershipView view{HaloMembershipView::Unknown};
    bool available{false};
};

void validate_halo_definition(
    std::span<const HaloCenterCandidate> center_candidates,
    const HaloCenterSelection& center_selection,
    const SOMassMeasurement& mass,
    const HaloMembership& membership);

} // namespace cosmo_nbody::halo
