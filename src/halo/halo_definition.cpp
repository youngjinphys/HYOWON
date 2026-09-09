#include "cosmo_nbody/halo/halo_definition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::halo {

std::string_view halo_center_kind_name(HaloCenterKind kind) noexcept {
    switch (kind) {
        case HaloCenterKind::Unknown:
            return "unknown_center_kind";
        case HaloCenterKind::PeriodicMassWeightedCircularCOM:
            return "periodic_mass_weighted_circular_com";
        case HaloCenterKind::DensityPeakParticle:
            return "density_peak_particle";
    }
    return "unknown_center_kind";
}

std::string_view standard_so_mass_definition_name(
    StandardSOMassDefinition definition) noexcept {
    switch (definition) {
        case StandardSOMassDefinition::Unknown: return "unknown_so_mass_definition";
        case StandardSOMassDefinition::M200m: return "M200m";
        case StandardSOMassDefinition::M200c: return "M200c";
        case StandardSOMassDefinition::MvirBN98: return "Mvir_BN98";
    }
    return "unknown_so_mass_definition";
}

std::vector<SOMassRequest> standard_so_mass_requests() {
    return {
        {StandardSOMassDefinition::M200m, SOReferenceDensity::MeanMatter, core::Real{200.0}},
        {StandardSOMassDefinition::M200c, SOReferenceDensity::Critical, core::Real{200.0}},
        {StandardSOMassDefinition::MvirBN98, SOReferenceDensity::Critical, std::nullopt},
    };
}

std::string_view halo_membership_view_name(HaloMembershipView view) noexcept {
    switch (view) {
        case HaloMembershipView::Unknown:
            return "unknown_membership_view";
        case HaloMembershipView::FoFCandidateAll:
            return "fof_candidate_all";
        case HaloMembershipView::DeblendedDensityBasinAll:
            return "deblended_density_basin_all";
        case HaloMembershipView::GeometricSOAll:
            return "geometric_so_all";
        case HaloMembershipView::BoundHostExcludingSubstructure:
            return "bound_host_excluding_substructure";
    }
    return "unknown_membership_view";
}

void validate_halo_definition(
    std::span<const HaloCenterCandidate> center_candidates,
    const HaloCenterSelection& center_selection,
    const SOMassMeasurement& mass,
    const HaloMembership& membership) {
    if (center_candidates.empty()) {
        throw std::invalid_argument("Halo definition requires at least one explicit center candidate");
    }
    if (center_selection.selected_kind == HaloCenterKind::Unknown) {
        throw std::invalid_argument("Halo definition requires an explicit center kind");
    }
    const auto selected = std::find_if(
        center_candidates.begin(), center_candidates.end(),
        [&](const HaloCenterCandidate& candidate) {
            return candidate.kind == center_selection.selected_kind;
        });
    if (selected == center_candidates.end()) {
        throw std::invalid_argument("Selected halo center kind is absent from the candidate set");
    }
    if (!selected->available) {
        throw std::invalid_argument("Selected halo center kind is not available");
    }
    if (!std::isfinite(selected->position.x)
        || !std::isfinite(selected->position.y)
        || !std::isfinite(selected->position.z)) {
        throw std::invalid_argument("Selected halo center position must be finite");
    }

    if (mass.definition == StandardSOMassDefinition::Unknown) {
        throw std::invalid_argument("SO mass measurement requires an explicit named definition");
    }
    const auto requests = standard_so_mass_requests();
    const auto request = std::find_if(
        requests.begin(), requests.end(), [&](const SOMassRequest& item) {
            return item.definition == mass.definition;
        });
    if (request == requests.end()) {
        throw std::logic_error("Named SO mass definition is missing from the standard request set");
    }
    if (!(mass.scale_factor > 0.0) || !std::isfinite(mass.scale_factor)) {
        throw std::invalid_argument("SO mass scale factor must be finite and positive");
    }
    if (mass.reference_density_kind != request->reference_density_kind) {
        throw std::invalid_argument(
            std::string(standard_so_mass_definition_name(mass.definition))
            + " reference-density kind disagrees with its named definition");
    }
    if (!(mass.reference_density > 0.0) || !std::isfinite(mass.reference_density)) {
        throw std::invalid_argument("SO reference density must be finite and positive");
    }
    if (!(mass.overdensity_threshold > 0.0) || !std::isfinite(mass.overdensity_threshold)) {
        throw std::invalid_argument("SO overdensity threshold must be finite and positive");
    }
    if (mass.radius < 0.0 || mass.mass < 0.0
        || !std::isfinite(mass.radius) || !std::isfinite(mass.mass)) {
        throw std::invalid_argument("SO radius and mass must be finite and non-negative");
    }
    if (mass.crossing_resolved) {
        if (!(mass.radius > 0.0) || !(mass.mass > 0.0)
            || mass.geometric_particle_count == 0) {
            throw std::invalid_argument(
                "A resolved SO measurement requires positive radius, mass, and particle count");
        }
    } else if (mass.radius != 0.0
               || mass.mass != 0.0
               || mass.geometric_particle_count != 0) {
        throw std::invalid_argument(
            "An unresolved SO measurement must carry zero payload");
    }
    if (mass.center_selection.selected_kind != center_selection.selected_kind) {
        throw std::invalid_argument(
            "SO mass center kind disagrees with the selected center");
    }
    if (request->fixed_overdensity_threshold.has_value()
        && mass.overdensity_threshold != *request->fixed_overdensity_threshold) {
        throw std::invalid_argument(
            std::string(standard_so_mass_definition_name(mass.definition))
            + " requires its named fixed overdensity threshold");
    }

    if (membership.view == HaloMembershipView::Unknown) {
        throw std::invalid_argument("Halo membership view must be explicit");
    }
    if (membership.view == HaloMembershipView::BoundHostExcludingSubstructure
        && !membership.available) {
        return;
    }
    if (!membership.available) {
        throw std::invalid_argument("Requested halo membership view is unavailable");
    }
    if (membership.view == HaloMembershipView::GeometricSOAll
        && !mass.crossing_resolved) {
        throw std::invalid_argument("Geometric SO membership requires a resolved SO crossing");
    }
}

} // namespace cosmo_nbody::halo
