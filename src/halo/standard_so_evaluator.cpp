#include "cosmo_nbody/halo/standard_so_evaluator.hpp"

#include "cosmo_nbody/halo/spherical_overdensity.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cosmo_nbody::halo {

StandardSOEvaluator::StandardSOEvaluator(SOContext context)
    : context_(std::move(context)) {}

std::vector<StandardSOSeedEvaluation> StandardSOEvaluator::evaluate(
    const core::ParticleStore& particles,
    const std::vector<NamedSOSeed>& seeds,
    core::Real scale_factor) const {
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "Standard SO evaluation requires a finite positive scale factor");
    }

    std::set<std::pair<std::size_t, std::size_t>> seed_identities;
    std::vector<core::Vec3> centers;
    centers.reserve(seeds.size());
    std::vector<StandardSOSeedEvaluation> result(seeds.size());
    for (std::size_t index = 0; index < seeds.size(); ++index) {
        const auto& seed = seeds[index];
        if (!std::isfinite(seed.center.x) || !std::isfinite(seed.center.y)
            || !std::isfinite(seed.center.z)) {
            throw std::invalid_argument(
                "Standard SO seed center must be finite");
        }
        if (!seed_identities.emplace(
                seed.candidate_id,
                seed.deblended_seed_id).second) {
            throw std::invalid_argument(
                "Standard SO seeds contain a duplicate candidate/seed ID pair");
        }
        centers.push_back(seed.center);
        result[index].seed = seed;
    }

    const auto requests = standard_so_mass_requests();
    if (requests.size() != 3) {
        throw std::logic_error(
            "Standard SO request set must contain exactly three definitions");
    }
    for (std::size_t definition_index = 0;
         definition_index < requests.size();
         ++definition_index) {
        const auto& request = requests[definition_index];

        std::unique_ptr<SphericalOverdensityFinder> finder;
        if (request.fixed_overdensity_threshold.has_value()) {
            finder = std::make_unique<SphericalOverdensityFinder>(
                context_,
                *request.fixed_overdensity_threshold,
                request.reference_density_kind);
        } else {
            // Bryan-Norman keeps a critical-density reference while Delta_vir(a)
            // is evaluated dynamically.
            const SphericalOverdensityFinder threshold_definition(
                context_, SOReferenceDensity::VirialCritical);
            const core::Real threshold =
                threshold_definition.effective_overdensity_threshold(
                    scale_factor);
            finder = std::make_unique<SphericalOverdensityFinder>(
                context_, threshold, request.reference_density_kind);
        }

        const core::Real reference_density = finder->compute_reference_density(
            particles, scale_factor);
        const core::Real effective_threshold =
            finder->effective_overdensity_threshold(scale_factor);
        const auto halos = finder->find_halos(
            particles, centers, scale_factor);

        std::vector<const SOHalo*> halo_by_seed(seeds.size(), nullptr);
        for (const auto& halo : halos) {
            if (halo.id >= halo_by_seed.size()) {
                throw std::logic_error(
                    "SO finder returned a center index outside the seed set");
            }
            if (halo_by_seed[halo.id] != nullptr) {
                throw std::logic_error(
                    "SO finder returned duplicate rows for one seed");
            }
            halo_by_seed[halo.id] = &halo;
        }

        for (std::size_t seed_index = 0;
             seed_index < seeds.size();
             ++seed_index) {
            const NamedSOSeed& seed = seeds[seed_index];
            SOMassMeasurement measurement;
            measurement.definition = request.definition;
            measurement.candidate_id = seed.candidate_id;
            measurement.deblended_seed_id = seed.deblended_seed_id;
            measurement.peak_particle_id = seed.peak_particle_id;
            measurement.center_selection = seed.center_selection;
            measurement.scale_factor = scale_factor;
            measurement.reference_density_kind =
                request.reference_density_kind;
            measurement.reference_density = reference_density;
            measurement.overdensity_threshold = effective_threshold;
            if (const SOHalo* halo = halo_by_seed[seed_index]) {
                if (halo->reference_density_kind
                        != request.reference_density_kind
                    || halo->reference_density != reference_density
                    || halo->overdensity_threshold != effective_threshold) {
                    throw std::logic_error(
                        "SO finder output disagrees with the named mass request");
                }
                measurement.radius = halo->radius;
                measurement.mass = halo->mass;
                measurement.geometric_particle_count = halo->particle_count;
                measurement.crossing_resolved = true;
            }
            result[seed_index].measurements[definition_index] =
                std::move(measurement);
        }
    }
    return result;
}

} // namespace cosmo_nbody::halo
