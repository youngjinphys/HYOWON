#include "cosmo_nbody/halo/spherical_overdensity.hpp"

#include "cosmo_nbody/cosmology/flat_matter_lambda.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/halo/so_scaled_density.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace halo {

namespace {

using RadiusRecord = PeriodicNeighbor;

constexpr std::size_t BYTES_PER_GIB = std::size_t{1} << 30;
constexpr std::uint8_t SO_FAILURE_NONE = 0;
constexpr std::uint8_t SO_FAILURE_NUMERICAL = 1;
constexpr std::uint8_t SO_FAILURE_ALLOCATION = 2;

std::size_t checked_add(
    std::size_t lhs,
    std::size_t rhs,
    const char* message) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(message);
    }
    return lhs + rhs;
}

std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs,
    const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

std::size_t floor_cube_root(std::size_t value) {
    if (value == 0) return 0;
    std::size_t root = static_cast<std::size_t>(
        std::cbrt(static_cast<long double>(value)));
    root = std::max<std::size_t>(1, root);
    const auto fits = [value](std::size_t candidate) {
        return candidate == 0 || candidate <= value / candidate / candidate;
    };
    while (root < std::numeric_limits<std::size_t>::max()
           && fits(root + 1)) {
        ++root;
    }
    while (!fits(root)) --root;
    return root;
}

core::Real particle_mass_at(
    const core::ParticleStore& particles,
    std::size_t index) {
    if (particles.get_uniform_mass().has_value()) {
        return *particles.get_uniform_mass();
    }
    return particles.get_masses()[index];
}

void add_positive_particle_mass(
    math::ExactPositiveDoubleSum& accumulator,
    core::Real mass) {
    if (!std::isfinite(mass) || mass <= 0.0) {
        throw std::invalid_argument(
            "SO finder requires finite positive particle masses");
    }
    accumulator.add(mass);
}

core::Real require_positive_exact_mass(
    const math::ExactPositiveDoubleSum& accumulator) {
    const core::Real result = accumulator.value();
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::overflow_error(
            "SO finder mass sum is not representable");
    }
    return result;
}

core::Real total_owned_mass(const core::ParticleStore& particles) {
    math::ExactPositiveDoubleSum total;
    for (std::size_t index = 0;
         index < particles.num_owned_particles();
         ++index) {
        add_positive_particle_mass(total, particle_mass_at(particles, index));
    }
    return require_positive_exact_mass(total);
}

void require_valid_background(const SOContext& context) {
    if (!std::isfinite(context.omega_m)
        || !std::isfinite(context.omega_lambda)
        || context.omega_m <= 0.0
        || context.omega_lambda < 0.0) {
        throw std::invalid_argument(
            "SO background requires finite Omega_m > 0 and Omega_lambda >= 0");
    }
    // Use the same stored-binary64 closure criterion as the simulation
    // cosmology. A separate empirical SO tolerance would make halo definitions
    // depend on which API admitted the same background.
    if (!cosmology::has_flat_matter_lambda_closure(
            context.omega_m, context.omega_lambda)) {
        throw std::invalid_argument(
            "SO background requires exact flat matter-plus-Lambda closure");
    }
}

long double comoving_critical_density_factor(
    const SOContext& context,
    core::Real scale_factor) {
    long double lambda_term =
        static_cast<long double>(context.omega_lambda);
    const long double a = static_cast<long double>(scale_factor);
    for (int power = 0; power < 3; ++power) {
        lambda_term *= a;
        if (!std::isfinite(lambda_term)) {
            throw std::overflow_error(
                "SO comoving critical-density factor is not representable");
        }
    }
    const long double factor =
        static_cast<long double>(context.omega_m) + lambda_term;
    if (!std::isfinite(factor) || factor <= 0.0L) {
        throw std::overflow_error(
            "SO comoving critical-density factor is not finite and positive");
    }
    return factor;
}

core::Real omega_m_at(
    const SOContext& context,
    core::Real scale_factor) {
    if (context.omega_lambda == 0.0) return 1.0;

    const long double log_ratio =
        std::log(static_cast<long double>(context.omega_lambda))
        - std::log(static_cast<long double>(context.omega_m))
        + 3.0L * std::log(static_cast<long double>(scale_factor));
    if (!std::isfinite(log_ratio)) {
        throw std::overflow_error(
            "SO Omega_m(a) ratio is not representable");
    }

    long double result = 0.0L;
    if (log_ratio >= 0.0L) {
        const long double inverse_ratio = std::exp(-log_ratio);
        result = inverse_ratio / (1.0L + inverse_ratio);
    } else {
        const long double ratio = std::exp(log_ratio);
        result = 1.0L / (1.0L + ratio);
    }
    if (!std::isfinite(result) || result < 0.0L || result > 1.0L) {
        throw std::overflow_error("SO Omega_m(a) is outside [0,1]");
    }
    return static_cast<core::Real>(result);
}

std::size_t requested_workers(const SOContext& context) {
    std::size_t result = 1;
#ifdef COSMO_NBODY_HAS_OPENMP
    const std::uint64_t configured = context.requested_threads;
    if (configured > 0) {
        result = static_cast<std::size_t>(std::min<std::uint64_t>(
            configured,
            static_cast<std::uint64_t>(
                std::numeric_limits<int>::max())));
    } else {
        result = static_cast<std::size_t>(
            std::max(1, omp_get_max_threads()));
    }
#else
    (void)context;
#endif
    return std::max<std::size_t>(1, result);
}

std::size_t explicit_workspace_cap(const SOContext& context) {
    const core::Real budget_gib = context.memory_policy.memory_budget_gib;
    if (!std::isfinite(budget_gib) || budget_gib < 0.0) {
        throw std::invalid_argument(
            "SO analysis memory budget must be finite and non-negative");
    }
    if (budget_gib == 0.0) return 0;
    const long double bytes =
        static_cast<long double>(budget_gib)
        * static_cast<long double>(BYTES_PER_GIB);
    if (!std::isfinite(bytes)
        || bytes < 1.0L
        || bytes > static_cast<long double>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "SO explicit analysis memory budget is not representable");
    }
    return static_cast<std::size_t>(bytes);
}

} // namespace

std::string_view so_reference_density_name(
    SOReferenceDensity reference) noexcept {
    switch (reference) {
        case SOReferenceDensity::Unknown:
            return "unknown";
        case SOReferenceDensity::BoxMeanMatter:
            return "box_mean_matter";
        case SOReferenceDensity::MeanMatter:
            return "mean_matter";
        case SOReferenceDensity::Critical:
            return "critical";
        case SOReferenceDensity::VirialCritical:
            return "virial_critical";
    }
    return "unknown";
}

SphericalOverdensityFinder::SphericalOverdensityFinder(
    SOContext context,
    core::Real overdensity_threshold,
    SOReferenceDensity reference_density)
    : context_(std::move(context)),
      overdensity_threshold_(overdensity_threshold),
      reference_density_(reference_density) {
    if (!std::isfinite(context_.box_size) || context_.box_size <= 0.0) {
        throw std::invalid_argument(
            "SO box size must be finite and positive");
    }
    require_valid_background(context_);
    if (!std::isfinite(overdensity_threshold_)
        || overdensity_threshold_ <= 0.0) {
        throw std::invalid_argument(
            "SO overdensity threshold must be finite and positive");
    }
    if (reference_density_ == SOReferenceDensity::Unknown
        || reference_density_ == SOReferenceDensity::VirialCritical) {
        throw std::invalid_argument(
            "Fixed-threshold SO constructor requires an explicit fixed reference-density mode");
    }
}

SOExecutionPlan SphericalOverdensityFinder::execution_plan(
    std::size_t particle_count,
    std::size_t center_count) const {
    SOExecutionPlan plan;
    plan.particle_count = particle_count;
    plan.center_count = center_count;
    if (particle_count == 0 || center_count == 0) return plan;

    plan.candidate_bytes_per_worker = checked_multiply(
        particle_count,
        sizeof(RadiusRecord),
        "SO exact candidate storage overflows size_t");
    plan.center_staging_bytes = checked_add(
        checked_multiply(
            center_count,
            sizeof(std::optional<SOHalo>),
            "SO center result staging overflows size_t"),
        checked_multiply(
            center_count,
            sizeof(std::uint8_t),
            "SO center failure staging overflows size_t"),
        "SO persistent center staging overflows size_t");
    plan.output_staging_bytes = checked_multiply(
        center_count,
        sizeof(SOHalo),
        "SO output staging overflows size_t");

    plan.query_axis_bytes_per_worker = checked_multiply(
        checked_multiply(
            floor_cube_root(particle_count),
            std::size_t{3},
            "SO query-axis element count overflows size_t"),
        sizeof(std::size_t),
        "SO query-axis storage overflows size_t");
    plan.worker_scratch_bytes_per_worker = checked_add(
        plan.candidate_bytes_per_worker,
        plan.query_axis_bytes_per_worker,
        "SO per-worker scratch overflows size_t");
    plan.requested_workers = requested_workers(context_);

    const auto natural_index_plan = PeriodicNeighborIndex::make_plan(
        particle_count, std::numeric_limits<std::size_t>::max());
    plan.spatial_index_cap_bytes = natural_index_plan.index_bytes;
    plan.spatial_index_bytes = natural_index_plan.index_bytes;
    plan.spatial_cells_per_dimension = natural_index_plan.cells_per_dimension;

    const std::size_t persistent_bytes = checked_add(
        plan.center_staging_bytes,
        plan.spatial_index_bytes,
        "SO persistent workspace overflows size_t");
    const std::size_t one_worker_phase = checked_add(
        persistent_bytes,
        plan.worker_scratch_bytes_per_worker,
        "SO one-worker phase overflows size_t");
    const std::size_t output_phase = checked_add(
        persistent_bytes,
        plan.output_staging_bytes,
        "SO output phase overflows size_t");
    const std::size_t derived_minimum = std::max(
        one_worker_phase, output_phase);

    const std::size_t explicit_cap = explicit_workspace_cap(context_);
    plan.aggregate_scratch_cap_bytes =
        explicit_cap == 0 ? derived_minimum : explicit_cap;
    if (explicit_cap != 0 && explicit_cap < derived_minimum) {
        throw std::runtime_error(
            "Explicit analysis memory budget cannot hold the exact SO index, "
            "center staging, one worker, and the result phase");
    }
    if (persistent_bytes > plan.aggregate_scratch_cap_bytes) {
        throw std::logic_error(
            "SO persistent workspace exceeds the selected execution ceiling");
    }
    plan.candidate_scratch_cap_bytes =
        plan.aggregate_scratch_cap_bytes - persistent_bytes;
    if (plan.worker_scratch_bytes_per_worker
            > plan.candidate_scratch_cap_bytes
        || plan.output_staging_bytes > plan.candidate_scratch_cap_bytes) {
        throw std::logic_error(
            "SO derived minimum does not admit one worker and the result phase");
    }

    std::size_t workers_by_memory = 1;
    if (explicit_cap != 0) {
        workers_by_memory = plan.candidate_scratch_cap_bytes
            / plan.worker_scratch_bytes_per_worker;
        if (workers_by_memory == 0) {
            throw std::logic_error(
                "Explicit SO workspace admitted no candidate worker");
        }
    }
    plan.worker_count = std::min({
        plan.requested_workers,
        center_count,
        workers_by_memory});
    plan.estimated_candidate_peak_bytes = checked_multiply(
        plan.worker_count,
        plan.candidate_bytes_per_worker,
        "SO candidate peak estimate overflows size_t");
    plan.estimated_query_axis_peak_bytes = checked_multiply(
        plan.worker_count,
        plan.query_axis_bytes_per_worker,
        "SO query-axis peak estimate overflows size_t");
    plan.estimated_worker_phase_peak_bytes = checked_add(
        persistent_bytes,
        checked_add(
            plan.estimated_candidate_peak_bytes,
            plan.estimated_query_axis_peak_bytes,
            "SO worker phase parallel peak overflows size_t"),
        "SO worker phase peak overflows size_t");
    plan.estimated_output_phase_peak_bytes = output_phase;
    plan.estimated_total_peak_bytes = std::max(
        plan.estimated_worker_phase_peak_bytes,
        plan.estimated_output_phase_peak_bytes);
    if (plan.estimated_total_peak_bytes
        > plan.aggregate_scratch_cap_bytes) {
        throw std::logic_error(
            "SO execution plan exceeds the selected workspace ceiling");
    }
    return plan;
}

core::Real SphericalOverdensityFinder::compute_reference_density(
    const core::ParticleStore& particles) const {
    if (reference_density_ != SOReferenceDensity::BoxMeanMatter) {
        throw std::invalid_argument(
            "Cosmological SO reference modes require an explicit scale factor");
    }
    const core::Real total_mass = total_owned_mass(particles);
    const core::Real L = context_.box_size;
    const std::array<core::Real, 1> numerators{total_mass};
    const std::array<core::Real, 3> denominators{L, L, L};
    return math::scaled_positive_product_quotient(
        numerators, denominators, "SO box mean density");
}

core::Real SphericalOverdensityFinder::compute_reference_density(
    const core::ParticleStore& particles,
    core::Real scale_factor) const {
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "SO scale factor must be finite and positive");
    }
    core::Real reference_density = 0.0;
    switch (reference_density_) {
        case SOReferenceDensity::Unknown:
            throw std::logic_error("SO reference-density mode is unresolved");
        case SOReferenceDensity::BoxMeanMatter:
            return compute_reference_density(particles);
        case SOReferenceDensity::MeanMatter:
            reference_density = cosmology::units::rho_crit0
                * context_.omega_m;
            break;
        case SOReferenceDensity::Critical:
        case SOReferenceDensity::VirialCritical: {
            const long double density =
                static_cast<long double>(cosmology::units::rho_crit0)
                * comoving_critical_density_factor(context_, scale_factor);
            if (!std::isfinite(density)
                || density <= 0.0L
                || density > static_cast<long double>(
                    std::numeric_limits<core::Real>::max())) {
                throw std::overflow_error(
                    "SO critical reference density is not representable");
            }
            reference_density = static_cast<core::Real>(density);
            break;
        }
    }
    if (!std::isfinite(reference_density)
        || reference_density <= 0.0) {
        throw std::overflow_error(
            "SO reference density is not finite and positive");
    }
    if (particles.num_owned_particles() > 0) {
        (void)total_owned_mass(particles);
    }
    return reference_density;
}

core::Real SphericalOverdensityFinder::effective_overdensity_threshold(
    core::Real scale_factor) const {
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "SO scale factor must be finite and positive");
    }
    if (reference_density_ != SOReferenceDensity::VirialCritical) {
        if (!std::isfinite(overdensity_threshold_)
            || overdensity_threshold_ <= 0.0) {
            throw std::logic_error("Fixed SO definition lost its positive threshold");
        }
        return overdensity_threshold_;
    }
    if (!cosmology::has_flat_matter_lambda_closure(
            context_.omega_m, context_.omega_lambda)) {
        throw std::invalid_argument(
            "Bryan-Norman virial overdensity requires exact flat LambdaCDM closure");
    }
    const core::Real omega_m_a = omega_m_at(context_, scale_factor);
    const core::Real x = omega_m_a - 1.0;
    const core::Real pi = std::acos(core::Real{-1.0});
    const core::Real delta_vir =
        18.0 * pi * pi + 82.0 * x - 39.0 * x * x;
    if (!std::isfinite(delta_vir) || delta_vir <= 0.0) {
        throw std::runtime_error(
            "SO Bryan-Norman virial overdensity is invalid");
    }
    return delta_vir;
}

std::vector<SOHalo> SphericalOverdensityFinder::find_halos(
    const core::ParticleStore& particles,
    const std::vector<core::Vec3>& centers) const {
    if (reference_density_ != SOReferenceDensity::BoxMeanMatter) {
        throw std::invalid_argument(
            "Cosmological SO reference modes require an explicit scale factor");
    }
    if (particles.num_owned_particles() == 0 || centers.empty()) return {};
    return find_halos_with_resolved_density(
        particles,
        centers,
        compute_reference_density(particles),
        overdensity_threshold_);
}

std::vector<SOHalo> SphericalOverdensityFinder::find_halos(
    const core::ParticleStore& particles,
    const std::vector<core::Vec3>& centers,
    core::Real scale_factor) const {
    const core::Real reference_density =
        compute_reference_density(particles, scale_factor);
    const core::Real effective_overdensity =
        effective_overdensity_threshold(scale_factor);
    if (particles.num_owned_particles() == 0 || centers.empty()) return {};
    return find_halos_with_resolved_density(
        particles,
        centers,
        reference_density,
        effective_overdensity);
}

std::vector<SOHalo>
SphericalOverdensityFinder::find_halos_with_resolved_density(
    const core::ParticleStore& particles,
    const std::vector<core::Vec3>& centers,
    core::Real reference_density,
    core::Real effective_overdensity) const {
    std::vector<SOHalo> halos;
    const std::size_t n = particles.num_owned_particles();
    if (n == 0 || centers.empty()) return halos;
    if (!std::isfinite(reference_density) || reference_density <= 0.0
        || !std::isfinite(effective_overdensity)
        || effective_overdensity <= 0.0) {
        throw std::invalid_argument(
            "SO reference density and overdensity must be finite and positive");
    }

    const core::Real L = context_.box_size;
    const core::Real max_radius = 0.5 * L;
    if (!std::isfinite(max_radius) || max_radius <= 0.0) {
        throw std::overflow_error(
            "SO maximum periodic aperture is invalid");
    }
    (void)total_owned_mass(particles);
    const auto target_density = detail::so_target_density(
        effective_overdensity,
        reference_density);

    const auto x = particles.get_positions_x().first(n);
    const auto y = particles.get_positions_y().first(n);
    const auto z = particles.get_positions_z().first(n);
    for (std::size_t particle = 0; particle < n; ++particle) {
        if (!std::isfinite(x[particle])
            || !std::isfinite(y[particle])
            || !std::isfinite(z[particle])) {
            throw std::invalid_argument(
                "SO particle positions must be finite");
        }
    }
    for (const auto& center : centers) {
        if (!std::isfinite(center.x)
            || !std::isfinite(center.y)
            || !std::isfinite(center.z)) {
            throw std::invalid_argument(
                "SO center coordinates must be finite");
        }
    }

    const SOExecutionPlan plan = execution_plan(n, centers.size());
    if (plan.worker_count == 0) return halos;
#ifdef COSMO_NBODY_HAS_OPENMP
    if (plan.worker_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "SO worker count exceeds the OpenMP integer range");
    }
    const int worker_count = static_cast<int>(plan.worker_count);
#endif
    const PeriodicNeighborIndex neighbor_index(
        particles,
        L,
        plan.spatial_index_cap_bytes);
    if (neighbor_index.plan().index_bytes != plan.spatial_index_bytes
        || neighbor_index.plan().cells_per_dimension
            != plan.spatial_cells_per_dimension) {
        throw std::logic_error(
            "SO periodic index disagrees with its execution plan");
    }

    std::vector<std::optional<SOHalo>> by_center(centers.size());
    std::vector<std::uint8_t> center_failure(
        centers.size(), SO_FAILURE_NONE);

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(worker_count) if(worker_count > 1)
#endif
    for (std::size_t center_index = 0;
         center_index < centers.size();
         ++center_index) {
        const core::Vec3 center = math::wrap(centers[center_index], L);
        const core::Real tolerance =
            64.0 * std::numeric_limits<core::Real>::epsilon() * L;
        core::Real query_radius = std::min(
            max_radius,
            neighbor_index.cell_width());
        std::vector<RadiusRecord> radii;

        while (true) {
            try {
                neighbor_index.collect_within(
                    center,
                    query_radius,
                    radii);
            } catch (const std::bad_alloc&) {
                center_failure[center_index] = SO_FAILURE_ALLOCATION;
                break;
            } catch (...) {
                center_failure[center_index] = SO_FAILURE_NUMERICAL;
                break;
            }

            std::sort(
                radii.begin(),
                radii.end(),
                [](const RadiusRecord& lhs, const RadiusRecord& rhs) {
                    if (lhs.radius != rhs.radius) {
                        return lhs.radius < rhs.radius;
                    }
                    return lhs.particle_index < rhs.particle_index;
                });

            if (radii.empty()) {
                if (query_radius >= max_radius - tolerance) break;
                const core::Real expanded = std::min(
                    max_radius,
                    std::max(
                        2.0 * query_radius,
                        query_radius + neighbor_index.cell_width()));
                if (!std::isfinite(expanded)
                    || expanded <= query_radius) {
                    center_failure[center_index] = SO_FAILURE_NUMERICAL;
                    break;
                }
                query_radius = expanded;
                continue;
            }

            math::ExactPositiveDoubleSum enclosed_mass_accumulator;
            core::Real enclosed_mass = 0.0;
            std::size_t enclosed_count = 0;
            core::Real last_above_radius = 0.0;
            core::Real last_above_mass = 0.0;
            std::size_t last_above_count = 0;
            bool found_above = false;
            bool found_first_below = false;

            std::size_t shell_begin = 0;
            while (shell_begin < radii.size()) {
                const core::Real shell_radius = radii[shell_begin].radius;
                std::size_t shell_end = shell_begin;
                try {
                    while (shell_end < radii.size()
                           && radii[shell_end].radius == shell_radius) {
                        add_positive_particle_mass(
                            enclosed_mass_accumulator,
                            particle_mass_at(
                                particles,
                                radii[shell_end].particle_index));
                        if (enclosed_count
                            == std::numeric_limits<std::size_t>::max()) {
                            throw std::overflow_error(
                                "SO enclosed particle count overflows size_t");
                        }
                        ++enclosed_count;
                        ++shell_end;
                    }
                    enclosed_mass = require_positive_exact_mass(
                        enclosed_mass_accumulator);
                } catch (...) {
                    center_failure[center_index] = SO_FAILURE_NUMERICAL;
                    break;
                }

                bool above_target = shell_radius == 0.0;
                if (!above_target) {
                    const auto density = detail::so_scaled_density_at_radius(
                        enclosed_mass,
                        shell_radius);
                    if (!density.has_value()) {
                        center_failure[center_index] = SO_FAILURE_NUMERICAL;
                        break;
                    }
                    const auto relation = detail::so_density_at_or_above_target(
                        *density,
                        target_density);
                    if (!relation.has_value()) {
                        center_failure[center_index] = SO_FAILURE_NUMERICAL;
                        break;
                    }
                    above_target = *relation;
                }

                if (above_target) {
                    found_above = true;
                    last_above_radius = shell_radius;
                    last_above_mass = enclosed_mass;
                    last_above_count = enclosed_count;
                    if (shell_end < radii.size()) {
                        const core::Real next_shell_radius =
                            radii[shell_end].radius;
                        if (!std::isfinite(next_shell_radius)
                            || !(next_shell_radius > shell_radius)) {
                            center_failure[center_index] =
                                SO_FAILURE_NUMERICAL;
                            break;
                        }
                        const auto pre_shell_density =
                            detail::so_scaled_density_at_radius(
                                enclosed_mass, next_shell_radius);
                        if (!pre_shell_density.has_value()) {
                            center_failure[center_index] =
                                SO_FAILURE_NUMERICAL;
                            break;
                        }
                        const auto pre_shell_relation =
                            detail::so_density_at_or_above_target(
                                *pre_shell_density, target_density);
                        if (!pre_shell_relation.has_value()) {
                            center_failure[center_index] =
                                SO_FAILURE_NUMERICAL;
                            break;
                        }
                        if (!*pre_shell_relation) {
                            found_first_below = true;
                            break;
                        }
                    }
                } else {
                    found_first_below = true;
                    break;
                }
                shell_begin = shell_end;
            }
            if (center_failure[center_index] != SO_FAILURE_NONE) break;
            if (!found_above) break;

            const auto crossing = detail::so_crossing_radius(
                last_above_mass,
                effective_overdensity,
                reference_density);
            if (!crossing.has_value()
                || *crossing + tolerance < last_above_radius) {
                center_failure[center_index] = SO_FAILURE_NUMERICAL;
                break;
            }

            const bool resolved = found_first_below
                || *crossing <= query_radius + tolerance;
            if (!resolved) {
                if (query_radius >= max_radius - tolerance) break;
                const core::Real expanded = std::min(
                    max_radius,
                    std::max(
                        2.0 * query_radius,
                        query_radius + neighbor_index.cell_width()));
                if (!std::isfinite(expanded)
                    || expanded <= query_radius) {
                    center_failure[center_index] = SO_FAILURE_NUMERICAL;
                    break;
                }
                query_radius = expanded;
                continue;
            }
            if (*crossing > max_radius + tolerance) break;

            SOHalo halo{};
            halo.id = center_index;
            halo.center = center;
            halo.radius = std::min(*crossing, max_radius);
            halo.mass = last_above_mass;
            halo.particle_count = last_above_count;
            halo.overdensity_threshold = effective_overdensity;
            halo.reference_density = reference_density;
            halo.reference_density_kind = reference_density_;
            by_center[center_index] = halo;
            break;
        }
    }

    for (std::size_t center_index = 0;
         center_index < center_failure.size();
         ++center_index) {
        if (center_failure[center_index] == SO_FAILURE_ALLOCATION) {
            throw std::runtime_error(
                "SO candidate allocation failed for centre index "
                + std::to_string(center_index)
                + " after the workload plan was formed");
        }
        if (center_failure[center_index] == SO_FAILURE_NUMERICAL) {
            throw std::runtime_error(
                "SO numerical failure while processing centre index "
                + std::to_string(center_index));
        }
    }

    halos.reserve(centers.size());
    for (auto& slot : by_center) {
        if (slot.has_value()) halos.push_back(std::move(*slot));
    }
    return halos;
}

} // namespace halo
} // namespace cosmo_nbody
