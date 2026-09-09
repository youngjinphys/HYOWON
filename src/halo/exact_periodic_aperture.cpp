#include "cosmo_nbody/halo/exact_periodic_aperture.hpp"

#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cosmo_nbody::halo {
namespace {

void require_valid_box(core::Real box_size) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "Exact periodic aperture requires a finite positive box size");
    }
}

void collect_validated_exact_periodic_aperture(
    const PeriodicNeighborIndex& index,
    const core::ParticleStore& particles,
    const core::Vec3& center,
    core::Real radius,
    core::Real box_size,
    std::vector<PeriodicNeighbor>& output) {
    if (!std::isfinite(radius) || radius <= 0.0
        || radius > 0.5 * box_size) {
        throw std::invalid_argument(
            "Exact periodic aperture requires radius in (0,L/2]");
    }
    if (index.plan().particle_count != particles.num_owned_particles()) {
        throw std::logic_error(
            "Validated exact periodic aperture query observed particle-count mutation");
    }

    const std::size_t count = particles.num_owned_particles();
    const auto x = particles.get_positions_x().first(count);
    const auto y = particles.get_positions_y().first(count);
    const auto z = particles.get_positions_z().first(count);
    const auto exact_neighbor = [&, radius](
        std::size_t particle_index,
        PeriodicNeighbor& neighbor) {
        const core::Vec3 displacement = math::minimum_image_displacement(
            center,
            core::Vec3{x[particle_index], y[particle_index], z[particle_index]},
            box_size);
        const core::Real distance = displacement.norm();
        if (!std::isfinite(distance)) {
            throw std::runtime_error(
                "Exact periodic aperture encountered invalid periodic geometry");
        }
        if (distance > radius
            || !core::scale_safe_norm3_leq(
                displacement.x,
                displacement.y,
                displacement.z,
                radius)) {
            return false;
        }
        neighbor = {distance, particle_index};
        return true;
    };

    const core::Real maximum_radius = 0.5 * box_size;
    const core::Real guard =
        64.0 * std::numeric_limits<core::Real>::epsilon()
        * std::max(box_size, radius);
    const core::Real guarded_query_radius = std::nextafter(
        radius + guard, std::numeric_limits<core::Real>::infinity());

    // Candidate generation is deliberately conservative so cell-edge rounding
    // cannot hide a point that the precise radius predicate admits. At L/2 the
    // guard cannot be enlarged without leaving the unique minimum-image domain,
    // so use one full scan and apply the same central comparison rule.
    if (!(guarded_query_radius < maximum_radius)) {
        if (output.capacity() < count) {
            std::vector<PeriodicNeighbor>{}.swap(output);
            output.reserve(count);
        } else {
            output.clear();
        }
        for (std::size_t particle_index = 0;
             particle_index < count;
             ++particle_index) {
            PeriodicNeighbor neighbor;
            if (exact_neighbor(particle_index, neighbor)) {
                output.push_back(neighbor);
            }
        }
        return;
    }

    index.collect_within(center, guarded_query_radius, output);
    std::size_t accepted = 0;
    for (const auto& candidate : output) {
        if (candidate.particle_index >= count) {
            throw std::logic_error(
                "Exact periodic aperture index returned a non-owned particle");
        }
        PeriodicNeighbor neighbor;
        if (exact_neighbor(candidate.particle_index, neighbor)) {
            output[accepted++] = neighbor;
        }
    }
    output.resize(accepted);
}

} // namespace

ExactPeriodicApertureQuery::ExactPeriodicApertureQuery(
    const PeriodicNeighborIndex& index,
    const core::ParticleStore& particles,
    core::Real box_size)
    : index_(&index),
      particles_(&particles),
      box_size_(box_size),
      particle_count_(particles.num_owned_particles()) {
    require_valid_box(box_size_);
    if (!index.describes(particles, box_size_)) {
        throw std::invalid_argument(
            "Exact periodic aperture index does not describe the supplied particle store and box");
    }
}

bool ExactPeriodicApertureQuery::describes(
    const core::ParticleStore& particles,
    core::Real box_size) const noexcept {
    return index_ != nullptr
        && particles_ == &particles
        && box_size_ == box_size
        && particle_count_ == particles.num_owned_particles()
        && index_->describes(particles, box_size);
}

core::Real ExactPeriodicApertureQuery::cell_width() const {
    if (index_ == nullptr || particles_ == nullptr) {
        throw std::logic_error(
            "Exact periodic aperture query is not initialized");
    }
    if (particle_count_ != particles_->num_owned_particles()) {
        throw std::logic_error(
            "Exact periodic aperture query observed particle-count mutation");
    }
    return index_->cell_width();
}

void ExactPeriodicApertureQuery::collect(
    const core::Vec3& center,
    core::Real radius,
    std::vector<PeriodicNeighbor>& output) const {
    if (index_ == nullptr || particles_ == nullptr) {
        throw std::logic_error(
            "Exact periodic aperture query is not initialized");
    }
    if (particle_count_ != particles_->num_owned_particles()) {
        throw std::logic_error(
            "Exact periodic aperture query observed particle-count mutation");
    }
    collect_validated_exact_periodic_aperture(
        *index_, *particles_, center, radius, box_size_, output);
}

void collect_exact_periodic_aperture(
    const PeriodicNeighborIndex& index,
    const core::ParticleStore& particles,
    const core::Vec3& center,
    core::Real radius,
    core::Real box_size,
    std::vector<PeriodicNeighbor>& output) {
    ExactPeriodicApertureQuery query(index, particles, box_size);
    query.collect(center, radius, output);
}

} // namespace cosmo_nbody::halo
