#include "cosmo_nbody/halo/exact_periodic_aperture.hpp"

#include <cmath>
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

    // PeriodicNeighborIndex traverses a conservative candidate-cell superset and
    // decides the closed boundary from the exact mathematical torus squared
    // distance of represented canonical binary64 endpoints. The cached rounded
    // displacement norm is descriptive only. No empirical epsilon expansion is
    // part of the aperture definition.
    index.collect_within(center, radius, output);
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
