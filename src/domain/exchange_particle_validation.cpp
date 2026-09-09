#include "cosmo_nbody/domain/exchange_particle_validation.hpp"

#include <cmath>
#include <stdexcept>

namespace cosmo_nbody::domain {
namespace {

template <typename ParticleRecord>
void validate_source_fields(
    const ParticleRecord& particle,
    core::Real box_size) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "Exchange validation box size must be finite and positive");
    }
    if (!std::isfinite(particle.x)
        || !std::isfinite(particle.y)
        || !std::isfinite(particle.z)) {
        throw std::runtime_error(
            "Exchange particle positions must be finite");
    }
    if (particle.x < 0.0 || particle.x >= box_size
        || particle.y < 0.0 || particle.y >= box_size
        || particle.z < 0.0 || particle.z >= box_size) {
        throw std::runtime_error(
            "Exchange particle positions must use the canonical periodic half-open interval [0,L)");
    }
    if (!std::isfinite(particle.mass) || particle.mass <= 0.0) {
        throw std::runtime_error(
            "Exchange particle mass must be finite and positive");
    }
}

} // namespace

void validate_received_exchange_particle(
    const ExchangeParticle& particle,
    core::Real box_size) {
    validate_source_fields(particle, box_size);
    if (!std::isfinite(particle.px)
        || !std::isfinite(particle.py)
        || !std::isfinite(particle.pz)) {
        throw std::runtime_error(
            "Exchange particle momenta must be finite");
    }
}

void validate_received_ghost_exchange_particle(
    const GhostExchangeParticle& particle,
    core::Real box_size) {
    validate_source_fields(particle, box_size);
}

} // namespace cosmo_nbody::domain
