#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"

#include <limits>

namespace cosmo_nbody::halo {

bool PeriodicNeighborIndex::describes(
    const core::ParticleStore& particles,
    core::Real box_size) const noexcept {
    if (particles_ != &particles
        || plan_.particle_count != particles.num_owned_particles()
        || box_size_ != box_size) {
        return false;
    }

    const std::size_t owned = particles.num_owned_particles();
    if (owned == 0) {
        return compact_heads_.empty() && compact_next_.empty()
            && wide_heads_.empty() && wide_next_.empty();
    }

    const auto x = particles.get_positions_x().first(owned);
    const auto y = particles.get_positions_y().first(owned);
    const auto z = particles.get_positions_z().first(owned);
    std::size_t visited = 0;

    try {
        if (plan_.compact_indices) {
            constexpr std::uint32_t invalid =
                std::numeric_limits<std::uint32_t>::max();
            if (compact_heads_.size() != plan_.cell_count
                || compact_next_.size() != owned
                || !wide_heads_.empty()
                || !wide_next_.empty()) {
                return false;
            }
            for (std::size_t cell = 0; cell < compact_heads_.size(); ++cell) {
                std::uint32_t particle = compact_heads_[cell];
                while (particle != invalid) {
                    const std::size_t index = static_cast<std::size_t>(particle);
                    if (index >= owned || index >= compact_next_.size()
                        || visited == owned) {
                        return false;
                    }
                    if (cell_id_for_position(x[index], y[index], z[index]) != cell) {
                        return false;
                    }
                    ++visited;
                    particle = compact_next_[index];
                }
            }
        } else {
            constexpr std::size_t invalid =
                std::numeric_limits<std::size_t>::max();
            if (wide_heads_.size() != plan_.cell_count
                || wide_next_.size() != owned
                || !compact_heads_.empty()
                || !compact_next_.empty()) {
                return false;
            }
            for (std::size_t cell = 0; cell < wide_heads_.size(); ++cell) {
                std::size_t particle = wide_heads_[cell];
                while (particle != invalid) {
                    if (particle >= owned || particle >= wide_next_.size()
                        || visited == owned) {
                        return false;
                    }
                    if (cell_id_for_position(
                            x[particle], y[particle], z[particle]) != cell) {
                        return false;
                    }
                    ++visited;
                    particle = wide_next_[particle];
                }
            }
        }
    } catch (...) {
        // Invalid or non-finite current coordinates, arithmetic failures, and
        // malformed private topology all make the index unusable. The public
        // aperture boundary translates false into an explicit rejection.
        return false;
    }

    return visited == owned;
}

} // namespace cosmo_nbody::halo
