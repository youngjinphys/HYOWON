#include "cosmo_nbody/domain/ghost_exchange.hpp"

#include "cosmo_nbody/domain/bounded_migration.hpp"
#include "cosmo_nbody/domain/exchange_particle_validation.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/mesh/mpi_slab_layout.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace domain {

namespace {

#ifdef COSMO_NBODY_HAS_MPI
struct SlabInterval {
    int rank{0};
    std::size_t start{0};
    std::size_t count{0};
    core::Real lo{0.0};
    core::Real hi{0.0};
};

SlabInterval slab_interval(core::Real box_size,
                           std::size_t mesh_size,
                           int rank,
                           int size) {
    const auto layout = mesh::mpi_slab_layout(mesh_size, rank, size);
    const core::Real mesh_real = static_cast<core::Real>(mesh_size);
    return {
        rank,
        layout.start,
        layout.count,
        box_size * static_cast<core::Real>(layout.start) / mesh_real,
        box_size * static_cast<core::Real>(layout.start + layout.count)
            / mesh_real};
}

std::size_t periodic_slab_gap_planes(
    const SlabInterval& lhs,
    const SlabInterval& rhs,
    std::size_t mesh_size) {
    const std::size_t lhs_end = lhs.start + lhs.count;
    const std::size_t rhs_end = rhs.start + rhs.count;
    if (lhs_end <= rhs.start) {
        const std::size_t direct = rhs.start - lhs_end;
        const std::size_t wrapped = mesh_size - (rhs_end - lhs.start);
        return std::min(direct, wrapped);
    }
    if (rhs_end <= lhs.start) {
        const std::size_t direct = lhs.start - rhs_end;
        const std::size_t wrapped = mesh_size - (lhs_end - rhs.start);
        return std::min(direct, wrapped);
    }
    return 0;
}

bool slabs_can_exchange_ghosts(
    const SlabInterval& local,
    const SlabInterval& remote,
    core::Real box_size,
    std::size_t mesh_size,
    core::Real r_search) {
    const std::size_t gap_planes = periodic_slab_gap_planes(
        local, remote, mesh_size);
    // Compare L*gap_planes <= r_search*N_mesh in long double so candidate
    // pruning is based on the exact integer mesh-plane separation rather than
    // a rounded subtraction of physical slab boundaries. The exact
    // per-particle binary64 distance test below remains authoritative.
    const long double gap_scaled = static_cast<long double>(box_size)
        * static_cast<long double>(gap_planes);
    const long double search_scaled = static_cast<long double>(r_search)
        * static_cast<long double>(mesh_size);
    return gap_scaled <= search_scaled;
}

std::vector<SlabInterval> candidate_remote_slabs(
    core::Real box_size,
    std::size_t mesh_size,
    int rank,
    int size,
    core::Real r_search) {
    const SlabInterval local = slab_interval(
        box_size, mesh_size, rank, size);
    std::vector<SlabInterval> candidates;
    if (size > 1) {
        candidates.reserve(static_cast<std::size_t>(size - 1));
    }
    for (int destination = 0; destination < size; ++destination) {
        if (destination == rank) continue;
        const SlabInterval remote = slab_interval(
            box_size, mesh_size, destination, size);
        if (slabs_can_exchange_ghosts(
                local, remote, box_size, mesh_size, r_search)) {
            candidates.push_back(remote);
        }
    }
    return candidates;
}

core::Real nearest_image_to_slab_center(
    core::Real x,
    core::Real box_size,
    const SlabInterval& slab) {
    const core::Real center = 0.5 * (slab.lo + slab.hi);
    core::Real delta = x - center;
    delta -= std::round(delta / box_size) * box_size;
    return center + delta;
}

core::Real distance_to_slab(
    core::Real x_image,
    const SlabInterval& slab) {
    if (x_image < slab.lo) return slab.lo - x_image;
    if (x_image > slab.hi) return x_image - slab.hi;
    return 0.0;
}
#endif

} // namespace

GhostExchange::GhostExchange(core::Real box_size,
                             std::size_t mesh_size,
                             int rank,
                             int size)
    : box_size_(box_size),
      mesh_size_(mesh_size),
      rank_(rank),
      size_(size) {
    if (!std::isfinite(box_size_) || box_size_ <= 0.0) {
        throw std::invalid_argument(
            "GhostExchange box size must be finite and positive");
    }
    if (mesh_size_ == 0) {
        throw std::invalid_argument(
            "GhostExchange mesh size must be positive");
    }
    if (size_ < 1 || rank_ < 0 || rank_ >= size_) {
        throw std::invalid_argument(
            "GhostExchange rank topology is invalid");
    }
    if (static_cast<std::size_t>(size_) > mesh_size_) {
        throw std::invalid_argument(
            "GhostExchange requires MPI rank count <= mesh size");
    }
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (mesh::mpi_slab_layout_has_empty_rank(mesh_size_, size_)) {
        throw std::invalid_argument(
            "Current FFTW-MPI slab layout requires every MPI rank to own at least one mesh plane");
    }
#endif
}

void GhostExchange::exchange(core::ParticleStore& particles,
                             core::Real r_search) const {
    if (!std::isfinite(r_search) || r_search < 0.0) {
        throw std::invalid_argument(
            "Ghost search radius must be finite and non-negative");
    }

    if (size_ == 1 || r_search == 0.0) {
        particles.clear_ghosts();
        particles.mark_ghost_start();
        particles.set_ghost_validity(core::FieldValidity::VALID);
        return;
    }

#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        "Multi-rank GhostExchange requires an MPI-enabled build");
#else
    // Preserve the currently admitted suffix until routing, receive validation,
    // and every rank's final-capacity allocation have all succeeded. The owned
    // prefix remains the sole routing basis throughout this transaction.
    const std::size_t owned = particles.num_owned_particles();
    std::vector<RoutedGhostExchangeParticle> routed;
    std::vector<SlabInterval> destinations;
    SlabInterval local_slab;
    std::exception_ptr stage_exception;
    try {
        // Precompute candidate slabs, then apply exact per-particle distances only
        // to those neighbors: O(ranks + N_local*neighbors), without topology or
        // search-radius assumptions.
        local_slab = slab_interval(box_size_, mesh_size_, rank_, size_);
        destinations = candidate_remote_slabs(
            box_size_, mesh_size_, rank_, size_, r_search);

        for (std::size_t i = 0; i < owned; ++i) {
            const core::Real x = math::wrap(
                particles.get_positions_x()[i], box_size_);
            for (const SlabInterval& destination : destinations) {
                const core::Real x_image = nearest_image_to_slab_center(
                    x, box_size_, destination);
                if (distance_to_slab(x_image, destination) > r_search) {
                    continue;
                }
                GhostExchangeParticle particle;
                // Store canonical periodic coordinates. The nearest image is used
                // only for visibility testing; force evaluation performs periodic
                // minimum-image displacement in the full-box tree.
                particle.x = x;
                particle.y = math::wrap(
                    particles.get_positions_y()[i], box_size_);
                particle.z = math::wrap(
                    particles.get_positions_z()[i], box_size_);
                particle.mass = particles.mass_at(i);
                particle.id = particles.get_ids()[i];
                routed.push_back({destination.rank, particle});
            }
        }
        ExchangeBuffer::sort_routed_ghosts_in_place(routed);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Ghost routing preparation");

    auto received = exchange_routed_ghosts_bounded(
        routed, rank_, size_);
    // The compact routing payload is dead before received records are converted
    // into the ParticleStore's compact source suffix.
    std::vector<RoutedGhostExchangeParticle>().swap(routed);

    stage_exception = nullptr;
    try {
        std::sort(
            received.begin(), received.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.id < rhs.id;
            });
        for (std::size_t i = 1; i < received.size(); ++i) {
            if (received[i - 1].id == received[i].id) {
                throw std::runtime_error(
                    "Duplicate ghost particle ID received");
            }
        }

        const auto uniform_mass = particles.get_uniform_mass();
        for (const auto& particle : received) {
            validate_received_ghost_exchange_particle(particle, box_size_);
            if (uniform_mass.has_value() && particle.mass != *uniform_mass) {
                throw std::runtime_error(
                    "Received ghost particle disagrees with uniform mass");
            }
            const core::Real x_image = nearest_image_to_slab_center(
                particle.x, box_size_, local_slab);
            if (distance_to_slab(x_image, local_slab) > r_search) {
                throw std::runtime_error(
                    "Received ghost particle lies outside the admitted search radius");
            }
        }

        const auto owned_ids = particles.get_ids().first(owned);
        for (std::size_t i = 1; i < owned_ids.size(); ++i) {
            if (owned_ids[i - 1] >= owned_ids[i]) {
                throw std::logic_error(
                    "Owned particle IDs must be strictly increasing before ghost admission");
            }
        }
        std::size_t owned_index = 0;
        std::size_t ghost_index = 0;
        while (owned_index < owned_ids.size()
               && ghost_index < received.size()) {
            const core::ParticleId owned_id = owned_ids[owned_index];
            const core::ParticleId ghost_id = received[ghost_index].id;
            if (owned_id < ghost_id) {
                ++owned_index;
            } else if (ghost_id < owned_id) {
                ++ghost_index;
            } else {
                throw std::runtime_error(
                    "Ghost particle ID collides with an owned particle ID");
            }
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Ghost receive validation");

    // Allocate only source fields (position, explicit mass, ID) before any rank
    // clears its currently valid suffix. Momentum and acceleration are owned
    // integration fields and deliberately do not scale with ghost count.
    stage_exception = nullptr;
    try {
        particles.reserve_ghost_capacity(received.size());
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Ghost publication capacity preflight");

    stage_exception = nullptr;
    try {
        particles.clear_ghosts();
        for (const auto& particle : received) {
            particles.append_ghost_particle(
                core::Position{{particle.x, particle.y, particle.z}},
                particle.mass,
                particle.id);
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Ghost publication");
    particles.set_ghost_validity(core::FieldValidity::VALID);
#endif
}

} // namespace domain
} // namespace cosmo_nbody
