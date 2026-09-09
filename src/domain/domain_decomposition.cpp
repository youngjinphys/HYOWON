#include "cosmo_nbody/domain/domain_decomposition.hpp"

#include "cosmo_nbody/core/id_uniqueness.hpp"
#include "cosmo_nbody/domain/bounded_migration.hpp"
#include "cosmo_nbody/domain/exchange_particle_validation.hpp"
#include "cosmo_nbody/domain/mpi_global_id_check.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/math/mpi_exact_uint64_sum.hpp"
#include "cosmo_nbody/mesh/mpi_slab_layout.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/mpi_execution_identity.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace domain {

namespace {

void validate_partition_input(const core::ParticleStore& particles) {
    const std::size_t owned = particles.num_owned_particles();
    const auto px = particles.get_positions_x().first(owned);
    const auto py = particles.get_positions_y().first(owned);
    const auto pz = particles.get_positions_z().first(owned);
    const auto mx = particles.get_momenta_x().first(owned);
    const auto my = particles.get_momenta_y().first(owned);
    const auto mz = particles.get_momenta_z().first(owned);
    const auto ids = particles.get_ids().first(owned);
    const auto uniform_mass = particles.get_uniform_mass();
    const auto masses = uniform_mass.has_value()
        ? std::span<const core::Real>{}
        : particles.get_masses().first(owned);

    if (uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
        throw std::invalid_argument(
            "Domain partition uniform mass must be finite and positive");
    }

    for (std::size_t i = 0; i < owned; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i])
            || !std::isfinite(pz[i])) {
            throw std::invalid_argument(
                "Domain partition positions must be finite");
        }
        if (!std::isfinite(mx[i]) || !std::isfinite(my[i])
            || !std::isfinite(mz[i])) {
            throw std::invalid_argument(
                "Domain partition momenta must be finite");
        }
        if (!uniform_mass.has_value()
            && (!std::isfinite(masses[i]) || masses[i] <= 0.0)) {
            throw std::invalid_argument(
                "Domain partition masses must be finite and positive");
        }
    }
    core::require_unique_particle_ids_bounded(
        ids, "Domain partition owned particle IDs");
}

#ifdef COSMO_NBODY_HAS_MPI
std::uint64_t collective_particle_population(
    std::size_t local_count,
    const char* context) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        const int local_invalid = local_count > static_cast<std::size_t>(
            std::numeric_limits<std::uint64_t>::max())
            ? 1 : 0;
        int any_invalid = 0;
        if (MPI_Allreduce(
                &local_invalid,
                &any_invalid,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                std::string("MPI_Allreduce failed for ") + context);
        }
        if (any_invalid != 0) {
            throw std::overflow_error(
                std::string(context) + " local count exceeds uint64_t");
        }
    }
    try {
        return math::mpi_exact_uint64_sum(
            static_cast<std::uint64_t>(local_count));
    } catch (const std::overflow_error&) {
        throw std::overflow_error(
            std::string(context) + " global count overflows uint64_t");
    }
}

std::optional<core::Real> collective_uniform_mass_representation(
    const core::ParticleStore& particles,
    int rank,
    int size) {
    const std::size_t owned = particles.num_owned_particles();
    const auto local_uniform_mass = particles.get_uniform_mass();

    const int local_explicit_nonempty =
        owned > 0 && !local_uniform_mass.has_value() ? 1 : 0;
    int any_explicit_nonempty = 0;
    if (MPI_Allreduce(
            &local_explicit_nonempty,
            &any_explicit_nonempty,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed while selecting domain-partition mass representation");
    }
    if (any_explicit_nonempty != 0) {
        return std::nullopt;
    }

    const int local_candidate =
        owned > 0 && local_uniform_mass.has_value() ? rank : size;
    int source_rank = size;
    if (MPI_Allreduce(
            &local_candidate,
            &source_rank,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed while validating uniform particle mass");
    }
    if (source_rank == size) {
        return std::nullopt;
    }

    core::Real agreed_mass = rank == source_rank
        ? *local_uniform_mass
        : 0.0;
    if (MPI_Bcast(
            &agreed_mass,
            1,
            MPI_DOUBLE,
            source_rank,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Bcast failed for domain-partition uniform mass");
    }

    const int local_mismatch =
        owned > 0
        && (!local_uniform_mass.has_value()
            || *local_uniform_mass != agreed_mass)
        ? 1 : 0;
    int any_mismatch = 0;
    if (MPI_Allreduce(
            &local_mismatch,
            &any_mismatch,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed while validating domain-partition uniform mass");
    }
    if (any_mismatch != 0) {
        throw std::runtime_error(
            "Non-empty MPI ranks disagree on the uniform particle mass");
    }
    return agreed_mass;
}
#endif

} // namespace

DomainDecomposition::DomainDecomposition(
    const config::SimulationParameters& config,
    int rank,
    int size)
    : config_(config),
      rank_(rank),
      size_(size),
      ghost_exchange_(
          config.get_box().L,
          static_cast<std::size_t>(config.get_box().N_mesh),
          rank,
          size) {
    if (size_ < 1 || rank_ < 0 || rank_ >= size_) {
        throw std::invalid_argument(
            "DomainDecomposition rank topology is invalid");
    }
    mesh_size_ = static_cast<std::size_t>(config_.get_box().N_mesh);
    if (static_cast<std::size_t>(size_) > mesh_size_) {
        throw std::invalid_argument(
            "DomainDecomposition requires MPI rank count <= N_mesh");
    }
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (mesh::mpi_slab_layout_has_empty_rank(mesh_size_, size_)) {
        throw std::invalid_argument(
            "Current FFTW-MPI slab layout requires every MPI rank to own at least one mesh plane");
    }
#endif

    const core::Real L = config_.get_box().L;
    mesh_cell_size_ = L / static_cast<core::Real>(mesh_size_);

    // Materialize the logical ownership topology once. Particle routing is a
    // hot path and should not rediscover every rank's FFT slab for every
    // particle. Exclusive end planes provide a compact monotone representation
    // that also handles any zero-width ranks without per-case branches.
    slab_end_planes_.reserve(static_cast<std::size_t>(size_));
    std::size_t expected_start = 0;
    mesh::MpiSlabLayout local_layout{};
    for (int owner = 0; owner < size_; ++owner) {
        const auto layout = mesh::mpi_slab_layout(
            mesh_size_, owner, size_);
        if (layout.start != expected_start
            || layout.start > mesh_size_
            || layout.count > mesh_size_ - layout.start) {
            throw std::logic_error(
                "MPI slab ownership does not form a contiguous bounded partition");
        }
        const std::size_t end = layout.start + layout.count;
        slab_end_planes_.push_back(end);
        expected_start = end;
        if (owner == rank_) local_layout = layout;
    }
    if (expected_start != mesh_size_) {
        throw std::logic_error(
            "MPI slab ownership does not cover the global mesh extent");
    }

    local_bounds_ = {
        L * static_cast<core::Real>(local_layout.start)
            / static_cast<core::Real>(mesh_size_),
        L * static_cast<core::Real>(local_layout.start + local_layout.count)
            / static_cast<core::Real>(mesh_size_),
        0.0, L,
        0.0, L
    };
}

int DomainDecomposition::owner_rank(core::Real x) const {
    if (!std::isfinite(x)) {
        throw std::invalid_argument(
            "Particle x coordinate must be finite");
    }
    const core::Real wrapped = math::wrap(x, config_.get_box().L);
    std::size_t cell = static_cast<std::size_t>(std::floor(
        wrapped / mesh_cell_size_));
    if (cell >= mesh_size_) cell = mesh_size_ - 1;

    const auto owner = std::upper_bound(
        slab_end_planes_.begin(), slab_end_planes_.end(), cell);
    if (owner == slab_end_planes_.end()) {
        throw std::logic_error(
            "No MPI owner rank found for represented mesh cell");
    }
    return static_cast<int>(owner - slab_end_planes_.begin());
}

std::ptrdiff_t DomainDecomposition::get_local_slab(
    std::ptrdiff_t N,
    std::ptrdiff_t& local_n0,
    std::ptrdiff_t& local_0_start) const {
    if (N <= 0) {
        throw std::invalid_argument(
            "Mesh slab dimension must be positive");
    }
    const auto layout = mesh::mpi_slab_layout(
        static_cast<std::size_t>(N), rank_, size_);
    local_n0 = static_cast<std::ptrdiff_t>(layout.count);
    local_0_start = static_cast<std::ptrdiff_t>(layout.start);

    const std::ptrdiff_t nz_complex = N / 2 + 1;
    if (local_n0 != 0
        && N > std::numeric_limits<std::ptrdiff_t>::max() / local_n0) {
        throw std::overflow_error(
            "Local complex slab size overflows ptrdiff_t");
    }
    const std::ptrdiff_t plane_product = local_n0 * N;
    if (plane_product != 0
        && nz_complex
            > std::numeric_limits<std::ptrdiff_t>::max() / plane_product) {
        throw std::overflow_error(
            "Local complex slab size overflows ptrdiff_t");
    }
    return plane_product * nz_complex;
}

std::string DomainDecomposition::collect_initial_condition_provenance(
    const core::ParticleStore& local_particles) const {
    if (initial_condition_provenance_synchronized_) {
        return local_particles.get_verified_snapshot_ic_sha256();
    }
    if (size_ == 1) {
        return local_particles.get_verified_snapshot_ic_sha256();
    }
#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        "Multi-rank provenance synchronization requires an MPI-enabled build");
#else
    return runtime::agree_optional_sha256_collective(
        local_particles.get_verified_snapshot_ic_sha256(),
        "initial-condition provenance");
#endif
}

void DomainDecomposition::partition_domain(
    core::ParticleStore& local_particles) {
    int local_validation_failed = 0;
    std::string local_validation_error;
    std::exception_ptr local_validation_exception;
    try {
        validate_partition_input(local_particles);
    } catch (const std::exception& error) {
        local_validation_failed = 1;
        local_validation_error = error.what();
        local_validation_exception = std::current_exception();
    } catch (...) {
        local_validation_failed = 1;
        local_validation_error = "unknown domain partition input failure";
        local_validation_exception = std::current_exception();
    }

#ifdef COSMO_NBODY_HAS_MPI
    if (size_ > 1) {
        runtime::require_active_mpi_main_thread("Domain partition");
    }
#endif

    if (size_ == 1) {
        if (local_validation_failed != 0) {
            if (local_validation_exception) {
                std::rethrow_exception(local_validation_exception);
            }
            throw std::runtime_error(local_validation_error);
        }
    } else {
#ifndef COSMO_NBODY_HAS_MPI
        throw std::runtime_error(
            "Multi-rank DomainDecomposition requires an MPI-enabled build");
#else
        int any_validation_failed = 0;
        if (MPI_Allreduce(
                &local_validation_failed,
                &any_validation_failed,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "MPI_Allreduce failed for domain partition input preflight");
        }
        if (any_validation_failed != 0) {
            if (local_validation_failed != 0
                && !local_validation_error.empty()) {
                throw std::runtime_error(
                    "Domain partition input preflight failed: "
                    + local_validation_error);
            }
            throw std::runtime_error(
                "Domain partition input preflight failed on another MPI rank");
        }
#endif
    }

    const std::string verified_snapshot_ic_sha256 =
        collect_initial_condition_provenance(local_particles);
    const core::Real L = config_.get_box().L;
    const std::size_t owned = local_particles.num_owned_particles();

    if (size_ == 1) {
        local_particles.clear_ghosts();
        auto px = local_particles.get_positions_x().first(owned);
        auto py = local_particles.get_positions_y().first(owned);
        auto pz = local_particles.get_positions_z().first(owned);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) \
            if(runtime::should_use_host_parallel_team(owned))
#endif
        for (std::size_t i = 0; i < owned; ++i) {
            px[i] = math::wrap(px[i], L);
            py[i] = math::wrap(py[i], L);
            pz[i] = math::wrap(pz[i], L);
        }
        local_particles.mark_ghost_start();
        local_particles.set_ghost_validity(core::FieldValidity::VALID);
        config_.set_verified_snapshot_ic_sha256(
            verified_snapshot_ic_sha256);
        local_particles.set_verified_snapshot_ic_sha256(
            verified_snapshot_ic_sha256);
        initial_condition_provenance_synchronized_ = true;
        canonical_partition_established_ = true;
        return;
    }

#ifndef COSMO_NBODY_HAS_MPI
    throw std::runtime_error(
        "Multi-rank DomainDecomposition requires an MPI-enabled build");
#else
    const auto migrated_uniform_mass =
        collective_uniform_mass_representation(
            local_particles, rank_, size_);
    const auto px = local_particles.get_positions_x().first(owned);
    const auto py = local_particles.get_positions_y().first(owned);
    const auto pz = local_particles.get_positions_z().first(owned);
    const auto mx = local_particles.get_momenta_x().first(owned);
    const auto my = local_particles.get_momenta_y().first(owned);
    const auto mz = local_particles.get_momenta_z().first(owned);
    const auto ids = local_particles.get_ids().first(owned);

    int local_ids_sorted = 1;
    for (std::size_t i = 1; i < owned; ++i) {
        if (ids[i - 1] >= ids[i]) {
            local_ids_sorted = 0;
            break;
        }
    }

    int local_can_retain = local_ids_sorted;
    const auto local_uniform_mass = local_particles.get_uniform_mass();
    if (owned > 0
        && (local_uniform_mass.has_value()
            != migrated_uniform_mass.has_value()
            || (local_uniform_mass.has_value()
                && *local_uniform_mass != *migrated_uniform_mass))) {
        local_can_retain = 0;
    }
    for (std::size_t i = 0; i < owned && local_can_retain != 0; ++i) {
        if (owner_rank(px[i]) != rank_) {
            local_can_retain = 0;
        }
    }

    int all_can_retain = 0;
    if (MPI_Allreduce(
            &local_can_retain,
            &all_can_retain,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed for domain no-migration admission");
    }
    if (all_can_retain != 0) {
        std::exception_ptr stage_exception;
        try {
            config_.set_verified_snapshot_ic_sha256(
                verified_snapshot_ic_sha256);
            local_particles.set_verified_snapshot_ic_sha256(
                verified_snapshot_ic_sha256);
        } catch (...) {
            stage_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            stage_exception, size_, "Domain no-migration publication");

        local_particles.clear_ghosts();
        local_particles.set_uniform_mass(migrated_uniform_mass);
        auto retained_x = local_particles.get_positions_x().first(owned);
        auto retained_y = local_particles.get_positions_y().first(owned);
        auto retained_z = local_particles.get_positions_z().first(owned);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) \
            if(runtime::should_use_host_parallel_team(owned))
#endif
        for (std::size_t i = 0; i < owned; ++i) {
            retained_x[i] = math::wrap(retained_x[i], L);
            retained_y[i] = math::wrap(retained_y[i], L);
            retained_z[i] = math::wrap(retained_z[i], L);
        }
        // Derived fields do not survive a partition boundary, and the full
        // migration path below marks the rebuilt store INVALID. The fast path
        // must land in the same state, otherwise skipping the rebuild would
        // silently change semantics rather than just skipping work.
        //
        // Retire the values but retain their storage. The force path explicitly
        // overwrites all owned entries after partition; releasing three O(N)
        // arrays here would otherwise allocate and free them at every refresh
        // even though ownership and order did not change.
        local_particles.set_acceleration_validity(core::FieldValidity::INVALID);
        local_particles.mark_ghost_start();
        local_particles.set_ghost_validity(core::FieldValidity::INVALID);
        initial_condition_provenance_synchronized_ = true;
        canonical_partition_established_ = true;
        return;
    }

    int all_ids_sorted = 0;
    if (MPI_Allreduce(
            &local_ids_sorted,
            &all_ids_sorted,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed for domain migration ordering admission");
    }

    // Once this DomainDecomposition has completed a collective partition, its
    // owned stores are published in stable-ID order. Subsequent force refreshes
    // can therefore migrate only the particles that actually leave a slab and
    // merge incoming records into the retained sorted subsequence in place.
    // The transport accounts for every outgoing record exactly once and the
    // global integer population is checked below, so repeating an O(N_local)
    // global-ID redistribution on every migration is unnecessary. First-use or
    // non-canonical inputs keep the exact route-all fallback below.
    if (canonical_partition_established_ && all_ids_sorted != 0) {
        const std::uint64_t population_before = collective_particle_population(
            owned, "domain migration input population");

        std::exception_ptr stage_exception;
        try {
            // Acceleration values and old ghosts are derived state that never
            // survive a partition boundary. Retire the values here; owned-store
            // publication below reuses their capacity when the new local count
            // fits and omits it before growth otherwise.
            local_particles.clear_ghosts();
            local_particles.set_acceleration_validity(
                core::FieldValidity::INVALID);
        } catch (...) {
            stage_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            stage_exception, size_, "Domain sparse migration derived-state release");

        std::size_t retained_count = 0;
        std::vector<RoutedExchangeParticle> outgoing;
        stage_exception = nullptr;
        try {
            std::size_t outgoing_count = 0;
            for (std::size_t i = 0; i < owned; ++i) {
                if (owner_rank(px[i]) == rank_) {
                    ++retained_count;
                } else {
                    ++outgoing_count;
                }
            }
            if (retained_count > owned || outgoing_count != owned - retained_count) {
                throw std::logic_error(
                    "Domain migration retained/outgoing accounting is inconsistent");
            }
            outgoing.reserve(outgoing_count);
            for (std::size_t i = 0; i < owned; ++i) {
                const core::Real wrapped_x = math::wrap(px[i], L);
                const int destination = owner_rank(wrapped_x);
                if (destination == rank_) continue;

                ExchangeParticle particle;
                particle.x = wrapped_x;
                particle.y = math::wrap(py[i], L);
                particle.z = math::wrap(pz[i], L);
                particle.px = mx[i];
                particle.py = my[i];
                particle.pz = mz[i];
                particle.mass = local_particles.mass_at(i);
                particle.id = ids[i];
                outgoing.push_back({destination, particle});
            }
            if (outgoing.size() != outgoing_count) {
                throw std::logic_error(
                    "Domain migration outgoing materialization is incomplete");
            }
            ExchangeBuffer::sort_routed_in_place(outgoing);
        } catch (...) {
            stage_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            stage_exception, size_, "Domain sparse migration routing");

        auto incoming = exchange_migration_bounded(
            outgoing, rank_, size_);

        std::size_t final_count = 0;
        stage_exception = nullptr;
        try {
            if (incoming.size()
                > std::numeric_limits<std::size_t>::max() - retained_count) {
                throw std::overflow_error(
                    "Domain sparse migration output population overflows size_t");
            }
            final_count = retained_count + incoming.size();

            std::sort(
                incoming.begin(), incoming.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.id < rhs.id;
                });
            for (std::size_t i = 0; i < incoming.size(); ++i) {
                const auto& particle = incoming[i];
                validate_received_exchange_particle(particle, L);
                if (owner_rank(particle.x) != rank_) {
                    throw std::runtime_error(
                        "Received migrated particle is outside local ownership slab");
                }
                if (migrated_uniform_mass.has_value()
                    && particle.mass != *migrated_uniform_mass) {
                    throw std::runtime_error(
                        "Received migrated particle disagrees with uniform mass");
                }
                if (i > 0 && incoming[i - 1].id >= particle.id) {
                    throw std::runtime_error(
                        "Duplicate owned particle ID in sparse migration receive set");
                }
            }

            // Retained IDs form a sorted subsequence of the pre-migration
            // store. Check the only new local collision class before mutating
            // the store: a retained ID equal to an incoming ID.
            std::size_t incoming_index = 0;
            for (std::size_t i = 0;
                 i < owned && incoming_index < incoming.size();
                 ++i) {
                if (owner_rank(px[i]) != rank_) continue;
                const core::ParticleId retained_id = ids[i];
                while (incoming_index < incoming.size()
                       && incoming[incoming_index].id < retained_id) {
                    ++incoming_index;
                }
                if (incoming_index < incoming.size()
                    && incoming[incoming_index].id == retained_id) {
                    throw std::runtime_error(
                        "Duplicate owned particle ID between retained and incoming migration sets");
                }
            }
        } catch (...) {
            stage_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            stage_exception, size_, "Domain sparse migration receive validation");

        const std::uint64_t population_after = collective_particle_population(
            final_count, "domain migration output population");
        if (population_after != population_before) {
            throw std::runtime_error(
                "Domain migration population changed across inter-rank transport");
        }

        stage_exception = nullptr;
        try {
            local_particles.set_uniform_mass(migrated_uniform_mass);

            auto compact_x = local_particles.get_positions_x().first(owned);
            auto compact_y = local_particles.get_positions_y().first(owned);
            auto compact_z = local_particles.get_positions_z().first(owned);
            auto compact_px = local_particles.get_momenta_x().first(owned);
            auto compact_py = local_particles.get_momenta_y().first(owned);
            auto compact_pz = local_particles.get_momenta_z().first(owned);
            auto compact_ids = local_particles.get_ids().first(owned);
            const bool explicit_mass = !migrated_uniform_mass.has_value();
            auto compact_masses = explicit_mass
                ? local_particles.get_masses().first(owned)
                : std::span<core::Real>{};

            std::size_t write = 0;
            for (std::size_t read = 0; read < owned; ++read) {
                if (owner_rank(compact_x[read]) != rank_) continue;

                const core::Real wrapped_x = math::wrap(compact_x[read], L);
                const core::Real wrapped_y = math::wrap(compact_y[read], L);
                const core::Real wrapped_z = math::wrap(compact_z[read], L);
                if (write != read) {
                    compact_px[write] = compact_px[read];
                    compact_py[write] = compact_py[read];
                    compact_pz[write] = compact_pz[read];
                    compact_ids[write] = compact_ids[read];
                    if (explicit_mass) {
                        compact_masses[write] = compact_masses[read];
                    }
                }
                compact_x[write] = wrapped_x;
                compact_y[write] = wrapped_y;
                compact_z[write] = wrapped_z;
                ++write;
            }
            if (write != retained_count) {
                throw std::logic_error(
                    "Domain sparse migration retained compaction count changed");
            }

            // resize() does not shrink primary-vector capacity. In the common
            // balanced case, growing from retained_count to final_count therefore
            // reuses the original local capacity instead of allocating a second
            // full ParticleStore.
            local_particles.resize_owned_reusing_acceleration_storage(
                retained_count);
            local_particles.resize_owned_reusing_acceleration_storage(
                final_count);

            auto merged_x = local_particles.get_positions_x().first(final_count);
            auto merged_y = local_particles.get_positions_y().first(final_count);
            auto merged_z = local_particles.get_positions_z().first(final_count);
            auto merged_px = local_particles.get_momenta_x().first(final_count);
            auto merged_py = local_particles.get_momenta_y().first(final_count);
            auto merged_pz = local_particles.get_momenta_z().first(final_count);
            auto merged_ids = local_particles.get_ids().first(final_count);
            auto merged_masses = explicit_mass
                ? local_particles.get_masses().first(final_count)
                : std::span<core::Real>{};

            const auto copy_retained = [&](std::size_t source, std::size_t destination) {
                if (source == destination) return;
                merged_x[destination] = merged_x[source];
                merged_y[destination] = merged_y[source];
                merged_z[destination] = merged_z[source];
                merged_px[destination] = merged_px[source];
                merged_py[destination] = merged_py[source];
                merged_pz[destination] = merged_pz[source];
                merged_ids[destination] = merged_ids[source];
                if (explicit_mass) {
                    merged_masses[destination] = merged_masses[source];
                }
            };
            const auto copy_incoming = [&](const ExchangeParticle& particle,
                                           std::size_t destination) {
                merged_x[destination] = particle.x;
                merged_y[destination] = particle.y;
                merged_z[destination] = particle.z;
                merged_px[destination] = particle.px;
                merged_py[destination] = particle.py;
                merged_pz[destination] = particle.pz;
                merged_ids[destination] = particle.id;
                if (explicit_mass) {
                    merged_masses[destination] = particle.mass;
                }
            };

            std::size_t retained_cursor = retained_count;
            std::size_t incoming_cursor = incoming.size();
            std::size_t output_cursor = final_count;
            while (retained_cursor > 0 && incoming_cursor > 0) {
                const core::ParticleId retained_id =
                    merged_ids[retained_cursor - 1];
                const core::ParticleId incoming_id =
                    incoming[incoming_cursor - 1].id;
                if (retained_id == incoming_id) {
                    throw std::logic_error(
                        "Domain sparse migration merge encountered duplicate IDs");
                }
                --output_cursor;
                if (retained_id > incoming_id) {
                    --retained_cursor;
                    copy_retained(retained_cursor, output_cursor);
                } else {
                    --incoming_cursor;
                    copy_incoming(incoming[incoming_cursor], output_cursor);
                }
            }
            while (incoming_cursor > 0) {
                --incoming_cursor;
                --output_cursor;
                copy_incoming(incoming[incoming_cursor], output_cursor);
            }
            if (output_cursor != retained_cursor) {
                throw std::logic_error(
                    "Domain sparse migration merge accounting is inconsistent");
            }
            for (std::size_t i = 1; i < final_count; ++i) {
                if (merged_ids[i - 1] >= merged_ids[i]) {
                    throw std::logic_error(
                        "Domain sparse migration did not publish strict stable-ID order");
                }
            }

            local_particles.set_verified_snapshot_ic_sha256(
                verified_snapshot_ic_sha256);
            local_particles.mark_ghost_start();
            local_particles.set_ghost_validity(core::FieldValidity::INVALID);
            config_.set_verified_snapshot_ic_sha256(
                verified_snapshot_ic_sha256);
        } catch (...) {
            stage_exception = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            stage_exception, size_, "Domain sparse migration publication");
        initial_condition_provenance_synchronized_ = true;
        canonical_partition_established_ = true;
        return;
    }

    const std::uint64_t population_before = collective_particle_population(
        owned, "domain migration input population");
    const GlobalParticleIdSetSnapshot id_set_before =
        capture_global_particle_id_set(ids);

    std::vector<RoutedExchangeParticle> routed;
    std::exception_ptr stage_exception;
    try {
        routed.reserve(owned);
        for (std::size_t i = 0; i < owned; ++i) {
            ExchangeParticle particle;
            particle.x = math::wrap(px[i], L);
            particle.y = math::wrap(py[i], L);
            particle.z = math::wrap(pz[i], L);
            particle.px = mx[i];
            particle.py = my[i];
            particle.pz = mz[i];
            particle.mass = local_particles.mass_at(i);
            particle.id = local_particles.get_ids()[i];
            routed.push_back({owner_rank(particle.x), particle});
        }
        ExchangeBuffer::sort_routed_in_place(routed);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Domain migration routing");

    auto received = exchange_migration_bounded(
        routed, rank_, size_);
    const std::uint64_t population_after = collective_particle_population(
        received.size(), "domain migration output population");
    if (population_after != population_before) {
        throw std::runtime_error(
            "Domain migration population changed across inter-rank transport");
    }

    core::ParticleStore rebuilt;
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
                    "Duplicate owned particle ID after migration");
            }
        }

        rebuilt.set_uniform_mass(migrated_uniform_mass);
        rebuilt.resize(received.size());
        rebuilt.set_verified_snapshot_ic_sha256(
            verified_snapshot_ic_sha256);
        for (std::size_t i = 0; i < received.size(); ++i) {
            const auto& particle = received[i];
            validate_received_exchange_particle(particle, L);
            if (owner_rank(particle.x) != rank_) {
                throw std::runtime_error(
                    "Received migrated particle is outside local ownership slab");
            }
            if (migrated_uniform_mass.has_value()
                && particle.mass != *migrated_uniform_mass) {
                throw std::runtime_error(
                    "Received migrated particle disagrees with uniform mass");
            }
            rebuilt.get_positions_x()[i] = particle.x;
            rebuilt.get_positions_y()[i] = particle.y;
            rebuilt.get_positions_z()[i] = particle.z;
            rebuilt.get_momenta_x()[i] = particle.px;
            rebuilt.get_momenta_y()[i] = particle.py;
            rebuilt.get_momenta_z()[i] = particle.pz;
            if (!migrated_uniform_mass.has_value()) {
                rebuilt.get_masses()[i] = particle.mass;
            }
            rebuilt.get_ids()[i] = particle.id;
        }
        rebuilt.set_acceleration_validity(core::FieldValidity::INVALID);
        rebuilt.set_ghost_validity(core::FieldValidity::INVALID);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Domain migration rebuild");

    // The complete received payload has been semantically checked and copied to
    // the private rebuilt store on every rank. Release its capacity before the
    // exact post-migration ID canonicalization allocates a second distributed
    // bucket; live state is still unpublished until all remaining checks pass.
    std::vector<ExchangeParticle>().swap(received);
    require_global_particle_id_set_preserved(
        id_set_before,
        rebuilt.get_ids().first(rebuilt.num_owned_particles()));

    stage_exception = nullptr;
    try {
        config_.set_verified_snapshot_ic_sha256(
            verified_snapshot_ic_sha256);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size_, "Domain migration publication");
    local_particles = std::move(rebuilt);
    initial_condition_provenance_synchronized_ = true;
    canonical_partition_established_ = true;
#endif
}

void DomainDecomposition::exchange_ghosts(
    core::ParticleStore& local_particles,
    core::Real r_search) {
    ghost_exchange_.exchange(local_particles, r_search);
}

} // namespace domain
} // namespace cosmo_nbody
