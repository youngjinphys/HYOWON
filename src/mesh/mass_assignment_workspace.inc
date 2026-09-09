#include "cosmo_nbody/mesh/mass_assignment.hpp"
#include "cosmo_nbody/math/deterministic_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace mesh {

std::uint64_t cic_stable_index_bytes(std::uint64_t particle_count) {
    constexpr std::uint64_t compact_capacity =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) + 1ULL;
    const std::uint64_t element_bytes = particle_count <= compact_capacity
        ? sizeof(std::uint32_t)
        : sizeof(std::size_t);
    if (particle_count != 0
        && element_bytes
            > std::numeric_limits<std::uint64_t>::max() / particle_count) {
        throw std::overflow_error(
            "CIC stable particle-index byte size overflows uint64_t");
    }
    return particle_count * element_bytes;
}

namespace detail {

void require_position_sizes(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z) {
    if (pos_x.size() != pos_y.size() || pos_x.size() != pos_z.size()) {
        throw std::invalid_argument(
            "CIC position component sizes must match");
    }
}

void require_finite_positions(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z) {
    int invalid = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|: invalid) \
        if(runtime::should_use_host_parallel_team(pos_x.size()))
#endif
    for (std::size_t idx = 0; idx < pos_x.size(); ++idx) {
        if (!std::isfinite(pos_x[idx])
            || !std::isfinite(pos_y[idx])
            || !std::isfinite(pos_z[idx])) {
            invalid = 1;
        }
    }
    if (invalid != 0) {
        throw std::invalid_argument("CIC positions must be finite");
    }
}

class StableParticleOrder {
public:
    void ensure(
        std::size_t particle_count,
        const config::MemoryPolicyParams& policy) {
        const bool compact =
            static_cast<std::uint64_t>(particle_count)
            <= static_cast<std::uint64_t>(
                   std::numeric_limits<std::uint32_t>::max()) + 1ULL;
        const std::uint64_t bytes_u64 = cic_stable_index_bytes(
            static_cast<std::uint64_t>(particle_count));
        if (bytes_u64 > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "CIC stable particle-index bytes exceed size_t");
        }
        const std::size_t bytes = static_cast<std::size_t>(bytes_u64);
        // A reusable evolution workspace is a capacity, not an exact current
        // particle count. Distributed slab ownership can shrink and grow after
        // migration; keep the previous high-water allocation whenever its index
        // representation and byte extent can hold the current population.
        if (storage_ && storage_->size_bytes() >= bytes && compact_ == compact) {
            return;
        }
        storage_.reset();
        compact_ = compact;
        storage_ = std::make_unique<runtime::RawScratchBuffer>(
            bytes,
            policy.evolution_scratch_mode,
            policy.scratch_directory,
            "cic_particle_order");
    }

    void release() noexcept { storage_.reset(); }

    std::size_t retained_bytes() const noexcept {
        return storage_ ? storage_->size_bytes() : 0;
    }

    void set(std::size_t cursor, std::size_t particle) {
        if (!storage_) {
            throw std::logic_error("CIC stable particle order is not allocated");
        }
        if (compact_) {
            if (particle > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "CIC compact particle index exceeds uint32_t");
            }
            static_cast<std::uint32_t*>(storage_->data())[cursor] =
                static_cast<std::uint32_t>(particle);
        } else {
            static_cast<std::size_t*>(storage_->data())[cursor] = particle;
        }
    }

    std::size_t at(std::size_t cursor) const {
        if (!storage_) {
            throw std::logic_error("CIC stable particle order is not allocated");
        }
        if (compact_) {
            return static_cast<std::size_t>(
                static_cast<const std::uint32_t*>(storage_->data())[cursor]);
        }
        return static_cast<const std::size_t*>(storage_->data())[cursor];
    }

private:
    bool compact_{true};
    std::unique_ptr<runtime::RawScratchBuffer> storage_;
};

std::size_t saturating_capacity_bytes(
    std::size_t capacity,
    std::size_t element_bytes) noexcept {
    if (capacity > std::numeric_limits<std::size_t>::max() / element_bytes) {
        return std::numeric_limits<std::size_t>::max();
    }
    return capacity * element_bytes;
}

std::size_t saturating_add(
    std::size_t left,
    std::size_t right) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return std::numeric_limits<std::size_t>::max();
    }
    return left + right;
}

struct XPlaneStorage {
    StableParticleOrder particles;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> thread_counts;
    std::vector<std::size_t> thread_offsets;

    void release() noexcept {
        particles.release();
        std::vector<std::size_t>().swap(offsets);
        std::vector<std::size_t>().swap(thread_counts);
        std::vector<std::size_t>().swap(thread_offsets);
    }

    std::size_t retained_bytes() const noexcept {
        std::size_t total = particles.retained_bytes();
        total = saturating_add(total, saturating_capacity_bytes(
            offsets.capacity(), sizeof(std::size_t)));
        total = saturating_add(total, saturating_capacity_bytes(
            thread_counts.capacity(), sizeof(std::size_t)));
        total = saturating_add(total, saturating_capacity_bytes(
            thread_offsets.capacity(), sizeof(std::size_t)));
        return total;
    }
};

struct XPlaneBins {
    const std::vector<std::size_t>& offsets;
    const StableParticleOrder& particles;
};

// Stable counting-sort by the wrapped base x-cell. Threads own contiguous
// particle ranges and their per-bin ranges are concatenated in thread order, so
// every bin preserves the original particle order independent of thread count.
// That makes plane accumulation reproducible while avoiding atomics in the
// eight-contribution CIC hot loop.
XPlaneBins build_x_plane_bins(
    std::span<const core::Real> pos_x,
    core::Real box_size,
    core::Real cell_size,
    const MeshGeometry& geom,
    const config::MemoryPolicyParams& policy,
    const CICDepositMemoryPlan& memory_plan,
    XPlaneStorage& storage) {
    const std::size_t particle_count = pos_x.size();
    const std::size_t mesh_size = geom.grid_size();

    storage.particles.ensure(particle_count, policy);
    storage.offsets.assign(mesh_size + 1, 0);
    if (particle_count == 0) {
        storage.thread_counts.clear();
        storage.thread_offsets.clear();
        return {storage.offsets, storage.particles};
    }

    if (memory_plan.binning_thread_count
        > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::logic_error(
            "CIC workspace plan returned a thread count outside the OpenMP int range");
    }
    if (memory_plan.binning_thread_count == 0) {
        throw std::logic_error(
            "Non-empty CIC binning received a zero-thread memory plan");
    }
    const int thread_count = static_cast<int>(
        memory_plan.binning_thread_count);

    if (mesh_size > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(thread_count)) {
        throw std::overflow_error(
            "CIC x-bin thread-count workspace overflows size_t");
    }

    storage.thread_counts.assign(
        static_cast<std::size_t>(thread_count) * mesh_size, 0);
    storage.thread_offsets.assign(
        static_cast<std::size_t>(thread_count) * mesh_size, 0);
    auto& thread_counts = storage.thread_counts;
    auto& thread_offsets = storage.thread_offsets;

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel num_threads(thread_count) if(thread_count > 1)
#endif
    {
#ifdef COSMO_NBODY_HAS_OPENMP
        const int raw_thread = omp_get_thread_num();
        const int raw_team = omp_get_num_threads();
#else
        const int raw_thread = 0;
        const int raw_team = 1;
#endif
        const std::size_t thread = static_cast<std::size_t>(raw_thread);
        const std::size_t team = static_cast<std::size_t>(raw_team);
        const std::size_t begin = particle_count * thread / team;
        const std::size_t end = particle_count * (thread + 1) / team;
        auto* local_counts =
            thread_counts.data() + thread * mesh_size;

        for (std::size_t particle = begin; particle < end; ++particle) {
            const auto coordinate = cic_coordinate_1d(
                pos_x[particle], box_size, cell_size, mesh_size);
            const auto base = static_cast<std::size_t>(coordinate.base);
            ++local_counts[base];
        }

#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp barrier
        #pragma omp single
#endif
        {
            for (std::size_t bin = 0; bin < mesh_size; ++bin) {
                std::size_t count = 0;
                for (int owner = 0; owner < thread_count; ++owner) {
                    count += thread_counts[
                        static_cast<std::size_t>(owner) * mesh_size + bin];
                }
                storage.offsets[bin + 1] = storage.offsets[bin] + count;
            }

            for (std::size_t bin = 0; bin < mesh_size; ++bin) {
                std::size_t cursor = storage.offsets[bin];
                for (int owner = 0; owner < thread_count; ++owner) {
                    const std::size_t index =
                        static_cast<std::size_t>(owner) * mesh_size + bin;
                    thread_offsets[index] = cursor;
                    cursor += thread_counts[index];
                }
            }
        }

#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp barrier
#endif
        auto* local_offsets =
            thread_offsets.data() + thread * mesh_size;
        for (std::size_t particle = begin; particle < end; ++particle) {
            const auto coordinate = cic_coordinate_1d(
                pos_x[particle], box_size, cell_size, mesh_size);
            const auto base = static_cast<std::size_t>(coordinate.base);
            storage.particles.set(local_offsets[base]++, particle);
        }
    }

    return {storage.offsets, storage.particles};
}

void zero_buffer(std::span<core::Real> values) {
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        if(runtime::should_use_host_parallel_team(values.size()))
#endif
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = core::Real{0.0};
    }
}

} // namespace detail

using detail::XPlaneBins;
using detail::XPlaneStorage;
using detail::build_x_plane_bins;
using detail::cic_coordinate_1d;
using detail::require_finite_positions;
using detail::require_position_sizes;
using detail::zero_buffer;

struct CICDepositWorkspace::Impl {
    XPlaneStorage bins;
};

CICDepositWorkspace::CICDepositWorkspace()
    : impl_(std::make_unique<Impl>()) {}

CICDepositWorkspace::~CICDepositWorkspace() = default;
CICDepositWorkspace::CICDepositWorkspace(CICDepositWorkspace&&) noexcept = default;
CICDepositWorkspace& CICDepositWorkspace::operator=(
    CICDepositWorkspace&&) noexcept = default;

void CICDepositWorkspace::release() noexcept {
    if (impl_) impl_->bins.release();
}

std::size_t CICDepositWorkspace::retained_bytes() const noexcept {
    return impl_ ? impl_->bins.retained_bytes() : 0;
}

CICMassAssignment::CICMassAssignment(
    const MeshGeometry& geom,
    config::MemoryPolicyParams memory_policy)
    : geom_(geom),
      dx_(geom.cell_size()),
      memory_policy_(std::move(memory_policy)) {}

void CICMassAssignment::deposit(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    RealField& field,
    RealField* right_ghost) const {
    deposit_impl(
        pos_x, pos_y, pos_z, masses, uniform_mass, nullptr,
        field, right_ghost);
}

void CICMassAssignment::deposit_reusing_workspace(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    CICDepositWorkspace& workspace,
    RealField& field,
    RealField* right_ghost) const {
    if (!workspace.impl_) {
        throw std::logic_error(
            "CIC reusable workspace is in a moved-from state");
    }
    deposit_impl(
        pos_x, pos_y, pos_z, masses, uniform_mass, &workspace,
        field, right_ghost);
}
