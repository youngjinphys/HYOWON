// Native HDF5 snapshot I/O. Internal p=a*v_pec is stored as declared peculiar
// velocity; unknown native conventions fail closed.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmo_nbody {
namespace io {

// Bounded internal hyperslab size; an I/O memory coordinate, not physics/output.
inline constexpr std::size_t SNAPSHOT_READ_BATCH_PARTICLES =
    std::size_t{1} << 20;

enum class SnapshotWritePolicy {
    RequireAbsent,
    ReplaceExisting,
};

enum class SnapshotReadPolicy {
    AnalysisSubset,
    ExactInitialConditions
};

// Immutable physical facts needed to admit a native snapshot payload.
struct SnapshotReadContext {
    core::Real box_size_Mpc_h{0.0};
    core::Real omega_m{0.0};
    core::Real omega_lambda{0.0};
    core::Real hubble_param{0.0};
    std::size_t population_limit{0};
};

// Per-read particle-count admission checked before allocation.
class SnapshotPopulationAdmission {
public:
    static SnapshotPopulationAdmission exact(std::size_t count) noexcept {
        return SnapshotPopulationAdmission(Kind::Exact, count);
    }

    static SnapshotPopulationAdmission maximum(std::size_t count) noexcept {
        return SnapshotPopulationAdmission(Kind::Maximum, count);
    }

    void require(std::size_t observed) const {
        if (kind_ == Kind::Exact && observed != count_) {
            throw std::runtime_error(
                "Snapshot particle count " + std::to_string(observed)
                + " does not match the admitted exact count "
                + std::to_string(count_));
        }
        if (kind_ == Kind::Maximum && observed > count_) {
            throw std::runtime_error(
                "Snapshot particle count " + std::to_string(observed)
                + " exceeds the admitted maximum count "
                + std::to_string(count_));
        }
    }

private:
    enum class Kind {
        Unbounded,
        Exact,
        Maximum,
    };

    constexpr SnapshotPopulationAdmission() noexcept = default;
    constexpr SnapshotPopulationAdmission(Kind kind, std::size_t count) noexcept
        : kind_(kind), count_(count) {}

    Kind kind_{Kind::Unbounded};
    std::size_t count_{0};

    friend class SnapshotIO;
};

// Analysis-side reader that does not require full SimulationParameters.
void read_snapshot_with_context(
    const SnapshotReadContext& context,
    const std::string& filename,
    core::ParticleStore& particles_out,
    core::Real& current_a_out,
    SnapshotReadPolicy policy,
    const SnapshotPopulationAdmission& population_admission);

class VerifiedSnapshotSource;
struct SnapshotDescriptor;
class ParallelSnapshotIO;
class Sha256Accumulator;

class SnapshotIO {
public:
    explicit SnapshotIO(const config::SimulationParameters& config);

    // Publish an indexed simulation snapshot only if its canonical path is absent.
    void write_snapshot(const core::ParticleStore& particles,
                        core::Real current_a,
                        int snapshot_index) const;

    // Transactional explicit-path write. Replacement is opt-in; an optional digest
    // is assigned only after the exact staged object is successfully published.
    void write_snapshot_to_path(
        const core::ParticleStore& particles,
        core::Real current_a,
        const std::string& filename,
        SnapshotWritePolicy policy = SnapshotWritePolicy::RequireAbsent,
        std::string* published_object_sha256_out = nullptr) const;

    // Native convention only; external formats require adapters.
    void read_snapshot(
        const std::string& filename,
        core::ParticleStore& particles_out,
        core::Real& current_a_out,
        SnapshotReadPolicy policy = SnapshotReadPolicy::AnalysisSubset,
        const SnapshotPopulationAdmission& population_admission = {}) const;

    // Exact-IC reads stay bound to the admitted HDF5 object despite pathname replacement.
    void read_snapshot(
        const VerifiedSnapshotSource& source,
        core::ParticleStore& particles_out,
        core::Real& current_a_out,
        SnapshotReadPolicy policy = SnapshotReadPolicy::ExactInitialConditions,
        const SnapshotPopulationAdmission& population_admission = {}) const;

    // Read one admitted contiguous range with local schema/finite/ID checks;
    // cross-range ID uniqueness and total mass remain the caller's responsibility.
    void read_exact_initial_condition_range(
        const VerifiedSnapshotSource& source,
        const SnapshotDescriptor& descriptor,
        std::size_t offset,
        std::size_t count,
        core::ParticleStore& particles_out) const;

    // Require the complete configured analysis population before allocation;
    // clear both outputs on failure.
    void read_complete_analysis_snapshot(
        const std::string& filename,
        core::ParticleStore& particles_out,
        core::Real& current_a_out) const {
        const std::uint64_t expected_raw = config_.num_particles();
        if (expected_raw > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            particles_out = core::ParticleStore{};
            current_a_out = core::Real{0.0};
            throw std::overflow_error(
                "Configured complete-analysis particle count does not fit size_t");
        }
        const std::size_t expected = static_cast<std::size_t>(expected_raw);
        try {
            read_snapshot(
                filename,
                particles_out,
                current_a_out,
                SnapshotReadPolicy::AnalysisSubset,
                SnapshotPopulationAdmission::exact(expected));
        } catch (...) {
            particles_out = core::ParticleStore{};
            current_a_out = core::Real{0.0};
            throw;
        }
    }

private:
    // Exact-range read with raw-velocity hashing, avoiding a second payload pass.
    void read_exact_initial_condition_range_with_raw_velocity_identity(
        const VerifiedSnapshotSource& source,
        const SnapshotDescriptor& descriptor,
        std::size_t offset,
        std::size_t count,
        core::ParticleStore& particles_out,
        Sha256Accumulator* raw_velocity_identity) const;

    friend class ParallelSnapshotIO;

    // Write one native payload at a staging path; the transaction owner publishes it.
    void write_snapshot_payload_to_path(
        const core::ParticleStore& particles,
        core::Real current_a,
        const std::string& filename) const;

    config::SimulationParameters config_;
};

} // namespace io
} // namespace cosmo_nbody
