#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/time/leapfrog_integrator.hpp"
#include "cosmo_nbody/time/time_stepper.hpp"
#include "cosmo_nbody/cosmology/cosmology_model.hpp"
#include "cosmo_nbody/cosmology/drift_kick_factors.hpp"
#include "cosmo_nbody/domain/domain_decomposition.hpp"
#include "cosmo_nbody/runtime/evolution_pm_solver_handle.hpp"
#include "cosmo_nbody/gravity/treepm_solver.hpp"
#include "cosmo_nbody/ic/realised_ic_evidence.hpp"
#include "cosmo_nbody/io/restart_lineage.hpp"
#include "cosmo_nbody/io/run_artifact_layout.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/restart_checkpoint_io.hpp"
#include "cosmo_nbody/runtime/mpi_execution_identity.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cosmo_nbody {
namespace runtime {

class PeakDriftMeasurement {
public:
    PeakDriftMeasurement(
        const core::Real* current_scale_factor,
        const std::size_t* current_step) noexcept
        : current_scale_factor_(current_scale_factor),
          current_step_(current_step) {}

    PeakDriftMeasurement(const PeakDriftMeasurement&) = delete;
    PeakDriftMeasurement& operator=(const PeakDriftMeasurement&) = delete;
    PeakDriftMeasurement(PeakDriftMeasurement&&) = delete;
    PeakDriftMeasurement& operator=(PeakDriftMeasurement&&) = delete;

    PeakDriftMeasurement& operator=(core::Real candidate) noexcept {
        if (!value_.has_value() || candidate > *value_) {
            value_ = candidate;
            scale_factor_ = *current_scale_factor_;
            step_ = *current_step_;
        }
        return *this;
    }

    void reset() noexcept {
        value_.reset();
        scale_factor_.reset();
        step_.reset();
    }

    bool has_value() const noexcept { return value_.has_value(); }
    core::Real value_or(core::Real fallback) const noexcept {
        return value_.value_or(fallback);
    }
    const core::Real& operator*() const { return value_.value(); }

    const std::optional<core::Real>& scale_factor() const noexcept {
        return scale_factor_;
    }
    const std::optional<std::size_t>& step() const noexcept { return step_; }

private:
    const core::Real* current_scale_factor_{nullptr};
    const std::size_t* current_step_{nullptr};
    std::optional<core::Real> value_;
    std::optional<core::Real> scale_factor_;
    std::optional<std::size_t> step_;
};

class SimulationRunner {
public:
    SimulationRunner(
        const config::SimulationParameters& config,
        io::RunArtifactLayout artifact_layout);

    void initialize();
    // Prepare only the canonical initial particle state for export tooling.
    // This deliberately omits the evolution force solver; calling run() after
    // this entry point is not a supported runner lifecycle.
    void initialize_initial_conditions_only() { setup_initial_conditions(); }
    void initialize_from_restart(const std::string& filename);
    void run();

    const core::ParticleStore& get_particles() const { return particles_; }
    const std::optional<ic::RealisedICEvidence>&
    initial_condition_evidence() const noexcept {
        return initial_condition_evidence_;
    }

    int rank() const noexcept { return runtime_context_.rank(); }
    int size() const noexcept { return runtime_context_.size(); }
    bool mpi_active() const noexcept { return runtime_context_.mpi_active(); }

private:
    struct MemoryTimelineSample {
        std::uint64_t sequence{0};
        std::uint64_t step{0};
        core::Real scale_factor{0.0};
        std::string phase;
        std::uint64_t observed_rank_count{0};
        std::uint64_t current_rss_bytes_min{0};
        std::uint64_t current_rss_bytes_max{0};
        std::uint64_t peak_rss_bytes_min{0};
        std::uint64_t peak_rss_bytes_max{0};
        std::uint64_t owned_particles_min{0};
        std::uint64_t owned_particles_max{0};
        std::uint64_t ghost_particles_min{0};
        std::uint64_t ghost_particles_max{0};
    };

    io::RestartLineageScope restart_lineage_scope_;
    config::SimulationParameters config_;
    core::ParticleStore particles_;

    RuntimeContext runtime_context_;
    MpiExecutionIdentityAgreement execution_identity_agreement_{
        config_, runtime_context_.rank(), runtime_context_.size()};
    cosmology::CosmologyModel cosmo_model_;
    cosmology::DriftKickIntegrator drift_kick_;
    mesh::MeshGeometry mesh_geom_;
    domain::DomainDecomposition domain_decomp_;
    EvolutionPMSolverHandle pm_solver_;
    std::unique_ptr<gravity::TreePMSolver> treepm_solver_;
    io::RunArtifactLayout artifact_layout_;
    io::SnapshotIO io_;
    io::RestartCheckpointIO restart_io_;

    core::Real current_a_;
    std::size_t current_step_;
    core::Real last_layzer_irvine_max_ratio_{0.0};
    PeakDriftMeasurement max_step_drift_displacement_{
        &current_a_, &current_step_};
    bool step_drift_measurement_invalid_{false};
    bool resumed_from_restart_{false};
    std::optional<ic::RealisedICEvidence> initial_condition_evidence_;
    std::vector<MemoryTimelineSample> memory_samples_;

    void setup_initial_conditions();
    void initialize_force_solver();
    core::Real required_ghost_width() const;
    void write_initial_condition_evidence() const;
    void write_validation_report(
        std::optional<core::Real> final_force_balance_relative_residual,
        std::optional<core::Real> max_force_balance_relative_residual) const;
    void write_snapshot_collective(core::Real a, int snapshot_index) const;
    void write_restart_collective(const time::TimeStepper& stepper) const;
    void record_memory_sample(std::string_view phase);
    void write_memory_timeline() const;

    void setup_initial_conditions_serial();
    void initialize_from_restart_serial(const std::string& filename);
};

} // namespace runtime
} // namespace cosmo_nbody
