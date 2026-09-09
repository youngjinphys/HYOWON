#include "cosmo_nbody/runtime/simulation_runner.hpp"

#include "cosmo_nbody/validation/conservation_checks.hpp"
#include "cosmo_nbody/validation/diagnostics.hpp"
#include "cosmo_nbody/domain/mpi_global_id_check.hpp"
#include "cosmo_nbody/ic/initial_conditions.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/io/parallel_snapshot_io.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::runtime {
namespace {

static_assert(
    std::is_nothrow_move_assignable_v<core::ParticleStore>,
    "Restart publication requires noexcept ParticleStore move assignment");

mesh::MeshGeometry create_mesh_geometry(
    core::Real L,
    std::size_t N_mesh) {
    return mesh::MeshGeometry(L, N_mesh);
}

void broadcast_root_error_if_needed(
    bool mpi_active,
    int rank,
    int size,
    int root_failed,
    std::string& root_message) {
    if (!mpi_active) {
        if (root_failed) throw std::runtime_error(root_message);
        return;
    }
#ifdef COSMO_NBODY_HAS_MPI
    require_active_mpi_main_thread("Initial-condition diagnostics status");
    if (MPI_Bcast(&root_failed, 1, MPI_INT, 0, MPI_COMM_WORLD)
        != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to broadcast initial-condition diagnostics status");
    }
    root_message = broadcast_string_collective(
        std::move(root_message),
        0,
        rank,
        size,
        "initial-condition diagnostics error");
    if (root_failed) throw std::runtime_error(root_message);
#else
    (void)rank;
    (void)size;
    throw std::logic_error("MPI active state in non-MPI build");
#endif
}

core::Real exact_count_as_real(std::uint64_t value, const char* label) {
    // IEEE-754 binary64 represents every integer through 2^53 exactly. Runtime
    // diagnostics are integer measurements, so do not silently round them when
    // inserting them into the existing Real-valued diagnostic container.
    constexpr std::uint64_t maximum_exact_binary64_integer =
        std::uint64_t{1} << 53U;
    if (value > maximum_exact_binary64_integer) {
        throw std::overflow_error(
            std::string(label)
            + " exceeds the exact-integer range of diagnostic core::Real");
    }
    return static_cast<core::Real>(value);
}

core::Real exact_size_as_real(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                std::string(label) + " exceeds uint64_t range");
        }
    }
    return exact_count_as_real(static_cast<std::uint64_t>(value), label);
}

} // namespace

SimulationRunner::SimulationRunner(
    const config::SimulationParameters& config,
    io::RunArtifactLayout artifact_layout)
    : config_(config),
      runtime_context_(config.get_runtime()),
      cosmo_model_(config.get_cosmology()),
      drift_kick_(cosmo_model_),
      mesh_geom_(create_mesh_geometry(
          config.get_box().L, config.get_box().N_mesh)),
      domain_decomp_(
          config,
          runtime_context_.rank(),
          runtime_context_.size()),
      artifact_layout_(std::move(artifact_layout)),
      io_(config),
      restart_io_(config),
      current_a_(0.0),
      current_step_(0) {}

void SimulationRunner::initialize_force_solver() {
    if (pm_solver_ || treepm_solver_) {
        throw std::logic_error(
            "Simulation force solver was initialized more than once");
    }
    record_memory_sample("runner_before_solver");
    switch (config_.solver_kind()) {
    case config::SolverKind::PM:
        pm_solver_ = std::make_unique<gravity::PMSolver>(
            mesh_geom_,
            mesh::PMForceMethod::pure_pm(
                config_.get_gravity().deconvolve_cic),
            config_.get_runtime().mpi_enabled,
            config_.get_memory_policy());
        break;
    case config::SolverKind::TreePM:
        treepm_solver_ = std::make_unique<gravity::TreePMSolver>(
            config_,
            mesh_geom_,
            domain_decomp_.get_local_bounds());
        break;
    }
    record_memory_sample("runner_after_solver");
}

core::Real SimulationRunner::required_ghost_width() const {
    switch (config_.solver_kind()) {
    case config::SolverKind::PM:
        return 0.0;
    case config::SolverKind::TreePM:
        return config_.r_cut();
    }
    throw std::logic_error("Unknown solver kind");
}

void SimulationRunner::setup_initial_conditions_serial() {
    initial_condition_evidence_.reset();
    if (config_.get_ic().mode == "snapshot") {
        ic::InitialConditions::load_snapshot(config_, particles_);
    } else {
        ic::InitialConditions::generate(
            config_,
            cosmo_model_,
            particles_,
            &initial_condition_evidence_);
    }

    domain_decomp_.partition_domain(particles_);
    domain_decomp_.exchange_ghosts(
        particles_, required_ghost_width());

    current_a_ = 1.0 / (1.0 + config_.get_time().z_start);
    current_step_ = 0;
    io::clear_process_restart_parent();
    resumed_from_restart_ = false;
    particles_.set_acceleration_validity(core::FieldValidity::INVALID);
}

void SimulationRunner::initialize_from_restart_serial(
    const std::string& filename) {
    // Restart particles carry the exact immediate source digest, but not the
    // source snapshot's bounded generator metadata. Re-read that immutable
    // object before restart validation so resumed outputs preserve the same
    // upstream seed/LPT/mesh/spectrum lineage as uninterrupted outputs.
    ic::InitialConditions::admit_snapshot_provenance(config_);

    const core::Real a_start =
        1.0 / (1.0 + config_.get_time().z_start);
    const core::Real a_end =
        1.0 / (1.0 + config_.get_time().z_final);
    time::TimeStepper restored_stepper(
        a_start,
        a_end,
        config_.get_time().delta_ln_a,
        config_.get_output().snapshot_scale_factors);

    core::ParticleStore restored_particles;
    const std::string parent_manifest_sha256 = restart_io_.read_restart(
        filename,
        restored_particles,
        restored_stepper);

    if (restored_particles.num_owned_particles()
        != config_.num_particles()) {
        throw std::runtime_error(
            "Serial restart particle count does not equal configured N^3");
    }

    restored_particles.clear_ghosts();
    domain_decomp_.partition_domain(restored_particles);
    domain_decomp_.exchange_ghosts(
        restored_particles, required_ghost_width());
    restored_particles.set_acceleration_validity(core::FieldValidity::INVALID);

    io::set_process_restart_parent(parent_manifest_sha256);
    resumed_from_restart_ = true;

    initial_condition_evidence_.reset();
    particles_ = std::move(restored_particles);
    current_a_ = restored_stepper.current_a();
    current_step_ = restored_stepper.current_step();
    initialize_force_solver();
    record_memory_sample("post_restart_initialization");

    if (runtime_context_.rank() == 0) {
        std::cout << "Loaded restart checkpoint at step " << current_step_
                  << " and a = " << std::setprecision(17)
                  << current_a_ << "\n";
    }
}

void SimulationRunner::write_initial_condition_evidence() const {
    int failed = 0;
    std::string message;
    if (runtime_context_.rank() == 0
        && initial_condition_evidence_.has_value()) {
        try {
            const auto output_path =
                artifact_layout_.diagnostics_directory()
                / "initial_conditions.json";
            io::write_text_durable_atomic(
                output_path,
                initial_condition_evidence_->to_json() + "\n",
                "initial-condition diagnostics");
            std::cout
                << "Wrote " << output_path.string()
                << " before evolution.\n";
        } catch (const std::exception& error) {
            failed = 1;
            message = std::string(
                "Failed to write generated initial-condition diagnostics before evolution: ")
                + error.what();
        }
    }
    broadcast_root_error_if_needed(
        runtime_context_.mpi_active(),
        runtime_context_.rank(),
        runtime_context_.size(),
        failed,
        message);
}

void SimulationRunner::write_validation_report(
    std::optional<core::Real> final_force_balance_relative_residual,
    std::optional<core::Real> max_force_balance_relative_residual) const {
    // This collective observes the execution environment but never participates
    // in the numerical method. In particular, heterogeneous thread capacities
    // are recorded rather than forced to match across hosts.
    const RuntimeTopologyDiagnostics runtime_topology =
        runtime_context_.collect_topology_diagnostics();

    if (runtime_context_.rank() != 0) return;

    validation::DiagnosticReport report(io::RunMetadata::from_config(
        config_, io::ProductLineage::DirectSimulation, {}, true));

    const auto& growth = cosmo_model_.growth_integration_diagnostics();
    const auto maximum_discrepancy = [](
        const cosmology::GrowthNestedGridDiscrepancy& discrepancy) {
        return std::max({
            discrepancy.D1,
            discrepancy.f1,
            discrepancy.D2,
            discrepancy.f2,
        });
    };
    report.add_check({
        "growth_ode_used_eds_shortcut",
        growth.used_eds_shortcut ? core::Real{1.0} : core::Real{0.0},
        "dimensionless",
        "One means the analytic Einstein-de Sitter branch represented the full supported interval because Lambda was zero or below binary64 distinguishability; measurement only"
    });
    report.add_check({
        "growth_ode_start_scale_factor",
        growth.start_scale_factor,
        "scale_factor",
        "Matter-era boundary where the Einstein-de Sitter growing mode initializes the flat matter-plus-Lambda growth operator; derived from cosmology and binary64 precision"
    });
    report.add_check({
        "growth_ode_coarse_step_count",
        exact_size_as_real(
            growth.coarse_step_count, "Growth ODE coarse step count"),
        "steps",
        "N/2 nested RK4 grid used to measure internal growth-operator resolution sensitivity; not a user-adjustable accuracy threshold"
    });
    report.add_check({
        "growth_ode_nominal_step_count",
        exact_size_as_real(
            growth.nominal_step_count, "Growth ODE nominal step count"),
        "steps",
        "N nested RK4 grid derived from binary64 precision and RK4 order; not a user-adjustable accuracy threshold"
    });
    report.add_check({
        "growth_ode_refined_step_count",
        exact_size_as_real(
            growth.refined_step_count, "Growth ODE refined step count"),
        "steps",
        "2N nested RK4 grid retained by the growth operator; not a user-adjustable accuracy threshold"
    });
    report.add_check({
        "growth_ode_coarse_to_nominal_D1_relative_discrepancy",
        growth.coarse_to_nominal.D1,
        "dimensionless",
        "Maximum common-node relative discrepancy in D1 between the N/2 and N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_coarse_to_nominal_f1_relative_discrepancy",
        growth.coarse_to_nominal.f1,
        "dimensionless",
        "Maximum common-node relative discrepancy in f1 between the N/2 and N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_coarse_to_nominal_D2_relative_discrepancy",
        growth.coarse_to_nominal.D2,
        "dimensionless",
        "Maximum common-node relative discrepancy in D2 between the N/2 and N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_coarse_to_nominal_f2_relative_discrepancy",
        growth.coarse_to_nominal.f2,
        "dimensionless",
        "Maximum common-node relative discrepancy in f2 between the N/2 and N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_coarse_to_nominal_max_relative_discrepancy",
        maximum_discrepancy(growth.coarse_to_nominal),
        "dimensionless",
        "Maximum of the four N/2-to-N component discrepancies; compact context only and not a convergence certificate"
    });
    report.add_check({
        "growth_ode_nominal_to_refined_D1_relative_discrepancy",
        growth.nominal_to_refined.D1,
        "dimensionless",
        "Maximum common-node relative discrepancy in D1 between the N and 2N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_nominal_to_refined_f1_relative_discrepancy",
        growth.nominal_to_refined.f1,
        "dimensionless",
        "Maximum common-node relative discrepancy in f1 between the N and 2N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_nominal_to_refined_D2_relative_discrepancy",
        growth.nominal_to_refined.D2,
        "dimensionless",
        "Maximum common-node relative discrepancy in D2 between the N and 2N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_nominal_to_refined_f2_relative_discrepancy",
        growth.nominal_to_refined.f2,
        "dimensionless",
        "Maximum common-node relative discrepancy in f2 between the N and 2N grids; measurement only, not an error bound or acceptance rule"
    });
    report.add_check({
        "growth_ode_nominal_to_refined_max_relative_discrepancy",
        maximum_discrepancy(growth.nominal_to_refined),
        "dimensionless",
        "Maximum of the four N-to-2N component discrepancies; compact context only and not a convergence certificate"
    });

    report.add_check({
        "runtime_rank_count",
        exact_count_as_real(runtime_topology.rank_count, "Runtime rank count"),
        "count",
        "Observed MPI rank count for this execution segment; measurement only"
    });
    report.add_check({
        "runtime_shared_memory_domain_count",
        exact_count_as_real(
            runtime_topology.shared_memory_domain_count,
            "Runtime shared-memory domain count"),
        "count",
        "Observed number of MPI shared-memory domains, derived from local-rank zero representatives; measurement only"
    });
    report.add_check({
        "runtime_local_size_min",
        exact_count_as_real(runtime_topology.local_size_min, "Runtime local size minimum"),
        "ranks/domain",
        "Minimum observed MPI ranks per shared-memory domain; measurement only"
    });
    report.add_check({
        "runtime_local_size_max",
        exact_count_as_real(runtime_topology.local_size_max, "Runtime local size maximum"),
        "ranks/domain",
        "Maximum observed MPI ranks per shared-memory domain; measurement only"
    });
    report.add_check({
        "runtime_effective_threads_min",
        exact_count_as_real(
            runtime_topology.effective_threads_min,
            "Runtime effective thread minimum"),
        "threads/rank",
        "Minimum actual OpenMP team capacity selected across ranks; heterogeneous values are recorded, not rejected"
    });
    report.add_check({
        "runtime_effective_threads_max",
        exact_count_as_real(
            runtime_topology.effective_threads_max,
            "Runtime effective thread maximum"),
        "threads/rank",
        "Maximum actual OpenMP team capacity selected across ranks; heterogeneous values are recorded, not rejected"
    });
    report.add_check({
        "runtime_visible_cpu_count_min",
        exact_count_as_real(
            runtime_topology.visible_cpu_count_min,
            "Runtime visible CPU minimum"),
        "cpus/rank",
        "Minimum OS CPU-affinity observation across ranks; zero means the observation was unavailable or not applicable"
    });
    report.add_check({
        "runtime_visible_cpu_count_max",
        exact_count_as_real(
            runtime_topology.visible_cpu_count_max,
            "Runtime visible CPU maximum"),
        "cpus/rank",
        "Maximum OS CPU-affinity observation across ranks; zero means the observation was unavailable or not applicable"
    });
    report.add_check({
        "runtime_automatic_thread_ceiling_min",
        exact_count_as_real(
            runtime_topology.automatic_thread_ceiling_min,
            "Runtime automatic thread ceiling minimum"),
        "threads/rank",
        "Minimum topology-derived automatic thread ceiling; zero means no synthetic ceiling was applied"
    });
    report.add_check({
        "runtime_automatic_thread_ceiling_max",
        exact_count_as_real(
            runtime_topology.automatic_thread_ceiling_max,
            "Runtime automatic thread ceiling maximum"),
        "threads/rank",
        "Maximum topology-derived automatic thread ceiling; zero means no synthetic ceiling was applied"
    });
    report.add_check({
        "runtime_shared_unbound_affinity_rank_count",
        exact_count_as_real(
            runtime_topology.shared_unbound_affinity_rank_count,
            "Runtime shared unbound affinity rank count"),
        "ranks",
        "Ranks whose colocated peers exposed an identical CPU affinity mask, causing automatic mode to divide that shared capacity; measurement only"
    });
    if (!memory_samples_.empty()) {
        const auto& memory = memory_samples_.back();
        report.add_check({
            "runtime_memory_observed_rank_count",
            exact_count_as_real(
                memory.observed_rank_count,
                "Runtime memory observed rank count"),
            "ranks",
            "Ranks contributing platform process resident-memory observations to the last memory sample; measurement only"
        });
        if (memory.observed_rank_count > 0) {
            report.add_check({
                "runtime_current_rss_bytes_min",
                exact_count_as_real(
                    memory.current_rss_bytes_min,
                    "Runtime current RSS minimum"),
                "bytes",
                "Minimum current resident set across observed ranks at the final sampled phase; measurement only"
            });
            report.add_check({
                "runtime_current_rss_bytes_max",
                exact_count_as_real(
                    memory.current_rss_bytes_max,
                    "Runtime current RSS maximum"),
                "bytes",
                "Maximum current resident set across observed ranks at the final sampled phase; measurement only"
            });
            report.add_check({
                "runtime_peak_rss_bytes_min",
                exact_count_as_real(
                    memory.peak_rss_bytes_min,
                    "Runtime peak RSS minimum"),
                "bytes",
                "Minimum process resident-set high-water mark across observed ranks; measurement only"
            });
            report.add_check({
                "runtime_peak_rss_bytes_max",
                exact_count_as_real(
                    memory.peak_rss_bytes_max,
                    "Runtime peak RSS maximum"),
                "bytes",
                "Maximum process resident-set high-water mark across observed ranks; use memory_timeline.json to locate the phase where it first appeared"
            });
        }
    }
    report.add_check({
        "layzer_irvine_max_ratio",
        last_layzer_irvine_max_ratio_,
        "dimensionless",
        resumed_from_restart_
            ? "Maximum global |residual|/max(|K|, |W|) since restart with the CIC self-energy subtracted from W; pre-restart diagnostic history is not persisted. Measurement only; interpret with layzer_irvine_timeline.json and an independent convergence comparison"
            : "Single-run maximum global |residual|/max(|K|, |W|) with the CIC self-energy subtracted from W. Measurement only; interpret with layzer_irvine_timeline.json and independent trajectory/quadrature convergence controls"
    });
    if (final_force_balance_relative_residual.has_value()) {
        report.add_check({
            "force_balance_final_relative_residual",
            *final_force_balance_relative_residual,
            "dimensionless",
            "Instantaneous ||sum_i m_i a_i||/sum_i m_i||a_i|| at the final sampled force solve; diagnostic only"
        });
    }
    if (max_force_balance_relative_residual.has_value()) {
        report.add_check({
            "force_balance_max_relative_residual",
            *max_force_balance_relative_residual,
            "dimensionless",
            "Maximum instantaneous internal-force residual over the current execution segment; diagnostic only"
        });
    }
    if (max_step_drift_displacement_.has_value()) {
        std::string drift_message =
            "Peak exact discrete KDK drift over the current execution segment; measurement only, not a timestep controller";
        if (max_step_drift_displacement_.step().has_value()) {
            drift_message += "; first observed at step "
                + std::to_string(*max_step_drift_displacement_.step());
        }
        report.add_check({
            "max_kdk_drift_displacement",
            *max_step_drift_displacement_,
            "Mpc/h",
            std::move(drift_message)
        });
        if (max_step_drift_displacement_.scale_factor().has_value()) {
            report.add_check({
                "max_kdk_drift_scale_factor",
                *max_step_drift_displacement_.scale_factor(),
                "scale_factor",
                "Scale factor at which the peak exact discrete KDK drift was first observed; context only"
            });
        }
        report.add_check({
            "max_kdk_drift_over_mean_spacing",
            *max_step_drift_displacement_ / config_.d_mean(),
            "dimensionless",
            "Peak exact KDK drift divided by the configured mean interparticle spacing; context only, not an acceptance threshold"
        });
        report.add_check({
            "max_kdk_drift_over_pm_cell",
            *max_step_drift_displacement_
                / (config_.get_box().L / config_.get_box().N_mesh),
            "dimensionless",
            "Peak exact KDK drift divided by the PM cell size; context only, not an acceptance threshold"
        });
        if (config_.solver_kind() == config::SolverKind::TreePM) {
            report.add_check({
                "max_kdk_drift_over_softening",
                *max_step_drift_displacement_ / config_.eps(),
                "dimensionless",
                "Peak exact KDK drift divided by the TreePM Plummer softening length; force-resolution context only, not a timestep criterion or convergence threshold"
            });
        }
    }

    io::write_text_durable_atomic(
        artifact_layout_.validation_report_path(),
        report.to_json() + "\n",
        "validation report");
    std::cout << report.summary_text();
}

} // namespace cosmo_nbody::runtime
