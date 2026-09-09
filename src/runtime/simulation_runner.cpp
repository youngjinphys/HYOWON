#include "cosmo_nbody/runtime/simulation_runner.hpp"

#include "cosmo_nbody/domain/mpi_global_id_check.hpp"
#include "cosmo_nbody/ic/distributed_snapshot_ingest.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/io/parallel_snapshot_io.hpp"
#include "cosmo_nbody/io/restart_checkpoint_io.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"
#include "cosmo_nbody/validation/conservation_checks.hpp"
#include "cosmo_nbody/validation/force_balance_diagnostics.hpp"
#include "cosmo_nbody/validation/layzer_irvine_ratio.hpp"
#include "cosmo_nbody/validation/layzer_irvine_timeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::runtime {
namespace {

void synchronize_exception_or_rethrow(
    bool mpi_active,
    std::exception_ptr local_exception,
    int mpi_size,
    const char* context) {
    if (mpi_active) {
#ifdef COSMO_NBODY_HAS_MPI
        synchronize_mpi_exception(local_exception, mpi_size, context);
        return;
#else
        (void)mpi_size;
        (void)context;
        throw std::logic_error("MPI active state in non-MPI build");
#endif
    }
    if (local_exception) std::rethrow_exception(local_exception);
}

std::uint64_t checked_size_to_u64(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds uint64_t");
        }
    }
    return static_cast<std::uint64_t>(value);
}

int checked_snapshot_index(std::size_t index) {
    if (index > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error("Snapshot index exceeds int range");
    }
    return static_cast<int>(index);
}

core::Real stable_layzer_irvine_log_interval(
    core::Real current_a,
    core::Real previous_a) {
    if (!std::isfinite(current_a) || !std::isfinite(previous_a)
        || current_a <= previous_a || previous_a <= 0.0) {
        throw std::invalid_argument(
            "Layzer-Irvine diagnostic scale interval must be finite and increasing");
    }
    const long double current = static_cast<long double>(current_a);
    const long double previous = static_cast<long double>(previous_a);
    const long double relative = (current - previous) / previous;
    const long double interval = std::isfinite(relative) && relative <= 0.5L
        ? std::log1p(relative)
        : std::log(current) - std::log(previous);
    if (!std::isfinite(interval) || interval <= 0.0L
        || interval > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(
            "Layzer-Irvine diagnostic logarithmic interval is not representable");
    }
    return static_cast<core::Real>(interval);
}

validation::ForceBalanceDiagnostics compute_runtime_force_balance(
    const core::ParticleStore& particles,
    std::span<const core::Real> acceleration_x,
    std::span<const core::Real> acceleration_y,
    std::span<const core::Real> acceleration_z,
    std::size_t owned,
    bool mpi_active,
    int mpi_size) {
    const auto uniform_mass = particles.get_uniform_mass();

    if (!mpi_active) {
        if (uniform_mass.has_value()) {
            return validation::compute_force_balance_diagnostics(
                *uniform_mass,
                acceleration_x.first(owned),
                acceleration_y.first(owned),
                acceleration_z.first(owned));
        }
        return validation::compute_force_balance_diagnostics(
            particles.get_masses().first(owned),
            acceleration_x.first(owned),
            acceleration_y.first(owned),
            acceleration_z.first(owned));
    }

#ifndef COSMO_NBODY_HAS_MPI
    (void)mpi_size;
    throw std::logic_error("MPI force balance requested in non-MPI build");
#else
    if (mpi_size <= 0) {
        throw std::logic_error("MPI force balance requires a positive rank count");
    }

    const int local_mass_mode = uniform_mass.has_value() ? 1 : 2;
    int minimum_mass_mode = 0;
    int maximum_mass_mode = 0;
    const int mass_min_status = MPI_Allreduce(
        &local_mass_mode,
        &minimum_mass_mode,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    const int mass_max_status = MPI_Allreduce(
        &local_mass_mode,
        &maximum_mass_mode,
        1,
        MPI_INT,
        MPI_MAX,
        MPI_COMM_WORLD);
    if (mass_min_status != MPI_SUCCESS || mass_max_status != MPI_SUCCESS) {
        throw std::runtime_error(
            "Force-balance mass-representation reduction failed");
    }
    if (minimum_mass_mode != maximum_mass_mode) {
        throw std::runtime_error(
            "Force-balance mass representation differs across MPI ranks");
    }

    std::exception_ptr local_exception;
    int local_exponent = std::numeric_limits<int>::min();
    try {
        if (uniform_mass.has_value()) {
            local_exponent = validation::force_balance_common_exponent(
                *uniform_mass,
                acceleration_x.first(owned),
                acceleration_y.first(owned),
                acceleration_z.first(owned));
        } else {
            local_exponent = validation::force_balance_common_exponent(
                particles.get_masses().first(owned),
                acceleration_x.first(owned),
                acceleration_y.first(owned),
                acceleration_z.first(owned));
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        local_exception,
        mpi_size,
        "Force-balance exponent construction");

    int global_exponent = std::numeric_limits<int>::min();
    if (MPI_Allreduce(
            &local_exponent,
            &global_exponent,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Force-balance common-exponent reduction failed");
    }

    validation::ForceBalanceScaledSums local_sums;
    local_exception = nullptr;
    try {
        if (uniform_mass.has_value()) {
            local_sums = validation::compute_force_balance_scaled_sums(
                *uniform_mass,
                acceleration_x.first(owned),
                acceleration_y.first(owned),
                acceleration_z.first(owned),
                global_exponent);
        } else {
            local_sums = validation::compute_force_balance_scaled_sums(
                particles.get_masses().first(owned),
                acceleration_x.first(owned),
                acceleration_y.first(owned),
                acceleration_z.first(owned),
                global_exponent);
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        local_exception,
        mpi_size,
        "Force-balance scaled summation");

    const std::array<long double, 4> local_values{
        local_sums.scaled_force_x,
        local_sums.scaled_force_y,
        local_sums.scaled_force_z,
        local_sums.scaled_denominator,
    };
    const auto force_balance_memory = validation::force_balance_memory_plan(
        local_sums.particle_count,
        true,
        static_cast<std::uint64_t>(mpi_size));
    std::vector<long double> gathered_values;
    std::vector<std::uint64_t> gathered_counts;
    std::vector<validation::ForceBalanceScaledSums> partials;
    std::exception_ptr allocation_exception;
    try {
        if (force_balance_memory.mpi_rank_count
                > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())
            || force_balance_memory.mpi_gathered_value_count
                > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "Force-balance gather storage size exceeds size_t range");
        }
        const std::size_t rank_count = static_cast<std::size_t>(
            force_balance_memory.mpi_rank_count);
        const std::size_t gathered_value_count = static_cast<std::size_t>(
            force_balance_memory.mpi_gathered_value_count);
        if (gathered_value_count != rank_count * local_values.size()) {
            throw std::logic_error(
                "Force-balance memory plan disagrees with the MPI value layout");
        }
        gathered_values.resize(gathered_value_count);
        gathered_counts.resize(rank_count);
        partials.resize(rank_count);
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        allocation_exception,
        mpi_size,
        "Force-balance rank-partial gather storage allocation");

    const std::uint64_t local_count = local_sums.particle_count;
    const int values_status = MPI_Allgather(
        local_values.data(),
        static_cast<int>(local_values.size()),
        MPI_LONG_DOUBLE,
        gathered_values.data(),
        static_cast<int>(local_values.size()),
        MPI_LONG_DOUBLE,
        MPI_COMM_WORLD);
    const int counts_status = MPI_Allgather(
        &local_count,
        1,
        MPI_UINT64_T,
        gathered_counts.data(),
        1,
        MPI_UINT64_T,
        MPI_COMM_WORLD);
    if (values_status != MPI_SUCCESS || counts_status != MPI_SUCCESS) {
        throw std::runtime_error(
            "Force-balance rank-partial gather failed");
    }

    for (int rank = 0; rank < mpi_size; ++rank) {
        const std::size_t rank_index = static_cast<std::size_t>(rank);
        const std::size_t offset = rank_index * local_values.size();
        partials[rank_index] = validation::ForceBalanceScaledSums{
            global_exponent,
            gathered_values[offset],
            gathered_values[offset + 1],
            gathered_values[offset + 2],
            gathered_values[offset + 3],
            gathered_counts[rank_index],
        };
    }
    return validation::force_balance_diagnostics_from_scaled_sums(
        validation::combine_force_balance_scaled_sums(partials));
#endif
}

#ifdef COSMO_NBODY_HAS_MPI
void require_rank_restart_state_agreement(
    core::Real current_a,
    std::size_t current_step) {
    const std::uint64_t local_step = static_cast<std::uint64_t>(current_step);
    std::uint64_t step_min = 0;
    std::uint64_t step_max = 0;
    core::Real a_min = 0.0;
    core::Real a_max = 0.0;
    const int step_min_status = MPI_Allreduce(
        &local_step, &step_min, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    const int step_max_status = MPI_Allreduce(
        &local_step, &step_max, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    const int a_min_status = MPI_Allreduce(
        &current_a, &a_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    const int a_max_status = MPI_Allreduce(
        &current_a, &a_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (step_min_status != MPI_SUCCESS
        || step_max_status != MPI_SUCCESS
        || a_min_status != MPI_SUCCESS
        || a_max_status != MPI_SUCCESS) {
        const int status = step_min_status != MPI_SUCCESS
            ? step_min_status
            : step_max_status != MPI_SUCCESS
                ? step_max_status
                : a_min_status != MPI_SUCCESS
                    ? a_min_status : a_max_status;
        (void)MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
    if (step_min != step_max || a_min != a_max) {
        throw std::runtime_error(
            "Restart checkpoint shards disagree on step or scale factor");
    }
}
#endif

} // namespace

void SimulationRunner::setup_initial_conditions() {
    if (!runtime_context_.mpi_active() || runtime_context_.size() == 1) {
        setup_initial_conditions_serial();
        return;
    }
    if (config_.get_ic().mode != "snapshot") {
        throw std::runtime_error(
            "Fresh MPI generated-IC ingestion is disabled because the current "
            "LPT pipeline is root-concentrated. Generate a serial immutable "
            "snapshot IC and start the distributed run from that file, or "
            "implement distributed FFT/LPT ingestion before enabling generated "
            "MPI ICs.");
    }

    ic::load_snapshot_distributed(
        config_,
        domain_decomp_,
        particles_,
        runtime_context_.rank(),
        runtime_context_.size());
    domain_decomp_.exchange_ghosts(
        particles_, required_ghost_width());
    current_a_ = 1.0 / (1.0 + config_.get_time().z_start);
    current_step_ = 0;
    io::clear_process_restart_parent();
    resumed_from_restart_ = false;
    particles_.set_acceleration_validity(core::FieldValidity::INVALID);
}

void SimulationRunner::initialize() {
    setup_initial_conditions();
    initialize_force_solver();
    record_memory_sample("post_initialization");
}

void SimulationRunner::initialize_from_restart(
    const std::string& filename) {
    if (runtime_context_.size() == 1) {
        initialize_from_restart_serial(filename);
        return;
    }

    if (!runtime_context_.mpi_active()) {
        throw std::logic_error(
            "Multi-rank restart requested without active MPI");
    }

#ifndef COSMO_NBODY_HAS_MPI
    throw std::logic_error("MPI restart requested in non-MPI build");
#else
    // Re-establish source-IC lineage from the exact configured object before
    // restoring dynamics. This is descriptor-only; rank 0 does not reread
    // particle phase space and peers never open the source path.
    ic::admit_snapshot_provenance_distributed(
        config_,
        runtime_context_.rank(),
        runtime_context_.size());

    std::optional<std::filesystem::path> checkpoint_directory;
    std::optional<time::TimeStepper> restored_stepper;
    std::exception_ptr preparation_exception;
    try {
        checkpoint_directory.emplace(
            std::filesystem::absolute(filename).lexically_normal());
        const core::Real a_start =
            1.0 / (1.0 + config_.get_time().z_start);
        const core::Real a_end =
            1.0 / (1.0 + config_.get_time().z_final);
        restored_stepper.emplace(
            a_start,
            a_end,
            config_.get_time().delta_ln_a,
            config_.get_output().snapshot_scale_factors);
    } catch (...) {
        preparation_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        preparation_exception,
        runtime_context_.size(),
        "Restart checkpoint state preparation");

    core::ParticleStore restored_particles;
    std::string parent_manifest_sha256;
    std::exception_ptr read_exception;
    try {
        parent_manifest_sha256 = io::read_restart_checkpoint_collective(
            config_,
            *checkpoint_directory,
            runtime_context_.rank(),
            runtime_context_.size(),
            restored_particles,
            *restored_stepper);
    } catch (...) {
        read_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        read_exception,
        runtime_context_.size(),
        "Restart checkpoint ingestion");

    domain::require_globally_unique_particle_ids(
        restored_particles.get_ids().first(
            restored_particles.num_owned_particles()));
    const core::Real restored_a = restored_stepper->current_a();
    const std::size_t restored_step = restored_stepper->current_step();
    require_rank_restart_state_agreement(restored_a, restored_step);

    restored_particles.clear_ghosts();
    domain_decomp_.partition_domain(restored_particles);
    domain_decomp_.exchange_ghosts(
        restored_particles, required_ghost_width());
    restored_particles.set_acceleration_validity(core::FieldValidity::INVALID);

    std::exception_ptr lineage_exception;
    try {
        io::set_process_restart_parent(parent_manifest_sha256);
        resumed_from_restart_ = true;
    } catch (...) {
        lineage_exception = std::current_exception();
    }
    synchronize_mpi_exception(
        lineage_exception,
        runtime_context_.size(),
        "Restart parent provenance publication");

    initial_condition_evidence_.reset();
    particles_ = std::move(restored_particles);
    current_a_ = restored_a;
    current_step_ = restored_step;
    initialize_force_solver();
    record_memory_sample("post_restart_initialization");

    if (runtime_context_.rank() == 0) {
        std::cout << "Loaded restart checkpoint at step "
                  << current_step_
                  << " and a = " << std::setprecision(17)
                  << current_a_ << "\n";
    }
    return;
#endif
}

void SimulationRunner::write_snapshot_collective(
    core::Real a,
    int snapshot_index) const {
    if (runtime_context_.size() == 1) {
        io_.write_snapshot_to_path(
            particles_,
            a,
            artifact_layout_.snapshot_path(snapshot_index).string());
        return;
    }
    if (!runtime_context_.mpi_active()) {
        throw std::logic_error(
            "Multi-rank snapshot requested without active MPI");
    }
    io::ParallelSnapshotIO parallel_io(
        config_, artifact_layout_.snapshots_directory());
    parallel_io.write_snapshot(particles_, a, snapshot_index);
}

void SimulationRunner::write_restart_collective(
    const time::TimeStepper& stepper) const {
    const auto restart_base = artifact_layout_.restart_base_path();
    if (runtime_context_.size() == 1) {
        restart_io_.write_restart(
            particles_, stepper, restart_base.string());
        return;
    }
    if (!runtime_context_.mpi_active()) {
        throw std::logic_error(
            "Multi-rank restart requested without active MPI");
    }
#ifndef COSMO_NBODY_HAS_MPI
    throw std::logic_error("MPI restart requested in non-MPI build");
#else
    io::write_restart_checkpoint_collective(
        config_,
        particles_,
        stepper,
        restart_base,
        runtime_context_.rank(),
        runtime_context_.size());
#endif
}

void SimulationRunner::record_memory_sample(std::string_view phase) {
    const bool diagnostics_enabled =
        config_.get_validation().write_diagnostics;
    const bool operational_milestone =
        phase == "runner_before_solver"
        || phase == "runner_after_solver"
        || phase == "post_initialization"
        || phase == "post_restart_initialization"
        || phase == "run_start"
        || phase == "post_force"
        || phase == "post_snapshot"
        || phase == "post_restart_checkpoint"
        || phase == "terminal";
    if (!diagnostics_enabled && !operational_milestone) return;

    const RuntimeMemoryObservation memory =
        runtime_context_.observe_process_memory();
    const std::uint64_t owned = checked_size_to_u64(
        particles_.num_owned_particles(), "Owned particle count");
    const std::uint64_t ghosts = checked_size_to_u64(
        particles_.num_ghost_particles(), "Ghost particle count");
    constexpr std::size_t fields_per_rank = 5;
    const std::array<std::uint64_t, fields_per_rank> local_fields{
        memory.available ? 1U : 0U,
        memory.available ? memory.current_rss_bytes : 0U,
        memory.available ? memory.peak_rss_bytes : 0U,
        owned,
        ghosts};
    std::vector<std::uint64_t> gathered_fields;

    if (runtime_context_.mpi_active() && runtime_context_.size() > 1) {
#ifndef COSMO_NBODY_HAS_MPI
        throw std::logic_error("MPI memory telemetry requested in non-MPI build");
#else
        require_active_mpi_main_thread("Runtime memory telemetry");
        int allocation_failed = 0;
        if (runtime_context_.rank() == 0) {
            try {
                gathered_fields.resize(
                    static_cast<std::size_t>(runtime_context_.size())
                    * fields_per_rank);
            } catch (...) {
                allocation_failed = 1;
            }
        }
        int any_allocation_failed = 0;
        if (MPI_Allreduce(
                &allocation_failed, &any_allocation_failed, 1,
                MPI_INT, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "Runtime memory telemetry allocation agreement failed");
        }
        if (any_allocation_failed != 0) {
            throw std::runtime_error(
                "Runtime memory telemetry root allocation failed");
        }
        if (MPI_Gather(
                local_fields.data(),
                static_cast<int>(local_fields.size()), MPI_UINT64_T,
                runtime_context_.rank() == 0
                    ? gathered_fields.data() : nullptr,
                static_cast<int>(local_fields.size()), MPI_UINT64_T,
                0, MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "Runtime memory telemetry MPI gather failed");
        }
#endif
    } else {
        gathered_fields.assign(local_fields.begin(), local_fields.end());
    }

    if (runtime_context_.rank() != 0) return;

    const std::uint64_t unavailable_min =
        std::numeric_limits<std::uint64_t>::max();
    std::array<std::uint64_t, 4> minima{
        unavailable_min, unavailable_min,
        unavailable_min, unavailable_min};
    std::array<std::uint64_t, 4> maxima{0, 0, 0, 0};
    std::uint64_t observed_rank_count = 0;
    const std::size_t gathered_ranks =
        gathered_fields.size() / fields_per_rank;
    for (std::size_t rank = 0; rank < gathered_ranks; ++rank) {
        const std::size_t base = rank * fields_per_rank;
        const bool available = gathered_fields[base] != 0;
        if (available) {
            ++observed_rank_count;
            minima[0] = std::min(minima[0], gathered_fields[base + 1]);
            minima[1] = std::min(minima[1], gathered_fields[base + 2]);
            maxima[0] = std::max(maxima[0], gathered_fields[base + 1]);
            maxima[1] = std::max(maxima[1], gathered_fields[base + 2]);
        }
        minima[2] = std::min(minima[2], gathered_fields[base + 3]);
        minima[3] = std::min(minima[3], gathered_fields[base + 4]);
        maxima[2] = std::max(maxima[2], gathered_fields[base + 3]);
        maxima[3] = std::max(maxima[3], gathered_fields[base + 4]);
    }
    if (observed_rank_count == 0) {
        minima[0] = 0;
        minima[1] = 0;
    }

    if (operational_milestone) {
        std::cout << std::setprecision(17);
        for (std::size_t rank = 0; rank < gathered_ranks; ++rank) {
            const std::size_t base = rank * fields_per_rank;
            std::cout
                << "RUNTIME_MEMORY"
                << " phase=" << phase
                << " step=" << current_step_
                << " scale_factor=" << current_a_
                << " rank=" << rank
#if defined(__linux__)
                << " source=linux_proc_self_status_vmrss_vmhwm"
#elif defined(__APPLE__)
                << " source=darwin_mach_task_resident_and_rusage_ru_maxrss"
#else
                << " source=unavailable"
#endif
                << " available="
                << (gathered_fields[base] != 0 ? "true" : "false")
                << " current_rss_bytes=" << gathered_fields[base + 1]
                << " peak_rss_bytes=" << gathered_fields[base + 2]
                << " owned_particles=" << gathered_fields[base + 3]
                << " ghost_particles=" << gathered_fields[base + 4]
                << '\n';
        }
        std::cout
            << "RUNTIME_MEMORY_AGGREGATE"
            << " phase=" << phase
            << " step=" << current_step_
            << " scale_factor=" << current_a_
            << " ranks=" << gathered_ranks
            << " observed_ranks=" << observed_rank_count
            << " current_rss_bytes_min=" << minima[0]
            << " current_rss_bytes_max=" << maxima[0]
            << " peak_rss_bytes_min=" << minima[1]
            << " peak_rss_bytes_max=" << maxima[1]
            << " owned_particles_min=" << minima[2]
            << " owned_particles_max=" << maxima[2]
            << " ghost_particles_min=" << minima[3]
            << " ghost_particles_max=" << maxima[3]
            << '\n';
        std::cout.flush();
    }

    if (!diagnostics_enabled) return;
    MemoryTimelineSample sample;
    sample.sequence = checked_size_to_u64(
        memory_samples_.size(), "Memory timeline sequence");
    sample.step = checked_size_to_u64(current_step_, "Memory timeline step");
    sample.scale_factor = current_a_;
    sample.phase.assign(phase.begin(), phase.end());
    sample.observed_rank_count = observed_rank_count;
    sample.current_rss_bytes_min = minima[0];
    sample.current_rss_bytes_max = maxima[0];
    sample.peak_rss_bytes_min = minima[1];
    sample.peak_rss_bytes_max = maxima[1];
    sample.owned_particles_min = minima[2];
    sample.owned_particles_max = maxima[2];
    sample.ghost_particles_min = minima[3];
    sample.ghost_particles_max = maxima[3];
    memory_samples_.push_back(std::move(sample));
}

void SimulationRunner::write_memory_timeline() const {
    if (!config_.get_validation().write_diagnostics
        || runtime_context_.rank() != 0) {
        return;
    }

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema\": \"hyowon.runtime_memory_timeline.v1\",\n"
        << "  \"measurement_only\": true,\n"
#if defined(__linux__)
        << "  \"rss_source\": \"linux_proc_self_status_vmrss_vmhwm\",\n"
#elif defined(__APPLE__)
        << "  \"rss_source\": \"darwin_mach_task_resident_and_rusage_ru_maxrss\",\n"
#else
        << "  \"rss_source\": \"unavailable\",\n"
#endif
        << "  \"samples\": [\n";
    for (std::size_t i = 0; i < memory_samples_.size(); ++i) {
        const auto& sample = memory_samples_[i];
        out << "    {\"sequence\":" << sample.sequence
            << ",\"step\":" << sample.step
            << ",\"scale_factor\":" << sample.scale_factor
            << ",\"phase\":" << std::quoted(sample.phase)
            << ",\"observed_rank_count\":" << sample.observed_rank_count
            << ",\"current_rss_bytes_min\":" << sample.current_rss_bytes_min
            << ",\"current_rss_bytes_max\":" << sample.current_rss_bytes_max
            << ",\"peak_rss_bytes_min\":" << sample.peak_rss_bytes_min
            << ",\"peak_rss_bytes_max\":" << sample.peak_rss_bytes_max
            << ",\"owned_particles_min\":" << sample.owned_particles_min
            << ",\"owned_particles_max\":" << sample.owned_particles_max
            << ",\"ghost_particles_min\":" << sample.ghost_particles_min
            << ",\"ghost_particles_max\":" << sample.ghost_particles_max
            << "}";
        if (i + 1 != memory_samples_.size()) out << ',';
        out << '\n';
    }
    out << "  ]\n}\n";
    io::write_text_durable_atomic(
        artifact_layout_.diagnostics_directory() / "memory_timeline.json",
        out.str(),
        "runtime memory timeline");
}

void SimulationRunner::run() {
    if (current_a_ <= 0.0) {
        throw std::logic_error(
            "SimulationRunner::run() called before initialization");
    }
    const int initialized_solver_count =
        (pm_solver_ ? 1 : 0) + (treepm_solver_ ? 1 : 0);
    if (initialized_solver_count != 1) {
        throw std::logic_error(
            "SimulationRunner::run() requires exactly one initialized force solver");
    }
    write_initial_condition_evidence();

    const bool diagnostics_enabled =
        config_.get_validation().write_diagnostics;
    const bool distributed_timing =
        runtime_context_.mpi_active() && runtime_context_.size() > 1;
    const bool lightweight_performance_timing =
        distributed_timing && !diagnostics_enabled;

    using TimingClock = std::chrono::steady_clock;
    using TimingDuration = TimingClock::duration;
    enum class TimingPhase : std::size_t {
        ForceRefreshTotal,
        DomainPartition,
        GhostExchange,
        GravitySolve,
        ForceBalance,
        SnapshotWrite,
        RestartWrite,
        Count,
    };
    struct PhaseTimingAccumulator {
        std::uint64_t sample_count{0};
        TimingDuration total{};
        TimingDuration maximum{};
        bool valid{true};
    };
    constexpr std::size_t timing_phase_count =
        static_cast<std::size_t>(TimingPhase::Count);
    static_assert(timing_phase_count == 7);
    std::array<PhaseTimingAccumulator, timing_phase_count> phase_timings{};

    const auto record_phase_timing = [&] (
        TimingPhase phase,
        TimingDuration elapsed) noexcept {
        auto& accumulator = phase_timings[static_cast<std::size_t>(phase)];
        if (!accumulator.valid) return;
        if (elapsed < TimingDuration::zero()
            || accumulator.sample_count
                == std::numeric_limits<std::uint64_t>::max()
            || elapsed > TimingDuration::max() - accumulator.total) {
            accumulator.valid = false;
            return;
        }
        ++accumulator.sample_count;
        accumulator.total += elapsed;
        if (elapsed > accumulator.maximum) accumulator.maximum = elapsed;
    };

    const auto report_phase_timings = [&] () noexcept {
        if (!distributed_timing) return;
#ifndef COSMO_NBODY_HAS_MPI
        return;
#else
        constexpr std::array<const char*, timing_phase_count> phase_names{
            "force_refresh_total",
            "domain_partition",
            "ghost_exchange",
            "gravity_solve",
            "force_balance",
            "snapshot_write",
            "restart_write",
        };
        constexpr std::array<const char*, timing_phase_count> phase_scopes{
            "lightweight_mpi_core",
            "lightweight_mpi_core",
            "lightweight_mpi_core",
            "lightweight_mpi_core",
            "diagnostic_only",
            "artifact_io",
            "artifact_io",
        };

        std::array<std::uint64_t, timing_phase_count> local_counts{};
        std::array<double, timing_phase_count> local_totals{};
        std::array<double, timing_phase_count> local_maxima{};
        std::array<int, timing_phase_count> local_valid{};
        for (std::size_t i = 0; i < timing_phase_count; ++i) {
            local_counts[i] = phase_timings[i].sample_count;
            local_totals[i] = std::chrono::duration<double>(
                phase_timings[i].total).count();
            local_maxima[i] = std::chrono::duration<double>(
                phase_timings[i].maximum).count();
            local_valid[i] = phase_timings[i].valid
                && std::isfinite(local_totals[i])
                && local_totals[i] >= 0.0
                && std::isfinite(local_maxima[i])
                && local_maxima[i] >= 0.0
                ? 1 : 0;
            if (local_valid[i] == 0) {
                local_totals[i] = 0.0;
                local_maxima[i] = 0.0;
            }
        }

        auto count_min = local_counts;
        auto count_max = local_counts;
        auto total_min = local_totals;
        auto total_max = local_totals;
        auto call_max = local_maxima;
        auto valid_min = local_valid;
        const std::uint64_t local_effective_threads =
            static_cast<std::uint64_t>(runtime_context_.thread_count());
        const std::uint64_t local_shared_size =
            static_cast<std::uint64_t>(runtime_context_.local_size());
        std::uint64_t effective_threads_min = local_effective_threads;
        std::uint64_t effective_threads_max = local_effective_threads;
        std::uint64_t shared_size_min = local_shared_size;
        std::uint64_t shared_size_max = local_shared_size;
        const int count = static_cast<int>(timing_phase_count);
        bool reductions_ok = true;
        reductions_ok &= MPI_Reduce(
            local_counts.data(), count_min.data(), count,
            MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            local_counts.data(), count_max.data(), count,
            MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            local_totals.data(), total_min.data(), count,
            MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            local_totals.data(), total_max.data(), count,
            MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            local_maxima.data(), call_max.data(), count,
            MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            local_valid.data(), valid_min.data(), count,
            MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            &local_effective_threads, &effective_threads_min, 1,
            MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            &local_effective_threads, &effective_threads_max, 1,
            MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            &local_shared_size, &shared_size_min, 1,
            MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
        reductions_ok &= MPI_Reduce(
            &local_shared_size, &shared_size_max, 1,
            MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD) == MPI_SUCCESS;

        if (!reductions_ok) {
            if (runtime_context_.rank() == 0) {
                std::cerr
                    << "WARNING: runtime phase timing aggregation failed after "
                       "scientific snapshot publication; timing telemetry is unavailable.\n";
            }
            return;
        }
        if (runtime_context_.rank() != 0) return;

        std::cout << std::setprecision(17);
        for (std::size_t i = 0; i < timing_phase_count; ++i) {
            const bool force_balance_phase =
                i == static_cast<std::size_t>(TimingPhase::ForceBalance);
            const bool artifact_io_phase =
                i == static_cast<std::size_t>(TimingPhase::SnapshotWrite)
                || i == static_cast<std::size_t>(TimingPhase::RestartWrite);
            const bool enabled = force_balance_phase
                ? diagnostics_enabled
                : artifact_io_phase
                    ? true
                    : lightweight_performance_timing;
            std::cout
                << "RUNTIME_PHASE_TIMING"
                << " phase=" << phase_names[i]
                << " scope=" << phase_scopes[i]
                << " enabled=" << (enabled ? "true" : "false")
                << " valid=" << (valid_min[i] != 0 ? "true" : "false")
                << " ranks=" << runtime_context_.size()
                << " effective_threads_min=" << effective_threads_min
                << " effective_threads_max=" << effective_threads_max
                << " shared_local_size_min=" << shared_size_min
                << " shared_local_size_max=" << shared_size_max
                << " samples_min=" << count_min[i]
                << " samples_max=" << count_max[i]
                << " total_seconds_min_rank=" << total_min[i]
                << " total_seconds_max_rank=" << total_max[i]
                << " max_call_seconds=" << call_max[i]
                << '\n';
        }
        std::cout.flush();
#endif
    };

    record_memory_sample("run_start");
    std::optional<core::Real> final_force_balance_relative_residual;
    std::optional<core::Real> max_force_balance_relative_residual;
    std::uint64_t force_balance_sample_count = 0;

    const auto sample_force_balance = [&] (
        std::span<const core::Real> acceleration_x,
        std::span<const core::Real> acceleration_y,
        std::span<const core::Real> acceleration_z,
        std::size_t owned) {
        if (!diagnostics_enabled) return;
        const auto timing_start = TimingClock::now();
        const auto diagnostics = compute_runtime_force_balance(
            particles_,
            acceleration_x,
            acceleration_y,
            acceleration_z,
            owned,
            runtime_context_.mpi_active(),
            runtime_context_.size());
        record_phase_timing(
            TimingPhase::ForceBalance,
            TimingClock::now() - timing_start);
        final_force_balance_relative_residual = diagnostics.relative_residual;
        max_force_balance_relative_residual = std::max(
            max_force_balance_relative_residual.value_or(core::Real{0.0}),
            diagnostics.relative_residual);
        if (force_balance_sample_count
            == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "Force-balance sample count exceeds uint64 range");
        }
        ++force_balance_sample_count;
    };

    if (runtime_context_.rank() == 0) {
        std::cout << "Starting KDK leapfrog loop (ranks "
                  << runtime_context_.size()
                  << ", threads/rank "
                  << runtime_context_.thread_count() << ")...\n";
        if (!diagnostics_enabled) {
            std::cout
                << "Memory-saving mode: in-run Layzer-Irvine workspace disabled; "
                   "inspect written snapshots offline when needed.\n";
        }
        std::cout.flush();
    }

    struct ForcePotentialSample {
        core::Real scale_factor{0.0};
        gravity::PMForceDiagnostics diagnostics{};
    };
    std::optional<ForcePotentialSample> latest_force_potential;
    const bool request_reused_pm_potential =
        diagnostics_enabled && static_cast<bool>(pm_solver_);

    auto force_func = [&] (
        std::span<const core::Real>,
        std::span<const core::Real>,
        std::span<const core::Real>,
        std::span<const core::Real>,
        std::span<core::Real>,
        std::span<core::Real>,
        std::span<core::Real>,
        core::Real force_scale_factor) {
        latest_force_potential.reset();
        record_memory_sample("force_entry");
        const auto force_refresh_start = lightweight_performance_timing
            ? TimingClock::now()
            : TimingClock::time_point{};
        const auto finish_force = [&] () {
            if (lightweight_performance_timing) {
                record_phase_timing(
                    TimingPhase::ForceRefreshTotal,
                    TimingClock::now() - force_refresh_start);
            }
            // Keep operational RSS collection outside the measured force
            // interval so the telemetry collective cannot inflate solver
            // timing. Peak RSS still includes the completed force work.
            record_memory_sample("post_force");
        };

        if (runtime_context_.mpi_active()) {
            particles_.clear_ghosts();
            if (lightweight_performance_timing) {
                const auto timing_start = TimingClock::now();
                domain_decomp_.partition_domain(particles_);
                record_phase_timing(
                    TimingPhase::DomainPartition,
                    TimingClock::now() - timing_start);
            } else {
                domain_decomp_.partition_domain(particles_);
            }
            record_memory_sample("post_partition");
            if (lightweight_performance_timing) {
                const auto timing_start = TimingClock::now();
                domain_decomp_.exchange_ghosts(
                    particles_, required_ghost_width());
                record_phase_timing(
                    TimingPhase::GhostExchange,
                    TimingClock::now() - timing_start);
            } else {
                domain_decomp_.exchange_ghosts(
                    particles_, required_ghost_width());
            }
            record_memory_sample("post_ghost_exchange");
        }

        auto ax = particles_.mutable_accelerations_x();
        auto ay = particles_.mutable_accelerations_y();
        auto az = particles_.mutable_accelerations_z();
        const std::size_t owned = particles_.num_owned_particles();
        if (pm_solver_) {
            // Leapfrog zeros the pre-partition storage before invoking this
            // callback. MPI partitioning can reorder or resize owned particles,
            // so overwrite the post-partition spans explicitly before the PM
            // gather-add operator consumes them.
            if (runtime_context_.mpi_active()) {
#ifdef COSMO_NBODY_HAS_OPENMP
                #pragma omp parallel for schedule(static) \
                    if(should_use_host_parallel_team(owned))
#endif
                for (std::size_t i = 0; i < owned; ++i) {
                    ax[i] = 0.0;
                    ay[i] = 0.0;
                    az[i] = 0.0;
                }
            }
            const auto timing_start = lightweight_performance_timing
                ? TimingClock::now()
                : TimingClock::time_point{};
            auto pm_diagnostics = pm_solver_->compute_forces(
                particles_.get_positions_x().first(owned),
                particles_.get_positions_y().first(owned),
                particles_.get_positions_z().first(owned),
                particles_.get_uniform_mass().has_value()
                    ? std::span<const core::Real>{}
                    : particles_.get_masses().first(owned),
                particles_.get_uniform_mass(),
                ax.first(owned),
                ay.first(owned),
                az.first(owned),
                request_reused_pm_potential);
            if (lightweight_performance_timing) {
                record_phase_timing(
                    TimingPhase::GravitySolve,
                    TimingClock::now() - timing_start);
            }
            if (pm_diagnostics.has_value()) {
                latest_force_potential = ForcePotentialSample{
                    force_scale_factor,
                    *pm_diagnostics};
            }
            sample_force_balance(ax, ay, az, owned);
            finish_force();
            return;
        }
        if (treepm_solver_) {
            const auto timing_start = lightweight_performance_timing
                ? TimingClock::now()
                : TimingClock::time_point{};
            treepm_solver_->compute_forces(particles_);
            if (lightweight_performance_timing) {
                record_phase_timing(
                    TimingPhase::GravitySolve,
                    TimingClock::now() - timing_start);
            }
            if (treepm_solver_->last_pm_diagnostics().has_value()) {
                latest_force_potential = ForcePotentialSample{
                    force_scale_factor,
                    *treepm_solver_->last_pm_diagnostics()};
            }
            const std::size_t current_owned = particles_.num_owned_particles();
            sample_force_balance(
                particles_.get_accelerations_x().first(current_owned),
                particles_.get_accelerations_y().first(current_owned),
                particles_.get_accelerations_z().first(current_owned),
                current_owned);
            finish_force();
            return;
        }
        throw std::logic_error(
            "SimulationRunner has no configured gravity solver");
    };

    const core::Real a_start =
        1.0 / (1.0 + config_.get_time().z_start);
    const core::Real a_end =
        1.0 / (1.0 + config_.get_time().z_final);

    time::LeapfrogIntegrator integrator(
        particles_, drift_kick_, force_func, config_.get_box().L,
        runtime_context_.mpi_active() ? runtime_context_.size() : 1);
    time::TimeStepper stepper(
        a_start, a_end, config_.get_time().delta_ln_a,
        config_.get_output().snapshot_scale_factors);
    if (resumed_from_restart_) {
        stepper.restore_state(current_step_, current_a_);
    } else {
        current_step_ = 0;
    }

    std::unique_ptr<validation::ConservationChecks> conservation;
    validation::ConservationState li_state;
    core::Real max_li_ratio = 0.0;
    bool li_ratio_undefined = false;
    std::uint64_t li_sample_count = 0;
    std::uint64_t li_finite_sample_count = 0;
    std::optional<validation::LayzerIrvineSampleRecord> max_li_sample;
    std::vector<validation::LayzerIrvineSampleRecord> li_samples;
    const bool collect_layzer_irvine = diagnostics_enabled;

    struct PotentialDiagnosticSample {
        core::Real raw{0.0};
        core::Real self{0.0};
        bool reused_force_potential{false};
    };
    const auto evaluate_potential_diagnostic = [&] (
        core::Real scale_factor) -> PotentialDiagnosticSample {
        if (!conservation) {
            throw std::logic_error(
                "Layzer-Irvine potential diagnostic requested without conservation state");
        }
        const bool reusable = latest_force_potential.has_value()
            && latest_force_potential->scale_factor == scale_factor;
        if (reusable) {
            core::Real raw =
                latest_force_potential->diagnostics.potential_energy_comoving
                / scale_factor;
            if (treepm_solver_) {
                raw += conservation->compute_treepm_short_potential_energy(
                    particles_,
                    treepm_solver_->last_force_tree(),
                    scale_factor);
            }
            const core::Real self =
                latest_force_potential->diagnostics.cic_self_energy_comoving
                    .has_value()
                ? *latest_force_potential->diagnostics.cic_self_energy_comoving
                    / scale_factor
                : conservation->compute_cic_self_energy(
                      particles_, scale_factor);
            if (!std::isfinite(raw) || !std::isfinite(self)) {
                throw std::overflow_error(
                    "Layzer-Irvine force-produced potential diagnostic is non-finite");
            }
            return PotentialDiagnosticSample{raw, self, true};
        }

        const core::Real raw = conservation->compute_potential_energy(
            particles_,
            scale_factor,
            treepm_solver_ ? &treepm_solver_->last_force_tree() : nullptr);
        const core::Real self = conservation->compute_cic_self_energy(
            particles_, scale_factor);
        return PotentialDiagnosticSample{raw, self, false};
    };

    if (collect_layzer_irvine) {
        std::exception_ptr diagnostic_setup_exception;
        try {
            if (current_step_ > stepper.num_steps()) {
                throw std::logic_error(
                    "Current step exceeds the materialized TimeStepper schedule");
            }
            if (runtime_context_.rank() == 0) {
                li_samples.reserve(stepper.num_steps() - current_step_);
            }
            conservation =
                std::make_unique<validation::ConservationChecks>(config_);
        } catch (...) {
            diagnostic_setup_exception = std::current_exception();
        }
        synchronize_exception_or_rethrow(
            runtime_context_.mpi_active(),
            diagnostic_setup_exception,
            runtime_context_.size(),
            "Layzer-Irvine diagnostic storage setup");

        if (!runtime_context_.mpi_active()) {
            // Only 27 impulse-response values are retained. Build them while
            // no PM force mesh is live, then release the temporary diagnostic
            // mesh before the first force refresh. Distributed PM supplies its
            // own self term and must not allocate this replicated preparation.
            record_memory_sample("before_self_kernel");
            conservation->prepare_cic_self_kernel();
            record_memory_sample("after_self_kernel");
        }
        if (request_reused_pm_potential || treepm_solver_) {
            integrator.ensure_force_at(current_a_);
        }
        const PotentialDiagnosticSample initial_potential =
            evaluate_potential_diagnostic(current_a_);
        const core::Real W0 =
            initial_potential.raw - initial_potential.self;
        conservation->reset_state(
            particles_, current_a_, W0, li_state);
        last_layzer_irvine_max_ratio_ = 0.0;
    } else {
        last_layzer_irvine_max_ratio_ =
            std::numeric_limits<core::Real>::quiet_NaN();
    }

    const auto& snapshot_targets =
        config_.get_output().snapshot_scale_factors;
    std::size_t next_snapshot_index = 0;
    while (next_snapshot_index < snapshot_targets.size()
           && snapshot_targets[next_snapshot_index] <= current_a_) {
        ++next_snapshot_index;
    }

    // A resumed execution publishes into its own artifact directory. A final
    // snapshot in the parent run is not an output of this execution segment.
    bool final_snapshot_written = false;
    const std::uint64_t restart_cadence =
        config_.get_output().restart_cadence_steps;

    while (const auto step = stepper.next_step()) {
        latest_force_potential.reset();
        integrator.step(*step);
        current_a_ = step->a_end();
        current_step_ = stepper.current_step();

        auto step_drift = integrator.last_step_max_drift_displacement();
        int local_invalid = step_drift.has_value() ? 0 : 1;
        core::Real global_step_drift = step_drift.value_or(0.0);
        if (runtime_context_.mpi_active()) {
#ifdef COSMO_NBODY_HAS_MPI
            int global_invalid = 0;
            core::Real reduced_step_drift = 0.0;
            const int validity_status = MPI_Allreduce(
                &local_invalid, &global_invalid, 1, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);
            const int displacement_status = MPI_Allreduce(
                &global_step_drift, &reduced_step_drift, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
            if (validity_status != MPI_SUCCESS
                || displacement_status != MPI_SUCCESS) {
                throw std::runtime_error(
                    "Per-step drift diagnostic MPI reduction failed");
            }
            local_invalid = global_invalid;
            global_step_drift = reduced_step_drift;
#else
            throw std::logic_error("MPI active state in non-MPI build");
#endif
        }
        if (local_invalid != 0 || !std::isfinite(global_step_drift)) {
            step_drift_measurement_invalid_ = true;
            max_step_drift_displacement_.reset();
        } else if (!step_drift_measurement_invalid_) {
            max_step_drift_displacement_ = std::max(
                max_step_drift_displacement_.value_or(0.0),
                global_step_drift);
        }

        std::optional<core::Real> W;
        std::optional<core::Real> li_ratio;
        if (collect_layzer_irvine) {
            const PotentialDiagnosticSample potential =
                evaluate_potential_diagnostic(current_a_);
            const core::Real W_raw = potential.raw;
            const core::Real W_self = potential.self;
            W = W_raw - W_self;
            const core::Real previous_a = li_state.a_prev;
            const core::Real source_start = li_state.source_prev;
            const core::Real residual =
                conservation->evaluate_layzer_irvine(
                    particles_, current_a_, *W, li_state);
            li_ratio = validation::layzer_irvine_ratio(
                residual, li_state.last_kinetic, *W);

            if (li_sample_count
                == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error(
                    "Layzer-Irvine sample count exceeds uint64 range");
            }
            ++li_sample_count;

            validation::LayzerIrvineSampleRecord sample;
            sample.step = static_cast<std::uint64_t>(current_step_);
            sample.scale_factor_start = previous_a;
            sample.scale_factor_end = current_a_;
            sample.delta_ln_a =
                stable_layzer_irvine_log_interval(current_a_, previous_a);
            sample.kinetic_energy = li_state.last_kinetic;
            sample.potential_energy_raw = W_raw;
            sample.cic_self_energy = W_self;
            sample.potential_energy_pair = *W;
            sample.initial_energy = li_state.initial_E;
            sample.source_start = source_start;
            sample.source = li_state.last_source;
            sample.integrated_source = li_state.integrated_source;
            sample.integrated_source_compensation =
                li_state.integrated_source_compensation;
            sample.residual = residual;
            sample.ratio = li_ratio;
            sample.reused_force_potential = potential.reused_force_potential;
            if (runtime_context_.rank() == 0) {
                li_samples.push_back(sample);
            }

            if (li_ratio.has_value()) {
                if (li_finite_sample_count
                    == std::numeric_limits<std::uint64_t>::max()) {
                    throw std::overflow_error(
                        "Finite Layzer-Irvine sample count exceeds uint64 range");
                }
                ++li_finite_sample_count;
                if (!max_li_sample.has_value()
                    || *li_ratio > max_li_ratio) {
                    max_li_ratio = *li_ratio;
                    max_li_sample = sample;
                }
            } else {
                li_ratio_undefined = true;
            }
        }

        if (current_step_ % 10 == 0 && runtime_context_.rank() == 0) {
            std::cout << "Step " << current_step_
                      << " | a = " << std::fixed
                      << std::setprecision(4) << current_a_
                      << " | z = " << (1.0 / current_a_ - 1.0);
            if (collect_layzer_irvine) {
                const core::Real K = li_state.last_kinetic;
                std::cout << " | K = " << std::scientific
                          << std::setprecision(3) << K
                          << " W = " << *W
                          << " | LI |resid|/max(|K|,|W|) = ";
                if (li_ratio.has_value()) {
                    std::cout << *li_ratio;
                } else {
                    std::cout
                        << "undefined(nonzero residual with vanishing energy scale)";
                }
            }
            std::cout << std::fixed << std::endl;
        }

        if (next_snapshot_index < snapshot_targets.size()
            && current_a_ == snapshot_targets[next_snapshot_index]) {
            const auto timing_start = distributed_timing
                ? TimingClock::now()
                : TimingClock::time_point{};
            write_snapshot_collective(
                current_a_, checked_snapshot_index(next_snapshot_index));
            if (distributed_timing) {
                record_phase_timing(
                    TimingPhase::SnapshotWrite,
                    TimingClock::now() - timing_start);
            }
            record_memory_sample("post_snapshot");
            if (current_a_ == a_end) {
                final_snapshot_written = true;
            }
            ++next_snapshot_index;
        }

        if (restart_cadence > 0
            && static_cast<std::uint64_t>(current_step_)
                % restart_cadence == 0) {
            const auto timing_start = distributed_timing
                ? TimingClock::now()
                : TimingClock::time_point{};
            write_restart_collective(stepper);
            if (distributed_timing) {
                record_phase_timing(
                    TimingPhase::RestartWrite,
                    TimingClock::now() - timing_start);
            }
            record_memory_sample("post_restart_checkpoint");
        }
    }

    if (collect_layzer_irvine) {
        if (li_sample_count == 0) {
            last_layzer_irvine_max_ratio_ =
                std::numeric_limits<core::Real>::quiet_NaN();
            if (runtime_context_.rank() == 0) {
                std::cout
                    << "Layzer-Irvine measurement unavailable: no "
                       "post-initialization evolution step contributed a sample.\n";
            }
        } else if (li_ratio_undefined) {
            last_layzer_irvine_max_ratio_ =
                std::numeric_limits<core::Real>::infinity();
            if (runtime_context_.rank() == 0) {
                std::cout
                    << "Layzer-Irvine ratio became undefined because both |K| and "
                       "|W| vanished while the residual remained nonzero.\n"
                    << "WARNING: inspect "
                    << artifact_layout_.layzer_irvine_timeline_path().string()
                    << " before interpretation.\n";
            }
        } else {
            last_layzer_irvine_max_ratio_ = max_li_ratio;
            if (runtime_context_.rank() == 0) {
                std::cout
                    << "Layzer-Irvine max |residual|/max(|K|,|W|) over run = "
                    << std::scientific << max_li_ratio
                    << " from " << li_sample_count << " samples";
                if (max_li_sample.has_value()) {
                    std::cout << " at step " << max_li_sample->step
                              << " (a=" << max_li_sample->scale_factor_end << ")";
                }
                std::cout
                    << " (measurement only; no universal single-run threshold)\n";
            }
        }
    } else if (diagnostics_enabled) {
        last_layzer_irvine_max_ratio_ =
            std::numeric_limits<core::Real>::quiet_NaN();
    }

    if (runtime_context_.rank() == 0) {
        if (force_balance_sample_count > 0) {
            std::cout
                << "Force-balance residual over current execution segment: final = "
                << std::scientific
                << *final_force_balance_relative_residual
                << ", max = " << *max_force_balance_relative_residual
                << ", samples = " << force_balance_sample_count
                << " (instantaneous measurement only)\n";
        } else if (diagnostics_enabled) {
            std::cout
                << "Force-balance diagnostic unavailable: no force solve "
                   "occurred in this execution segment.\n";
        }

        if (max_step_drift_displacement_.has_value()) {
            std::cout
                << "Peak exact KDK drift over current execution segment = "
                << std::scientific << *max_step_drift_displacement_
                << " Mpc/h; /mean-spacing = "
                << *max_step_drift_displacement_ / config_.d_mean()
                << "; /PM-cell = "
                << *max_step_drift_displacement_
                    / (config_.get_box().L / config_.get_box().N_mesh)
                << std::fixed
                << " (diagnostic only; not a timestep controller)\n";
        } else {
            std::cout
                << "Exact KDK drift diagnostic unavailable for this execution "
                   "segment; no zero-valued proxy was recorded.\n";
        }
    }

    // Publish the Layzer-Irvine series before the final snapshot so the
    // evolved-state diagnostic has an independently durable artifact. The
    // aggregate diagnostic report is deliberately deferred until after the
    // final snapshot and terminal memory sample so its memory summary covers
    // the complete execution, including final-output high-water usage.
    std::exception_ptr li_timeline_exception;
    if (diagnostics_enabled && runtime_context_.rank() == 0) {
        try {
            validation::write_layzer_irvine_timeline_atomic(
                li_samples, artifact_layout_.layzer_irvine_timeline_path());
        } catch (...) {
            li_timeline_exception = std::current_exception();
        }
    }
    synchronize_exception_or_rethrow(
        runtime_context_.mpi_active(),
        li_timeline_exception,
        runtime_context_.size(),
        "Layzer-Irvine timeline publication");

    if (!final_snapshot_written) {
        const auto timing_start = distributed_timing
            ? TimingClock::now()
            : TimingClock::time_point{};
        const std::size_t final_snapshot_index = next_snapshot_index > 0
                && snapshot_targets[next_snapshot_index - 1] == current_a_
            ? next_snapshot_index - 1 : next_snapshot_index;
        write_snapshot_collective(
            current_a_, checked_snapshot_index(final_snapshot_index));
        if (distributed_timing) {
            record_phase_timing(
                TimingPhase::SnapshotWrite,
                TimingClock::now() - timing_start);
        }
        record_memory_sample("post_snapshot");
    }

    // This sample is independent of physical diagnostics. Process high-water
    // survives transient force, snapshot, and restart allocations and therefore
    // records final operational memory telemetry without controlling execution.
    record_memory_sample("terminal");

    if (diagnostics_enabled) {
        write_validation_report(
            final_force_balance_relative_residual,
            max_force_balance_relative_residual);
    }

    std::exception_ptr memory_timeline_exception;
    if (diagnostics_enabled && runtime_context_.rank() == 0) {
        try {
            write_memory_timeline();
        } catch (...) {
            memory_timeline_exception = std::current_exception();
        }
    }
    synchronize_exception_or_rethrow(
        runtime_context_.mpi_active(),
        memory_timeline_exception,
        runtime_context_.size(),
        "Runtime memory timeline publication");

    // Performance telemetry is deliberately published only after the final
    // scientific snapshot is durable. Failure to aggregate the measurements
    // cannot prevent the simulation product from being written.
    report_phase_timings();
}

} // namespace cosmo_nbody::runtime
