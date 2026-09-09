#include "cosmo_nbody/io/metadata.hpp"

#include "bounded_text_serialization.hpp"

#include "cosmo_nbody/analysis/analysis_resource_policy.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/runtime_provenance.hpp"
#include "cosmo_nbody/io/restart_lineage.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace io {

namespace {

void append_json_escaped(std::ostream& out, std::string_view value) {
    static constexpr char hexadecimal[] = "0123456789abcdef";
    for (const char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: {
                const unsigned char byte = static_cast<unsigned char>(c);
                if (byte < 0x20U) {
                    out << "\\u00"
                        << hexadecimal[(byte >> 4U) & 0x0fU]
                        << hexadecimal[byte & 0x0fU];
                } else {
                    out << c;
                }
                break;
            }
        }
    }
}

void append_json_string(
    std::ostream& out,
    std::string_view key,
    std::string_view value,
    bool comma = true) {
    out << "\"" << key << "\":\"";
    append_json_escaped(out, value);
    out << "\"";
    if (comma) out << ",";
}

template <typename T>
void append_json_number(
    std::ostream& out,
    std::string_view key,
    T value,
    bool comma = true) {
    out << "\"" << key << "\":" << value;
    if (comma) out << ",";
}

void append_json_bool(
    std::ostream& out,
    std::string_view key,
    bool value,
    bool comma = true) {
    out << "\"" << key << "\":" << (value ? "true" : "false");
    if (comma) out << ",";
}

void append_json_null(
    std::ostream& out,
    std::string_view key,
    bool comma = true) {
    out << "\"" << key << "\":null";
    if (comma) out << ",";
}

void append_real_array(
    std::ostream& out,
    std::string_view key,
    const std::vector<core::Real>& values,
    bool comma = true) {
    out << "\"" << key << "\":[";
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) out << ",";
        out << values[idx];
    }
    out << "]";
    if (comma) out << ",";
}

template <typename StringRange>
void append_string_array(
    std::ostream& out,
    std::string_view key,
    const StringRange& values,
    bool comma = true) {
    out << "\"" << key << "\":[";
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) out << ",";
        out << "\"";
        append_json_escaped(out, std::string_view(values[idx]));
        out << "\"";
    }
    out << "]";
    if (comma) out << ",";
}

void append_fftw_planning_record(
    std::ostream& out,
    const mesh::FftwPlanningRecord& record) {
    out << "{";
    append_json_string(out, "backend", record.backend);
    append_json_string(out, "planner_rigor", record.planner_rigor);
    append_json_string(out, "planner_schema", record.planner_schema);
    append_json_string(out, "planner_flags", record.planner_flags);
    append_json_string(
        out, "wisdom_cache_policy", record.wisdom_cache_policy);
    append_json_string(
        out, "plan_identity_scope", record.plan_identity_scope);
    append_json_number(out, "grid_size", record.grid_size);
    append_json_number(out, "thread_count", record.thread_count);
    append_json_number(out, "mpi_rank_count", record.mpi_rank_count);
    append_json_string(out, "fftw_version", record.fftw_version);
    append_json_string(
        out,
        "fftw_build_identity_sha256",
        record.fftw_build_identity_sha256);
    append_json_string(
        out,
        "fftw_provider_path_content_sha256",
        record.fftw_provider_path_content_sha256);
    append_json_string(
        out,
        "runtime_system_identity_sha256",
        record.runtime_system_identity_sha256);
    append_json_string(out, "wisdom_path", record.wisdom_path);
    append_json_bool(out, "wisdom_imported", record.wisdom_imported);
    append_json_string(
        out,
        "imported_wisdom_sha256",
        record.imported_wisdom_sha256);
    append_json_string(
        out,
        "canonical_wisdom_sha256",
        record.canonical_wisdom_sha256);
    append_json_string(
        out,
        "planned_wisdom_corpus_sha256",
        record.planned_wisdom_corpus_sha256);
    append_json_string(
        out,
        "plan_representation_sha256",
        record.plan_representation_sha256);
    append_json_string(
        out,
        "wisdom_publication_outcome",
        record.wisdom_publication_outcome);
    append_json_bool(
        out, "wisdom_published", record.wisdom_published, false);
    out << "}";
}

void append_fftw_planning_record_array(
    std::ostream& out,
    const std::vector<mesh::FftwPlanningRecord>& records,
    bool comma = true) {
    out << "\"fftw_planning_records\":[";
    for (std::size_t index = 0; index < records.size(); ++index) {
        if (index != 0U) out << ',';
        append_fftw_planning_record(out, records[index]);
    }
    out << ']';
    if (comma) out << ',';
}

void append_ic_support(std::ostream& out, const RunMetadata& meta) {
    if (meta.ic_max_mode_per_axis) {
        append_json_number(out, "ic_max_mode_per_axis", *meta.ic_max_mode_per_axis);
    } else {
        append_json_null(out, "ic_max_mode_per_axis");
    }
    if (meta.ic_effective_max_mode_per_axis) {
        append_json_number(out, "ic_effective_max_mode_per_axis", *meta.ic_effective_max_mode_per_axis);
    } else {
        append_json_null(out, "ic_effective_max_mode_per_axis");
    }
}

void append_common_prefix(
    std::ostream& out,
    const RunMetadata& meta) {
    append_json_string(out, "product_kind", "run_metadata");
    append_json_string(out, "lineage", lineage_to_string(meta.lineage));
    append_json_number(out, "h", meta.h);
    append_json_number(out, "omega_m", meta.omega_m);
    append_json_number(out, "omega_lambda", meta.omega_lambda);
    append_json_number(out, "omega_b", meta.omega_b);
    append_json_number(out, "sigma8", meta.sigma8);
    append_json_number(out, "n_s", meta.n_s);
    append_json_number(out, "box_size_Mpc_h", meta.box_size_Mpc_h);
    append_json_number(out, "particles_per_dimension", meta.particles_per_dimension);
    append_json_number(out, "pm_mesh_per_dimension", meta.pm_mesh_per_dimension);
    append_json_number(out, "ic_mesh_per_dimension", meta.ic_mesh_per_dimension);
    append_ic_support(out, meta);
    append_json_number(out, "mean_spacing_Mpc_h", meta.mean_spacing_Mpc_h);
    append_json_number(out, "particle_mass", meta.particle_mass);
    append_json_number(out, "r_s", meta.r_s);
    append_json_number(out, "r_cut", meta.r_cut);
    append_json_number(out, "k_nyq", meta.k_nyq);
    append_json_string(out, "solver", meta.solver);
    append_json_number(out, "softening_Mpc_h", meta.softening_Mpc_h);
    append_json_number(out, "theta", meta.theta);
    append_json_number(out, "split_scale_cells", meta.split_scale_cells);
    append_json_bool(out, "deconvolve_cic", meta.deconvolve_cic);
}

void append_common_suffix(std::ostream& out, const RunMetadata& meta) {
    append_json_string(
        out, "treepm_short_range_composition", meta.treepm_short_range_composition);
    append_json_number(out, "z_start", meta.z_start);
    append_json_number(out, "z_final", meta.z_final);
    append_json_number(out, "delta_ln_a", meta.delta_ln_a);
    append_json_string(out, "step_policy", meta.step_policy);
    append_real_array(out, "snapshot_scale_factors", meta.snapshot_scale_factors);
    append_json_string(out, "output_format", meta.output_format);
    append_json_number(out, "restart_cadence_steps", meta.restart_cadence_steps);
    append_json_string(out, "output_root_directory", meta.output_root_directory);
    append_json_string(out, "output_run_label", meta.output_run_label);
    append_json_bool(
        out, "timestamped_run_directory", meta.timestamped_run_directory);
    append_json_number(
        out, "snapshot_batch_particles", meta.snapshot_batch_particles);
    append_json_number(out, "runtime_num_threads", meta.runtime_num_threads);
    append_json_bool(out, "runtime_mpi_enabled", meta.runtime_mpi_enabled);
    append_json_string(
        out, "runtime_ic_scratch_mode", meta.runtime_ic_scratch_mode);
    append_json_string(
        out, "runtime_scratch_directory", meta.runtime_scratch_directory);
    append_fftw_planning_record_array(out, meta.fftw_planning_records);
    append_json_bool(
        out,
        "fftw_planning_records_complete",
        meta.fftw_planning_records_complete);
    append_json_bool(
        out, "validation_write_diagnostics", meta.validation_write_diagnostics, false);
    out << "}";
}

void append_snapshot_derived_analysis_json(
    std::ostream& out,
    const RunMetadata& meta) {
    out << "{";
    append_json_string(
        out, "product_kind", "snapshot_derived_analysis_metadata");
    append_json_string(out, "lineage", "derived_analysis");
    append_json_bool(out, "resumed_from_restart", false);
    append_json_bool(out, "restart_parent_identity_available", false);
    append_json_string(
        out,
        "restart_parent_identity_scope",
        "inherited_via_input_snapshot_run_metadata");
    append_json_string(
        out,
        "restart_parent_identity_unavailable_reason",
        "refer_to_input_snapshot_run_metadata");
    append_json_null(out, "admitted_parent_restart_sha256");
    append_json_string(out, "physical_source", "input_snapshot_descriptor");
    append_json_string(out, "runtime_role", "analysis_execution");
    append_json_string(
        out,
        "trajectory_metadata_location",
        "input_snapshot:/Config/RunMetadataJson");
    append_json_bool(out, "parent_snapshot_run_metadata_embedded", false);

    append_json_number(out, "h", meta.h);
    append_json_number(out, "omega_m", meta.omega_m);
    append_json_number(out, "omega_lambda", meta.omega_lambda);
    append_json_number(out, "omega_b", meta.omega_b);
    append_json_number(out, "sigma8", meta.sigma8);
    append_json_number(out, "n_s", meta.n_s);
    append_json_number(out, "box_size_Mpc_h", meta.box_size_Mpc_h);
    append_json_number(out, "particles_per_dimension", meta.particles_per_dimension);
    append_json_number(out, "pm_mesh_per_dimension", meta.pm_mesh_per_dimension);
    append_json_number(out, "ic_mesh_per_dimension", meta.ic_mesh_per_dimension);
    append_ic_support(out, meta);
    append_json_number(out, "mean_spacing_Mpc_h", meta.mean_spacing_Mpc_h);
    append_json_null(out, "particle_mass");
    append_json_null(out, "r_s");
    append_json_null(out, "r_cut");
    append_json_number(out, "k_nyq", meta.k_nyq);
    append_json_null(out, "solver");
    append_json_number(out, "softening_Mpc_h", meta.softening_Mpc_h);
    append_json_null(out, "theta");
    append_json_null(out, "split_scale_cells");
    append_json_null(out, "deconvolve_cic");

    append_json_string(out, "ic_mode", "snapshot");
    append_json_string(
        out,
        "ic_provenance_status",
        "input_snapshot_descriptor_applied_upstream_run_metadata_not_republished");
    append_json_bool(out, "ic_generation_parameters_applied", false);
    append_json_number(out, "seed", meta.seed);
    append_json_number(out, "lpt_order", meta.lpt_order);
    append_json_null(out, "power_spectrum_file");
    append_json_null(out, "power_spectrum_sha256");
    append_json_null(out, "power_spectrum_redshift");
    if (meta.power_spectrum_fidelity.empty()) {
        append_json_null(out, "power_spectrum_fidelity");
    } else {
        append_json_string(
            out, "power_spectrum_fidelity", meta.power_spectrum_fidelity);
    }
    append_json_string(out, "snapshot_file", meta.snapshot_file);
    if (meta.snapshot_sha256.empty()) {
        append_json_null(out, "snapshot_sha256");
    } else {
        append_json_string(out, "snapshot_sha256", meta.snapshot_sha256);
    }
    append_json_string(out, "ic_amplitude_mode", meta.ic_amplitude_mode);
    append_json_string(out, "ic_phase_pairing", meta.ic_phase_pairing);
    append_json_null(out, "ic_lattice_convention");

    append_json_null(out, "treepm_short_range_composition");
    append_json_null(out, "z_start");
    append_json_null(out, "z_final");
    append_json_null(out, "delta_ln_a");
    append_json_null(out, "step_policy");
    append_json_null(out, "snapshot_scale_factors");
    append_json_null(out, "output_format");
    append_json_null(out, "restart_cadence_steps");
    append_json_null(out, "output_root_directory");
    append_json_null(out, "output_run_label");
    append_json_null(out, "timestamped_run_directory");
    append_json_null(out, "snapshot_batch_particles");

    // Requested analysis controls are retained here. Actual host execution
    // details are emitted once by analysis_runtime.json and are not duplicated
    // into this snapshot-derived scientific metadata product.
    append_json_number(out, "runtime_num_threads", meta.runtime_num_threads);
    append_json_bool(out, "runtime_mpi_enabled", meta.runtime_mpi_enabled);
    append_json_string(
        out, "runtime_ic_scratch_mode", meta.runtime_ic_scratch_mode);
    append_json_string(
        out, "runtime_scratch_directory", meta.runtime_scratch_directory);
    append_fftw_planning_record_array(out, meta.fftw_planning_records);
    append_json_bool(
        out,
        "fftw_planning_records_complete",
        meta.fftw_planning_records_complete);

    append_json_bool(out, "build_openmp_enabled", meta.build_openmp_enabled);
    append_json_bool(
        out, "build_fftw_threads_enabled", meta.build_fftw_threads_enabled);
    append_json_bool(out, "build_mpi_enabled", meta.build_mpi_enabled);
    append_json_bool(out, "build_fftw_mpi_enabled", meta.build_fftw_mpi_enabled);
    append_json_bool(
        out, "build_parallel_hdf5_enabled", meta.build_parallel_hdf5_enabled);
    append_json_null(out, "validation_write_diagnostics");
    std::vector<std::string_view> unavailable_fields = {
            "particle_mass",
            "r_s",
            "r_cut",
            "solver",
            "theta",
            "split_scale_cells",
            "deconvolve_cic",
            "power_spectrum_file",
            "power_spectrum_sha256",
            "power_spectrum_redshift",
            "ic_lattice_convention",
            "treepm_short_range_composition",
            "z_start",
            "z_final",
            "delta_ln_a",
            "step_policy",
            "snapshot_scale_factors",
            "output_format",
            "restart_cadence_steps",
            "output_root_directory",
            "output_run_label",
            "timestamped_run_directory",
            "snapshot_batch_particles",
            "validation_write_diagnostics",
    };
    if (meta.power_spectrum_fidelity.empty()) {
        unavailable_fields.push_back("power_spectrum_fidelity");
    }
    append_string_array(
        out,
        "unavailable_fields",
        unavailable_fields,
        false);
    out << "}";
}

std::uint64_t checked_u64(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                std::string(label) + " exceeds uint64 metadata range");
        }
    }
    return static_cast<std::uint64_t>(value);
}

void apply_build_capabilities(RunMetadata& meta) {
    (void)meta;
#ifdef COSMO_NBODY_HAS_OPENMP
    meta.build_openmp_enabled = true;
#endif
#ifdef COSMO_NBODY_HAS_FFTW_THREADS
    meta.build_fftw_threads_enabled = true;
#endif
#ifdef COSMO_NBODY_HAS_MPI
    meta.build_mpi_enabled = true;
#endif
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    meta.build_fftw_mpi_enabled = true;
#endif
#ifdef COSMO_NBODY_HAS_PARALLEL_HDF5
    meta.build_parallel_hdf5_enabled = true;
#endif
}

} // namespace

namespace {

std::uint64_t checked_metadata_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* context) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string(context) + " overflows uint64");
    }
    return lhs + rhs;
}

std::uint64_t checked_metadata_multiply(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* context) {
    if (lhs != 0U
        && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(
            std::string(context) + " overflows uint64");
    }
    return lhs * rhs;
}

std::uint64_t string_storage_bytes(std::string_view value) {
    return checked_metadata_add(
        static_cast<std::uint64_t>(value.size()),
        1U,
        "RunMetadata string storage");
}

void add_fftw_planning_record_storage(
    std::uint64_t& bytes,
    const std::vector<mesh::FftwPlanningRecord>& records,
    const char* context) {
    bytes = checked_metadata_add(
        bytes,
        checked_metadata_multiply(
            checked_u64(records.size(), "FFTW planning record count"),
            sizeof(mesh::FftwPlanningRecord),
            "RunMetadata FFTW planning record objects"),
        context);
    const auto add_string = [&](std::string_view value) {
        bytes = checked_metadata_add(
            bytes, string_storage_bytes(value), context);
    };
    for (const mesh::FftwPlanningRecord& record : records) {
        add_string(record.backend);
        add_string(record.planner_rigor);
        add_string(record.planner_schema);
        add_string(record.planner_flags);
        add_string(record.wisdom_cache_policy);
        add_string(record.plan_identity_scope);
        add_string(record.fftw_version);
        add_string(record.fftw_build_identity_sha256);
        add_string(record.fftw_provider_path_content_sha256);
        add_string(record.runtime_system_identity_sha256);
        add_string(record.wisdom_path);
        add_string(record.imported_wisdom_sha256);
        add_string(record.canonical_wisdom_sha256);
        add_string(record.planned_wisdom_corpus_sha256);
        add_string(record.plan_representation_sha256);
        add_string(record.wisdom_publication_outcome);
    }
}

void require_fftw_planning_record(
    const mesh::FftwPlanningRecord& record) {
    const bool serial = record.backend == "serial_fftw";
    const bool distributed = record.backend == "fftw_mpi";
    if ((!serial && !distributed)
        || record.planner_rigor.empty()
        || record.planner_schema.empty()
        || record.planner_flags.empty()
        || record.wisdom_cache_policy.empty()
        || record.plan_identity_scope.empty()
        || record.grid_size == 0U
        || record.thread_count < 1
        || record.mpi_rank_count == 0U
        || record.fftw_version.empty()
        || !is_canonical_sha256(record.fftw_build_identity_sha256)
        || !is_canonical_sha256(
            record.fftw_provider_path_content_sha256)
        || !is_canonical_sha256(record.runtime_system_identity_sha256)
        || record.wisdom_publication_outcome.empty()) {
        throw std::invalid_argument(
            "RunMetadata received an incomplete FFTW planning record");
    }
    if (record.wisdom_imported
        != is_canonical_sha256(record.imported_wisdom_sha256)) {
        throw std::invalid_argument(
            "FFTW imported-wisdom flag and identity disagree");
    }
    if ((record.wisdom_imported || record.wisdom_published)
        && record.wisdom_path.empty()) {
        throw std::invalid_argument(
            "FFTW wisdom provenance requires a source/destination path");
    }
    if (record.wisdom_path.empty()
        != record.canonical_wisdom_sha256.empty()
        || (!record.canonical_wisdom_sha256.empty()
            && !is_canonical_sha256(
                record.canonical_wisdom_sha256))) {
        throw std::invalid_argument(
            "FFTW canonical wisdom path and identity disagree");
    }
    if (record.wisdom_imported
        && record.canonical_wisdom_sha256
            != record.imported_wisdom_sha256) {
        throw std::invalid_argument(
            "Imported FFTW wisdom differs from the admitted canonical bytes");
    }
    if (record.wisdom_published
        && record.canonical_wisdom_sha256
            != record.planned_wisdom_corpus_sha256) {
        throw std::invalid_argument(
            "Published FFTW wisdom differs from the planned corpus");
    }
    if (serial) {
        if (!is_canonical_sha256(
                record.planned_wisdom_corpus_sha256)
            || !is_canonical_sha256(record.plan_representation_sha256)
            || record.mpi_rank_count != 1U) {
            throw std::invalid_argument(
                "Serial FFTW planning record lacks concrete plan identities");
        }
    } else if (!record.planned_wisdom_corpus_sha256.empty()
               || !record.plan_representation_sha256.empty()) {
        throw std::invalid_argument(
            "FFTW-MPI policy-only record must not invent serial plan identities");
    }
}

std::vector<mesh::FftwPlanningRecord>
snapshot_fftw_planning_records_bounded() {
    auto source = mesh::fftw_planning_records_snapshot();
    std::uint64_t planning_bytes = 0U;
    if (source.size()
        > MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES
              / sizeof(mesh::FftwPlanningRecord)) {
        throw std::length_error(
            "FFTW planning record count exceeds RunMetadata bounds");
    }
    for (const mesh::FftwPlanningRecord& record : source) {
        require_fftw_planning_record(record);
    }
    add_fftw_planning_record_storage(
        planning_bytes,
        source,
        "RunMetadata FFTW planning record payload");
    if (planning_bytes > MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES) {
        throw std::length_error(
            "FFTW planning record payload exceeds RunMetadata bounds");
    }
    return source;
}

void require_config_metadata_source_within_bound(
    const config::SimulationParameters& config,
    std::string_view verified_snapshot_ic_sha256,
    const std::vector<mesh::FftwPlanningRecord>& fftw_planning_records) {
    // The default object is the exact bounded allowance for every untouched
    // string field. Each potential replacement is then added in full, which
    // is conservative by the replaced field's previous storage and avoids
    // copying any caller-owned payload during admission.
    const RunMetadata default_metadata;
    std::uint64_t source_bytes =
        run_metadata_dynamic_payload_bytes(default_metadata);
    const auto add_source_string = [&](std::string_view value) {
        source_bytes = checked_metadata_add(
            source_bytes,
            string_storage_bytes(value),
            "RunMetadata configuration source payload");
    };

    const auto& gravity = config.get_gravity();
    const auto& time = config.get_time();
    const auto& ic = config.get_ic();
    const auto& output = config.get_output();
    const auto& memory_policy = config.get_memory_policy();

    add_source_string(gravity.solver);
    add_source_string(time.step_policy);
    add_source_string(ic.mode);
    add_source_string(ic.power_spectrum_file);
    add_source_string(ic.power_spectrum_sha256);
    add_source_string(ic.power_spectrum_fidelity);
    add_source_string(ic.snapshot_file);
    add_source_string(ic.snapshot_sha256);
    add_source_string(ic.amplitude_mode);
    add_source_string(ic.phase_pairing);
    add_source_string(verified_snapshot_ic_sha256);
    add_source_string(config.verified_snapshot_ic_sha256());
    add_source_string(config.verified_snapshot_ic_run_metadata_json());
    if (!output.formats.empty()) {
        add_source_string(output.formats.front());
    }
    add_source_string(output.root_directory);
    add_source_string(output.run_label);
    add_source_string(memory_policy.scratch_directory);
    add_source_string(config::scratch_mode_name(memory_policy.ic_scratch_mode));
    add_source_string("restart_checkpoint_manifest");
    add_source_string("inherited_via_input_snapshot_run_metadata");
    add_source_string("refer_to_input_snapshot_run_metadata");
    add_source_string(
        "external_snapshot_source_provenance_not_verified_input_hash_unavailable");
    add_source_string(
        "internal_generation_parameters_and_input_hash_applied");
    add_source_string(
        "external_snapshot_bytes_verified_upstream_generation_provenance_inherited");
    add_source_string("corner");
    add_source_string("plummer_total_minus_newtonian_long");
    source_bytes = checked_metadata_add(
        source_bytes, 65U, "RunMetadata restart parent identity allowance");
    source_bytes = checked_metadata_add(
        source_bytes,
        checked_metadata_multiply(
            checked_u64(
                output.snapshot_scale_factors.size(),
                "RunMetadata snapshot target count"),
            sizeof(core::Real),
            "RunMetadata configuration snapshot targets"),
        "RunMetadata configuration source payload");
    add_fftw_planning_record_storage(
        source_bytes,
        fftw_planning_records,
        "RunMetadata configuration FFTW planning payload");

    if (source_bytes > MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES) {
        throw std::length_error(
            "RunMetadata configuration source payload exceeds its enforced bound");
    }
}

void require_snapshot_analysis_metadata_source_within_bound(
    const SnapshotDescriptor& descriptor,
    std::string_view snapshot_file,
    const analysis::AnalysisResourcePolicy& resource_policy,
    const std::vector<mesh::FftwPlanningRecord>& fftw_planning_records) {
    const RunMetadata default_metadata;
    std::uint64_t source_bytes =
        run_metadata_dynamic_payload_bytes(default_metadata);
    const auto add_source_string = [&](std::string_view value) {
        source_bytes = checked_metadata_add(
            source_bytes,
            string_storage_bytes(value),
            "Snapshot-analysis RunMetadata source payload");
    };

    add_source_string("inherited_via_input_snapshot_run_metadata");
    add_source_string("refer_to_input_snapshot_run_metadata");
    add_source_string("snapshot");
    add_source_string(
        "input_snapshot_descriptor_applied_upstream_run_metadata_not_republished");
    add_source_string(snapshot_file);
    add_source_string(descriptor.ic_amplitude_mode);
    add_source_string(descriptor.ic_phase_pairing);
    add_source_string(descriptor.power_spectrum_fidelity);
    add_source_string(config::scratch_mode_name(
        resource_policy.memory.ic_scratch_mode));
    add_source_string(resource_policy.memory.scratch_directory);
    add_fftw_planning_record_storage(
        source_bytes,
        fftw_planning_records,
        "Snapshot-analysis RunMetadata FFTW planning payload");

    if (source_bytes > MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES) {
        throw std::length_error(
            "Snapshot-analysis RunMetadata source payload exceeds its enforced bound");
    }
}

} // namespace

RunMetadataJsonMemoryPlan run_metadata_json_memory_plan(
    std::uint64_t metadata_dynamic_payload_bytes,
    std::uint64_t json_logical_bytes) {
    if (metadata_dynamic_payload_bytes
            > MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES
        || json_logical_bytes == 0U
        || json_logical_bytes > MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES) {
        throw std::length_error(
            "RunMetadata JSON memory plan exceeds its enforced bounds");
    }
    const std::uint64_t json_storage_bytes = checked_metadata_add(
        json_logical_bytes, 1U, "RunMetadata JSON storage");
    return RunMetadataJsonMemoryPlan{
        metadata_dynamic_payload_bytes,
        json_storage_bytes,
        checked_metadata_add(
            metadata_dynamic_payload_bytes,
            json_storage_bytes,
            "RunMetadata serialization peak")};
}

std::uint64_t run_metadata_dynamic_payload_bytes(
    const RunMetadata& metadata) {
    std::uint64_t bytes = 0U;
    const auto add_string = [&](std::string_view value) {
        bytes = checked_metadata_add(
            bytes,
            string_storage_bytes(value),
            "RunMetadata dynamic string payload");
    };

    if (metadata.admitted_parent_restart_sha256.has_value()) {
        add_string(*metadata.admitted_parent_restart_sha256);
    }
    add_string(metadata.restart_parent_identity_scope);
    add_string(metadata.restart_parent_identity_unavailable_reason);
    add_string(metadata.solver);
    add_string(metadata.ic_mode);
    add_string(metadata.ic_provenance_status);
    add_string(metadata.power_spectrum_file);
    add_string(metadata.power_spectrum_sha256);
    add_string(metadata.power_spectrum_fidelity);
    add_string(metadata.snapshot_file);
    add_string(metadata.snapshot_sha256);
    add_string(metadata.ic_amplitude_mode);
    add_string(metadata.ic_phase_pairing);
    add_string(metadata.ic_lattice_convention);
    add_string(metadata.treepm_short_range_composition);
    add_string(metadata.step_policy);
    add_string(metadata.output_format);
    add_string(metadata.output_root_directory);
    add_string(metadata.output_run_label);
    add_string(metadata.runtime_ic_scratch_mode);
    add_string(metadata.runtime_scratch_directory);
    add_fftw_planning_record_storage(
        bytes,
        metadata.fftw_planning_records,
        "RunMetadata dynamic FFTW planning payload");
    bytes = checked_metadata_add(
        bytes,
        checked_metadata_multiply(
            static_cast<std::uint64_t>(
                metadata.snapshot_scale_factors.size()),
            sizeof(core::Real),
            "RunMetadata snapshot target vector"),
        "RunMetadata dynamic snapshot targets");
    if (bytes > MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES) {
        throw std::length_error(
            "RunMetadata dynamic payload exceeds its enforced bound");
    }
    return bytes;
}

std::string lineage_to_string(ProductLineage lineage) {
    switch (lineage) {
        case ProductLineage::DirectSimulation: return "direct_simulation";
        case ProductLineage::DerivedAnalysis: return "derived_analysis";
        case ProductLineage::Unknown: break;
    }
    return "unknown";
}

RunMetadata RunMetadata::from_config(
    const config::SimulationParameters& config,
    ProductLineage lineage_value,
    const std::string& verified_snapshot_ic_sha256,
    bool planning_records_complete) {
    if (lineage_value == ProductLineage::Unknown) {
        throw std::invalid_argument(
            "RunMetadata requires an explicit product lineage");
    }

    std::vector<mesh::FftwPlanningRecord> fftw_planning_records =
        snapshot_fftw_planning_records_bounded();

    // Reject the complete caller-owned dynamic source before RunMetadata copies
    // any configuration vector or string. The final materialized-object count
    // below also covers literals and derived provenance fields.
    require_config_metadata_source_within_bound(
        config,
        verified_snapshot_ic_sha256,
        fftw_planning_records);

    RunMetadata meta;
    meta.lineage = lineage_value;
    if (lineage_value == ProductLineage::DirectSimulation) {
        const auto parent_manifest_sha256 =
            process_restart_parent_manifest_sha256();
        meta.resumed_from_restart = parent_manifest_sha256.has_value();
        if (parent_manifest_sha256.has_value()) {
            // Keep the persisted JSON key name for snapshot-format continuity;
            // the value is the verified direct parent checkpoint manifest.
            meta.admitted_parent_restart_sha256 = *parent_manifest_sha256;
            meta.restart_parent_identity_scope =
                "restart_checkpoint_manifest";
            meta.restart_parent_identity_unavailable_reason.clear();
        } else {
            meta.restart_parent_identity_scope = "unavailable";
            meta.restart_parent_identity_unavailable_reason =
                "not_resumed_from_restart";
        }
    } else {
        meta.resumed_from_restart = false;
        meta.admitted_parent_restart_sha256.reset();
        meta.restart_parent_identity_scope =
            "inherited_via_input_snapshot_run_metadata";
        meta.restart_parent_identity_unavailable_reason =
            "refer_to_input_snapshot_run_metadata";
    }

    const auto& c = config.get_cosmology();
    meta.h = c.h;
    meta.omega_m = c.omega_m;
    meta.omega_lambda = c.omega_lambda;
    meta.omega_b = c.omega_b;
    meta.sigma8 = c.sigma8;
    meta.n_s = c.n_s;

    const auto& b = config.get_box();
    meta.box_size_Mpc_h = b.L;
    meta.particles_per_dimension = b.N;
    meta.pm_mesh_per_dimension = b.N_mesh;
    meta.mean_spacing_Mpc_h = config.d_mean();
    meta.particle_mass = config.particle_mass();
    meta.r_s = config.r_s();
    meta.r_cut = config.r_cut();
    meta.k_nyq = config.k_Nyq();

    const auto& g = config.get_gravity();
    meta.solver = g.solver;
    meta.softening_Mpc_h = config.eps();
    meta.theta = g.theta;
    meta.split_scale_cells = g.split_scale_cells;
    meta.deconvolve_cic = g.deconvolve_cic;

    const auto& ic = config.get_ic();
    const std::string effective_snapshot_sha256 =
        resolve_verified_snapshot_ic_sha256(
            config, verified_snapshot_ic_sha256);
    meta.ic_mode = ic.mode;
    meta.snapshot_file = ic.snapshot_file;
    if (ic.mode == "generate") {
        meta.ic_provenance_status =
            ic.power_spectrum_sha256.empty()
            ? "internal_generation_parameters_applied_input_hash_unavailable"
            : "internal_generation_parameters_and_input_hash_applied";
        meta.ic_generation_parameters_applied = true;
        meta.ic_mesh_per_dimension = config.ic_mesh_per_dimension();
        meta.ic_max_mode_per_axis = ic.max_mode_per_axis;
        meta.ic_effective_max_mode_per_axis = config.ic_effective_max_mode_per_axis();
        meta.seed = ic.seed;
        meta.lpt_order = ic.lpt_order;
        meta.power_spectrum_file = ic.power_spectrum_file;
        meta.power_spectrum_sha256 = ic.power_spectrum_sha256;
        meta.power_spectrum_redshift = ic.power_spectrum_redshift;
        meta.power_spectrum_fidelity = ic.power_spectrum_fidelity;
        meta.snapshot_sha256.clear();
        meta.ic_amplitude_mode = ic.amplitude_mode;
        meta.ic_phase_pairing = ic.phase_pairing;
        meta.ic_lattice_convention = "corner";
    } else if (ic.mode == "snapshot") {
        meta.ic_generation_parameters_applied = false;
        const std::string& source_run_metadata_json =
            config.verified_snapshot_ic_run_metadata_json();
        const SnapshotGenerationProvenance inherited =
            source_run_metadata_json.empty()
            ? SnapshotGenerationProvenance{}
            : read_snapshot_generation_provenance(
                source_run_metadata_json);
        if (inherited.available) {
            meta.ic_provenance_status =
                "external_snapshot_bytes_verified_upstream_generation_provenance_inherited";
            meta.ic_mesh_per_dimension =
                inherited.ic_mesh_per_dimension;
            meta.ic_max_mode_per_axis = inherited.ic_max_mode_per_axis;
            meta.ic_effective_max_mode_per_axis = inherited.ic_effective_max_mode_per_axis;
            meta.seed = inherited.seed;
            meta.lpt_order = inherited.lpt_order;
            meta.power_spectrum_sha256 =
                inherited.power_spectrum_sha256;
            meta.power_spectrum_redshift =
                inherited.power_spectrum_redshift;
            meta.power_spectrum_fidelity =
                inherited.power_spectrum_fidelity;
            meta.ic_amplitude_mode = inherited.ic_amplitude_mode;
            meta.ic_phase_pairing = inherited.ic_phase_pairing;
            meta.ic_lattice_convention =
                inherited.ic_lattice_convention;
        } else {
            meta.ic_provenance_status = effective_snapshot_sha256.empty()
                ? "external_snapshot_source_provenance_not_verified_input_hash_unavailable"
                : "external_snapshot_bytes_verified_upstream_generation_not_verified";
            meta.ic_mesh_per_dimension = 0;
            meta.seed = 0;
            meta.ic_amplitude_mode.clear();
            meta.ic_phase_pairing.clear();
            meta.lpt_order = 0;
            meta.power_spectrum_sha256.clear();
            meta.power_spectrum_redshift = 0.0;
            meta.power_spectrum_fidelity.clear();
            meta.ic_lattice_convention.clear();
        }
        meta.power_spectrum_file.clear();
        meta.snapshot_sha256 = effective_snapshot_sha256;
    } else {
        throw std::logic_error("RunMetadata encountered unsupported IC mode");
    }
    meta.treepm_short_range_composition =
        "plummer_total_minus_newtonian_long";

    const auto& t = config.get_time();
    meta.z_start = t.z_start;
    meta.z_final = t.z_final;
    meta.delta_ln_a = t.delta_ln_a;
    meta.step_policy = t.step_policy;

    const auto& output = config.get_output();
    meta.snapshot_scale_factors = output.snapshot_scale_factors;
    meta.output_format = output.formats.empty()
        ? "" : output.formats.front();
    meta.restart_cadence_steps = output.restart_cadence_steps;
    meta.output_root_directory = output.root_directory;
    meta.output_run_label = output.run_label;
    meta.timestamped_run_directory = output.timestamped_run_directory;
    meta.snapshot_batch_particles = output.snapshot_batch_particles;

    const auto& configured_runtime = config.get_runtime();
    meta.runtime_num_threads = configured_runtime.num_threads;
    meta.runtime_mpi_enabled = configured_runtime.mpi_enabled;

    const auto& memory_policy = config.get_memory_policy();
    meta.runtime_ic_scratch_mode = std::string(
        config::scratch_mode_name(memory_policy.ic_scratch_mode));
    meta.runtime_scratch_directory = memory_policy.scratch_directory;
    meta.fftw_planning_records = std::move(fftw_planning_records);
    meta.fftw_planning_records_complete = planning_records_complete;

    apply_build_capabilities(meta);

    const auto& validation = config.get_validation();
    meta.validation_write_diagnostics = validation.write_diagnostics;

    (void)run_metadata_dynamic_payload_bytes(meta);

    return meta;
}

RunMetadata RunMetadata::from_snapshot_analysis(
    const SnapshotDescriptor& descriptor,
    const std::string& snapshot_file,
    const analysis::AnalysisResourcePolicy& resource_policy,
    std::size_t effective_threads,
    std::size_t automatic_thread_capacity,
    bool thread_count_automatic,
    std::string_view thread_selection_reason) {
    if (snapshot_file.empty()) {
        throw std::invalid_argument(
            "Snapshot-derived analysis metadata requires the input snapshot path");
    }
    if (!std::isfinite(descriptor.box_size_Mpc_h)
        || descriptor.box_size_Mpc_h <= 0.0
        || descriptor.particles_per_dimension == 0
        || descriptor.pm_mesh_per_dimension == 0
        || !std::isfinite(descriptor.hubble_param)
        || descriptor.hubble_param <= 0.0
        || !std::isfinite(descriptor.omega_m)
        || descriptor.omega_m <= 0.0
        || !std::isfinite(descriptor.omega_lambda)
        || descriptor.omega_lambda < 0.0
        || !std::isfinite(descriptor.omega_b)
        || descriptor.omega_b < 0.0
        || descriptor.omega_b > descriptor.omega_m
        || (descriptor.lpt_order != 0
            && descriptor.lpt_order != 1
            && descriptor.lpt_order != 2)
        || (descriptor.lpt_order != 0
            && descriptor.ic_mesh_per_dimension == 0)
        || descriptor.physics_fingerprint.empty()
        || effective_threads == 0
        || automatic_thread_capacity == 0
        || thread_selection_reason.empty()) {
        throw std::invalid_argument(
            "Snapshot-derived analysis metadata received an incomplete descriptor or execution context");
    }
    if (thread_count_automatic) {
        if (resource_policy.requested_threads != 0
            || effective_threads != automatic_thread_capacity) {
            throw std::invalid_argument(
                "Automatic analysis thread context is internally inconsistent");
        }
    } else if (resource_policy.requested_threads != effective_threads) {
        throw std::invalid_argument(
            "Explicit analysis thread context is internally inconsistent");
    }

    std::vector<mesh::FftwPlanningRecord> fftw_planning_records =
        snapshot_fftw_planning_records_bounded();
    require_snapshot_analysis_metadata_source_within_bound(
        descriptor,
        snapshot_file,
        resource_policy,
        fftw_planning_records);

    RunMetadata meta;
    meta.lineage = ProductLineage::DerivedAnalysis;
    meta.restart_parent_identity_scope =
        "inherited_via_input_snapshot_run_metadata";
    meta.restart_parent_identity_unavailable_reason =
        "refer_to_input_snapshot_run_metadata";
    meta.h = descriptor.hubble_param;
    meta.omega_m = descriptor.omega_m;
    meta.omega_lambda = descriptor.omega_lambda;
    meta.omega_b = descriptor.omega_b;
    meta.sigma8 = descriptor.sigma8;
    meta.n_s = descriptor.spectral_index_ns;
    meta.box_size_Mpc_h = descriptor.box_size_Mpc_h;
    meta.particles_per_dimension = descriptor.particles_per_dimension;
    meta.pm_mesh_per_dimension = descriptor.pm_mesh_per_dimension;
    meta.ic_mesh_per_dimension = descriptor.lpt_order == 0
        ? 0 : descriptor.ic_mesh_per_dimension;
    meta.ic_max_mode_per_axis = descriptor.generation_provenance.ic_max_mode_per_axis;
    meta.ic_effective_max_mode_per_axis =
        descriptor.generation_provenance.ic_effective_max_mode_per_axis;
    meta.mean_spacing_Mpc_h = descriptor.box_size_Mpc_h
        / static_cast<core::Real>(descriptor.particles_per_dimension);
    meta.k_nyq = std::numbers::pi_v<core::Real>
        / meta.mean_spacing_Mpc_h;
    meta.softening_Mpc_h = descriptor.softening_comoving_Mpc_h;
    meta.ic_mode = "snapshot";
    meta.ic_provenance_status =
        "input_snapshot_descriptor_applied_upstream_run_metadata_not_republished";
    meta.ic_generation_parameters_applied = false;
    meta.seed = descriptor.ic_seed;
    meta.lpt_order = descriptor.lpt_order;
    meta.power_spectrum_fidelity = descriptor.power_spectrum_fidelity;
    meta.snapshot_file = snapshot_file;
    meta.ic_amplitude_mode = descriptor.ic_amplitude_mode;
    meta.ic_phase_pairing = descriptor.ic_phase_pairing;

    meta.runtime_num_threads = checked_u64(
        resource_policy.requested_threads, "Requested analysis thread count");
    meta.runtime_mpi_enabled = false;

    const auto& memory_policy = resource_policy.memory;
    meta.runtime_ic_scratch_mode = std::string(
        config::scratch_mode_name(memory_policy.ic_scratch_mode));
    meta.runtime_scratch_directory = memory_policy.scratch_directory;
    meta.fftw_planning_records = std::move(fftw_planning_records);
    meta.fftw_planning_records_complete = true;

    apply_build_capabilities(meta);
    (void)run_metadata_dynamic_payload_bytes(meta);
    return meta;
}

void RunMetadata::append_json(std::ostream& out) const {
    if (lineage == ProductLineage::DerivedAnalysis && ic_mode == "snapshot") {
        append_snapshot_derived_analysis_json(out, *this);
        return;
    }

    out << "{";
    append_common_prefix(out, *this);
    append_json_bool(out, "resumed_from_restart", resumed_from_restart);
    append_json_bool(
        out,
        "restart_parent_identity_available",
        admitted_parent_restart_sha256.has_value());
    append_json_string(
        out, "restart_parent_identity_scope", restart_parent_identity_scope);
    if (restart_parent_identity_unavailable_reason.empty()) {
        append_json_null(out, "restart_parent_identity_unavailable_reason");
    } else {
        append_json_string(
            out,
            "restart_parent_identity_unavailable_reason",
            restart_parent_identity_unavailable_reason);
    }
    if (admitted_parent_restart_sha256.has_value()) {
        append_json_string(
            out,
            "admitted_parent_restart_sha256",
            *admitted_parent_restart_sha256);
    } else {
        append_json_null(out, "admitted_parent_restart_sha256");
    }
    append_json_string(out, "ic_mode", ic_mode);
    append_json_string(out, "ic_provenance_status", ic_provenance_status);
    append_json_bool(
        out, "ic_generation_parameters_applied", ic_generation_parameters_applied);
    append_json_number(out, "seed", seed);
    append_json_number(out, "lpt_order", lpt_order);
    append_json_string(out, "power_spectrum_file", power_spectrum_file);
    append_json_string(out, "power_spectrum_sha256", power_spectrum_sha256);
    append_json_number(out, "power_spectrum_redshift", power_spectrum_redshift);
    append_json_string(out, "power_spectrum_fidelity", power_spectrum_fidelity);
    append_json_string(out, "snapshot_file", snapshot_file);
    append_json_string(out, "snapshot_sha256", snapshot_sha256);
    append_json_string(out, "ic_amplitude_mode", ic_amplitude_mode);
    append_json_string(out, "ic_phase_pairing", ic_phase_pairing);
    append_json_string(out, "ic_lattice_convention", ic_lattice_convention);
    append_json_bool(out, "build_openmp_enabled", build_openmp_enabled);
    append_json_bool(
        out, "build_fftw_threads_enabled", build_fftw_threads_enabled);
    append_json_bool(out, "build_mpi_enabled", build_mpi_enabled);
    append_json_bool(out, "build_fftw_mpi_enabled", build_fftw_mpi_enabled);
    append_json_bool(
        out, "build_parallel_hdf5_enabled", build_parallel_hdf5_enabled);
    append_common_suffix(out, *this);
}

std::string RunMetadata::to_json() const {
    return to_json_bounded(MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES);
}

std::string RunMetadata::to_json_bounded(std::size_t maximum_bytes) const {
    if (maximum_bytes > MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES) {
        throw std::length_error(
            "RunMetadata JSON requested bound exceeds the shared maximum");
    }
    const std::uint64_t metadata_bytes =
        run_metadata_dynamic_payload_bytes(*this);
    std::string json = detail::render_bounded_text(
        maximum_bytes,
        [this](std::ostream& out) { append_json(out); });
    (void)run_metadata_json_memory_plan(metadata_bytes, json.size());
    return json;
}

} // namespace io
} // namespace cosmo_nbody
