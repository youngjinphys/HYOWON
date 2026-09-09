#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/io/persisted_text_limits.hpp"
#include "cosmo_nbody/mesh/fftw_planning_record.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cosmo_nbody {
namespace analysis {
struct AnalysisResourcePolicy;
}
namespace io {

// Bounds logical dynamic metadata payload from the persisted JSON limit; allocator
// overhead/capacity and the fixed RunMetadata object are excluded.
inline constexpr std::size_t MAXIMUM_RUN_METADATA_DYNAMIC_PAYLOAD_BYTES =
    MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES * (sizeof(std::string) + 1U);

struct SnapshotDescriptor;

enum class ProductLineage {
    Unknown,
    DirectSimulation,
    DerivedAnalysis
};

struct RunMetadataJsonMemoryPlan {
    std::uint64_t metadata_dynamic_payload_bytes{0};
    std::uint64_t json_storage_bytes{0};
    std::uint64_t serialization_peak_bytes{0};
};

// Restart/snapshot writer byte model; string payload includes trailing NUL while
// caller-owned configuration, fixed stack frames, and HDF5 internals are excluded.
RunMetadataJsonMemoryPlan run_metadata_json_memory_plan(
    std::uint64_t metadata_dynamic_payload_bytes,
    std::uint64_t json_logical_bytes);

std::string lineage_to_string(ProductLineage lineage);

struct RunMetadata {
    ProductLineage lineage{ProductLineage::Unknown};

    bool resumed_from_restart{false};
    std::optional<std::string> admitted_parent_restart_sha256;
    std::string restart_parent_identity_scope{"unavailable"};
    std::string restart_parent_identity_unavailable_reason{
        "not_resumed_from_restart"};

    core::Real h{0.0};
    core::Real omega_m{0.0};
    core::Real omega_lambda{0.0};
    core::Real omega_b{0.0};
    core::Real sigma8{0.0};
    core::Real n_s{0.0};

    core::Real box_size_Mpc_h{0.0};
    std::uint64_t particles_per_dimension{0};
    std::uint64_t pm_mesh_per_dimension{0};
    std::uint64_t ic_mesh_per_dimension{0};
    // Requested K is null for the default particle window; effective K is null
    // when upstream support was not recorded.
    std::optional<std::uint64_t> ic_max_mode_per_axis;
    std::optional<std::uint64_t> ic_effective_max_mode_per_axis;
    core::Real mean_spacing_Mpc_h{0.0};
    core::Real particle_mass{0.0};
    core::Real r_s{0.0};
    core::Real r_cut{0.0};
    core::Real k_nyq{0.0};

    std::string solver;
    core::Real softening_Mpc_h{0.0};
    core::Real theta{0.0};
    core::Real split_scale_cells{0.0};
    bool deconvolve_cic{false};

    std::string ic_mode;
    std::string ic_provenance_status;
    bool ic_generation_parameters_applied{false};
    std::uint64_t seed{0};
    int lpt_order{0};
    std::string power_spectrum_file;
    std::string power_spectrum_sha256;
    core::Real power_spectrum_redshift{0.0};
    std::string power_spectrum_fidelity;
    std::string snapshot_file;
    std::string snapshot_sha256;

    // Semantic fields remain unavailable unless established by admitted config or
    // immutable input; defaults must not invent IC or TreePM semantics.
    std::string ic_amplitude_mode;
    std::string ic_phase_pairing;
    std::string ic_lattice_convention;
    std::string treepm_short_range_composition;

    core::Real z_start{0.0};
    core::Real z_final{0.0};
    core::Real delta_ln_a{0.0};
    std::string step_policy;

    std::vector<core::Real> snapshot_scale_factors;
    std::string output_format;
    std::uint64_t restart_cadence_steps{0};
    std::string output_root_directory;
    std::string output_run_label;
    bool timestamped_run_directory{true};
    std::uint64_t snapshot_batch_particles{0};

    // Runtime controls describe the process producing this metadata object;
    // analysis metadata does not invent unavailable trajectory controls.
    std::uint64_t runtime_num_threads{1};
    bool runtime_mpi_enabled{false};

    std::string runtime_ic_scratch_mode;
    std::string runtime_scratch_directory;

    // Process-order FFTW planning records are serialized only inside RunMetadataJson.
    std::vector<mesh::FftwPlanningRecord> fftw_planning_records;
    bool fftw_planning_records_complete{false};

    bool build_openmp_enabled{false};
    bool build_fftw_threads_enabled{false};
    bool build_mpi_enabled{false};
    bool build_fftw_mpi_enabled{false};
    bool build_parallel_hdf5_enabled{false};
    bool validation_write_diagnostics{false};

    static RunMetadata from_config(
        const config::SimulationParameters& config,
        ProductLineage lineage,
        const std::string& verified_snapshot_ic_sha256,
        bool fftw_planning_records_complete);

    // Republish descriptor-backed analysis provenance; fields absent from the
    // immutable parent descriptor remain unavailable.
    static RunMetadata from_snapshot_analysis(
        const SnapshotDescriptor& descriptor,
        const std::string& snapshot_file,
        const analysis::AnalysisResourcePolicy& resource_policy,
        std::size_t effective_threads,
        std::size_t automatic_thread_capacity,
        bool thread_count_automatic,
        std::string_view thread_selection_reason);

    std::string to_json() const;
    std::string to_json_bounded(std::size_t maximum_bytes) const;

private:
    void append_json(std::ostream& output) const;
};

// Exact logical dynamic payload: vector elements and string payloads including
// trailing NUL, excluding fixed members and allocator capacity/bookkeeping.
std::uint64_t run_metadata_dynamic_payload_bytes(
    const RunMetadata& metadata);

} // namespace io
} // namespace cosmo_nbody
