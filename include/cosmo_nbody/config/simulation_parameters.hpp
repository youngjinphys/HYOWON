#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/explicit_boolean.hpp"
#include "cosmo_nbody/core/explicit_value.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace config {

// Physical coordinates default invalid so omission cannot silently select a run.
struct CosmologyParams {
    core::Real h{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real omega_m{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real omega_lambda{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real omega_b{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real sigma8{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real n_s{std::numeric_limits<core::Real>::quiet_NaN()};
};

struct BoxParams {
    core::Real L{std::numeric_limits<core::Real>::quiet_NaN()};
    std::uint64_t N{0};
    std::uint64_t N_mesh{0};
};

enum class SolverKind { PM, TreePM };

struct GravityParams {
    // Sentinels prevent hidden force-method choices; active coordinates are explicit.
    std::string solver{""};
    core::Real eps{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real theta{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real split_scale_cells{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real cutoff_multiplier{std::numeric_limits<core::Real>::quiet_NaN()};
    core::ExplicitBoolean deconvolve_cic{};
};

struct TimeParams {
    core::Real z_start{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real z_final{std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real delta_ln_a{std::numeric_limits<core::Real>::quiet_NaN()};
    // Sole implemented stepping policy; an algorithm invariant, not a default choice.
    std::string step_policy{"global"};
};

struct ICParams {
    std::string mode{""};
    int lpt_order{0};

    // Explicit wrappers distinguish valid zero values from omitted coordinates.
    core::ExplicitValue<std::uint64_t> seed{};
    core::ExplicitValue<std::uint64_t> mesh_per_dimension{};
    // Optional inclusive Cartesian support for first-order density and final LPT source.
    std::optional<std::uint64_t> max_mode_per_axis;

    std::string power_spectrum_file{""};
    core::Real power_spectrum_redshift{
        std::numeric_limits<core::Real>::quiet_NaN()};
    std::string power_spectrum_fidelity{""};
    std::string snapshot_file{""};
    // Empty accepts the opened object's digest; non-empty requires exact identity.
    std::string expected_snapshot_sha256{""};
    std::string amplitude_mode{""};
    std::string phase_pairing{""};

    // Derived immutable input identities; not public TOML coordinates.
    std::string power_spectrum_sha256;
    std::string snapshot_sha256;
};

struct OutputParams {
    std::vector<core::Real> snapshot_scale_factors;
    // Sole implemented writer; an implementation invariant.
    std::vector<std::string> formats{"hdf5"};
    std::uint64_t restart_cadence_steps{0};
    std::string root_directory{"runs"};
    std::string run_label{""};
    bool timestamped_run_directory{true};
    // I/O batching only; does not alter numerical state.
    std::uint64_t snapshot_batch_particles{1ULL << 20};
};

struct ValidationParams {
    bool write_diagnostics{false};
};

// Execution topology only; scratch backing is configured independently.
struct RuntimeParams {
    // Zero selects automatic host-thread policy; positive values are exact overrides.
    std::uint64_t num_threads{0};
    bool mpi_enabled{false};
};

class SimulationParameters {
public:
    SimulationParameters(CosmologyParams c,
                         BoxParams b,
                         GravityParams g,
                         TimeParams t,
                         ICParams i,
                         OutputParams o,
                         RuntimeParams r = {},
                         ValidationParams v = {},
                         MemoryPolicyParams m = {});

    const CosmologyParams& get_cosmology() const noexcept { return cosmo; }
    const BoxParams& get_box() const noexcept { return box; }
    const GravityParams& get_gravity() const noexcept { return gravity; }
    const TimeParams& get_time() const noexcept { return time; }
    const ICParams& get_ic() const noexcept { return ic; }
    const OutputParams& get_output() const noexcept { return output; }
    const ValidationParams& get_validation() const noexcept { return validation; }
    const RuntimeParams& get_runtime() const noexcept { return runtime; }
    const MemoryPolicyParams& get_memory_policy() const noexcept {
        return memory_policy;
    }

    SolverKind solver_kind() const noexcept { return derived_solver_kind; }
    core::Real d_mean() const noexcept { return derived_d_mean; }
    core::Real particle_mass() const noexcept { return derived_m_p; }
    core::Real r_s() const noexcept { return derived_r_s; }
    core::Real r_cut() const noexcept { return derived_r_cut; }
    core::Real k_Nyq() const noexcept { return derived_k_Nyq; }
    core::Real eps() const noexcept { return derived_eps; }
    std::uint64_t ic_mesh_per_dimension() const noexcept { return derived_ic_mesh; }
    std::uint64_t ic_effective_max_mode_per_axis() const {
        if (ic.mode != "generate" || box.N == 0) {
            throw std::logic_error("IC Fourier support requires a generated-IC configuration");
        }
        return ic.max_mode_per_axis.value_or((box.N - 1) / 2);
    }
    std::uint64_t num_particles() const noexcept { return derived_num_particles; }

    // Resolved generated-IC seed; snapshot mode exposes canonical inactive zero.
    std::uint64_t ic_seed() const noexcept { return derived_ic_seed; }

    // Run-local provenance populated only after exact snapshot ingestion.
    const std::string& verified_snapshot_ic_sha256() const noexcept {
        return runtime_provenance->verified_snapshot_ic_sha256;
    }

    // Exact admitted parent RunMetadataJson retained with the verified snapshot digest.
    const std::string& verified_snapshot_ic_run_metadata_json() const noexcept {
        return runtime_provenance->verified_snapshot_ic_run_metadata_json;
    }

    void set_verified_snapshot_ic_sha256(std::string value) const {
        if (!value.empty()) {
            if (ic.mode != "snapshot") {
                throw std::invalid_argument(
                    "Generated IC configuration cannot carry external snapshot provenance");
            }
            if (value.size() != 64) {
                throw std::invalid_argument(
                    "Verified snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
            }
            for (const char c : value) {
                const bool decimal = c >= '0' && c <= '9';
                const bool lowercase_hex = c >= 'a' && c <= 'f';
                if (!decimal && !lowercase_hex) {
                    throw std::invalid_argument(
                        "Verified snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
                }
            }
            if (!ic.expected_snapshot_sha256.empty()
                && ic.expected_snapshot_sha256 != value) {
                throw std::invalid_argument(
                    "Verified snapshot IC SHA-256 disagrees with the requested input identity");
            }
            if (!ic.snapshot_sha256.empty() && ic.snapshot_sha256 != value) {
                throw std::invalid_argument(
                    "Verified snapshot IC SHA-256 disagrees with immutable configuration identity");
            }
            if (!runtime_provenance->verified_snapshot_ic_sha256.empty()
                && runtime_provenance->verified_snapshot_ic_sha256 != value) {
                throw std::logic_error(
                    "Run-local snapshot IC provenance cannot change after initialization");
            }
        } else if (!runtime_provenance->verified_snapshot_ic_sha256.empty()) {
            throw std::logic_error(
                "Run-local snapshot IC provenance cannot be cleared after initialization");
        }
        runtime_provenance->verified_snapshot_ic_sha256 = std::move(value);
    }

    void set_verified_snapshot_ic_provenance(
        std::string sha256,
        std::string run_metadata_json) const {
        if (run_metadata_json.empty()) {
            throw std::invalid_argument(
                "Verified snapshot IC provenance requires non-empty source RunMetadataJson");
        }
        if (ic.mode != "snapshot") {
            throw std::invalid_argument(
                "Generated IC configuration cannot carry external snapshot provenance");
        }
        if (sha256.size() != 64) {
            throw std::invalid_argument(
                "Verified snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
        }
        for (const char c : sha256) {
            const bool decimal = c >= '0' && c <= '9';
            const bool lowercase_hex = c >= 'a' && c <= 'f';
            if (!decimal && !lowercase_hex) {
                throw std::invalid_argument(
                    "Verified snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
            }
        }
        if ((!ic.expected_snapshot_sha256.empty()
             && ic.expected_snapshot_sha256 != sha256)
            || (!ic.snapshot_sha256.empty() && ic.snapshot_sha256 != sha256)) {
            throw std::invalid_argument(
                "Verified snapshot IC provenance disagrees with the immutable input identity");
        }
        if ((!runtime_provenance->verified_snapshot_ic_sha256.empty()
             && runtime_provenance->verified_snapshot_ic_sha256 != sha256)
            || (!runtime_provenance->verified_snapshot_ic_run_metadata_json.empty()
                && runtime_provenance->verified_snapshot_ic_run_metadata_json
                    != run_metadata_json)) {
            throw std::logic_error(
                "Run-local snapshot IC provenance cannot change after initialization");
        }

        // Copies are complete at call entry; publish both components with noexcept swaps.
        if (runtime_provenance->verified_snapshot_ic_sha256.empty()) {
            runtime_provenance->verified_snapshot_ic_sha256.swap(sha256);
        }
        if (runtime_provenance->verified_snapshot_ic_run_metadata_json.empty()) {
            runtime_provenance->verified_snapshot_ic_run_metadata_json.swap(
                run_metadata_json);
        }
    }

private:
    struct RuntimeProvenanceState {
        std::string verified_snapshot_ic_sha256;
        std::string verified_snapshot_ic_run_metadata_json;
    };

    CosmologyParams cosmo;
    BoxParams box;
    GravityParams gravity;
    TimeParams time;
    ICParams ic;
    OutputParams output;
    ValidationParams validation;
    RuntimeParams runtime;
    MemoryPolicyParams memory_policy;

    SolverKind derived_solver_kind{SolverKind::PM};
    core::Real derived_d_mean{0.0};
    core::Real derived_m_p{0.0};
    core::Real derived_r_s{0.0};
    core::Real derived_r_cut{0.0};
    core::Real derived_k_Nyq{0.0};
    core::Real derived_eps{0.0};
    std::uint64_t derived_ic_mesh{0};
    std::uint64_t derived_ic_seed{0};
    std::uint64_t derived_num_particles{0};
    std::shared_ptr<RuntimeProvenanceState> runtime_provenance{
        std::make_shared<RuntimeProvenanceState>()};

    static std::string ascii_lower(std::string value);
    static void require_finite_positive(core::Real value, const char* label);

    void derive_ic_input_hashes();
    void validate_cosmology();
    void resolve_and_validate_resolution();
    void validate_time() const;
    void validate_gravity();
    void validate_initial_conditions();
    void normalize_and_validate_output();
    void normalize_runtime();
    void derive_physical_scales();
    void validate_and_derive();
};

} // namespace config
} // namespace cosmo_nbody
