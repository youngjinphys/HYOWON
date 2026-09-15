#include "cosmo_nbody/io/snapshot_physics.hpp"
#include "bounded_text_serialization.hpp"
#include "cosmo_nbody/io/runtime_provenance.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cosmo_nbody {
namespace io {

namespace {

std::uint64_t checked_snapshot_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* context) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string(context) + " overflows uint64");
    }
    return lhs + rhs;
}

std::uint64_t snapshot_string_storage(
    std::uint64_t logical_bytes,
    std::uint64_t maximum_bytes,
    const char* context) {
    if (logical_bytes > maximum_bytes) {
        throw std::length_error(
            std::string(context) + " exceeds its enforced bound");
    }
    return checked_snapshot_add(logical_bytes, 1U, context);
}

} // namespace

SnapshotMetadataMemoryPlan snapshot_metadata_memory_plan(
    std::uint64_t metadata_dynamic_payload_bytes,
    std::uint64_t metadata_json_logical_bytes,
    std::uint64_t physics_fingerprint_logical_bytes) {
    const RunMetadataJsonMemoryPlan metadata_plan =
        run_metadata_json_memory_plan(
            metadata_dynamic_payload_bytes,
            metadata_json_logical_bytes);
    const std::uint64_t physics_storage = snapshot_string_storage(
        physics_fingerprint_logical_bytes,
        MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES,
        "Snapshot physics fingerprint");
    const std::uint64_t retained_attributes = checked_snapshot_add(
        metadata_plan.json_storage_bytes,
        physics_storage,
        "Snapshot retained metadata attributes");
    const std::uint64_t rank_reference = std::max(
        metadata_json_logical_bytes,
        physics_fingerprint_logical_bytes);
    const std::uint64_t rank_agreement = checked_snapshot_add(
        retained_attributes,
        rank_reference,
        "Parallel snapshot rank agreement");
    const std::uint64_t reader_attribute_peak = std::max({
        metadata_plan.json_storage_bytes,
        physics_storage});
    return SnapshotMetadataMemoryPlan{
        metadata_dynamic_payload_bytes,
        metadata_plan.json_storage_bytes,
        physics_storage,
        metadata_plan.serialization_peak_bytes,
        retained_attributes,
        reader_attribute_peak,
        std::max(
            metadata_plan.serialization_peak_bytes,
            retained_attributes),
        rank_reference,
        rank_agreement,
        std::max(
            metadata_plan.serialization_peak_bytes,
            rank_agreement)};
}

std::string snapshot_physics_fingerprint(
    const config::SimulationParameters& config,
    const std::string& verified_snapshot_ic_sha256) {
    return detail::render_bounded_text(
        MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES,
        [&](std::ostream& out) {
            out << "snapshot_physics\n";

            auto append_string = [&](const char* key, const std::string& value) {
                out << key << '=' << value.size() << ':' << value << '\n';
            };
            auto append_real = [&](const char* key, core::Real value) {
                out << key << '=' << value << '\n';
            };
            auto append_uint = [&](const char* key, std::uint64_t value) {
                out << key << '=' << value << '\n';
            };
            auto append_bool = [&](const char* key, bool value) {
                out << key << '=' << (value ? 1 : 0) << '\n';
            };

            const auto& cosmology = config.get_cosmology();
            append_real("h", cosmology.h);
            append_real("omega_m", cosmology.omega_m);
            append_real("omega_lambda", cosmology.omega_lambda);
            append_real("omega_b", cosmology.omega_b);
            append_real("sigma8", cosmology.sigma8);
            append_real("n_s", cosmology.n_s);

            const auto& box = config.get_box();
            append_real("box_L", box.L);
            append_uint("box_N", box.N);
            append_uint("box_N_mesh", box.N_mesh);

            const auto& gravity = config.get_gravity();
            append_string("gravity_solver", gravity.solver);
            append_bool("gravity_deconvolve_cic", gravity.deconvolve_cic);
            if (gravity.solver == "TreePM") {
                append_real("gravity_eps", config.eps());
                append_real("gravity_theta", gravity.theta);
                append_real("gravity_split_scale_cells", gravity.split_scale_cells);
                append_real("gravity_r_s", config.r_s());
                append_real("gravity_r_cut", config.r_cut());
            } else {
                // Pure PM force resolution is defined by the solver, mesh, and CIC policy.
                append_string(
                    "gravity_force_resolution_model",
                    "pm_mesh_cic_discrete_operator");
            }

            const auto& time = config.get_time();
            append_real("time_z_start", time.z_start);
            append_real("time_z_final", time.z_final);
            append_real("time_delta_ln_a", time.delta_ln_a);
            append_string("time_step_policy", time.step_policy);

            const auto& ic = config.get_ic();
            append_string("ic_mode", ic.mode);
            append_uint("ic_lpt_order", static_cast<std::uint64_t>(ic.lpt_order));
            append_uint("ic_seed", ic.seed);
            append_string("ic_power_spectrum_sha256", ic.power_spectrum_sha256);
            append_real("ic_power_spectrum_redshift", ic.power_spectrum_redshift);
            append_string("ic_power_spectrum_fidelity", ic.power_spectrum_fidelity);
            append_string(
                "ic_snapshot_sha256",
                resolve_verified_snapshot_ic_sha256(
                    config, verified_snapshot_ic_sha256));
            append_uint("ic_mesh_per_dimension", config.ic_mesh_per_dimension());
            // Unspecified support is not serialized; an explicit window changes IC identity.
            if (ic.max_mode_per_axis.has_value()) {
                append_uint("ic_max_mode_per_axis", *ic.max_mode_per_axis);
            }
            append_string("ic_amplitude_mode", ic.amplitude_mode);
            append_string("ic_phase_pairing", ic.phase_pairing);

            const auto& output = config.get_output();
            append_uint(
                "snapshot_target_count",
                static_cast<std::uint64_t>(output.snapshot_scale_factors.size()));
            append_string("snapshot_target_encoding", "indexed_key");
            for (std::size_t index = 0;
                 index < output.snapshot_scale_factors.size();
                 ++index) {
                out << "snapshot_target_" << index << '='
                    << output.snapshot_scale_factors[index] << '\n';
            }

            const auto& runtime = config.get_runtime();
            append_bool("runtime_mpi_enabled", runtime.mpi_enabled);
        });
}

} // namespace io
} // namespace cosmo_nbody
