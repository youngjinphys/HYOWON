#pragma once

#include "cosmo_nbody/analysis/bispectrum.hpp"
#include "cosmo_nbody/analysis/halo_mass_function.hpp"
#include "cosmo_nbody/analysis/matter_power_spectrum.hpp"
#include "cosmo_nbody/analysis/two_point_correlation.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/halo/pair_links.hpp"

#include <filesystem>
#include <cstddef>
#include <iosfwd>
#include <string_view>
#include <vector>

namespace cosmo_nbody::app::nbody_analyze {

void write_particle_tracer_csv_metadata(std::ostream& output);

void write_single_snapshot_csv_metadata(
    std::ostream& output,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a);

void write_particle_tracer_json_metadata(std::ostream& output);

void write_pk_csv(
    const std::filesystem::path& path,
    const std::vector<analysis::PowerSpectrumBin>& bins,
    int mesh_per_dimension,
    core::Real box_size_Mpc_h,
    const analysis::PowerSpectrumOptions& options,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a);

void write_hmf_csv(
    const std::filesystem::path& path,
    const analysis::HaloMassFunctionResult& hmf,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles);

void write_xi_csv(
    const std::filesystem::path& path,
    const analysis::TwoPointResult& xi,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles);

void write_bispectrum_csv(
    const std::filesystem::path& path,
    const std::vector<analysis::BispectrumBin>& bins,
    int analysis_mesh,
    core::Real box_size_Mpc_h,
    const analysis::BispectrumOptions& options,
    const std::filesystem::path& source_snapshot,
    std::string_view source_native_snapshot_object_sha256,
    core::Real snapshot_a);

void write_pair_links_json(
    const std::filesystem::path& path,
    const halo::PairLinkSet& pair_links,
    const std::filesystem::path& earlier_snapshot,
    std::string_view earlier_native_snapshot_object_sha256,
    core::Real earlier_scale_factor,
    const std::filesystem::path& later_snapshot,
    std::string_view later_native_snapshot_object_sha256,
    core::Real later_scale_factor,
    core::Real fof_linking_length_b,
    std::size_t fof_min_particles);

} // namespace cosmo_nbody::app::nbody_analyze
