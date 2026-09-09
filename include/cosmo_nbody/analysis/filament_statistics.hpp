// T-web classifier on the CIC overdensity mesh, without CIC deconvolution or
// interlacing: T_ij(k;R)=(k_i k_j/k^2) delta(k) exp[-(kR)^2/2], k!=0.
// At even-grid Nyquist axes, mixed derivatives use the reflection-compatible
// real trigonometric convention (zero if either axis is Nyquist); diagonal
// derivatives retain their Nyquist contribution.
// The number of eigenvalues above lambda_threshold labels void/sheet/filament/node.
#pragma once

#include "cosmo_nbody/analysis/periodic_domain.hpp"
#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cosmo_nbody {
namespace analysis {

enum class WebEnvironment : int {
    Void = 0,
    Sheet = 1,
    Filament = 2,
    Node = 3
};

struct TidalWebOptions {
    int mesh_size{0};
    core::Real gaussian_smoothing_radius{
        std::numeric_limits<core::Real>::quiet_NaN()};
    core::Real lambda_threshold{
        std::numeric_limits<core::Real>::quiet_NaN()};
    // Storage choice only; summary callers may omit filament directions.
    bool materialize_filament_directions{true};
    config::MemoryPolicyParams memory_policy{};
};

struct TidalWebSummary {
    std::size_t void_cells{0};
    std::size_t sheet_cells{0};
    std::size_t filament_cells{0};
    std::size_t node_cells{0};
    core::Real void_volume_fraction{0.0};
    core::Real sheet_volume_fraction{0.0};
    core::Real filament_volume_fraction{0.0};
    core::Real node_volume_fraction{0.0};
    core::Real gaussian_smoothing_radius{0.0};
    core::Real lambda_threshold{0.0};
};

struct TidalWebField {
    int mesh_size{0};
    core::Real gaussian_smoothing_radius{0.0};
    core::Real lambda_threshold{0.0};
    std::vector<WebEnvironment> environment;
    std::vector<core::Vec3> filament_direction;
    std::vector<std::uint8_t> filament_direction_valid;
};

struct TidalWebCellSample {
    WebEnvironment environment{WebEnvironment::Void};
    core::Vec3 filament_direction{0.0, 0.0, 0.0};
    bool filament_direction_valid{false};
};

class FilamentStatistics {
public:
    static TidalWebSummary classify_tidal_web(
        const PeriodicDomain& domain,
        const core::ParticleStore& particles,
        const TidalWebOptions& options);

    static TidalWebField build_tidal_web_field(
        const PeriodicDomain& domain,
        const core::ParticleStore& particles,
        const TidalWebOptions& options);

    static TidalWebCellSample sample_cell(
        const TidalWebField& field,
        core::Real box_size,
        core::Vec3 position);
};

} // namespace analysis
} // namespace cosmo_nbody
