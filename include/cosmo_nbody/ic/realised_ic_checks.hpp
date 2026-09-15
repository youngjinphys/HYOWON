#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cstddef>
#include <span>

namespace cosmo_nbody {
namespace ic {

// Deterministic generated-IC structural checks reject invalid geometry/numerics;
// statistical/spectral behavior is recorded separately without universal thresholds.
struct RealisedICSummary {
    std::size_t particle_count{0};
    core::Real displacement_rms_Mpc_h{0.0};
    core::Real displacement_max_Mpc_h{0.0};
    core::Real momentum_rms{0.0};
    core::Real momentum_max{0.0};

    // Finite forward-neighbour cell-edge orientation diagnostic. This is a
    // discrete lattice observable, not det(I + dPsi/dq), a global injectivity
    // test, or a shell-crossing criterion. `evaluated_count` and
    // `unevaluable_count` are kept separately so one ambiguous periodic edge
    // cannot erase valid observations made elsewhere.
    bool forward_cell_edge_determinant_requested{false};
    std::size_t forward_cell_edge_determinant_evaluated_count{0};
    std::size_t forward_cell_edge_determinant_unevaluable_count{0};
    core::Real forward_cell_edge_determinant_min{0.0};
    core::Real forward_cell_edge_determinant_max{0.0};
    std::size_t forward_cell_edge_determinant_nonpositive_count{0};
};

RealisedICSummary validate_realised_lattice_ic(
    std::size_t particles_per_dimension,
    core::Real box_size_Mpc_h,
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> mom_x,
    std::span<const core::Real> mom_y,
    std::span<const core::Real> mom_z);

} // namespace ic
} // namespace cosmo_nbody
