// Conservation/reversibility measurements, not scientific pass/fail gates.
// PM uses its unfiltered mesh operator for W; TreePM adds the exact shifted
// Plummer-total-minus-Gaussian-long short-pair correction on the canonical force
// tree. The short-energy traversal is exact, so the Layzer-Irvine residual remains
// sensitive to TreePM opening-angle force error.
//
// CIC-gathered centered-grid force is generally not the particle-coordinate
// derivative of the CIC mesh energy, even after self subtraction. Therefore a
// nonzero Layzer-Irvine floor need not vanish under timestep refinement; force,
// timestep, mesh, softening, split/cutoff, and resolution effects must remain distinct.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/gravity/octree.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/mass_assignment.hpp"
#include "cosmo_nbody/mesh/green_function.hpp"
#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/real_scratch_buffer.hpp"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace cosmo_nbody {
namespace validation {

struct ConservationState {
    core::Real a_prev{0.0};
    core::Real integrated_source{0.0};
    core::Real integrated_source_compensation{0.0};
    core::Real initial_E{0.0};
    core::Real source_prev{0.0};
    core::Real last_kinetic{0.0};
    core::Real last_potential{0.0};
    core::Real last_energy{0.0};
    core::Real last_source{0.0};
    core::Real last_residual{0.0};
};

class ConservationChecks {
public:
    ConservationChecks(const config::SimulationParameters& config);

    core::Real compute_kinetic_energy(
        const core::ParticleStore& particles,
        core::Real current_a) const;

    // PM: unfiltered mean-subtracted CIC mesh energy. TreePM adds exact shifted
    // short-pair energy from the canonical tree at the same force state.
    core::Real compute_potential_energy(
        const core::ParticleStore& particles,
        core::Real current_a,
        const gravity::Octree* treepm_force_tree = nullptr) const;

    // Exact local-tree short-pair energy; stable IDs count each global unordered
    // pair once, requiring current r_cut ghosts under MPI.
    core::Real compute_treepm_short_potential_energy(
        const core::ParticleStore& particles,
        const gravity::Octree& treepm_force_tree,
        core::Real current_a) const;

    // CIC cloud self-interaction through the same mesh operator. This position-
    // dependent term can dominate coarse-resolution raw W and is subtracted from
    // Layzer-Irvine tracking; its configured 27-lag impulse kernel is cached.
    core::Real compute_cic_self_energy(
        const core::ParticleStore& particles,
        core::Real current_a) const;

    // Prepare only the cached 27-lag self kernel, borrowing an existing potential
    // workspace or releasing a temporary one. Lazy diagnostic caches are not concurrent.
    void prepare_cic_self_kernel() const;

    static void reset_layzer_irvine_state(
        core::Real current_a,
        core::Real current_K,
        core::Real current_W,
        ConservationState& state);
    static core::Real update_layzer_irvine_state(
        core::Real current_a,
        core::Real current_K,
        core::Real current_W,
        ConservationState& state);

    core::Real evaluate_layzer_irvine(
        const core::ParticleStore& particles,
        core::Real current_a,
        core::Real current_W,
        ConservationState& state) const;

    void reset_state(
        const core::ParticleStore& particles,
        core::Real current_a,
        core::Real current_W,
        ConservationState& state) const;

private:
    struct PotentialWorkspace {
        mesh::MeshGeometry geometry;
        mesh::FFTBackend fft;
        mesh::CICMassAssignment assignment;
        mesh::GreenFunction green;
        // Owners precede field views so reverse destruction releases views first.
        std::unique_ptr<runtime::RealScratchBuffer> real_backing;
        std::unique_ptr<runtime::RawScratchBuffer> complex_backing;
        mesh::RealField field;
        mesh::ComplexField modes;

        explicit PotentialWorkspace(
            const config::SimulationParameters& config);
    };

    PotentialWorkspace& potential_workspace() const;

    config::SimulationParameters config_;
    mutable std::unique_ptr<PotentialWorkspace> potential_workspace_;
    mutable std::vector<core::Real> phi_at_particles_;
    mutable bool self_kernel_ready_{false};
    // If unit-density 1/dx^3 is unrepresentable, cache the mass-space impulse
    // response (dx^3 times the normal kernel) and divide only after combining m^2.
    mutable bool self_kernel_scaled_by_cell_volume_{false};
    mutable std::array<core::Real, 27> self_kernel_{};
};

} // namespace validation
} // namespace cosmo_nbody
