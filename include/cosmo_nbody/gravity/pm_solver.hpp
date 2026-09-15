#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/gravity/distributed_pm_solver.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/green_function.hpp"
#include "cosmo_nbody/mesh/mass_assignment.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/pm_force_method.hpp"
#include "cosmo_nbody/runtime/real_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace cosmo_nbody {
namespace runtime {
class EvolutionPMSolverHandle;
}
namespace gravity {

class TreePMSolver;

struct PMForceDiagnostics {
    // Sum_i 0.5 m_i phi_i before cosmological 1/a conversion.
    core::Real potential_energy_comoving{0.0};
    // Distributed backend can evaluate the CIC self term in slab layout.
    std::optional<core::Real> cic_self_energy_comoving;
};

// Measurement from the real-space pure-PM potential produced by the immediately
// preceding force refresh. Directional quantities use caller-supplied endpoint
// canonical momenta and are derivatives of the reported CIC particle-position
// energies, not derivatives of the centered-grid gathered force operator.
struct PMPostForceEnergyDiagnostics {
    core::Real potential_energy_comoving{0.0};
    core::Real directional_potential_energy_comoving{0.0};
    std::optional<core::Real> cic_self_energy_comoving;
    std::optional<core::Real> directional_cic_self_energy_comoving;
};

class PMSolver {
public:
    // PMForceMethod binds the Green function, gradient, split, and CIC choices.
    // mpi_global_reduce makes force entrypoints collective over MPI_COMM_WORLD;
    // distributed FFT workspace destruction is likewise collective before finalize.
    PMSolver(
        const mesh::MeshGeometry& geom,
        mesh::PMForceMethod method,
        bool mpi_global_reduce,
        config::MemoryPolicyParams memory_policy);

    // Standalone path assembles and validates private force storage before adding
    // it to caller-visible acceleration.
    std::optional<PMForceDiagnostics> compute_forces(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<core::Real> acc_x,
        std::span<core::Real> acc_y,
        std::span<core::Real> acc_z,
        bool collect_potential_energy = false);

    bool using_distributed_backend() const noexcept {
        return static_cast<bool>(distributed_solver_);
    }
    bool workspace_initialized() const noexcept {
        return workspace_initialized_;
    }

private:
    friend class TreePMSolver;
    friend class runtime::EvolutionPMSolverHandle;

    // Internal zero-destination path avoids a second 3*N force allocation.
    std::optional<PMForceDiagnostics> compute_forces_in_place(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<core::Real> force_x,
        std::span<core::Real> force_y,
        std::span<core::Real> force_z,
        bool collect_potential_energy,
        mesh::CICDepositWorkspace* deposition_workspace = nullptr);

    // Consume the current pure-PM real-space potential immediately after its
    // force refresh. This path performs scalar reductions only and does not
    // materialize a particle-sized energy-gradient vector.
    PMPostForceEnergyDiagnostics measure_post_force_energy_diagnostics_in_place(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<const core::Real> momentum_x,
        std::span<const core::Real> momentum_y,
        std::span<const core::Real> momentum_z);

    int mpi_force_stage_size(const char* context) const;
    void synchronize_mpi_force_stage(
        std::exception_ptr local_exception,
        int mpi_size,
        const char* context) const;

    const mesh::MeshGeometry& geom_;
    mesh::PMForceMethod method_;
    bool mpi_global_reduce_{false};
    config::MemoryPolicyParams memory_policy_;
    bool workspace_initialized_{false};

    std::unique_ptr<DistributedPMSolver> distributed_solver_;
    std::unique_ptr<mesh::FFTBackend> fft_;
    std::unique_ptr<mesh::CICMassAssignment> mass_assign_;
    std::unique_ptr<mesh::GreenFunction> green_fn_;
    // Scratch owners precede non-owning views so reverse destruction is safe.
    std::unique_ptr<runtime::RealScratchBuffer> real_scratch_1_;
    std::unique_ptr<runtime::RealScratchBuffer> real_scratch_2_;
    std::unique_ptr<runtime::RawScratchBuffer> complex_scratch_;
    std::unique_ptr<runtime::RawScratchBuffer> phi_scratch_;
    std::unique_ptr<mesh::RealField> real_buf_1_;
    std::unique_ptr<mesh::RealField> real_buf_2_;
    std::unique_ptr<mesh::ComplexField> complex_buf_;
    std::unique_ptr<mesh::ComplexField> phi_k_buf_;

    void ensure_workspace(std::size_t particle_count);
};

} // namespace gravity
} // namespace cosmo_nbody
