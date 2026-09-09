#pragma once

#include "cosmo_nbody/gravity/pm_solver.hpp"
#include "cosmo_nbody/mesh/mass_assignment.hpp"

#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::runtime {

// Thin runtime adapter around the canonical PM implementation. Backend
// selection belongs to PMSolver; this layer only owns the reusable evolution CIC
// deposition scratch used by the evolution path.
class EvolutionPMSolverHandle {
public:
    EvolutionPMSolverHandle() = default;
    explicit EvolutionPMSolverHandle(
        std::unique_ptr<gravity::PMSolver> solver) noexcept
        : solver_(std::move(solver)) {}

    EvolutionPMSolverHandle(const EvolutionPMSolverHandle&) = delete;
    EvolutionPMSolverHandle& operator=(const EvolutionPMSolverHandle&) = delete;
    EvolutionPMSolverHandle(EvolutionPMSolverHandle&&) noexcept = default;
    EvolutionPMSolverHandle& operator=(EvolutionPMSolverHandle&&) noexcept = default;

    EvolutionPMSolverHandle& operator=(
        std::unique_ptr<gravity::PMSolver> solver) noexcept {
        deposition_workspace_.release();
        solver_ = std::move(solver);
        return *this;
    }

    explicit operator bool() const noexcept {
        return static_cast<bool>(solver_);
    }

    EvolutionPMSolverHandle* operator->() noexcept { return this; }
    const EvolutionPMSolverHandle* operator->() const noexcept { return this; }

    std::optional<gravity::PMForceDiagnostics> compute_forces(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<core::Real> acc_x,
        std::span<core::Real> acc_y,
        std::span<core::Real> acc_z,
        bool collect_potential_energy = false) {
        if (!solver_) {
            throw std::logic_error(
                "EvolutionPMSolverHandle has no configured PM solver");
        }
        return solver_->compute_forces_in_place(
            pos_x,
            pos_y,
            pos_z,
            masses,
            uniform_mass,
            acc_x,
            acc_y,
            acc_z,
            collect_potential_energy,
            &deposition_workspace_);
    }

    void release_transient_deposition_workspace() noexcept {
        deposition_workspace_.release();
    }

private:
    std::unique_ptr<gravity::PMSolver> solver_;
    mesh::CICDepositWorkspace deposition_workspace_;
};

} // namespace cosmo_nbody::runtime
