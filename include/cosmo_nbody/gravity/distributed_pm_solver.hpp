#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/destructive_complex_field.hpp"
#include "cosmo_nbody/mesh/green_function.hpp"
#include "cosmo_nbody/mesh/mass_assignment.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/mpi_fft_backend.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>

namespace cosmo_nbody {
namespace gravity {

class PMSolver;

struct DistributedPMForceDiagnostics {
    // Sum_i 0.5 m_i phi_i before cosmological 1/a conversion.
    core::Real potential_energy_comoving{0.0};
    // Same-operator CIC self interaction, also before 1/a conversion.
    core::Real cic_self_energy_comoving{0.0};
};

struct DistributedPMPostForceEnergyDiagnostics {
    core::Real potential_energy_comoving{0.0};
    core::Real directional_potential_energy_comoving{0.0};
    core::Real cic_self_energy_comoving{0.0};
    core::Real directional_cic_self_energy_comoving{0.0};
};

// Internal MPI backend; PMSolver alone resolves the named PMForceMethod and may
// construct this class, preventing unsupported lower-level operator hybrids.
class DistributedPMSolver {
public:
    std::optional<DistributedPMForceDiagnostics> compute_forces(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<core::Real> acc_x,
        std::span<core::Real> acc_y,
        std::span<core::Real> acc_z,
        bool collect_potential_energy = false,
        mesh::CICDepositWorkspace* deposition_workspace = nullptr);

    std::size_t local_n0() const noexcept { return geometry_.local_n0(); }
    std::size_t local_0_start() const noexcept {
        return geometry_.local_0_start();
    }

private:
    friend class PMSolver;
    struct LocalConstructionTag {};

    // Construction/destruction are collective over MPI_COMM_WORLD.
    static std::unique_ptr<DistributedPMSolver> create(
        core::Real box_size,
        std::size_t mesh_size,
        bool use_tree_pm,
        core::Real r_s,
        bool deconvolve_cic,
        bool use_exact_gradient);

    DistributedPMSolver(
        LocalConstructionTag,
        core::Real box_size,
        std::size_t mesh_size,
        bool use_tree_pm,
        core::Real r_s,
        bool deconvolve_cic,
        bool use_exact_gradient,
        const mesh::MpiFFTAllocationLayout& layout,
        int rank,
        int size);

    core::Real box_size_;
    std::size_t mesh_size_;
    bool use_exact_gradient_{false};
    int rank_{0};
    int size_{1};

    mesh::MeshGeometry geometry_;
    mesh::CICMassAssignment mass_assignment_;
    mesh::GreenFunction green_;

    // density_ views the compact prefix of FFTW-padded storage; fft_ is declared
    // last so its plans are released before the shared real allocation.
    mesh::MpiFFTRealStorage fft_real_storage_;
    mesh::RealField density_;
    mesh::RealField real_force_;
    mesh::DestructiveComplexField phi_modes_;
    std::unique_ptr<mesh::DestructiveComplexField> force_modes_;

    mesh::RealField exchange_plane_;
    mesh::RealField recv_left_plane_;
    mesh::RealField recv_right_plane_;

    std::optional<mesh::MpiFFTBackend> fft_;

    bool self_kernel_ready_{false};
    bool self_kernel_scaled_by_cell_volume_{false};
    std::array<core::Real, 27> self_kernel_{};

    core::Real exchange_deposit_planes();
    void exchange_field_planes(const mesh::RealField& field);
    void interpolate_force_field(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<core::Real> acceleration);
    void compute_axis_spectral(
        int axis,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<core::Real> acceleration);
    void ensure_self_kernel();
    core::Real compute_cic_self_energy_comoving(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass) const;
    core::Real compute_cic_self_energy_momentum_directional_comoving(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<const core::Real> momentum_x,
        std::span<const core::Real> momentum_y,
        std::span<const core::Real> momentum_z) const;
    DistributedPMForceDiagnostics collect_force_diagnostics(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass);
    DistributedPMPostForceEnergyDiagnostics
    measure_post_force_energy_diagnostics(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<const core::Real> momentum_x,
        std::span<const core::Real> momentum_y,
        std::span<const core::Real> momentum_z);
};

} // namespace gravity
} // namespace cosmo_nbody
