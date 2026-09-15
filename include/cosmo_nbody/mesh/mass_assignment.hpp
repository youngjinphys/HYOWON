#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace cosmo_nbody {
namespace mesh {

std::uint64_t cic_stable_index_bytes(std::uint64_t particle_count);

// Shared runtime/admission estimate for one CIC deposit. Counts are requested
// payload only; allocator overhead remains a runtime high-water concern.
struct CICDepositMemoryPlan {
    std::uint64_t binning_thread_count{0};
    std::uint64_t requested_plane_worker_count{0};
    std::uint64_t active_plane_worker_count{0};
    std::uint64_t stable_index_bytes{0};
    std::uint64_t bin_offset_bytes{0};
    std::uint64_t bin_thread_metadata_bytes{0};
    std::uint64_t retained_bin_bytes{0};
    std::uint64_t task_particle_visit_bytes{0};
    std::uint64_t worker_boundary_bytes{0};
    std::uint64_t call_local_bytes{0};
    std::uint64_t peak_bytes{0};
};

CICDepositMemoryPlan cic_deposit_memory_plan(
    std::uint64_t particle_count,
    std::uint64_t mesh_size,
    std::uint64_t plane_tasks,
    std::uint64_t maximum_threads);

// Reusable deterministic-deposit storage for sequential force refreshes. A
// workspace must not be shared by concurrent deposits.
class CICDepositWorkspace {
public:
    CICDepositWorkspace();
    ~CICDepositWorkspace();

    CICDepositWorkspace(const CICDepositWorkspace&) = delete;
    CICDepositWorkspace& operator=(const CICDepositWorkspace&) = delete;
    CICDepositWorkspace(CICDepositWorkspace&&) noexcept;
    CICDepositWorkspace& operator=(CICDepositWorkspace&&) noexcept;

    void release() noexcept;
    std::size_t retained_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend class CICMassAssignment;
};

class CICMassAssignment {
public:
    explicit CICMassAssignment(
        const MeshGeometry& geom,
        config::MemoryPolicyParams memory_policy = {});

    // Interpret deposited masses in units of deposition_mass_unit without
    // changing the physical-mass default.
    CICMassAssignment(
        const MeshGeometry& geom,
        config::MemoryPolicyParams memory_policy,
        core::Real deposition_mass_unit);

    void deposit(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        RealField& field,
        RealField* right_ghost = nullptr) const;

    // Same operator and accumulation order as deposit(), with reusable caller-owned
    // particle-index/bin storage.
    void deposit_reusing_workspace(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        CICDepositWorkspace& workspace,
        RealField& field,
        RealField* right_ghost = nullptr) const;

    // Overwrite output with gathered field values.
    void interpolate(
        const RealField& field,
        const RealField* right_ghost,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<core::Real> out_values) const;

    // Add gathered values directly, avoiding an N-particle temporary.
    void interpolate_add(
        const RealField& field,
        const RealField* right_ghost,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<core::Real> out_values) const;

    // Gather with local/right-plane CIC arithmetic and reduce scale*m_i*f_i in
    // deterministic particle order without materializing gathered values.
    core::Accum interpolate_mass_weighted_sum(
        const RealField& field,
        const RealField* right_ghost,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        core::Accum scale = core::Accum{1.0}) const;

    // Differentiate the CIC-gathered scalar field with respect to particle
    // position and contract it with a caller-supplied particle direction. The
    // result is sum_i m_i direction_i . grad S_i[field], without a particle-sized
    // gradient buffer. On an exact CIC cell boundary the selected-cell one-sided
    // derivative is used; callers must preserve that interpretation in diagnostics.
    core::Accum interpolate_mass_weighted_directional_derivative(
        const RealField& field,
        const RealField* right_ghost,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        std::span<const core::Real> direction_x,
        std::span<const core::Real> direction_y,
        std::span<const core::Real> direction_z) const;

    // Full-mesh counterpart returning sum(m_i f_i) with team-size-independent
    // blocked reduction order.
    core::Accum interpolate_mass_weighted_sum_full_mesh(
        const RealField& field,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass) const;

    // Validate logical N^3 mesh values only; FFT padding is not physical data.
    void require_finite_full_mesh_field(const RealField& field) const;

    // Full-periodic-mesh fused gather of all three centered-gradient components,
    // algebraically matching three materialized gradient meshes. It is not used
    // for MPI slabs because one ghost plane cannot supply every required neighbor.
    void interpolate_centered_gradient3_add_full_mesh(
        const RealField& potential,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<core::Real> out_x,
        std::span<core::Real> out_y,
        std::span<core::Real> out_z) const;

private:
    const MeshGeometry& geom_;
    core::Real dx_;
    config::MemoryPolicyParams memory_policy_;
    core::Real deposition_mass_unit_{1.0};

    void deposit_impl(
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<const core::Real> masses,
        std::optional<core::Real> uniform_mass,
        CICDepositWorkspace* workspace,
        RealField& field,
        RealField* right_ghost) const;

    void interpolate_impl(
        const RealField& field,
        const RealField* right_ghost,
        std::span<const core::Real> pos_x,
        std::span<const core::Real> pos_y,
        std::span<const core::Real> pos_z,
        std::span<core::Real> out_values,
        bool add) const;
};

} // namespace mesh
} // namespace cosmo_nbody
