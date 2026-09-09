// Single-rank binary64 FFTW3 r2c/c2r backend. Forward is unnormalized; inverse
// applies 1/N^3. Planning is serialized, thread-aware, provider-keyed, and records
// observed FFTW workspace; external arrays reuse plans only with matching alignment.
#pragma once

#include "cosmo_nbody/config/memory_policy.hpp"
#include "cosmo_nbody/mesh/fftw_planning_record.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"
#include <fftw3.h>

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace cosmo_nbody {
namespace mesh {

class FFTBackend {
public:
    // Scratch mode changes planner-array placement only, not rigor or identity.
    explicit FFTBackend(
        const MeshGeometry& geom,
        config::ScratchMode planning_scratch_mode = config::ScratchMode::Memory,
        const std::string& scratch_directory = {});
    ~FFTBackend();

    FFTBackend(const FFTBackend&) = delete;
    FFTBackend& operator=(const FFTBackend&) = delete;
    FFTBackend(FFTBackend&&) = delete;
    FFTBackend& operator=(FFTBackend&&) = delete;

    std::size_t grid_size() const noexcept { return N_; }

    void forward(RealField& in, ComplexField& out);
    void inverse(ComplexField& in, RealField& out);

    // Transform directly into caller-owned complex storage; matching alignment
    // reuses the regular plan, otherwise the unaligned fallback is used.
    void forward_external_output(
        RealField& in,
        std::complex<core::Real>* out,
        std::size_t out_size);

    // External contiguous buffers (for example mmap-backed IC scratch) must match
    // the compact logical N^3 extent; alignment selects regular/fallback plans.
    void forward_external(
        core::Real* in,
        std::size_t in_size,
        ComplexField& out);
    void inverse_external(
        ComplexField& in,
        core::Real* out,
        std::size_t out_size);

private:
    void normalize_inverse_buffer(core::Real* out, std::size_t out_size) const;

    MeshGeometry geom_;
    std::size_t N_;
    std::size_t real_size_;
    std::size_t complex_size_;
    core::Real norm_factor_;
    int real_alignment_{-1};
    int complex_alignment_{-1};

    fftw_plan forward_plan_;
    fftw_plan inverse_plan_;
    fftw_plan forward_unaligned_plan_;
    fftw_plan inverse_unaligned_plan_;
};

// Linked FFTW planner implementation identity.
std::string fftw_build_identity_sha256();

// Stable current-path content identity for resolved FFTW providers; not a proof
// of the already-loaded image after a pathname is replaced.
std::string fftw_provider_path_content_sha256();
std::string fftw_runtime_system_identity_sha256();

// Shared provenance channel for completed planning transactions, including MPI.
void append_fftw_planning_record(FftwPlanningRecord record);

// Reserve under the planner mutex before MPI failure agreement so append cannot
// allocate on only a subset of ranks.
void reserve_fftw_planning_record_slot();
void append_reserved_fftw_planning_record(
    FftwPlanningRecord record) noexcept;

// Process-order history across all distinct IC/evolution/analysis plans.
std::vector<FftwPlanningRecord> fftw_planning_records_snapshot();

} // namespace mesh
} // namespace cosmo_nbody
