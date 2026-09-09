// FFTW-MPI slab backend. Real storage uses FFTW's padded last dimension;
// caller-owned real/Fourier allocations are reused by the solver.
#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/mesh/mpi_fft_allocation_layout.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cosmo_nbody {
namespace gravity {
class DistributedPMSolver;
}
namespace mesh {

// Shared FFTW-aligned real allocation. The solver uses its leading compact view;
// transforms expand/compact that data in place to/from FFTW's padded layout.
class MpiFFTRealStorage {
public:
    explicit MpiFFTRealStorage(
        const MpiFFTAllocationLayout& layout);
    ~MpiFFTRealStorage();

    MpiFFTRealStorage(const MpiFFTRealStorage&) = delete;
    MpiFFTRealStorage& operator=(const MpiFFTRealStorage&) = delete;
    MpiFFTRealStorage(MpiFFTRealStorage&&) = delete;
    MpiFFTRealStorage& operator=(MpiFFTRealStorage&&) = delete;

    const MpiFFTAllocationLayout& allocation_layout() const noexcept {
        return layout_;
    }

private:
    friend class MpiFFTBackend;
    friend class gravity::DistributedPMSolver;

    std::span<core::Real> compact_real() noexcept {
        return {data_, compact_elements_};
    }
    std::span<const core::Real> compact_real() const noexcept {
        return {data_, compact_elements_};
    }

    // Admit destructive shared forward only after CIC has overwritten compact data.
    void declare_compact_overwrite_complete() noexcept {
        compact_ready_ = true;
    }

    std::span<core::Real> padded_real() noexcept {
        return {data_, padded_elements_};
    }

    MpiFFTAllocationLayout layout_{};
    std::size_t compact_elements_{0};
    std::size_t padded_elements_{0};
    core::Real* data_{nullptr};
    bool compact_ready_{false};
};

class MpiFFTBackend {
public:
    // Collective construction retains plans for new-array execution while reusing
    // caller-owned FFTW-compatible complex planning storage.
    MpiFFTBackend(
        std::size_t grid_size,
        MpiFFTRealStorage& real_storage,
        std::span<std::complex<core::Real>> planning_complex_storage);
    ~MpiFFTBackend();

    MpiFFTBackend(const MpiFFTBackend&) = delete;
    MpiFFTBackend& operator=(const MpiFFTBackend&) = delete;
    MpiFFTBackend(MpiFFTBackend&&) = delete;
    MpiFFTBackend& operator=(MpiFFTBackend&&) = delete;

    const MpiFFTAllocationLayout& allocation_layout() const noexcept {
        return layout_;
    }
    std::size_t grid_size() const noexcept { return layout_.grid_size; }
    std::size_t local_n0() const noexcept { return layout_.local_n0; }
    std::size_t local_0_start() const noexcept {
        return layout_.local_0_start;
    }
    std::size_t local_real_size() const {
        return layout_.compact_real_elements();
    }
    std::size_t local_complex_size() const noexcept {
        return layout_.alloc_local_complex_elements;
    }
    // Bytes in the shared padded real-transform allocation.
    std::uint64_t backend_persistent_bytes() const {
        return layout_.padded_real_bytes();
    }

    // Collective unnormalized forward transform from compact real data to the
    // FFTW-MPI complex slab; size/alignment mismatch rejects all ranks first.
    void forward(
        std::span<const core::Real> local_real,
        std::span<std::complex<core::Real>> local_complex);

    // Destructive forward is restricted to the shared allocation's compact view.
    void forward_shared_destructive(
        std::span<core::Real> local_real,
        std::span<std::complex<core::Real>> local_complex);

    // Preserve ordinary Fourier input with bounded scratch; registered destructive
    // storage may be consumed directly when its modes are dead.
    void inverse(
        std::span<const std::complex<core::Real>> local_complex,
        std::span<core::Real> local_real);

private:
    void require_complex_alignment(
        const std::complex<core::Real>* complex_data) const;
    bool overlaps_real_storage(
        std::span<const core::Real> local_real) const noexcept;
    void copy_real_for_forward(
        std::span<const core::Real> local_real);
    void pack_shared_real_for_forward();
    void unpack_normalized_real(std::span<core::Real> local_real) const;

    MpiFFTAllocationLayout layout_{};
    core::Real norm_factor_{1.0};
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    int complex_alignment_{-1};
#endif

    core::Real* real_buffer_{nullptr};
    MpiFFTRealStorage* real_storage_{nullptr};
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    void* forward_plan_{nullptr};
    void* inverse_plan_{nullptr};
#endif
};

} // namespace mesh
} // namespace cosmo_nbody
