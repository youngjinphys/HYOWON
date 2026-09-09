#include "cosmo_nbody/mesh/mpi_fft_allocation_layout.hpp"

#include "cosmo_nbody/mesh/mpi_fftw_runtime.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <complex>
#include <exception>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef COSMO_NBODY_HAS_FFTW_MPI
#include <fftw3-mpi.h>
#include <mpi.h>
#endif

namespace cosmo_nbody::mesh {
namespace {

std::size_t checked_mul_size(
    std::size_t lhs,
    std::size_t rhs,
    const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_mul_bytes(
    std::size_t elements,
    std::size_t element_bytes,
    const char* label) {
    const auto max = std::numeric_limits<std::uint64_t>::max();
    if (elements != 0
        && static_cast<std::uint64_t>(element_bytes)
            > max / static_cast<std::uint64_t>(elements)) {
        throw std::overflow_error(std::string(label) + " overflows uint64_t");
    }
    return static_cast<std::uint64_t>(elements)
        * static_cast<std::uint64_t>(element_bytes);
}

#ifdef COSMO_NBODY_HAS_FFTW_MPI
bool collective_any_failure(int local_failure, const char* stage) {
    int any_failure = 0;
    if (MPI_Allreduce(
            &local_failure,
            &any_failure,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("Failed to synchronize FFTW-MPI ") + stage);
    }
    return any_failure != 0;
}

std::size_t agreed_grid_size(std::size_t grid_size) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (collective_any_failure(
                grid_size > static_cast<std::size_t>(
                    std::numeric_limits<std::uint64_t>::max()),
                "grid-size representation")) {
            throw std::overflow_error(
                "FFTW-MPI grid size exceeds uint64_t on at least one rank");
        }
    }
    const std::uint64_t local = static_cast<std::uint64_t>(grid_size);
    std::uint64_t minimum = 0;
    std::uint64_t maximum = 0;
    const int minimum_status = MPI_Allreduce(
        &local, &minimum, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    const int maximum_status = MPI_Allreduce(
        &local, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    if (minimum_status != MPI_SUCCESS || maximum_status != MPI_SUCCESS) {
        throw std::runtime_error("Failed to synchronize FFTW-MPI grid size");
    }
    if (minimum != maximum) {
        throw std::invalid_argument(
            "FFTW-MPI grid size differs across MPI ranks");
    }
    if (minimum < 2) {
        throw std::invalid_argument("FFTW-MPI grid size must be at least two");
    }
    if (minimum > static_cast<std::uint64_t>(
            std::numeric_limits<ptrdiff_t>::max())) {
        throw std::overflow_error("FFTW-MPI grid size exceeds ptrdiff_t");
    }
    return static_cast<std::size_t>(minimum);
}
#endif

} // namespace

std::size_t MpiFFTAllocationLayout::compact_real_elements() const {
    return checked_mul_size(
        checked_mul_size(local_n0, grid_size, "compact real plane extent"),
        grid_size,
        "compact real extent");
}

std::size_t MpiFFTAllocationLayout::required_complex_elements() const {
    return checked_mul_size(
        checked_mul_size(local_n0, grid_size, "complex plane extent"),
        grid_size / 2 + 1,
        "complex slab extent");
}

std::size_t MpiFFTAllocationLayout::padded_real_elements() const {
    return checked_mul_size(
        2,
        alloc_local_complex_elements,
        "padded real extent");
}

std::uint64_t MpiFFTAllocationLayout::compact_real_bytes() const {
    return checked_mul_bytes(
        compact_real_elements(), sizeof(core::Real), "compact real bytes");
}

std::uint64_t MpiFFTAllocationLayout::padded_real_bytes() const {
    return checked_mul_bytes(
        padded_real_elements(), sizeof(core::Real), "padded real bytes");
}

std::uint64_t MpiFFTAllocationLayout::complex_bytes() const {
    return checked_mul_bytes(
        alloc_local_complex_elements,
        sizeof(std::complex<core::Real>),
        "complex bytes");
}

std::uint64_t MpiFFTAllocationLayout::transform_workspace_bytes() const {
    const std::uint64_t real = padded_real_bytes();
    const std::uint64_t complex = complex_bytes();
    if (complex > std::numeric_limits<std::uint64_t>::max() - real) {
        throw std::overflow_error(
            "FFTW-MPI transform workspace bytes overflow uint64_t");
    }
    return real + complex;
}

MpiFFTAllocationLayout make_mpi_fft_allocation_layout(
    std::size_t grid_size,
    int rank,
    int communicator_size,
    std::size_t local_n0,
    std::size_t local_0_start,
    std::size_t alloc_local_complex_elements) {
    if (grid_size < 2) {
        throw std::invalid_argument("FFTW-MPI grid size must be at least two");
    }
    if (communicator_size < 1 || rank < 0 || rank >= communicator_size) {
        throw std::invalid_argument("FFTW-MPI rank topology is invalid");
    }
    if (local_n0 == 0) {
        throw std::invalid_argument(
            "FFTW-MPI empty local slabs are unsupported");
    }
    if (local_0_start >= grid_size
        || local_n0 > grid_size - local_0_start) {
        throw std::invalid_argument("FFTW-MPI local slab is out of range");
    }

    MpiFFTAllocationLayout result;
    result.grid_size = grid_size;
    result.rank = rank;
    result.communicator_size = communicator_size;
    result.local_n0 = local_n0;
    result.local_0_start = local_0_start;
    result.alloc_local_complex_elements = alloc_local_complex_elements;
    result.padded_last_dim = checked_mul_size(
        2, grid_size / 2 + 1, "padded last dimension");

    if (result.alloc_local_complex_elements
            < result.required_complex_elements()
        || result.padded_real_elements()
            < result.compact_real_elements()) {
        throw std::invalid_argument(
            "FFTW-MPI allocation extent is smaller than its visible slab");
    }
    (void)result.transform_workspace_bytes();
    return result;
}

const MpiFFTAllocationLayout& MpiFFTCommunicatorLayout::rank_layout(
    int rank) const {
    if (rank < 0 || static_cast<std::size_t>(rank) >= ranks.size()) {
        throw std::out_of_range("FFTW-MPI rank is outside communicator layout");
    }
    return ranks[static_cast<std::size_t>(rank)];
}

int MpiFFTCommunicatorLayout::owner_rank(std::size_t global_plane) const {
    if (global_plane >= grid_size) {
        throw std::out_of_range("Global mesh plane is outside FFTW-MPI layout");
    }
    const auto it = std::upper_bound(
        ranks.begin(), ranks.end(), global_plane,
        [](std::size_t plane, const MpiFFTAllocationLayout& layout) {
            return plane < layout.local_0_start;
        });
    const auto candidate = it == ranks.begin() ? ranks.begin() : std::prev(it);
    if (candidate == ranks.end()
        || global_plane < candidate->local_0_start
        || global_plane >= candidate->local_0_start + candidate->local_n0) {
        throw std::logic_error("No FFTW-MPI owner for global mesh plane");
    }
    return candidate->rank;
}

std::size_t MpiFFTCommunicatorLayout::total_planes() const {
    std::size_t total = 0;
    for (const auto& layout : ranks) {
        if (layout.local_n0 > std::numeric_limits<std::size_t>::max() - total) {
            throw std::overflow_error("FFTW-MPI total slab extent overflows size_t");
        }
        total += layout.local_n0;
    }
    return total;
}

std::size_t MpiFFTCommunicatorLayout::minimum_local_n0() const {
    if (ranks.empty()) throw std::logic_error("FFTW-MPI communicator is empty");
    return std::min_element(
        ranks.begin(), ranks.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.local_n0 < rhs.local_n0;
        })->local_n0;
}

MpiFFTCommunicatorLayout make_mpi_fft_communicator_layout(
    std::size_t grid_size,
    std::vector<MpiFFTAllocationLayout> ranks) {
    if (grid_size < 2 || ranks.empty()) {
        throw std::invalid_argument("FFTW-MPI communicator layout is empty");
    }
    std::sort(
        ranks.begin(), ranks.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.rank < rhs.rank; });
    if (ranks.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error("FFTW-MPI communicator size exceeds int");
    }
    const int communicator_size = static_cast<int>(ranks.size());
    std::size_t expected_start = 0;
    for (int rank = 0; rank < communicator_size; ++rank) {
        const auto& layout = ranks[static_cast<std::size_t>(rank)];
        if (layout.grid_size != grid_size
            || layout.rank != rank
            || layout.communicator_size != communicator_size
            || layout.local_0_start != expected_start) {
            throw std::invalid_argument(
                "FFTW-MPI communicator slabs are inconsistent or non-contiguous");
        }
        if (layout.local_n0 > grid_size - expected_start) {
            throw std::invalid_argument("FFTW-MPI communicator slab exceeds grid");
        }
        expected_start += layout.local_n0;
    }
    if (expected_start != grid_size) {
        throw std::invalid_argument(
            "FFTW-MPI communicator slabs do not cover the global grid");
    }
    return {grid_size, std::move(ranks)};
}

MpiFFTAllocationLayout query_mpi_fft_allocation_layout(
    std::size_t grid_size) {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)grid_size;
    throw std::runtime_error(
        "Exact FFT allocation layout requires FFTW-MPI support");
#else
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized) {
        throw std::runtime_error(
            "Exact FFT allocation layout requires initialized MPI");
    }
    require_mpi_main_thread("FFTW-MPI allocation layout query");
    const std::size_t agreed = agreed_grid_size(grid_size);
    int rank = 0;
    int communicator_size = 0;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS
        || MPI_Comm_size(MPI_COMM_WORLD, &communicator_size) != MPI_SUCCESS) {
        throw std::runtime_error("Failed to query FFTW-MPI communicator topology");
    }

    const auto planner_session = configure_mpi_fftw_planner();
    ptrdiff_t local_n0 = 0;
    ptrdiff_t local_0_start = 0;
    const ptrdiff_t alloc_local = fftw_mpi_local_size_3d(
        static_cast<ptrdiff_t>(agreed),
        static_cast<ptrdiff_t>(agreed),
        static_cast<ptrdiff_t>(agreed / 2 + 1),
        MPI_COMM_WORLD,
        &local_n0,
        &local_0_start);
    const int invalid = alloc_local < 0 || local_n0 <= 0 || local_0_start < 0;
    if (collective_any_failure(invalid, "allocation-layout query")) {
        throw std::invalid_argument(
            "FFTW-MPI returned an invalid or empty local allocation layout");
    }
    std::optional<MpiFFTAllocationLayout> local_layout;
    try {
        local_layout = make_mpi_fft_allocation_layout(
            agreed,
            rank,
            communicator_size,
            static_cast<std::size_t>(local_n0),
            static_cast<std::size_t>(local_0_start),
            static_cast<std::size_t>(alloc_local));
    } catch (const std::exception&) {
        // A rank-local extent/overflow failure must not let peers advance to
        // the following Allgather while this rank unwinds independently.
    }
    if (collective_any_failure(
            !local_layout.has_value(), "allocation-layout validation")) {
        throw std::invalid_argument(
            "FFTW-MPI allocation layout validation failed on at least one rank");
    }
    return *local_layout;
#endif
}

MpiFFTCommunicatorLayout query_mpi_fft_communicator_layout(
    std::size_t grid_size) {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)grid_size;
    throw std::runtime_error(
        "Exact FFT communicator layout requires FFTW-MPI support");
#else
    const MpiFFTAllocationLayout local =
        query_mpi_fft_allocation_layout(grid_size);
    std::uint64_t local_values[3] = {
        static_cast<std::uint64_t>(local.local_n0),
        static_cast<std::uint64_t>(local.local_0_start),
        static_cast<std::uint64_t>(local.alloc_local_complex_elements)};
    std::vector<std::uint64_t> gathered;
    std::exception_ptr gather_storage_exception;
    try {
        gathered.resize(
            static_cast<std::size_t>(local.communicator_size) * 3);
    } catch (...) {
        gather_storage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        gather_storage_exception,
        local.communicator_size,
        "FFTW-MPI communicator gather storage allocation");
    if (MPI_Allgather(
            local_values,
            3,
            MPI_UINT64_T,
            gathered.data(),
            3,
            MPI_UINT64_T,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("Failed to gather FFTW-MPI allocation layouts");
    }

    std::optional<MpiFFTCommunicatorLayout> result;
    std::exception_ptr construction_exception;
    try {
        std::vector<MpiFFTAllocationLayout> layouts;
        layouts.reserve(static_cast<std::size_t>(local.communicator_size));
        for (int rank = 0; rank < local.communicator_size; ++rank) {
            const std::size_t offset = static_cast<std::size_t>(rank) * 3;
            const auto checked_size = [](std::uint64_t value, const char* label) {
                if (value > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
                    throw std::overflow_error(
                        std::string(label) + " exceeds size_t");
                }
                return static_cast<std::size_t>(value);
            };
            layouts.push_back(make_mpi_fft_allocation_layout(
                local.grid_size,
                rank,
                local.communicator_size,
                checked_size(gathered[offset], "local_n0"),
                checked_size(gathered[offset + 1], "local_0_start"),
                checked_size(gathered[offset + 2], "alloc_local")));
        }
        result = make_mpi_fft_communicator_layout(
            local.grid_size, std::move(layouts));
    } catch (...) {
        construction_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        construction_exception,
        local.communicator_size,
        "FFTW-MPI communicator layout construction");
    return std::move(*result);
#endif
}

} // namespace cosmo_nbody::mesh
