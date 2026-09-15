#include "cosmo_nbody/mesh/mpi_fft_backend.hpp"

#include "cosmo_nbody/mesh/destructive_complex_field.hpp"
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/fftw_runtime.hpp"
#include "cosmo_nbody/mesh/mpi_fftw_runtime.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/runtime/mpi_execution_identity.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_FFTW_MPI
#include <fftw3-mpi.h>
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace mesh {

namespace {

#ifdef COSMO_NBODY_HAS_FFTW_MPI
static_assert(std::is_same_v<core::Real, double>,
              "FFTW-MPI backend requires double-precision core::Real");
static_assert(sizeof(std::complex<core::Real>) == sizeof(fftw_complex),
              "std::complex layout must match FFTW interleaved complex storage");

bool collective_any_failure(int local_failed, const char* stage) {
    int any_failed = 0;
    if (MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MpiFFTBackend failed to synchronize ") + stage);
    }
    return any_failed != 0;
}

bool collective_agreed_flag(bool local_value, const char* stage) {
    const int local = local_value ? 1 : 0;
    int minimum = 0;
    int maximum = 0;
    const int minimum_status = MPI_Allreduce(
        &local,
        &minimum,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD);
    const int maximum_status = MPI_Allreduce(
        &local,
        &maximum,
        1,
        MPI_INT,
        MPI_MAX,
        MPI_COMM_WORLD);
    if (minimum_status != MPI_SUCCESS || maximum_status != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MpiFFTBackend failed to synchronize ") + stage);
    }
    if (minimum != maximum) {
        throw std::invalid_argument(
            std::string("MpiFFTBackend rank-asymmetric ") + stage);
    }
    return maximum != 0;
}

fftw_complex* fftw_data(std::complex<core::Real>* data) noexcept {
    return reinterpret_cast<fftw_complex*>(data);
}

std::string collective_rank_ordered_sha256(
    const std::string& local_sha256,
    int rank_count,
    const char* context) {
    std::vector<char> gathered;
    std::exception_ptr preparation_exception;
    try {
        if (!io::is_canonical_sha256(local_sha256) || rank_count < 1) {
            throw std::invalid_argument(
                std::string(context)
                + " requires one canonical local SHA-256 per rank");
        }
        if (static_cast<std::uintmax_t>(rank_count)
            > std::numeric_limits<std::size_t>::max()
                  / io::SHA256_HEX_CHARACTER_COUNT) {
            throw std::overflow_error(
                std::string(context) + " rank count exceeds local storage");
        }
        gathered.assign(
            static_cast<std::size_t>(rank_count)
                * io::SHA256_HEX_CHARACTER_COUNT,
            '\0');
    } catch (...) {
        preparation_exception = std::current_exception();
    }
    if (collective_any_failure(
            preparation_exception ? 1 : 0,
            "rank-ordered SHA-256 gather preparation")) {
        if (preparation_exception) {
            std::rethrow_exception(preparation_exception);
        }
        throw std::runtime_error(
            std::string(context) + " preparation failed on another rank");
    }
    if (MPI_Allgather(
            local_sha256.data(),
            static_cast<int>(io::SHA256_HEX_CHARACTER_COUNT),
            MPI_CHAR,
            gathered.data(),
            static_cast<int>(io::SHA256_HEX_CHARACTER_COUNT),
            MPI_CHAR,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            std::string("MpiFFTBackend failed to gather ") + context);
    }
    std::string digest;
    std::exception_ptr hashing_exception;
    try {
        digest = io::sha256_text(std::string_view(
            gathered.data(), gathered.size()));
    } catch (...) {
        hashing_exception = std::current_exception();
    }
    if (collective_any_failure(
            hashing_exception ? 1 : 0,
            "rank-ordered SHA-256 aggregation")) {
        if (hashing_exception) std::rethrow_exception(hashing_exception);
        throw std::runtime_error(
            std::string(context) + " hashing failed on another rank");
    }
    return digest;
}
#endif

} // namespace

MpiFFTRealStorage::MpiFFTRealStorage(
    const MpiFFTAllocationLayout& layout)
    : layout_(layout) {
    // Cache the validated sizes so the noexcept span accessors never repeat
    // overflow-checking arithmetic.
    compact_elements_ = layout_.compact_real_elements();
    padded_elements_ = layout_.padded_real_elements();
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    throw std::runtime_error(
        "MpiFFTRealStorage requires a build with FFTW-MPI support");
#else
    // Validate every derived extent before asking FFTW for storage.  In
    // particular, alloc_local may exceed the visible r2c slab because FFTW's
    // transpose workspace is part of the same allocation.
    data_ = fftw_alloc_real(padded_elements_);
    if (!data_) throw std::bad_alloc();
#endif
}

MpiFFTRealStorage::~MpiFFTRealStorage() {
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (data_) fftw_free(data_);
#endif
}

MpiFFTBackend::MpiFFTBackend(
    std::size_t grid_size,
    MpiFFTRealStorage& real_storage,
    std::span<std::complex<core::Real>> planning_complex_storage) {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)grid_size;
    (void)real_storage;
    (void)planning_complex_storage;
    throw std::runtime_error(
        "MpiFFTBackend requires a build with FFTW-MPI support");
#else
    layout_ = shared_mpi_fft_allocation_layout(grid_size);
    // The layout registry may be a cache hit, while FFTW's planner thread count
    // is process-global mutable state. Reassert the collective rank-local policy
    // and retain its shared lock through both distributed planner calls.
    const auto planner_session = configure_mpi_fftw_planner();
    // FFTW_ESTIMATE still consumes any process-global wisdom that happens to
    // be present. Clear it explicitly so distributed plan choice cannot depend
    // on an earlier serial backend or other hidden process state.
    fftw_forget_wisdom();

    const auto& supplied_real_layout = real_storage.allocation_layout();
    const int planning_storage_invalid =
        planning_complex_storage.size()
            != layout_.alloc_local_complex_elements
        || planning_complex_storage.data() == nullptr
        || supplied_real_layout.grid_size != layout_.grid_size
        || supplied_real_layout.rank != layout_.rank
        || supplied_real_layout.communicator_size
            != layout_.communicator_size
        || supplied_real_layout.local_n0 != layout_.local_n0
        || supplied_real_layout.local_0_start != layout_.local_0_start
        || supplied_real_layout.alloc_local_complex_elements
            != layout_.alloc_local_complex_elements
        || supplied_real_layout.padded_last_dim != layout_.padded_last_dim
        || real_storage.padded_real().size()
            != layout_.padded_real_elements()
        || real_storage.padded_real().data() == nullptr;
    if (collective_any_failure(
            planning_storage_invalid,
            "FFTW-MPI planning complex-storage validation")) {
        throw std::invalid_argument(
            "MpiFFTBackend planning complex storage does not match the allocation layout");
    }

    real_storage_ = &real_storage;
    real_buffer_ = real_storage.padded_real().data();

    const ptrdiff_t n = static_cast<ptrdiff_t>(layout_.grid_size);
    auto forward = fftw_mpi_plan_dft_r2c_3d(
        n,
        n,
        n,
        real_buffer_,
        fftw_data(planning_complex_storage.data()),
        MPI_COMM_WORLD,
        FFTW_ESTIMATE);
    bool any_forward_plan_failed = false;
    try {
        any_forward_plan_failed = collective_any_failure(
            forward == nullptr,
            "FFTW-MPI forward-plan creation");
    } catch (...) {
        // The synchronization itself failed, so no later FFTW-MPI collective is
        // safe here. A locally returned plan is intentionally leaked until
        // process teardown rather than conditionally destroying it on only a
        // subset of the communicator.
        throw;
    }
    if (any_forward_plan_failed) {
        // fftw_destroy_plan is collective for an MPI plan. If plan creation
        // returned nullptr on only some ranks, `if (forward) destroy` would
        // deadlock the ranks that did receive a plan. Leave any partial plan to
        // process teardown; this constructor cannot publish a usable backend.
        throw std::runtime_error(
            "Failed to create FFTW-MPI forward plan on at least one rank");
    }

    auto inverse = fftw_mpi_plan_dft_c2r_3d(
        n,
        n,
        n,
        fftw_data(planning_complex_storage.data()),
        real_buffer_,
        MPI_COMM_WORLD,
        FFTW_ESTIMATE);
    bool any_inverse_plan_failed = false;
    try {
        any_inverse_plan_failed = collective_any_failure(
            inverse == nullptr,
            "FFTW-MPI inverse-plan creation");
    } catch (...) {
        // Communicator synchronization failed. Do not attempt collective plan
        // destruction from an error path whose rank participation is unknown.
        throw;
    }
    if (any_inverse_plan_failed) {
        // The forward-plan stage already proved a non-null forward plan on every
        // rank, so that one plan can be destroyed collectively. The inverse
        // plan may exist on only a subset and therefore must not be destroyed
        // conditionally here.
        fftw_destroy_plan(forward);
        throw std::runtime_error(
            "Failed to create FFTW-MPI inverse plan on at least one rank");
    }

    int rank_count = 0;
    std::string local_fftw_version;
    std::string local_build_sha256;
    std::string local_loaded_sha256;
    std::string local_runtime_sha256;
    std::exception_ptr local_identity_exception;
    try {
        if (MPI_Comm_size(MPI_COMM_WORLD, &rank_count) != MPI_SUCCESS
            || rank_count < 1) {
            throw std::runtime_error(
                "MpiFFTBackend could not determine the communicator size for planning provenance");
        }
        local_fftw_version = std::string(fftw_version);
        local_build_sha256 = fftw_build_identity_sha256();
        local_loaded_sha256 = fftw_provider_path_content_sha256();
        local_runtime_sha256 = fftw_runtime_system_identity_sha256();
    } catch (...) {
        local_identity_exception = std::current_exception();
    }
    if (collective_any_failure(
            local_identity_exception ? 1 : 0,
            "FFTW-MPI local planning-provenance identity")) {
        fftw_destroy_plan(inverse);
        fftw_destroy_plan(forward);
        if (local_identity_exception) {
            std::rethrow_exception(local_identity_exception);
        }
        throw std::runtime_error(
            "FFTW-MPI local planning identity failed on another rank");
    }

    std::string agreed_loaded_sha256;
    std::string aggregate_runtime_sha256;
    try {
        agreed_loaded_sha256 = runtime::agree_optional_sha256_collective(
            local_loaded_sha256,
            "FFTW-MPI loaded-library identity");
        aggregate_runtime_sha256 = collective_rank_ordered_sha256(
            local_runtime_sha256,
            rank_count,
            "FFTW-MPI runtime-system identities");
    } catch (...) {
        // A collective failure does not prove that every rank can safely enter
        // collective FFTW plan destruction. Leave plans to process teardown.
        throw;
    }

    FftwPlanningRecord planning_record;
    std::exception_ptr planning_record_exception;
    try {
        planning_record.backend = "fftw_mpi";
        planning_record.planner_rigor = "estimate";
        planning_record.planner_schema =
            "fftw_mpi_r2c_c2r_3d_out_of_place";
        planning_record.planner_flags =
            "FFTW_ESTIMATE;wisdom_cleared_before_collective_planning";
        planning_record.wisdom_cache_policy = "disabled_and_cleared";
        planning_record.plan_identity_scope =
            "planner_policy_only_no_collective_mpi_wisdom";
        planning_record.grid_size = layout_.grid_size;
        planning_record.thread_count =
            planner_session.configured_thread_count();
        planning_record.mpi_rank_count =
            static_cast<std::size_t>(rank_count);
        planning_record.fftw_version = std::move(local_fftw_version);
        planning_record.fftw_build_identity_sha256 =
            std::move(local_build_sha256);
        planning_record.fftw_provider_path_content_sha256 =
            std::move(agreed_loaded_sha256);
        planning_record.runtime_system_identity_sha256 =
            std::move(aggregate_runtime_sha256);
        planning_record.wisdom_publication_outcome = "not_requested";
        reserve_fftw_planning_record_slot();
    } catch (...) {
        planning_record_exception = std::current_exception();
    }

    bool any_planning_record_failed = false;
    try {
        any_planning_record_failed = collective_any_failure(
            planning_record_exception ? 1 : 0,
            "FFTW-MPI planning-provenance publication");
    } catch (...) {
        // The collective state is unknown, so conditional plan destruction is
        // unsafe. Leave both plans to process teardown.
        throw;
    }
    if (any_planning_record_failed) {
        fftw_destroy_plan(inverse);
        fftw_destroy_plan(forward);
        if (planning_record_exception) {
            std::rethrow_exception(planning_record_exception);
        }
        throw std::runtime_error(
            "FFTW-MPI planning provenance failed on another rank");
    }
    append_reserved_fftw_planning_record(std::move(planning_record));

    complex_alignment_ = fftw_alignment_of(
        reinterpret_cast<double*>(planning_complex_storage.data()));
    forward_plan_ = reinterpret_cast<void*>(forward);
    inverse_plan_ = reinterpret_cast<void*>(inverse);
    real_storage_->compact_ready_ = false;

    const core::Real n_real = static_cast<core::Real>(layout_.grid_size);
    norm_factor_ = 1.0 / (n_real * n_real * n_real);
#endif
}

MpiFFTBackend::~MpiFFTBackend() {
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    std::lock_guard<std::mutex> planner_lock(shared_fftw_planner_mutex());
    if (forward_plan_) {
        fftw_destroy_plan(
            reinterpret_cast<fftw_plan>(forward_plan_));
    }
    if (inverse_plan_) {
        fftw_destroy_plan(
            reinterpret_cast<fftw_plan>(inverse_plan_));
    }
#endif
}

void MpiFFTBackend::require_complex_alignment(
    const std::complex<core::Real>* complex_data) const {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)complex_data;
#else
    const int alignment = fftw_alignment_of(
        const_cast<double*>(reinterpret_cast<const double*>(complex_data)));
    if (collective_any_failure(
            alignment != complex_alignment_,
            "FFTW-MPI new-array alignment")) {
        throw std::invalid_argument(
            "MpiFFTBackend complex-array alignment differs from its plan");
    }
#endif
}

void MpiFFTBackend::unpack_normalized_real(
    std::span<core::Real> local_real) const {
    for (std::size_t ix = 0; ix < layout_.local_n0; ++ix) {
        for (std::size_t iy = 0; iy < layout_.grid_size; ++iy) {
            const std::size_t compact_base =
                (ix * layout_.grid_size + iy) * layout_.grid_size;
            const std::size_t padded_base =
                (ix * layout_.grid_size + iy) * layout_.padded_last_dim;
            for (std::size_t iz = 0; iz < layout_.grid_size; ++iz) {
                local_real[compact_base + iz] =
                    real_buffer_[padded_base + iz] * norm_factor_;
            }
        }
    }
}

bool MpiFFTBackend::overlaps_real_storage(
    std::span<const core::Real> local_real) const noexcept {
    if (local_real.data() == nullptr) return false;
    const auto* input_begin = local_real.data();
    const auto* input_end = input_begin + local_real.size();
    const auto* storage_begin = real_buffer_;
    const auto* storage_end =
        storage_begin + layout_.padded_real_elements();
    const std::less<const core::Real*> less;
    return less(input_begin, storage_end)
        && less(storage_begin, input_end);
}

void MpiFFTBackend::copy_real_for_forward(
    std::span<const core::Real> local_real) {
    const std::size_t n = layout_.grid_size;
    const std::size_t rows = layout_.compact_real_elements() / n;
    const std::size_t padded_row = layout_.padded_last_dim;

    std::fill(
        real_buffer_,
        real_buffer_ + layout_.padded_real_elements(),
        core::Real{0.0});
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(
            local_real.data() + row * n,
            n,
            real_buffer_ + row * padded_row);
    }
}

void MpiFFTBackend::pack_shared_real_for_forward() {
    const std::size_t n = layout_.grid_size;
    const std::size_t rows = layout_.compact_real_elements() / n;
    const std::size_t padded_row = layout_.padded_last_dim;
    // Layout construction already proves that twice the required complex
    // extent fits and does not exceed the actual FFTW allocation.
    const std::size_t visible_padded =
        2 * layout_.required_complex_elements();

    // The compact and padded representations share one allocation.  Expanding
    // rows from high addresses to low addresses ensures that a widened row can
    // overwrite only compact rows whose values have already been moved.
    // Padding and any FFTW alloc_local tail are explicitly zeroed just as in
    // the disjoint-buffer path.
    std::fill(
        real_buffer_ + visible_padded,
        real_buffer_ + layout_.padded_real_elements(),
        core::Real{0.0});
    for (std::size_t row = rows; row > 1;) {
        --row;
        const std::size_t compact_base = row * n;
        const std::size_t padded_base = row * padded_row;
        std::copy_backward(
            real_buffer_ + compact_base,
            real_buffer_ + compact_base + n,
            real_buffer_ + padded_base + n);
        std::fill(
            real_buffer_ + padded_base + n,
            real_buffer_ + padded_base + padded_row,
            core::Real{0.0});
    }
    // Row zero has the same compact and padded base. Avoid even a self-copy;
    // only its FFTW padding needs materialization.
    std::fill(
        real_buffer_ + n,
        real_buffer_ + padded_row,
        core::Real{0.0});
}

void MpiFFTBackend::forward(
    std::span<const core::Real> local_real,
    std::span<std::complex<core::Real>> local_complex) {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)local_real;
    (void)local_complex;
    throw std::runtime_error(
        "MpiFFTBackend requires a build with FFTW-MPI support");
#else
    require_mpi_main_thread("FFTW-MPI forward execution");
    const int local_size_invalid =
        local_real.size() != local_real_size()
        || local_complex.size()
            != layout_.alloc_local_complex_elements
        || overlaps_real_storage(local_real);
    if (collective_any_failure(
            local_size_invalid,
            "FFTW-MPI forward slab-size validation")) {
        throw std::invalid_argument(
            "MpiFFTBackend forward slab size mismatch on at least one rank");
    }
    require_complex_alignment(local_complex.data());

    copy_real_for_forward(local_real);
    real_storage_->compact_ready_ = false;

    fftw_mpi_execute_dft_r2c(
        reinterpret_cast<fftw_plan>(forward_plan_),
        real_buffer_,
        fftw_data(local_complex.data()));
#endif
}

void MpiFFTBackend::forward_shared_destructive(
    std::span<core::Real> local_real,
    std::span<std::complex<core::Real>> local_complex) {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)local_real;
    (void)local_complex;
    throw std::runtime_error(
        "MpiFFTBackend requires a build with FFTW-MPI support");
#else
    require_mpi_main_thread("FFTW-MPI destructive shared forward execution");
    const int local_state_invalid =
        local_real.size() != local_real_size()
        || local_complex.size()
            != layout_.alloc_local_complex_elements
        || local_real.data() != real_buffer_
        || !real_storage_->compact_ready_;
    if (collective_any_failure(
            local_state_invalid,
            "FFTW-MPI destructive shared-forward state validation")) {
        throw std::invalid_argument(
            "MpiFFTBackend destructive shared forward requires a published compact overwrite on every rank");
    }
    require_complex_alignment(local_complex.data());

    pack_shared_real_for_forward();
    real_storage_->compact_ready_ = false;
    fftw_mpi_execute_dft_r2c(
        reinterpret_cast<fftw_plan>(forward_plan_),
        real_buffer_,
        fftw_data(local_complex.data()));
#endif
}

void MpiFFTBackend::inverse(
    std::span<const std::complex<core::Real>> local_complex,
    std::span<core::Real> local_real) {
#ifndef COSMO_NBODY_HAS_FFTW_MPI
    (void)local_complex;
    (void)local_real;
    throw std::runtime_error(
        "MpiFFTBackend requires a build with FFTW-MPI support");
#else
    require_mpi_main_thread("FFTW-MPI inverse execution");
    const bool exact_shared_output = local_real.data() == real_buffer_;
    const int local_size_invalid =
        local_complex.size()
            != layout_.alloc_local_complex_elements
        || local_real.size() != local_real_size()
        || (overlaps_real_storage(local_real) && !exact_shared_output);
    if (collective_any_failure(
            local_size_invalid,
            "FFTW-MPI inverse slab-size validation")) {
        throw std::invalid_argument(
            "MpiFFTBackend inverse slab size mismatch on at least one rank");
    }

    const bool destructive = collective_agreed_flag(
        is_destructive_fft_complex_storage(
            local_complex.data(), local_complex.size()),
        "inverse Fourier-ownership mode");
    if (destructive) {
        require_complex_alignment(local_complex.data());
        auto* mutable_modes = const_cast<std::complex<core::Real>*>(
            local_complex.data());
        fftw_mpi_execute_dft_c2r(
            reinterpret_cast<fftw_plan>(inverse_plan_),
            fftw_data(mutable_modes),
            real_buffer_);
    } else {
        auto* scratch = fftw_alloc_complex(
            layout_.alloc_local_complex_elements);
        bool any_allocation_failed = false;
        try {
            any_allocation_failed = collective_any_failure(
                scratch == nullptr,
                "FFTW-MPI preserving inverse scratch allocation");
        } catch (...) {
            if (scratch) fftw_free(scratch);
            throw;
        }
        if (any_allocation_failed) {
            if (scratch) fftw_free(scratch);
            throw std::bad_alloc();
        }

        try {
            require_complex_alignment(
                reinterpret_cast<std::complex<core::Real>*>(scratch));
        } catch (...) {
            fftw_free(scratch);
            throw;
        }
        for (std::size_t i = 0;
             i < layout_.alloc_local_complex_elements;
             ++i) {
            scratch[i][0] = local_complex[i].real();
            scratch[i][1] = local_complex[i].imag();
        }

        fftw_mpi_execute_dft_c2r(
            reinterpret_cast<fftw_plan>(inverse_plan_),
            scratch,
            real_buffer_);
        fftw_free(scratch);
    }

    unpack_normalized_real(local_real);
    real_storage_->compact_ready_ = exact_shared_output;
#endif
}

} // namespace mesh
} // namespace cosmo_nbody
