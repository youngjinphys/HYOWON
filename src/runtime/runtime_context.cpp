#include "cosmo_nbody/runtime/runtime_context.hpp"

#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(COSMO_NBODY_HAS_MPI) && defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace runtime {

namespace {

std::uint64_t size_to_u64(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                std::string(label) + " exceeds uint64_t");
        }
    }
    return static_cast<std::uint64_t>(value);
}

RuntimeMemoryObservation observe_local_process_memory() noexcept {
    RuntimeMemoryObservation observation;
#if defined(__linux__)
    try {
        std::ifstream status("/proc/self/status");
        if (!status) return observation;

        bool have_current = false;
        bool have_peak = false;
        std::uint64_t current_kib = 0;
        std::uint64_t peak_kib = 0;
        std::string key;
        while (status >> key) {
            if (key == "VmRSS:" || key == "VmHWM:") {
                std::uint64_t value_kib = 0;
                std::string unit;
                if (!(status >> value_kib >> unit) || unit != "kB") {
                    return RuntimeMemoryObservation{};
                }
                if (key == "VmRSS:") {
                    current_kib = value_kib;
                    have_current = true;
                } else {
                    peak_kib = value_kib;
                    have_peak = true;
                }
            }
            status.ignore(
                std::numeric_limits<std::streamsize>::max(), '\n');
        }
        constexpr std::uint64_t bytes_per_kib = 1024;
        const std::uint64_t maximum_kib =
            std::numeric_limits<std::uint64_t>::max() / bytes_per_kib;
        if (!have_current || !have_peak
            || current_kib > maximum_kib || peak_kib > maximum_kib) {
            return RuntimeMemoryObservation{};
        }
        observation.available = true;
        observation.current_rss_bytes = current_kib * bytes_per_kib;
        observation.peak_rss_bytes = peak_kib * bytes_per_kib;
    } catch (...) {
        return RuntimeMemoryObservation{};
    }
#elif defined(__APPLE__)
    try {
        mach_task_basic_info_data_t info{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        const kern_return_t info_status = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            reinterpret_cast<task_info_t>(&info),
            &count);
        struct rusage usage{};
        const int usage_status = getrusage(RUSAGE_SELF, &usage);
        if (info_status != KERN_SUCCESS || usage_status != 0
            || usage.ru_maxrss < 0) {
            return RuntimeMemoryObservation{};
        }
        observation.available = true;
        observation.current_rss_bytes =
            static_cast<std::uint64_t>(info.resident_size);
        observation.peak_rss_bytes =
            static_cast<std::uint64_t>(usage.ru_maxrss);
        if (observation.peak_rss_bytes < observation.current_rss_bytes) {
            observation.peak_rss_bytes = observation.current_rss_bytes;
        }
    } catch (...) {
        return RuntimeMemoryObservation{};
    }
#endif
    return observation;
}

#ifdef COSMO_NBODY_HAS_MPI
struct CpuAffinityObservation {
    std::size_t visible_cpu_count{0};
    std::size_t automatic_thread_ceiling{0};
    bool shared_unbound_affinity{false};
};

struct SharedMemoryLocality {
    int rank{0};
    int size{1};
    CpuAffinityObservation affinity;
};

#if defined(__linux__)
CpuAffinityObservation observe_shared_cpu_affinity(
    MPI_Comm local_comm,
    int local_size) {
    CpuAffinityObservation result;
    if (local_size < 1) return result;

    const std::size_t local_size_u = static_cast<std::size_t>(local_size);
    int processor_count = 0;
    int available = 0;
    std::vector<unsigned char> local_mask;
    std::vector<unsigned char> all_masks;
    try {
        const long configured_processors = sysconf(_SC_NPROCESSORS_CONF);
        if (configured_processors > 0
            && configured_processors <= std::numeric_limits<int>::max()) {
            processor_count = static_cast<int>(configured_processors);
            const auto count = static_cast<std::size_t>(processor_count);
            const std::size_t set_bytes = CPU_ALLOC_SIZE(processor_count);
            const auto release_affinity = [](cpu_set_t* value) {
                if (value != nullptr) CPU_FREE(value);
            };
            std::unique_ptr<cpu_set_t, decltype(release_affinity)> affinity(
                CPU_ALLOC(processor_count), release_affinity);
            if (affinity && count <= std::numeric_limits<std::size_t>::max()
                    / local_size_u) {
                CPU_ZERO_S(set_bytes, affinity.get());
                if (sched_getaffinity(0, set_bytes, affinity.get()) == 0) {
                    local_mask.resize(count, 0);
                    for (int processor = 0; processor < processor_count;
                         ++processor) {
                        if (CPU_ISSET_S(processor, set_bytes, affinity.get())) {
                            local_mask[static_cast<std::size_t>(processor)] = 1;
                            ++result.visible_cpu_count;
                        }
                    }
                    all_masks.resize(count * local_size_u, 0);
                    available = result.visible_cpu_count != 0 ? 1 : 0;
                }
            }
        }
    } catch (...) {
        // Affinity is optional telemetry; allocation failure must make every
        // participant skip the payload collective, not strand healthy peers.
        available = 0;
    }
    const auto require_collective_success = [](int status) {
        if (status != MPI_SUCCESS) {
            (void)MPI_Abort(MPI_COMM_WORLD, status);
            std::abort();
        }
    };
    int all_available = 0;
    require_collective_success(MPI_Allreduce(
        &available, &all_available, 1, MPI_INT, MPI_MIN, local_comm));
    if (!all_available) return {};
    int minimum_count = 0;
    int maximum_count = 0;
    require_collective_success(MPI_Allreduce(
        &processor_count, &minimum_count, 1, MPI_INT, MPI_MIN, local_comm));
    require_collective_success(MPI_Allreduce(
        &processor_count, &maximum_count, 1, MPI_INT, MPI_MAX, local_comm));
    if (minimum_count != maximum_count) return {};
    if (local_size == 1) return result;
    const auto processor_count_u = static_cast<std::size_t>(processor_count);
    if (MPI_Allgather(
            local_mask.data(),
            processor_count,
            MPI_UNSIGNED_CHAR,
            all_masks.data(),
            processor_count,
            MPI_UNSIGNED_CHAR,
            local_comm) != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }

    bool identical = true;
    for (std::size_t rank = 1; rank < local_size_u && identical; ++rank) {
        const auto begin = all_masks.begin()
            + static_cast<std::ptrdiff_t>(rank * processor_count_u);
        identical = std::equal(
            all_masks.begin(),
            all_masks.begin() + static_cast<std::ptrdiff_t>(processor_count_u),
            begin);
    }
    if (!identical) {
        // Non-identical masks are evidence that the launcher/scheduler already
        // expressed a placement policy. Do not second-guess it with a synthetic
        // CPU ownership model: OpenMP will see this process's actual affinity.
        return result;
    }

    result.shared_unbound_affinity = true;
    result.automatic_thread_ceiling = std::max<std::size_t>(
        std::size_t{1},
        result.visible_cpu_count / local_size_u);
    return result;
}
#endif

SharedMemoryLocality query_shared_memory_locality() {
    MPI_Comm local_comm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(
            MPI_COMM_WORLD,
            MPI_COMM_TYPE_SHARED,
            0,
            MPI_INFO_NULL,
            &local_comm) != MPI_SUCCESS
        || local_comm == MPI_COMM_NULL) {
        throw std::runtime_error(
            "Failed to construct the MPI shared-memory communicator");
    }

    const auto free_local = [&]() noexcept {
        if (local_comm != MPI_COMM_NULL) {
            (void)MPI_Comm_free(&local_comm);
        }
    };

    SharedMemoryLocality result;
    if (MPI_Comm_set_errhandler(local_comm, MPI_ERRORS_RETURN) != MPI_SUCCESS
        || MPI_Comm_rank(local_comm, &result.rank) != MPI_SUCCESS
        || MPI_Comm_size(local_comm, &result.size) != MPI_SUCCESS) {
        free_local();
        throw std::runtime_error(
            "Failed to query MPI shared-memory rank topology");
    }
    if (result.size < 1 || result.rank < 0 || result.rank >= result.size) {
        free_local();
        throw std::runtime_error(
            "MPI shared-memory communicator returned an invalid topology");
    }

    try {
#if defined(__linux__)
        result.affinity = observe_shared_cpu_affinity(local_comm, result.size);
#endif
    } catch (...) {
        free_local();
        throw;
    }
    free_local();
    return result;
}
#endif

} // namespace

struct RuntimeContext::State {
    explicit State(const config::RuntimeParams& params)
        : requested_num_threads(params.num_threads),
          mpi_requested(params.mpi_enabled) {
        if (!params.mpi_enabled) {
            host_threads.emplace(params.num_threads);
            return;
        }

#ifndef COSMO_NBODY_HAS_MPI
        throw std::invalid_argument(
            "runtime.mpi_enabled=true requires a build with MPI support");
#else
        int initialized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Initialized failed");
        }
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Finalized failed");
        }
        if (finalized) {
            throw std::runtime_error(
                "MPI-active RuntimeContext cannot attach after MPI_Finalize");
        }

        if (!initialized) {
            int argc = 0;
            char** argv = nullptr;
            int provided = MPI_THREAD_SINGLE;
            const int status = MPI_Init_thread(
                &argc, &argv, MPI_THREAD_FUNNELED, &provided);
            if (status != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Init_thread failed");
            }
            owns_mpi_lifecycle = true;
            if (provided < MPI_THREAD_FUNNELED) {
                (void)MPI_Finalize();
                owns_mpi_lifecycle = false;
                throw std::runtime_error(
                    "MPI implementation does not provide MPI_THREAD_FUNNELED");
            }
        } else {
            int is_main_thread = 0;
            if (MPI_Is_thread_main(&is_main_thread) != MPI_SUCCESS
                || !is_main_thread) {
                throw std::runtime_error(
                    "MPI-active RuntimeContext must be created on the MPI main thread");
            }

            int provided = MPI_THREAD_SINGLE;
            if (MPI_Query_thread(&provided) != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Query_thread failed");
            }
            if (provided < MPI_THREAD_FUNNELED) {
                throw std::runtime_error(
                    "Existing MPI runtime does not provide MPI_THREAD_FUNNELED");
            }
        }

        MPI_Errhandler captured_handler = MPI_ERRHANDLER_NULL;
        if (MPI_Comm_get_errhandler(
                MPI_COMM_WORLD, &captured_handler) != MPI_SUCCESS
            || captured_handler == MPI_ERRHANDLER_NULL) {
            release_mpi_noexcept();
            throw std::runtime_error(
                "Failed to capture MPI_COMM_WORLD error handler");
        }
        if (MPI_Comm_set_errhandler(
                MPI_COMM_WORLD, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
            (void)MPI_Errhandler_free(&captured_handler);
            release_mpi_noexcept();
            throw std::runtime_error("Failed to install MPI_ERRORS_RETURN");
        }
        previous_mpi_errhandler = captured_handler;
        mpi_errhandler_changed = true;

        try {
            if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS
                || MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
                throw std::runtime_error("Failed to query MPI rank/size");
            }
            if (size < 1 || rank < 0 || rank >= size) {
                throw std::runtime_error("MPI returned an invalid rank topology");
            }

            const SharedMemoryLocality locality =
                query_shared_memory_locality();
            local_rank = locality.rank;
            local_size = locality.size;
            visible_cpu_count = locality.affinity.visible_cpu_count;
            automatic_thread_ceiling =
                locality.affinity.automatic_thread_ceiling;
            shared_unbound_cpu_affinity =
                locality.affinity.shared_unbound_affinity;

            // Local policy rejection or allocation failure must be agreed
            // before any rank returns or begins MPI lifecycle cleanup. The
            // locality collectives above are already complete at this boundary.
            std::exception_ptr local_startup_error;
            try {
                host_threads.emplace(
                    params.num_threads,
                    params.num_threads == 0 ? automatic_thread_ceiling : 0);

                char processor_buffer[MPI_MAX_PROCESSOR_NAME]{};
                int processor_length = 0;
                if (MPI_Get_processor_name(
                        processor_buffer, &processor_length) == MPI_SUCCESS
                    && processor_length >= 0
                    && processor_length <= MPI_MAX_PROCESSOR_NAME) {
                    processor.assign(
                        processor_buffer,
                        static_cast<std::size_t>(processor_length));
                }

                char library_buffer[MPI_MAX_LIBRARY_VERSION_STRING]{};
                int library_length = 0;
                if (MPI_Get_library_version(
                        library_buffer, &library_length) == MPI_SUCCESS
                    && library_length >= 0
                    && library_length <= MPI_MAX_LIBRARY_VERSION_STRING) {
                    mpi_library.assign(
                        library_buffer,
                        static_cast<std::size_t>(library_length));
                }
            } catch (...) {
                local_startup_error = std::current_exception();
            }
            synchronize_mpi_exception(
                local_startup_error, size, "MPI runtime startup");
            mpi_active = true;
        } catch (...) {
            host_threads.reset();
            release_mpi_noexcept();
            throw;
        }
#endif
    }

    ~State() {
        // OpenMP state belongs to this process runtime lease and must be restored
        // before an MPI implementation supplied by either this object or the
        // caller is torn down.
        host_threads.reset();
        release_mpi_noexcept();
    }

    void require_compatible(const config::RuntimeParams& params) const {
        if (params.mpi_enabled != mpi_requested
            || params.num_threads != requested_num_threads) {
            throw std::invalid_argument(
                "Concurrent RuntimeContext leases must use identical runtime parameters");
        }
    }

    void release_mpi_noexcept() noexcept {
#ifdef COSMO_NBODY_HAS_MPI
        int finalized = 0;
        const bool mpi_is_live =
            MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized;
        if (mpi_errhandler_changed) {
            if (mpi_is_live
                && previous_mpi_errhandler != MPI_ERRHANDLER_NULL) {
                (void)MPI_Comm_set_errhandler(
                    MPI_COMM_WORLD, previous_mpi_errhandler);
                (void)MPI_Errhandler_free(&previous_mpi_errhandler);
            }
            previous_mpi_errhandler = MPI_ERRHANDLER_NULL;
            mpi_errhandler_changed = false;
        }
        if (owns_mpi_lifecycle && mpi_is_live) {
            (void)MPI_Finalize();
        }
#endif
        mpi_active = false;
        owns_mpi_lifecycle = false;
    }

    std::optional<HostThreadContext> host_threads;
    std::uint64_t requested_num_threads{0};
    bool mpi_requested{false};
    int rank{0};
    int size{1};
    int local_rank{0};
    int local_size{1};
    std::size_t visible_cpu_count{0};
    std::size_t automatic_thread_ceiling{0};
    bool shared_unbound_cpu_affinity{false};
    std::string processor;
    std::string mpi_library;
#ifdef COSMO_NBODY_HAS_MPI
    MPI_Errhandler previous_mpi_errhandler{MPI_ERRHANDLER_NULL};
    bool mpi_errhandler_changed{false};
#endif
    bool mpi_active{false};
    bool owns_mpi_lifecycle{false};
};

RuntimeContext::RuntimeContext(const config::RuntimeParams& params) {
    static std::mutex state_mutex;
    static std::weak_ptr<State> active_state;

    std::lock_guard<std::mutex> lock(state_mutex);
    if (auto existing = active_state.lock()) {
        existing->require_compatible(params);
        state_ = std::move(existing);
        return;
    }

    auto created = std::make_shared<State>(params);
    active_state = created;
    state_ = std::move(created);
}

RuntimeContext::~RuntimeContext() = default;

int RuntimeContext::rank() const noexcept {
    return state_->rank;
}

int RuntimeContext::size() const noexcept {
    return state_->size;
}

int RuntimeContext::local_rank() const noexcept {
    return state_->local_rank;
}

int RuntimeContext::local_size() const noexcept {
    return state_->local_size;
}

std::size_t RuntimeContext::thread_count() const noexcept {
    return state_->host_threads.has_value()
        ? state_->host_threads->thread_count()
        : std::size_t{1};
}

std::size_t RuntimeContext::visible_cpu_count() const noexcept {
    return state_->visible_cpu_count;
}

std::size_t RuntimeContext::automatic_thread_ceiling() const noexcept {
    return state_->automatic_thread_ceiling;
}

bool RuntimeContext::shared_unbound_cpu_affinity() const noexcept {
    return state_->shared_unbound_cpu_affinity;
}

bool RuntimeContext::mpi_active() const noexcept {
    return state_->mpi_active;
}

std::string_view RuntimeContext::thread_selection_reason() const noexcept {
    return state_->host_threads.has_value()
        ? state_->host_threads->thread_selection_reason()
        : std::string_view{"unresolved"};
}

std::string_view RuntimeContext::processor_name() const noexcept {
    return state_->processor;
}

std::string_view RuntimeContext::mpi_library_version() const noexcept {
    return state_->mpi_library;
}

RuntimeMemoryObservation RuntimeContext::observe_process_memory() const noexcept {
    return observe_local_process_memory();
}

RuntimeTopologyDiagnostics RuntimeContext::collect_topology_diagnostics() const {
    RuntimeTopologyDiagnostics result;
    const std::uint64_t local_threads = size_to_u64(
        thread_count(), "Runtime effective thread count");
    const std::uint64_t local_size_u64 = static_cast<std::uint64_t>(
        std::max(1, local_size()));
    const std::uint64_t local_visible_cpus = size_to_u64(
        visible_cpu_count(), "Runtime visible CPU count");
    const std::uint64_t local_automatic_ceiling = size_to_u64(
        automatic_thread_ceiling(), "Runtime automatic thread ceiling");

    result.rank_count = static_cast<std::uint64_t>(
        std::max(1, size()));
    result.shared_memory_domain_count = 1;
    result.local_size_min = local_size_u64;
    result.local_size_max = local_size_u64;
    result.effective_threads_min = local_threads;
    result.effective_threads_max = local_threads;
    result.visible_cpu_count_min = local_visible_cpus;
    result.visible_cpu_count_max = local_visible_cpus;
    result.automatic_thread_ceiling_min = local_automatic_ceiling;
    result.automatic_thread_ceiling_max = local_automatic_ceiling;
    result.shared_unbound_affinity_rank_count =
        shared_unbound_cpu_affinity() ? 1U : 0U;

    if (!mpi_active() || size() == 1) return result;

#ifndef COSMO_NBODY_HAS_MPI
    throw std::logic_error(
        "MPI runtime topology diagnostics requested in non-MPI build");
#else
    int is_main_thread = 0;
    if (MPI_Is_thread_main(&is_main_thread) != MPI_SUCCESS
        || is_main_thread == 0) {
        throw std::runtime_error(
            "Runtime topology diagnostics require the MPI main thread");
    }

    const std::array<std::uint64_t, 4> local_metrics{
        local_size_u64,
        local_threads,
        local_visible_cpus,
        local_automatic_ceiling};
    std::array<std::uint64_t, 4> minima{};
    std::array<std::uint64_t, 4> maxima{};
    if (MPI_Allreduce(
            local_metrics.data(), minima.data(),
            static_cast<int>(local_metrics.size()),
            MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS
        || MPI_Allreduce(
            local_metrics.data(), maxima.data(),
            static_cast<int>(local_metrics.size()),
            MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Runtime topology min/max reduction failed");
    }

    const std::array<std::uint64_t, 2> local_counts{
        local_rank() == 0 ? std::uint64_t{1} : std::uint64_t{0},
        shared_unbound_cpu_affinity()
            ? std::uint64_t{1} : std::uint64_t{0}};
    std::array<std::uint64_t, 2> sums{};
    if (MPI_Allreduce(
            local_counts.data(), sums.data(),
            static_cast<int>(local_counts.size()),
            MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Runtime topology count reduction failed");
    }

    result.shared_memory_domain_count = sums[0];
    result.local_size_min = minima[0];
    result.local_size_max = maxima[0];
    result.effective_threads_min = minima[1];
    result.effective_threads_max = maxima[1];
    result.visible_cpu_count_min = minima[2];
    result.visible_cpu_count_max = maxima[2];
    result.automatic_thread_ceiling_min = minima[3];
    result.automatic_thread_ceiling_max = maxima[3];
    result.shared_unbound_affinity_rank_count = sums[1];
    return result;
#endif
}

} // namespace runtime
} // namespace cosmo_nbody
