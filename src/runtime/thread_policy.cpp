#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody::runtime {
namespace {

std::size_t positive_or_one(std::size_t value) noexcept {
    return std::max<std::size_t>(value, 1);
}

HostThreadPolicyInput capture_host_thread_policy_input(
    std::size_t requested_threads,
    std::size_t automatic_capacity_ceiling) noexcept {
    HostThreadPolicyInput input;
    input.requested_threads = requested_threads;
    input.automatic_capacity_ceiling = automatic_capacity_ceiling;
#ifdef COSMO_NBODY_HAS_OPENMP
    input.openmp_available = true;
    input.runtime_max_threads = static_cast<std::size_t>(
        std::max(1, omp_get_max_threads()));
    input.available_processors = static_cast<std::size_t>(
        std::max(1, omp_get_num_procs()));
    input.thread_limit = static_cast<std::size_t>(
        std::max(1, omp_get_thread_limit()));
#else
    input.openmp_available = false;
#endif
    return input;
}

} // namespace

HostThreadPolicy resolve_host_thread_policy(
    const HostThreadPolicyInput& input) {
    HostThreadPolicy policy;
    policy.requested_threads = input.requested_threads;
    policy.automatic = input.requested_threads == 0;

    if (!input.openmp_available) {
        if (input.requested_threads > 1) {
            throw std::invalid_argument(
                "runtime.num_threads > 1 requires a build with OpenMP support");
        }
        policy.effective_threads = 1;
        policy.automatic_capacity = 1;
        policy.reason = policy.automatic
            ? "auto_serial_no_openmp"
            : "explicit_serial_no_openmp";
        return policy;
    }

    const std::size_t runtime_max = positive_or_one(
        input.runtime_max_threads);
    const std::size_t processors = positive_or_one(
        input.available_processors);
    const std::size_t limit = positive_or_one(input.thread_limit);
    policy.automatic_capacity = std::min({runtime_max, processors, limit});
    if (input.automatic_capacity_ceiling > 0) {
        policy.automatic_capacity = std::min(
            policy.automatic_capacity,
            input.automatic_capacity_ceiling);
    }

    if (policy.automatic) {
        policy.effective_threads = policy.automatic_capacity;
        policy.reason = input.automatic_capacity_ceiling > 0
            ? "auto_openmp_capacity_with_execution_topology_ceiling"
            : "auto_openmp_runtime_capacity";
        return policy;
    }

    if (input.requested_threads
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "runtime.num_threads is outside the supported OpenMP int range");
    }
    if (input.requested_threads > limit) {
        throw std::invalid_argument(
            "runtime.num_threads exceeds the OpenMP thread-limit ICV");
    }

    policy.effective_threads = input.requested_threads;
    policy.reason = "explicit_runtime_num_threads";
    return policy;
}

std::size_t host_parallel_worker_capacity(
    std::size_t independent_items) noexcept {
    if (independent_items == 0) return 1;
#ifdef COSMO_NBODY_HAS_OPENMP
    const std::size_t runtime_threads = static_cast<std::size_t>(
        std::max(1, omp_get_max_threads()));
    return std::max<std::size_t>(
        1,
        std::min(runtime_threads, independent_items));
#else
    return 1;
#endif
}

bool should_use_host_parallel_team(
    std::size_t independent_items) noexcept {
#ifdef COSMO_NBODY_HAS_OPENMP
    const std::size_t runtime_threads = static_cast<std::size_t>(
        std::max(1, omp_get_max_threads()));
    return runtime_threads > 1 && independent_items >= runtime_threads;
#else
    (void)independent_items;
    return false;
#endif
}

HostThreadContext::HostThreadContext(
    std::size_t requested_threads,
    std::size_t automatic_capacity_ceiling) {
    const HostThreadPolicy policy = resolve_host_thread_policy(
        capture_host_thread_policy_input(
            requested_threads,
            automatic_capacity_ceiling));

    // Complete all potentially allocating assignments before changing OpenMP
    // state. A failed constructor must never leak a partially applied policy.
    requested_thread_count_ = policy.requested_threads;
    automatic_thread_capacity_ = policy.automatic_capacity;
    automatic_thread_count_ = policy.automatic;
    thread_selection_reason_ = policy.reason;
    thread_count_ = policy.effective_threads;

#ifdef COSMO_NBODY_HAS_OPENMP
    previous_dynamic_ = omp_get_dynamic();
    previous_max_active_levels_ = omp_get_max_active_levels();
    previous_max_threads_ = omp_get_max_threads();

    const int requested = static_cast<int>(policy.effective_threads);
    omp_set_dynamic(0);
    // Kernels use the outer parallel level; nested teams can oversubscribe when
    // callbacks or FFT implementations also use threads.
    omp_set_max_active_levels(1);
    omp_set_num_threads(requested);

    int observed = 0;
    #pragma omp parallel num_threads(requested)
    {
        #pragma omp single
        observed = omp_get_num_threads();
    }
    if (observed != requested) {
        omp_set_dynamic(previous_dynamic_);
        omp_set_max_active_levels(previous_max_active_levels_);
        omp_set_num_threads(previous_max_threads_);
        throw std::runtime_error(
            "OpenMP runtime formed " + std::to_string(observed)
            + " threads but the resolved runtime policy requested "
            + std::to_string(requested)
            + "; reduce runtime.num_threads, inspect process affinity, or remove the external thread limit");
    }
    thread_count_ = static_cast<std::size_t>(observed);
    changed_ = true;
#endif
}

HostThreadContext::~HostThreadContext() {
    restore();
}

void HostThreadContext::restore() noexcept {
#ifdef COSMO_NBODY_HAS_OPENMP
    if (changed_) {
        omp_set_dynamic(previous_dynamic_);
        omp_set_max_active_levels(previous_max_active_levels_);
        omp_set_num_threads(previous_max_threads_);
        changed_ = false;
    }
#else
    changed_ = false;
#endif
}

} // namespace cosmo_nbody::runtime
