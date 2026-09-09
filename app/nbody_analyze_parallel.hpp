#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::app::nbody_analyze {

template <typename Function>
void parallel_for_indices(std::size_t count, Function&& function) {
    if (count == 0) return;

    std::atomic<bool> worker_failed{false};
    std::exception_ptr worker_exception;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 1) if(count > 1)
#endif
    for (std::size_t index = 0; index < count; ++index) {
        if (worker_failed.load(std::memory_order_relaxed)) continue;
        try {
            function(index);
        } catch (...) {
            worker_failed.store(true, std::memory_order_relaxed);
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp critical(cosmo_nbody_analysis_worker_failure)
#endif
            {
                if (!worker_exception) {
                    worker_exception = std::current_exception();
                }
            }
        }
    }
    if (worker_exception) std::rethrow_exception(worker_exception);
}

template <typename Function>
void parallel_for_indices_limited(
    std::size_t count,
    std::size_t maximum_workers,
    Function&& function) {
    if (count == 0) return;
    if (maximum_workers == 0
        || maximum_workers > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Analysis batch worker count is outside the OpenMP integer range");
    }
    const int worker_count = static_cast<int>(
        std::min(count, maximum_workers));
#ifndef COSMO_NBODY_HAS_OPENMP
    (void)worker_count;
#endif

    std::atomic<bool> worker_failed{false};
    std::exception_ptr worker_exception;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(worker_count) \
        if(count > 1 && worker_count > 1)
#endif
    for (std::size_t index = 0; index < count; ++index) {
        if (worker_failed.load(std::memory_order_relaxed)) continue;
        try {
            function(index);
        } catch (...) {
            worker_failed.store(true, std::memory_order_relaxed);
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp critical(cosmo_nbody_limited_analysis_worker_failure)
#endif
            {
                if (!worker_exception) {
                    worker_exception = std::current_exception();
                }
            }
        }
    }
    if (worker_exception) std::rethrow_exception(worker_exception);
}

} // namespace cosmo_nbody::app::nbody_analyze
