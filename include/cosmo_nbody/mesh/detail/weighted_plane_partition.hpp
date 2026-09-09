#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace cosmo_nbody::mesh::detail {

inline std::size_t contiguous_groups_required(
    std::span<const std::uint64_t> weights,
    std::uint64_t maximum_group_load) {
    std::size_t groups = 1;
    std::uint64_t current = 0;
    for (const std::uint64_t weight : weights) {
        if (weight > maximum_group_load) {
            return std::numeric_limits<std::size_t>::max();
        }
        if (current > maximum_group_load - weight) {
            if (groups == std::numeric_limits<std::size_t>::max()) {
                return groups;
            }
            ++groups;
            current = weight;
        } else {
            current += weight;
        }
    }
    return groups;
}

// Divide ordered destination-plane tasks into non-empty contiguous worker ranges.
// The chosen partition minimizes the largest worker weight exactly for
// non-negative integer weights. Plane ownership and each plane's particle
// accumulation order are unchanged, so changing the OpenMP team does not change
// floating-point order within any destination cell.
inline std::vector<std::size_t> weighted_contiguous_plane_boundaries(
    std::span<const std::uint64_t> weights,
    std::size_t requested_workers) {
    if (weights.empty()) {
        throw std::invalid_argument(
            "CIC weighted partition requires at least one plane task");
    }
    if (requested_workers == 0) {
        throw std::invalid_argument(
            "CIC weighted partition requires at least one worker");
    }

    const std::size_t task_count = weights.size();
    const std::size_t worker_count = std::min(task_count, requested_workers);
    std::uint64_t total = 0;
    std::uint64_t largest = 0;
    for (const std::uint64_t weight : weights) {
        if (weight > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error(
                "CIC weighted partition work overflows uint64_t");
        }
        total += weight;
        largest = std::max(largest, weight);
    }

    std::vector<std::size_t> boundaries(worker_count + 1, 0);
    boundaries.back() = task_count;
    if (total == 0) {
        const std::size_t base = task_count / worker_count;
        const std::size_t remainder = task_count % worker_count;
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            boundaries[worker + 1] = boundaries[worker]
                + base + (worker < remainder ? 1 : 0);
        }
        return boundaries;
    }

    const std::uint64_t average_floor = total / worker_count;
    std::uint64_t lower = std::max(
        largest,
        average_floor + (total % worker_count != 0 ? 1 : 0));
    std::uint64_t upper = total;
    while (lower < upper) {
        const std::uint64_t midpoint = lower + (upper - lower) / 2;
        if (contiguous_groups_required(weights, midpoint) <= worker_count) {
            upper = midpoint;
        } else {
            lower = midpoint + 1;
        }
    }
    const std::uint64_t optimal_maximum_load = lower;

    std::size_t task = 0;
    for (std::size_t worker = 0; worker + 1 < worker_count; ++worker) {
        const std::size_t remaining_workers = worker_count - worker - 1;
        const std::size_t maximum_end = task_count - remaining_workers;
        std::uint64_t load = 0;
        while (task < maximum_end) {
            const std::uint64_t weight = weights[task];
            if (load > optimal_maximum_load - weight) break;
            load += weight;
            ++task;
        }
        if (task == boundaries[worker]) {
            throw std::logic_error(
                "CIC minimax partition failed to assign a non-empty range");
        }
        boundaries[worker + 1] = task;
    }
    return boundaries;
}

} // namespace cosmo_nbody::mesh::detail
