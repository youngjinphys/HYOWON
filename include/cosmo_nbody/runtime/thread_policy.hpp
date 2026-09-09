#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace cosmo_nbody::runtime {

// Inputs are captured before OpenMP control variables change. Zero requested
// threads selects automatic mode; zero capacity ceiling means no topology limit.
struct HostThreadPolicyInput {
    std::size_t requested_threads{0};
    bool openmp_available{false};
    std::size_t runtime_max_threads{1};
    std::size_t available_processors{1};
    std::size_t thread_limit{1};
    std::size_t automatic_capacity_ceiling{0};
};

struct HostThreadPolicy {
    std::size_t requested_threads{0};
    std::size_t effective_threads{1};
    std::size_t automatic_capacity{1};
    bool automatic{true};
    std::string reason;
};

// Automatic mode selects the minimum positive runtime capacity and optional
// topology ceiling; positive requests remain exact overrides.
HostThreadPolicy resolve_host_thread_policy(
    const HostThreadPolicyInput& input);

// Use a team only when multiple workers are available and each can receive an
// independent iteration; no machine-specific item threshold is used.
bool should_use_host_parallel_team(std::size_t independent_items) noexcept;

// Useful workers for independent items under current OpenMP state, bounded by
// the item count and equal to one in serial builds.
std::size_t host_parallel_worker_capacity(
    std::size_t independent_items) noexcept;

// Scoped host OpenMP policy shared by simulation and analysis. It does not
// initialize MPI or select devices; callers may provide an automatic topology ceiling.
class HostThreadContext {
public:
    explicit HostThreadContext(
        std::size_t requested_threads,
        std::size_t automatic_capacity_ceiling = 0);
    ~HostThreadContext();

    HostThreadContext(const HostThreadContext&) = delete;
    HostThreadContext& operator=(const HostThreadContext&) = delete;
    HostThreadContext(HostThreadContext&&) = delete;
    HostThreadContext& operator=(HostThreadContext&&) = delete;

    std::size_t thread_count() const noexcept { return thread_count_; }
    std::size_t requested_thread_count() const noexcept {
        return requested_thread_count_;
    }
    std::size_t automatic_thread_capacity() const noexcept {
        return automatic_thread_capacity_;
    }
    bool thread_count_is_automatic() const noexcept {
        return automatic_thread_count_;
    }
    std::string_view thread_selection_reason() const noexcept {
        return thread_selection_reason_;
    }

private:
    void restore() noexcept;

    std::size_t thread_count_{1};
    std::size_t requested_thread_count_{0};
    std::size_t automatic_thread_capacity_{1};
    bool automatic_thread_count_{true};
    std::string thread_selection_reason_{"unresolved"};
#ifdef COSMO_NBODY_HAS_OPENMP
    int previous_dynamic_{0};
    int previous_max_active_levels_{1};
    int previous_max_threads_{1};
#endif
    bool changed_{false};
};

} // namespace cosmo_nbody::runtime
