#pragma once

#include "cosmo_nbody/config/simulation_parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace cosmo_nbody {
namespace runtime {

// Observed execution topology only; unavailable/not-applicable counts are zero.
struct RuntimeTopologyDiagnostics {
    std::uint64_t rank_count{1};
    std::uint64_t shared_memory_domain_count{1};
    std::uint64_t local_size_min{1};
    std::uint64_t local_size_max{1};
    std::uint64_t effective_threads_min{1};
    std::uint64_t effective_threads_max{1};
    std::uint64_t visible_cpu_count_min{0};
    std::uint64_t visible_cpu_count_max{0};
    std::uint64_t automatic_thread_ceiling_min{0};
    std::uint64_t automatic_thread_ceiling_max{0};
    std::uint64_t shared_unbound_affinity_rank_count{0};
};

// Local process-memory telemetry; unavailable platforms report available=false.
struct RuntimeMemoryObservation {
    bool available{false};
    std::uint64_t current_rss_bytes{0};
    std::uint64_t peak_rss_bytes{0};
};

// Lease on process-wide runtime state; MPI/OpenMP initialization and locality
// resolution occur once and are released with the final lease. Transport/NIC
// selection remains owned by MPI and is not inferred here.
class RuntimeContext {
public:
    explicit RuntimeContext(const config::RuntimeParams& params);
    ~RuntimeContext();

    RuntimeContext(const RuntimeContext&) = delete;
    RuntimeContext& operator=(const RuntimeContext&) = delete;
    RuntimeContext(RuntimeContext&&) = delete;
    RuntimeContext& operator=(RuntimeContext&&) = delete;

    int rank() const noexcept;
    int size() const noexcept;
    int local_rank() const noexcept;
    int local_size() const noexcept;
    std::size_t thread_count() const noexcept;
    std::size_t visible_cpu_count() const noexcept;
    std::size_t automatic_thread_ceiling() const noexcept;
    bool shared_unbound_cpu_affinity() const noexcept;
    bool mpi_active() const noexcept;
    std::string_view thread_selection_reason() const noexcept;
    std::string_view processor_name() const noexcept;
    std::string_view mpi_library_version() const noexcept;

    // Non-collective telemetry; observation failure never fails the run.
    RuntimeMemoryObservation observe_process_memory() const noexcept;

    // Collective exact-integer topology aggregation with no policy threshold.
    RuntimeTopologyDiagnostics collect_topology_diagnostics() const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace runtime
} // namespace cosmo_nbody
