#include "cosmo_nbody/domain/mpi_global_id_check.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace domain {

namespace {

static_assert(std::is_same_v<core::ParticleId, std::uint64_t>,
              "MPI stable-ID transport requires uint64 ParticleId");

struct CanonicalIdBucket {
    std::vector<core::ParticleId> ids;
    int communicator_size{1};
};

CanonicalIdBucket canonical_local_bucket(
    std::span<const core::ParticleId> local_ids) {
    CanonicalIdBucket result;
    result.ids.assign(local_ids.begin(), local_ids.end());
    std::sort(result.ids.begin(), result.ids.end());
    if (std::adjacent_find(result.ids.begin(), result.ids.end())
        != result.ids.end()) {
        throw std::invalid_argument(
            "Duplicate stable ParticleID detected in local set");
    }
    return result;
}

#ifdef COSMO_NBODY_HAS_MPI
constexpr int id_exchange_tag = 27183;
constexpr int sorted_id_count_exchange_tag = 27184;
constexpr int sorted_id_payload_exchange_tag = 27185;

static_assert(
    global_id_exchange_chunk_records
        <= static_cast<std::size_t>(INT_MAX),
    "MPI sorted-ID chunks must fit the int count interface");

void require_mpi_call_success(int status, const char* context) {
    if (status == MPI_SUCCESS) return;
    // MPI_ERRORS_RETURN is useful for deterministic validation failures, but
    // ordinary MPI has no ULFM continuation semantics after a communicator or
    // transport error. Abort the job instead of throwing on only one rank and
    // leaving another rank inside a matching collective or Sendrecv.
    (void)MPI_Abort(MPI_COMM_WORLD, status);
    throw std::runtime_error(std::string(context) + " failed");
}

void require_global_id_mpi_main_thread(const char* context) {
    int finalized = 0;
    require_mpi_call_success(
        MPI_Finalized(&finalized),
        "MPI_Finalized during global ParticleID validation");
    if (finalized != 0) {
        throw std::runtime_error(
            std::string(context) + " is unavailable after MPI_Finalize");
    }
    int is_main_thread = 0;
    require_mpi_call_success(
        MPI_Is_thread_main(&is_main_thread),
        "MPI_Is_thread_main during global ParticleID validation");
    if (is_main_thread == 0) {
        throw std::runtime_error(
            std::string(context) + " requires the MPI main thread");
    }
}

std::uint64_t mix_id(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::size_t checked_size_t(
    std::uint64_t value,
    const char* context) {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (value > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                std::string(context) + " exceeds size_t range");
        }
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t checked_u64(
    std::size_t value,
    const char* context) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                std::string(context) + " exceeds uint64_t range");
        }
    }
    return static_cast<std::uint64_t>(value);
}

std::uint64_t checked_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* context) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string(context) + " exceeds uint64 range");
    }
    return lhs + rhs;
}

std::size_t checked_add_size(
    std::size_t lhs,
    std::size_t rhs,
    const char* context) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(
            std::string(context) + " exceeds size_t range");
    }
    return lhs + rhs;
}

void require_collective_success(
    int local_failed,
    const char* context) {
    int any_failed = 0;
    const int status = MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
    require_mpi_call_success(status, "MPI_Allreduce for ParticleID validation");
    if (any_failed != 0) {
        throw std::runtime_error(
            std::string("Global ParticleID validation failed during ")
            + context);
    }
}

bool collective_any_flag(int local_flag, const char* context) {
    int any_flag = 0;
    require_mpi_call_success(
        MPI_Allreduce(
            &local_flag,
            &any_flag,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD),
        context);
    return any_flag != 0;
}

void require_mpi_topology(int& rank, int& size, const char* context) {
    require_mpi_call_success(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank for global ParticleID validation");
    require_mpi_call_success(
        MPI_Comm_size(MPI_COMM_WORLD, &size),
        "MPI_Comm_size for global ParticleID validation");
    if (rank < 0 || size < 1 || rank >= size) {
        (void)MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        throw std::runtime_error(
            std::string(context) + " has invalid MPI topology");
    }
}

void require_sorted_mpi_ids_unique(
    std::span<const core::ParticleId> local_ids,
    int rank,
    int size) {
    int local_order_invalid = 0;
    for (std::size_t index = 1; index < local_ids.size(); ++index) {
        if (local_ids[index - 1] >= local_ids[index]) {
            local_order_invalid = 1;
            break;
        }
    }
    if (collective_any_flag(
            local_order_invalid,
            "MPI_Allreduce for sorted global ParticleID order")) {
        throw std::invalid_argument(
            "Canonical distributed ParticleIDs are not strictly increasing on every rank");
    }

    std::uint64_t local_count = 0;
    int local_count_invalid = 0;
    try {
        local_count = checked_u64(
            local_ids.size(),
            "Canonical local ParticleID population");
    } catch (...) {
        local_count_invalid = 1;
    }
    require_collective_success(
        local_count_invalid,
        "canonical local ParticleID population conversion");

    std::uint64_t global_max_local_count = 0;
    require_mpi_call_success(
        MPI_Allreduce(
            &local_count,
            &global_max_local_count,
            1,
            MPI_UINT64_T,
            MPI_MAX,
            MPI_COMM_WORLD),
        "MPI_Allreduce for sorted global ParticleID maximum population");

    const std::uint64_t chunk = static_cast<std::uint64_t>(
        global_id_exchange_chunk_records);
    const std::size_t receive_capacity = static_cast<std::size_t>(
        std::min(global_max_local_count, chunk));
    std::unique_ptr<core::ParticleId[]> receive_chunk;
    int local_allocation_failed = 0;
    try {
        if (receive_capacity != 0) {
            receive_chunk =
                std::make_unique<core::ParticleId[]>(receive_capacity);
        }
    } catch (...) {
        local_allocation_failed = 1;
    }
    require_collective_success(
        local_allocation_failed,
        "sorted global ParticleID receive-buffer allocation");

    // All allocation and all locally detectable preconditions are complete
    // before entering this protocol. Every rank executes identical ring
    // offsets and obtains the matching source population before streaming.
    // The global round count keeps Sendrecv call counts identical even when
    // the three participating local populations differ for an offset.
    const std::uint64_t rounds = global_max_local_count == 0
        ? 0
        : 1 + (global_max_local_count - 1) / chunk;
    int local_protocol_invalid = 0;
    int local_duplicate = 0;

    for (int offset = 1; offset < size; ++offset) {
        const int destination = offset < size - rank
            ? rank + offset
            : rank - (size - offset);
        const int source = rank >= offset
            ? rank - offset
            : rank + (size - offset);
        std::uint64_t source_count = 0;
        MPI_Status count_status{};
        require_mpi_call_success(
            MPI_Sendrecv(
                &local_count,
                1,
                MPI_UINT64_T,
                destination,
                sorted_id_count_exchange_tag,
                &source_count,
                1,
                MPI_UINT64_T,
                source,
                sorted_id_count_exchange_tag,
                MPI_COMM_WORLD,
                &count_status),
            "MPI_Sendrecv for sorted global ParticleID counts");
        int observed_count_fields = 0;
        require_mpi_call_success(
            MPI_Get_count(
                &count_status,
                MPI_UINT64_T,
                &observed_count_fields),
            "MPI_Get_count for sorted global ParticleID counts");
        if (observed_count_fields != 1) {
            local_protocol_invalid = 1;
        }
        if (source_count > global_max_local_count) {
            local_protocol_invalid = 1;
            source_count = global_max_local_count;
        }

        std::uint64_t sent = 0;
        std::uint64_t received = 0;
        std::size_t local_cursor = 0;
        bool have_source_previous = false;
        core::ParticleId source_previous = 0;

        for (std::uint64_t round = 0; round < rounds; ++round) {
            const std::uint64_t send_remaining = local_count - sent;
            const std::uint64_t receive_remaining = source_count - received;
            const std::uint64_t send_this_u64 =
                std::min(chunk, send_remaining);
            const std::uint64_t receive_this_u64 =
                std::min(chunk, receive_remaining);
            const int send_this = static_cast<int>(send_this_u64);
            const int receive_this = static_cast<int>(receive_this_u64);
            const auto* send_data = send_this == 0
                ? nullptr
                : local_ids.data() + static_cast<std::size_t>(sent);

            MPI_Status payload_status{};
            require_mpi_call_success(
                MPI_Sendrecv(
                    send_data,
                    send_this,
                    MPI_UINT64_T,
                    destination,
                    sorted_id_payload_exchange_tag,
                    receive_this == 0 ? nullptr : receive_chunk.get(),
                    receive_this,
                    MPI_UINT64_T,
                    source,
                    sorted_id_payload_exchange_tag,
                    MPI_COMM_WORLD,
                    &payload_status),
                "MPI_Sendrecv for sorted global ParticleID payload");

            int observed_payload_count = 0;
            require_mpi_call_success(
                MPI_Get_count(
                    &payload_status,
                    MPI_UINT64_T,
                    &observed_payload_count),
                "MPI_Get_count for sorted global ParticleID payload");
            std::size_t observed_payload_size = 0;
            if (observed_payload_count < 0
                || observed_payload_count != receive_this) {
                local_protocol_invalid = 1;
            } else {
                observed_payload_size = static_cast<std::size_t>(
                    observed_payload_count);
            }

            for (std::size_t index = 0;
                 index < observed_payload_size;
                 ++index) {
                const core::ParticleId received_id = receive_chunk[index];
                if (have_source_previous
                    && source_previous >= received_id) {
                    local_protocol_invalid = 1;
                }
                source_previous = received_id;
                have_source_previous = true;

                while (local_cursor < local_ids.size()
                       && local_ids[local_cursor] < received_id) {
                    ++local_cursor;
                }
                if (local_cursor < local_ids.size()
                    && local_ids[local_cursor] == received_id) {
                    local_duplicate = 1;
                }
            }

            sent += send_this_u64;
            received += receive_this_u64;
        }
        if (sent != local_count || received != source_count) {
            local_protocol_invalid = 1;
        }
    }

    const bool any_protocol_invalid = collective_any_flag(
        local_protocol_invalid,
        "MPI_Allreduce for sorted global ParticleID protocol integrity");
    const bool any_duplicate = collective_any_flag(
        local_duplicate,
        "MPI_Allreduce for sorted global ParticleID duplicate status");
    if (any_protocol_invalid) {
        throw std::runtime_error(
            "Sorted global ParticleID exchange violated its exact stream protocol");
    }
    if (any_duplicate) {
        throw std::invalid_argument(
            "Duplicate stable ParticleID detected across MPI ranks");
    }
}

CanonicalIdBucket canonical_mpi_bucket(
    std::span<const core::ParticleId> local_ids,
    int rank,
    int size) {
    // Hash ownership is exact: equal IDs always select the same destination.
    // Counts use uint64 collectives, while payload traffic is split into fixed
    // int-sized chunks. IDs are grouped by owner once in O(N_local + P), then
    // transmitted in deterministic ring order without rescanning the local set
    // for every destination. This avoids MPI_Alltoallv's total INT displacement
    // limit and produces one complete sorted canonical bucket per checker rank.
    std::vector<std::uint64_t> send_counts;
    std::vector<std::uint64_t> recv_counts;
    std::vector<std::uint64_t> recv_offsets;
    std::vector<std::uint64_t> received_from_peer;
    std::vector<std::size_t> send_offsets;
    int local_small_allocation_failed = 0;
    try {
        const std::size_t rank_count = static_cast<std::size_t>(size);
        send_counts.assign(rank_count, 0);
        recv_counts.assign(rank_count, 0);
        recv_offsets.assign(rank_count, 0);
        received_from_peer.assign(rank_count, 0);
        send_offsets.assign(rank_count, 0);
    } catch (...) {
        local_small_allocation_failed = 1;
    }
    require_collective_success(
        local_small_allocation_failed,
        "global ParticleID rank-table allocation");

    int local_count_failed = 0;
    for (const core::ParticleId id : local_ids) {
        const int destination = static_cast<int>(
            mix_id(id) % static_cast<std::uint64_t>(size));
        std::uint64_t& count = send_counts[static_cast<std::size_t>(destination)];
        if (count == std::numeric_limits<std::uint64_t>::max()) {
            local_count_failed = 1;
            break;
        }
        ++count;
    }
    require_collective_success(
        local_count_failed,
        "global ParticleID destination counting");

    require_mpi_call_success(
        MPI_Alltoall(
            send_counts.data(),
            1,
            MPI_UINT64_T,
            recv_counts.data(),
            1,
            MPI_UINT64_T,
            MPI_COMM_WORLD),
        "MPI_Alltoall for global ParticleID uint64 counts");

    std::uint64_t total_recv = 0;
    std::size_t total_send = 0;
    std::uint64_t local_max_peer_count = 0;
    int local_layout_failed = 0;
    try {
        for (int peer = 0; peer < size; ++peer) {
            const std::size_t index = static_cast<std::size_t>(peer);
            send_offsets[index] = total_send;
            total_send = checked_add_size(
                total_send,
                checked_size_t(
                    send_counts[index],
                    "Global ParticleID send population"),
                "Global ParticleID send population");
            recv_offsets[index] = total_recv;
            total_recv = checked_add(
                total_recv,
                recv_counts[index],
                "Global ParticleID receive population");
            local_max_peer_count = std::max({
                local_max_peer_count,
                send_counts[index],
                recv_counts[index]});
        }
        if (total_send != local_ids.size()) {
            throw std::logic_error(
                "Global ParticleID send layout does not cover the local set");
        }
    } catch (...) {
        local_layout_failed = 1;
    }
    require_collective_success(
        local_layout_failed,
        "global ParticleID receive/send layout");

    std::uint64_t global_max_peer_count = 0;
    require_mpi_call_success(
        MPI_Allreduce(
            &local_max_peer_count,
            &global_max_peer_count,
            1,
            MPI_UINT64_T,
            MPI_MAX,
            MPI_COMM_WORLD),
        "MPI_Allreduce for global ParticleID chunk count");
    const std::uint64_t chunk = static_cast<std::uint64_t>(
        global_id_exchange_chunk_records);
    const std::uint64_t rounds = global_max_peer_count == 0
        ? 0
        : 1 + (global_max_peer_count - 1) / chunk;

    CanonicalIdBucket result;
    result.communicator_size = size;
    std::vector<core::ParticleId> send_order;
    std::vector<core::ParticleId> send_chunk;
    std::vector<core::ParticleId> recv_chunk;
    int local_allocation_failed = 0;
    try {
        result.ids.resize(checked_size_t(
            total_recv, "Global ParticleID receive population"));
        send_order.resize(local_ids.size());
        send_chunk.resize(checked_size_t(
            global_id_exchange_chunk_capacity(
                checked_u64(
                    local_ids.size(),
                    "Global ParticleID local send population")),
            "Global ParticleID send chunk capacity"));
        recv_chunk.resize(checked_size_t(
            global_id_exchange_chunk_capacity(total_recv),
            "Global ParticleID receive chunk capacity"));
    } catch (...) {
        local_allocation_failed = 1;
    }
    require_collective_success(
        local_allocation_failed,
        "global ParticleID bounded workspace allocation");

    int local_send_order_failed = 0;
    try {
        for (const core::ParticleId id : local_ids) {
            const std::size_t destination = static_cast<std::size_t>(
                mix_id(id) % static_cast<std::uint64_t>(size));
            std::size_t& cursor = send_offsets[destination];
            if (cursor >= send_order.size()) {
                throw std::logic_error(
                    "Global ParticleID send-order cursor exceeds storage");
            }
            send_order[cursor++] = id;
        }

        std::size_t expected_start = 0;
        for (int peer = 0; peer < size; ++peer) {
            const std::size_t index = static_cast<std::size_t>(peer);
            const std::size_t count = checked_size_t(
                send_counts[index],
                "Global ParticleID send-order peer count");
            const std::size_t expected_end = checked_add_size(
                expected_start,
                count,
                "Global ParticleID send-order extent");
            if (send_offsets[index] != expected_end) {
                throw std::logic_error(
                    "Global ParticleID send-order materialization is incomplete");
            }
            send_offsets[index] = expected_start;
            expected_start = expected_end;
        }
        if (expected_start != send_order.size()) {
            throw std::logic_error(
                "Global ParticleID send-order extent differs from local population");
        }
    } catch (...) {
        local_send_order_failed = 1;
    }
    require_collective_success(
        local_send_order_failed,
        "global ParticleID send-order materialization");

    for (int offset = 0; offset < size; ++offset) {
        const int destination = (rank + offset) % size;
        const int source = (rank - offset + size) % size;
        const std::size_t destination_index =
            static_cast<std::size_t>(destination);
        const std::size_t source_index = static_cast<std::size_t>(source);
        std::uint64_t sent_to_destination = 0;

        for (std::uint64_t round = 0; round < rounds; ++round) {
            const std::uint64_t send_remaining =
                send_counts[destination_index] - sent_to_destination;
            const std::uint64_t recv_remaining =
                recv_counts[source_index] - received_from_peer[source_index];
            const std::uint64_t send_this_u64 = std::min(chunk, send_remaining);
            const std::uint64_t recv_this_u64 = std::min(chunk, recv_remaining);
            const int send_this = static_cast<int>(send_this_u64);
            const int recv_this = static_cast<int>(recv_this_u64);

            const std::size_t send_begin = send_offsets[destination_index]
                + static_cast<std::size_t>(sent_to_destination);
            std::copy_n(
                send_order.begin() + static_cast<std::ptrdiff_t>(send_begin),
                static_cast<std::size_t>(send_this),
                send_chunk.begin());

            require_mpi_call_success(
                MPI_Sendrecv(
                    send_this == 0 ? nullptr : send_chunk.data(),
                    send_this,
                    MPI_UINT64_T,
                    destination,
                    id_exchange_tag,
                    recv_this == 0 ? nullptr : recv_chunk.data(),
                    recv_this,
                    MPI_UINT64_T,
                    source,
                    id_exchange_tag,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE),
                "MPI_Sendrecv for bounded global ParticleID exchange");

            const std::uint64_t output_offset = checked_add(
                recv_offsets[source_index],
                received_from_peer[source_index],
                "Global ParticleID receive offset");
            const std::size_t output_index = checked_size_t(
                output_offset, "Global ParticleID receive offset");
            std::copy_n(
                recv_chunk.begin(),
                static_cast<std::size_t>(recv_this),
                result.ids.begin() + static_cast<std::ptrdiff_t>(output_index));
            sent_to_destination += send_this_u64;
            received_from_peer[source_index] += recv_this_u64;
        }

        const int local_count_mismatch =
            sent_to_destination != send_counts[destination_index]
                || received_from_peer[source_index] != recv_counts[source_index]
            ? 1 : 0;
        require_collective_success(
            local_count_mismatch,
            "global ParticleID bounded exchange accounting");
    }

    std::sort(result.ids.begin(), result.ids.end());
    const int local_duplicate =
        std::adjacent_find(result.ids.begin(), result.ids.end())
            != result.ids.end()
        ? 1 : 0;
    int any_duplicate = 0;
    require_mpi_call_success(
        MPI_Allreduce(
            &local_duplicate,
            &any_duplicate,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD),
        "MPI_Allreduce for global ParticleID duplicate status");
    if (any_duplicate != 0) {
        throw std::invalid_argument(
            "Duplicate stable ParticleID detected across MPI ranks");
    }
    return result;
}
#endif

CanonicalIdBucket canonical_global_bucket(
    std::span<const core::ParticleId> local_ids) {
#ifndef COSMO_NBODY_HAS_MPI
    return canonical_local_bucket(local_ids);
#else
    int initialized = 0;
    require_mpi_call_success(
        MPI_Initialized(&initialized),
        "MPI_Initialized during global ParticleID validation");
    if (!initialized) return canonical_local_bucket(local_ids);

    require_global_id_mpi_main_thread("Global ParticleID validation");

    int rank = 0;
    int size = 0;
    require_mpi_topology(rank, size, "Global ParticleID validation");
    if (size == 1) return canonical_local_bucket(local_ids);
    return canonical_mpi_bucket(local_ids, rank, size);
#endif
}

} // namespace

GlobalParticleIdSetSnapshot capture_global_particle_id_set(
    std::span<const core::ParticleId> local_ids) {
    auto canonical = canonical_global_bucket(local_ids);
    return {
        std::move(canonical.ids),
        canonical.communicator_size};
}

void require_global_particle_id_set_preserved(
    const GlobalParticleIdSetSnapshot& before,
    std::span<const core::ParticleId> local_after_ids) {
    auto after = canonical_global_bucket(local_after_ids);
    const int local_mismatch =
        before.communicator_size != after.communicator_size
        || before.canonical_local_bucket != after.ids
        ? 1 : 0;
#ifdef COSMO_NBODY_HAS_MPI
    if (after.communicator_size > 1) {
        int any_mismatch = 0;
        require_mpi_call_success(
            MPI_Allreduce(
                &local_mismatch,
                &any_mismatch,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD),
            "MPI_Allreduce for global ParticleID set preservation");
        if (any_mismatch != 0) {
            throw std::runtime_error(
                "Global stable ParticleID set changed across distributed transport");
        }
        return;
    }
#endif
    if (local_mismatch != 0) {
        throw std::runtime_error(
            "Stable ParticleID set changed across local transport");
    }
}

void require_globally_unique_particle_ids(
    std::span<const core::ParticleId> local_ids) {
    (void)capture_global_particle_id_set(local_ids);
}

void require_globally_unique_sorted_particle_ids(
    std::span<const core::ParticleId> local_ids) {
#ifndef COSMO_NBODY_HAS_MPI
    for (std::size_t index = 1; index < local_ids.size(); ++index) {
        if (local_ids[index - 1] >= local_ids[index]) {
            throw std::invalid_argument(
                "Canonical local ParticleIDs are not strictly increasing");
        }
    }
#else
    int initialized = 0;
    require_mpi_call_success(
        MPI_Initialized(&initialized),
        "MPI_Initialized during sorted global ParticleID validation");
    if (!initialized) {
        for (std::size_t index = 1; index < local_ids.size(); ++index) {
            if (local_ids[index - 1] >= local_ids[index]) {
                throw std::invalid_argument(
                    "Canonical local ParticleIDs are not strictly increasing");
            }
        }
        return;
    }

    require_global_id_mpi_main_thread(
        "Sorted global ParticleID validation");
    int rank = 0;
    int size = 0;
    require_mpi_topology(
        rank, size, "Sorted global ParticleID validation");
    if (size == 1) {
        for (std::size_t index = 1; index < local_ids.size(); ++index) {
            if (local_ids[index - 1] >= local_ids[index]) {
                throw std::invalid_argument(
                    "Canonical local ParticleIDs are not strictly increasing");
            }
        }
        return;
    }
    require_sorted_mpi_ids_unique(local_ids, rank, size);
#endif
}

} // namespace domain
} // namespace cosmo_nbody
