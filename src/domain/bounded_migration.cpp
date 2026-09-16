#include "cosmo_nbody/domain/bounded_migration.hpp"

#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody::domain {
namespace {

#ifdef COSMO_NBODY_HAS_MPI
bool valid_payload_kind(ExchangePayloadKind kind) noexcept {
    return kind == ExchangePayloadKind::Migration
        || kind == ExchangePayloadKind::Ghost;
}
#endif

std::size_t u64_to_size(std::uint64_t value, const char* label) {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (value > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds size_t");
        }
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_add_size(
    std::size_t lhs,
    std::size_t rhs,
    const char* label) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return lhs + rhs;
}

struct RoutingLayout {
    std::vector<std::uint64_t> counts;
    std::vector<std::size_t> offsets;
};

template <typename RoutedRecord>
RoutingLayout validate_routing_layout(
    std::span<const RoutedRecord> sorted_routed,
    int size) {
    if (size < 1) {
        throw std::invalid_argument(
            "Bounded particle exchange communicator size must be positive");
    }

    RoutingLayout result;
    result.counts.assign(static_cast<std::size_t>(size), 0);
    result.offsets.assign(static_cast<std::size_t>(size), 0);
    for (const auto& route : sorted_routed) {
        if (route.destination_rank < 0 || route.destination_rank >= size) {
            throw std::invalid_argument(
                "Bounded particle route destination lies outside communicator");
        }
        auto& count = result.counts[static_cast<std::size_t>(
            route.destination_rank)];
        if (count == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "Bounded particle destination record count overflows uint64_t");
        }
        ++count;
    }

    std::size_t cursor = 0;
    for (int destination = 0; destination < size; ++destination) {
        const std::size_t index = static_cast<std::size_t>(destination);
        result.offsets[index] = cursor;
        const std::size_t count = u64_to_size(
            result.counts[index],
            "bounded particle destination record count");
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t routed_index = checked_add_size(
                cursor, local, "bounded particle routing index");
            if (routed_index >= sorted_routed.size()
                || sorted_routed[routed_index].destination_rank != destination) {
                throw std::logic_error(
                    "Bounded particle routing input is not destination-sorted");
            }
            if (local > 0
                && sorted_routed[routed_index - 1].particle.id
                    >= sorted_routed[routed_index].particle.id) {
                throw std::logic_error(
                    "Bounded particle destination IDs are not strictly increasing");
            }
        }
        cursor = checked_add_size(
            cursor, count, "bounded particle routing extent");
    }
    if (cursor != sorted_routed.size()) {
        throw std::logic_error(
            "Bounded particle routing layout did not cover every record");
    }
    return result;
}

template <typename Payload, typename RoutedRecord>
std::vector<Payload> copy_single_rank(
    std::span<const RoutedRecord> sorted_routed) {
    const RoutingLayout routing = validate_routing_layout(sorted_routed, 1);
    (void)routing;
    std::vector<Payload> result(sorted_routed.size());
    for (std::size_t index = 0; index < sorted_routed.size(); ++index) {
        result[index] = sorted_routed[index].particle;
    }
    return result;
}

#ifdef COSMO_NBODY_HAS_MPI
constexpr std::uint64_t mpi_int_record_limit =
    static_cast<std::uint64_t>(std::numeric_limits<int>::max());
static_assert(mpi_int_record_limit > 0);

std::uint64_t rounds_required(std::uint64_t records) noexcept {
    return records == 0
        ? 0
        : (records - 1) / mpi_int_record_limit + 1;
}

std::uint64_t round_begin(std::uint64_t round) {
    if (round > std::numeric_limits<std::uint64_t>::max()
                    / mpi_int_record_limit) {
        throw std::overflow_error(
            "Bounded particle exchange round offset overflows uint64_t");
    }
    return round * mpi_int_record_limit;
}

int round_count(std::uint64_t records, std::uint64_t round) {
    const std::uint64_t begin = round_begin(round);
    if (begin >= records) return 0;
    return static_cast<int>(std::min(
        records - begin,
        mpi_int_record_limit));
}

class MpiDatatypeHandle {
public:
    MpiDatatypeHandle() = default;
    explicit MpiDatatypeHandle(MPI_Datatype value) noexcept
        : value_(value) {}
    ~MpiDatatypeHandle() {
        if (value_ != MPI_DATATYPE_NULL) {
            (void)MPI_Type_free(&value_);
        }
    }

    MpiDatatypeHandle(const MpiDatatypeHandle&) = delete;
    MpiDatatypeHandle& operator=(const MpiDatatypeHandle&) = delete;

    MpiDatatypeHandle(MpiDatatypeHandle&& other) noexcept
        : value_(std::exchange(other.value_, MPI_DATATYPE_NULL)) {}
    MpiDatatypeHandle& operator=(MpiDatatypeHandle&& other) noexcept {
        if (this != &other) {
            if (value_ != MPI_DATATYPE_NULL) {
                (void)MPI_Type_free(&value_);
            }
            value_ = std::exchange(other.value_, MPI_DATATYPE_NULL);
        }
        return *this;
    }

    MPI_Datatype get() const noexcept { return value_; }

private:
    MPI_Datatype value_{MPI_DATATYPE_NULL};
};

MpiDatatypeHandle commit_resized_struct(
    int field_count,
    const int* block_lengths,
    const MPI_Aint* displacements,
    const MPI_Datatype* field_types,
    std::size_t record_extent,
    const char* label) {
    constexpr std::size_t maximum_aint = static_cast<std::size_t>(
        std::numeric_limits<MPI_Aint>::max());
    if (record_extent > maximum_aint) {
        throw std::overflow_error(
            std::string(label) + " extent exceeds MPI_Aint range");
    }

    MPI_Datatype raw = MPI_DATATYPE_NULL;
    if (MPI_Type_create_struct(
            field_count,
            block_lengths,
            displacements,
            field_types,
            &raw) != MPI_SUCCESS
        || raw == MPI_DATATYPE_NULL) {
        throw std::runtime_error(
            std::string("Failed to create ") + label + " MPI datatype");
    }

    MPI_Datatype resized = MPI_DATATYPE_NULL;
    const int resize_status = MPI_Type_create_resized(
        raw, 0, static_cast<MPI_Aint>(record_extent), &resized);
    (void)MPI_Type_free(&raw);
    if (resize_status != MPI_SUCCESS || resized == MPI_DATATYPE_NULL) {
        throw std::runtime_error(
            std::string("Failed to resize ") + label + " MPI datatype");
    }
    if (MPI_Type_commit(&resized) != MPI_SUCCESS) {
        (void)MPI_Type_free(&resized);
        throw std::runtime_error(
            std::string("Failed to commit ") + label + " MPI datatype");
    }
    return MpiDatatypeHandle(resized);
}

MpiDatatypeHandle make_particle_datatype(
    std::size_t particle_base,
    std::size_t record_extent) {
    static_assert(std::is_same_v<core::Real, double>);
    static_assert(std::is_same_v<core::ParticleId, std::uint64_t>);
    static_assert(std::is_standard_layout_v<ExchangeParticle>);
    static_assert(std::is_standard_layout_v<RoutedExchangeParticle>);

    constexpr std::size_t maximum_aint = static_cast<std::size_t>(
        std::numeric_limits<MPI_Aint>::max());
    if (particle_base > maximum_aint || record_extent > maximum_aint) {
        throw std::overflow_error(
            "MPI particle datatype layout exceeds MPI_Aint range");
    }
    const auto displacement = [particle_base](std::size_t field) {
        if (field > maximum_aint
            || particle_base > maximum_aint - field) {
            throw std::overflow_error(
                "MPI particle datatype displacement exceeds MPI_Aint range");
        }
        return static_cast<MPI_Aint>(particle_base + field);
    };
    const std::array<int, 8> block_lengths{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<MPI_Aint, 8> displacements{
        displacement(offsetof(ExchangeParticle, x)),
        displacement(offsetof(ExchangeParticle, y)),
        displacement(offsetof(ExchangeParticle, z)),
        displacement(offsetof(ExchangeParticle, px)),
        displacement(offsetof(ExchangeParticle, py)),
        displacement(offsetof(ExchangeParticle, pz)),
        displacement(offsetof(ExchangeParticle, mass)),
        displacement(offsetof(ExchangeParticle, id))};
    const std::array<MPI_Datatype, 8> field_types{
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_UINT64_T};
    return commit_resized_struct(
        static_cast<int>(block_lengths.size()),
        block_lengths.data(),
        displacements.data(),
        field_types.data(),
        record_extent,
        "full particle");
}

MpiDatatypeHandle make_ghost_datatype(
    std::size_t particle_base,
    std::size_t record_extent) {
    static_assert(std::is_same_v<core::Real, double>);
    static_assert(std::is_same_v<core::ParticleId, std::uint64_t>);
    static_assert(std::is_standard_layout_v<GhostExchangeParticle>);
    static_assert(std::is_standard_layout_v<RoutedGhostExchangeParticle>);

    constexpr std::size_t maximum_aint = static_cast<std::size_t>(
        std::numeric_limits<MPI_Aint>::max());
    if (particle_base > maximum_aint || record_extent > maximum_aint) {
        throw std::overflow_error(
            "MPI ghost datatype layout exceeds MPI_Aint range");
    }
    const auto displacement = [particle_base](std::size_t field) {
        if (field > maximum_aint
            || particle_base > maximum_aint - field) {
            throw std::overflow_error(
                "MPI ghost datatype displacement exceeds MPI_Aint range");
        }
        return static_cast<MPI_Aint>(particle_base + field);
    };
    const std::array<int, 5> block_lengths{1, 1, 1, 1, 1};
    const std::array<MPI_Aint, 5> displacements{
        displacement(offsetof(GhostExchangeParticle, x)),
        displacement(offsetof(GhostExchangeParticle, y)),
        displacement(offsetof(GhostExchangeParticle, z)),
        displacement(offsetof(GhostExchangeParticle, mass)),
        displacement(offsetof(GhostExchangeParticle, id))};
    const std::array<MPI_Datatype, 5> field_types{
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_UINT64_T};
    return commit_resized_struct(
        static_cast<int>(block_lengths.size()),
        block_lengths.data(),
        displacements.data(),
        field_types.data(),
        record_extent,
        "ghost source");
}

MpiDatatypeHandle make_exchange_particle_datatype() {
    return make_particle_datatype(0, sizeof(ExchangeParticle));
}

MpiDatatypeHandle make_routed_particle_datatype() {
    return make_particle_datatype(
        offsetof(RoutedExchangeParticle, particle),
        sizeof(RoutedExchangeParticle));
}

MpiDatatypeHandle make_ghost_exchange_particle_datatype() {
    return make_ghost_datatype(0, sizeof(GhostExchangeParticle));
}

MpiDatatypeHandle make_routed_ghost_particle_datatype() {
    return make_ghost_datatype(
        offsetof(RoutedGhostExchangeParticle, particle),
        sizeof(RoutedGhostExchangeParticle));
}

template <typename Payload, typename RoutedRecord>
std::vector<Payload> exchange_records_bounded(
    std::span<const RoutedRecord> sorted_routed,
    ExchangePayloadKind kind,
    int rank,
    int size,
    MpiDatatypeHandle (*make_payload_type)(),
    MpiDatatypeHandle (*make_routed_type)()) {
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Initialized failed for bounded particle exchange");
    }
    if (!initialized) {
        if (!valid_payload_kind(kind)) {
            throw std::invalid_argument(
                "Bounded particle exchange payload kind is invalid");
        }
        if (rank != 0 || size != 1) {
            throw std::invalid_argument(
                "Uninitialized-MPI bounded particle exchange requires rank 0 of size 1");
        }
        return copy_single_rank<Payload>(sorted_routed);
    }

    runtime::require_active_mpi_main_thread("Bounded particle exchange");

    int actual_rank = 0;
    int actual_size = 0;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &actual_rank) != MPI_SUCCESS
        || MPI_Comm_size(MPI_COMM_WORLD, &actual_size) != MPI_SUCCESS
        || actual_size < 1 || actual_rank < 0 || actual_rank >= actual_size) {
        throw std::runtime_error(
            "Could not resolve MPI_COMM_WORLD topology for bounded particle exchange");
    }

    const int local_topology_invalid =
        !valid_payload_kind(kind)
        || rank != actual_rank
        || size != actual_size;
    int any_topology_invalid = 0;
    if (MPI_Allreduce(
            &local_topology_invalid,
            &any_topology_invalid,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed for bounded particle exchange topology validation");
    }
    if (any_topology_invalid != 0) {
        throw std::invalid_argument(
            "Bounded particle exchange rank, size, or payload kind disagrees with MPI_COMM_WORLD");
    }

    const int local_kind = static_cast<int>(kind);
    int minimum_kind = 0;
    int maximum_kind = 0;
    const int minimum_kind_status = MPI_Allreduce(
        &local_kind, &minimum_kind, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const int maximum_kind_status = MPI_Allreduce(
        &local_kind, &maximum_kind, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (minimum_kind_status != MPI_SUCCESS
        || maximum_kind_status != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Allreduce failed for bounded particle payload-kind agreement");
    }
    if (minimum_kind != maximum_kind) {
        throw std::invalid_argument(
            "Bounded particle payload kind differs across MPI ranks");
    }

    rank = actual_rank;
    size = actual_size;
    if (size == 1) return copy_single_rank<Payload>(sorted_routed);

    RoutingLayout routing;
    std::vector<std::uint64_t> receive_counts;
    std::exception_ptr stage_exception;
    try {
        routing = validate_routing_layout(sorted_routed, size);
        receive_counts.assign(static_cast<std::size_t>(size), 0);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size, "Bounded particle routing/count allocation");

    if (MPI_Alltoall(
            routing.counts.data(), 1, MPI_UINT64_T,
            receive_counts.data(), 1, MPI_UINT64_T,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "MPI_Alltoall failed for bounded particle record counts");
    }

    std::vector<std::size_t> receive_offsets;
    std::vector<Payload> received;
    stage_exception = nullptr;
    try {
        receive_offsets.assign(static_cast<std::size_t>(size), 0);
        std::size_t total_receive = 0;
        for (int source = 0; source < size; ++source) {
            const std::size_t index = static_cast<std::size_t>(source);
            receive_offsets[index] = total_receive;
            total_receive = checked_add_size(
                total_receive,
                u64_to_size(
                    receive_counts[index],
                    "bounded particle receive count"),
                "bounded particle total receive records");
        }
        received.resize(total_receive);
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size, "Bounded particle receive storage allocation");

    const std::size_t self = static_cast<std::size_t>(rank);
    stage_exception = nullptr;
    try {
        if (routing.counts[self] != receive_counts[self]) {
            throw std::logic_error(
                "Bounded particle self record count changed in MPI_Alltoall");
        }
        const std::size_t self_count = u64_to_size(
            routing.counts[self], "bounded particle self record count");
        for (std::size_t index = 0; index < self_count; ++index) {
            received[receive_offsets[self] + index] =
                sorted_routed[routing.offsets[self] + index].particle;
        }
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size, "Bounded particle local copy");

    MpiDatatypeHandle payload_type;
    MpiDatatypeHandle routed_type;
    std::vector<MPI_Request> requests;
    stage_exception = nullptr;
    try {
        payload_type = make_payload_type();
        routed_type = make_routed_type();
        if (size > std::numeric_limits<int>::max() / 2 + 1) {
            throw std::overflow_error(
                "MPI rank count is too large for bounded exchange request storage");
        }
        requests.reserve(static_cast<std::size_t>(2 * (size - 1)));
    } catch (...) {
        stage_exception = std::current_exception();
    }
    runtime::synchronize_mpi_exception(
        stage_exception, size, "Bounded particle MPI datatype setup");

    std::uint64_t local_rounds = 0;
    for (int peer = 0; peer < size; ++peer) {
        if (peer == rank) continue;
        const std::size_t index = static_cast<std::size_t>(peer);
        local_rounds = std::max({
            local_rounds,
            rounds_required(routing.counts[index]),
            rounds_required(receive_counts[index])});
    }

    // MPI_Alltoall already established the exact directed record counts. For
    // every rank pair, the sender's routed count equals the receiver's incoming
    // count, so both endpoints independently execute enough count-limited
    // rounds for that edge. Ranks with no remaining edges need not participate
    // in unrelated peers' later rounds; a global max-round Allreduce would add
    // a synchronization dependency without strengthening message matching.
    const int message_tag = static_cast<int>(kind);
    for (std::uint64_t round = 0; round < local_rounds; ++round) {
        requests.clear();
        const std::uint64_t begin_u64 = round_begin(round);

        int launch_status = MPI_SUCCESS;
        for (int source = 0; source < size; ++source) {
            if (source == rank) continue;
            const std::size_t index = static_cast<std::size_t>(source);
            const int count = round_count(receive_counts[index], round);
            if (count == 0) continue;
            const std::size_t begin = static_cast<std::size_t>(begin_u64);
            MPI_Request request{};
            launch_status = MPI_Irecv(
                received.data() + receive_offsets[index] + begin,
                count,
                payload_type.get(),
                source,
                message_tag,
                MPI_COMM_WORLD,
                &request);
            if (launch_status != MPI_SUCCESS) break;
            requests.push_back(request);
        }

        if (launch_status == MPI_SUCCESS) {
            for (int destination = 0; destination < size; ++destination) {
                if (destination == rank) continue;
                const std::size_t index = static_cast<std::size_t>(destination);
                const int count = round_count(routing.counts[index], round);
                if (count == 0) continue;
                const std::size_t begin = static_cast<std::size_t>(begin_u64);
                MPI_Request request{};
                launch_status = MPI_Isend(
                    sorted_routed.data() + routing.offsets[index] + begin,
                    count,
                    routed_type.get(),
                    destination,
                    message_tag,
                    MPI_COMM_WORLD,
                    &request);
                if (launch_status != MPI_SUCCESS) break;
                requests.push_back(request);
            }
        }

        if (launch_status != MPI_SUCCESS) {
            if (!requests.empty()) {
                (void)MPI_Waitall(
                    static_cast<int>(requests.size()),
                    requests.data(),
                    MPI_STATUSES_IGNORE);
            }
            throw std::runtime_error(
                "Failed to post bounded particle nonblocking transfer");
        }
        if (!requests.empty()
            && MPI_Waitall(
                    static_cast<int>(requests.size()),
                    requests.data(),
                    MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
            throw std::runtime_error(
                "MPI_Waitall failed for bounded particle payload");
        }
    }

    return received;
}
#endif

} // namespace

std::vector<ExchangeParticle> exchange_migration_bounded(
    std::vector<RoutedExchangeParticle>& sorted_routed,
    int rank,
    int size) {
#ifndef COSMO_NBODY_HAS_MPI
    if (rank != 0 || size != 1) {
        throw std::invalid_argument(
            "Non-MPI bounded migration requires rank 0 of size 1");
    }
    const std::span<const RoutedExchangeParticle> routed_view(sorted_routed);
    auto received = copy_single_rank<ExchangeParticle>(routed_view);
#else
    const std::span<const RoutedExchangeParticle> routed_view(sorted_routed);
    auto received = exchange_records_bounded<ExchangeParticle>(
        routed_view,
        ExchangePayloadKind::Migration,
        rank,
        size,
        &make_exchange_particle_datatype,
        &make_routed_particle_datatype);
#endif
    std::vector<RoutedExchangeParticle>().swap(sorted_routed);
    return received;
}

std::vector<GhostExchangeParticle> exchange_routed_ghosts_bounded(
    std::span<const RoutedGhostExchangeParticle> sorted_routed,
    int rank,
    int size) {
#ifndef COSMO_NBODY_HAS_MPI
    if (rank != 0 || size != 1) {
        throw std::invalid_argument(
            "Non-MPI bounded ghost exchange requires rank 0 of size 1");
    }
    return copy_single_rank<GhostExchangeParticle>(sorted_routed);
#else
    return exchange_records_bounded<GhostExchangeParticle>(
        sorted_routed,
        ExchangePayloadKind::Ghost,
        rank,
        size,
        &make_ghost_exchange_particle_datatype,
        &make_routed_ghost_particle_datatype);
#endif
}

} // namespace cosmo_nbody::domain
