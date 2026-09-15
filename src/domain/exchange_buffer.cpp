#include "cosmo_nbody/domain/exchange_buffer.hpp"

#include <algorithm>
#include <stdexcept>

namespace cosmo_nbody {
namespace domain {
namespace {

template <typename RoutedRecord>
void sort_route_records(std::span<RoutedRecord> routed) {
    for (const auto& item : routed) {
        if (item.destination_rank < 0) {
            throw std::invalid_argument(
                "Exchange destination rank must be non-negative");
        }
    }
    std::sort(routed.begin(), routed.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.destination_rank != rhs.destination_rank) {
            return lhs.destination_rank < rhs.destination_rank;
        }
        return lhs.particle.id < rhs.particle.id;
    });
    for (std::size_t i = 1; i < routed.size(); ++i) {
        if (routed[i - 1].destination_rank == routed[i].destination_rank
            && routed[i - 1].particle.id == routed[i].particle.id) {
            throw std::invalid_argument(
                "Duplicate (destination rank, particle ID) in exchange route");
        }
    }
}

} // namespace

void ExchangeBuffer::sort_routed_in_place(
    std::span<RoutedExchangeParticle> routed) {
    sort_route_records(routed);
}

void ExchangeBuffer::sort_routed_ghosts_in_place(
    std::span<RoutedGhostExchangeParticle> routed) {
    sort_route_records(routed);
}

} // namespace domain
} // namespace cosmo_nbody
