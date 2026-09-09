#include "cosmo_nbody/math/cic_mass_conservation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::math {
namespace {

void require_positive_finite_pair(
    core::Real lhs,
    core::Real rhs,
    const char* context) {
    if (!std::isfinite(lhs) || lhs <= 0.0
        || !std::isfinite(rhs) || rhs <= 0.0) {
        throw std::runtime_error(
            std::string(context) + " requires finite positive masses");
    }
}

} // namespace

void require_cic_mass_conservation(
    core::Real particle_mass,
    core::Real deposited_mass,
    std::uint64_t particle_count) {
    require_positive_finite_pair(
        particle_mass, deposited_mass, "CIC mass conservation");
    if (particle_count == 0) {
        throw std::invalid_argument(
            "CIC mass conservation requires a positive particle count");
    }

    // Conservative binary64 forward-error bound. One CIC record forms eight
    // trilinear weights and accumulates eight non-negative cell contributions.
    // Thirty-two rounded operations per particle bounds the weight products and
    // a sequentialized worst-case view of the cell additions; exact positive
    // sums remove any additional MPI reduction-order error from this comparison.
    const long double epsilon =
        static_cast<long double>(std::numeric_limits<core::Real>::epsilon());
    const long double rounded_operations =
        32.0L * static_cast<long double>(particle_count) + 1.0L;
    const long double nu = rounded_operations * epsilon;
    if (!std::isfinite(nu) || nu >= 0.25L) {
        throw std::overflow_error(
            "CIC mass conservation forward-error bound is not informative at this particle count");
    }
    const long double gamma = nu / (1.0L - nu);
    const long double scale = std::max(
        static_cast<long double>(particle_mass),
        static_cast<long double>(deposited_mass));
    const long double tolerance =
        (gamma + 64.0L * epsilon) * scale;
    const long double difference = std::abs(
        static_cast<long double>(deposited_mass)
        - static_cast<long double>(particle_mass));
    if (difference > tolerance) {
        throw std::runtime_error(
            "Distributed CIC deposited mass violates the binary64 forward-error bound");
    }
}

void require_cic_transport_conservation(
    core::Real pre_exchange_mass,
    core::Real post_exchange_mass) {
    require_positive_finite_pair(
        pre_exchange_mass,
        post_exchange_mass,
        "CIC transport conservation");

    // The pre-exchange representation already contains every rounded local-cell
    // and outgoing-plane value. The exchange only adds non-negative received
    // values to at most two boundary planes. Even when one local plane receives
    // both neighbours, the total absolute first-order addition error is bounded
    // by a small multiple of epsilon times total mass. The factor 64 also covers
    // the two final exact-accumulator value() roundings without scaling with the
    // particle population; a dropped or duplicated finite plane contribution is
    // therefore not hidden by the broader deposition bound.
    const long double epsilon =
        static_cast<long double>(std::numeric_limits<core::Real>::epsilon());
    const long double scale = std::max(
        static_cast<long double>(pre_exchange_mass),
        static_cast<long double>(post_exchange_mass));
    const long double tolerance = 64.0L * epsilon * scale;
    const long double difference = std::abs(
        static_cast<long double>(post_exchange_mass)
        - static_cast<long double>(pre_exchange_mass));
    if (difference > tolerance) {
        throw std::runtime_error(
            "Distributed CIC plane transport changed the globally represented mass");
    }
}

} // namespace cosmo_nbody::math
