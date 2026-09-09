#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/core/portable_bit_cast.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmo_nbody::analysis {

// Generic same-epoch particle-state measurement. This object deliberately
// carries no study, execution-topology, convergence, or pass/fail meaning.
// A study layer may decide why two snapshots were selected after preserving the
// native snapshot identities emitted by hyowon_analyze.
struct PhaseSpaceCrossSummary {
    std::uint64_t particle_count{0};
    core::Real position_rms_mean_spacing{0.0};
    core::Real position_max_mean_spacing{0.0};
    core::Real canonical_momentum_rms_a_km_s{0.0};
    core::Real canonical_momentum_max_a_km_s{0.0};
    core::Real velocity_rms_km_s{0.0};
    core::Real velocity_max_km_s{0.0};
    // Stable-ID matched position and canonical-momentum components have exactly
    // the same binary64 object representations. Mass equality is admitted
    // separately below before this summary is returned.
    bool exact_binary64_phase_space_equal{false};
    std::size_t matching_workspace_bytes{0};
};

namespace detail {

inline void require_phase_space_store_shape(
    const core::ParticleStore& particles,
    const char* label) {
    const std::size_t n = particles.num_owned_particles();
    if (particles.get_ids().size() != n
        || particles.get_positions_x().size() != n
        || particles.get_positions_y().size() != n
        || particles.get_positions_z().size() != n
        || particles.get_momenta_x().size() != n
        || particles.get_momenta_y().size() != n
        || particles.get_momenta_z().size() != n) {
        throw std::invalid_argument(
            std::string(label) + " phase-space storage is incomplete");
    }
}

inline bool binary64_bits_equal(core::Real lhs, core::Real rhs) noexcept {
    static_assert(sizeof(core::Real) == sizeof(std::uint64_t));
    return core::portable_bit_cast<std::uint64_t>(lhs)
        == core::portable_bit_cast<std::uint64_t>(rhs);
}

struct DifferenceMagnitude {
    core::Real position_mean_spacing{0.0};
    core::Real canonical_momentum_a_km_s{0.0};
};

inline DifferenceMagnitude difference_magnitude(
    const core::ParticleStore& reference,
    std::size_t reference_index,
    const core::ParticleStore& candidate,
    std::size_t candidate_index,
    core::Real box_size,
    core::Real mean_spacing) {
    const core::Real dx = math::minimum_image_displacement(
        reference.get_positions_x()[reference_index],
        candidate.get_positions_x()[candidate_index],
        box_size) / mean_spacing;
    const core::Real dy = math::minimum_image_displacement(
        reference.get_positions_y()[reference_index],
        candidate.get_positions_y()[candidate_index],
        box_size) / mean_spacing;
    const core::Real dz = math::minimum_image_displacement(
        reference.get_positions_z()[reference_index],
        candidate.get_positions_z()[candidate_index],
        box_size) / mean_spacing;
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
        throw std::invalid_argument(
            "phase-space comparison periodic position difference is non-finite");
    }

    const core::Real dpx = candidate.get_momenta_x()[candidate_index]
        - reference.get_momenta_x()[reference_index];
    const core::Real dpy = candidate.get_momenta_y()[candidate_index]
        - reference.get_momenta_y()[reference_index];
    const core::Real dpz = candidate.get_momenta_z()[candidate_index]
        - reference.get_momenta_z()[reference_index];
    if (!std::isfinite(dpx) || !std::isfinite(dpy) || !std::isfinite(dpz)) {
        throw std::invalid_argument(
            "phase-space comparison canonical momentum difference is non-finite");
    }

    const core::Real dr = std::hypot(std::hypot(dx, dy), dz);
    const core::Real dp = std::hypot(std::hypot(dpx, dpy), dpz);
    if (!std::isfinite(dr) || !std::isfinite(dp)) {
        throw std::invalid_argument(
            "phase-space comparison magnitude is non-finite");
    }
    return {dr, dp};
}

inline void require_unique_ids_sorted_by_index(
    const core::ParticleStore& particles,
    const std::vector<std::size_t>& order,
    const char* label) {
    const auto ids = particles.get_ids();
    for (std::size_t i = 1; i < order.size(); ++i) {
        if (ids[order[i - 1]] == ids[order[i]]) {
            throw std::invalid_argument(
                std::string(label)
                + " contains duplicate stable ParticleIDs");
        }
    }
}

} // namespace detail

inline PhaseSpaceCrossSummary compare_phase_space_by_stable_id(
    const core::ParticleStore& reference,
    core::Real reference_a,
    const core::ParticleStore& candidate,
    core::Real candidate_a,
    core::Real box_size_Mpc_h,
    std::uint64_t particles_per_dimension,
    std::size_t matching_workspace_limit_bytes = 0) {
    detail::require_phase_space_store_shape(reference, "reference");
    detail::require_phase_space_store_shape(candidate, "candidate");
    const std::size_t n = reference.num_owned_particles();
    if (n == 0 || candidate.num_owned_particles() != n) {
        throw std::invalid_argument(
            "phase-space comparison requires equal non-empty particle populations");
    }
    if (n > static_cast<std::size_t>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error(
            "phase-space comparison particle count exceeds uint64_t");
    }
    if (!std::isfinite(reference_a) || reference_a <= 0.0
        || !std::isfinite(candidate_a) || candidate_a <= 0.0
        || reference_a != candidate_a
        || !std::isfinite(box_size_Mpc_h) || box_size_Mpc_h <= 0.0
        || particles_per_dimension == 0) {
        throw std::invalid_argument(
            "phase-space comparison requires one exact positive epoch and valid periodic coordinates");
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(std::size_t)) {
        throw std::overflow_error(
            "phase-space comparison matching workspace size overflows size_t");
    }
    const std::size_t matching_workspace_bytes = n * sizeof(std::size_t);
    if (matching_workspace_limit_bytes != 0
        && matching_workspace_bytes > matching_workspace_limit_bytes) {
        throw std::invalid_argument(
            "phase-space comparison stable-ID index exceeds the analysis workspace budget");
    }
    const core::Real mean_spacing = box_size_Mpc_h
        / static_cast<core::Real>(particles_per_dimension);
    if (!std::isfinite(mean_spacing) || mean_spacing <= 0.0) {
        throw std::invalid_argument(
            "phase-space comparison mean spacing is invalid");
    }

    const auto reference_ids = reference.get_ids();
    const auto candidate_ids = candidate.get_ids();
    std::vector<std::size_t> reference_order(n);
    std::iota(
        reference_order.begin(), reference_order.end(), std::size_t{0});

    // Verify candidate uniqueness without retaining a second O(N) index array.
    std::sort(
        reference_order.begin(), reference_order.end(),
        [&](std::size_t left, std::size_t right) {
            if (candidate_ids[left] != candidate_ids[right]) {
                return candidate_ids[left] < candidate_ids[right];
            }
            return left < right;
        });
    detail::require_unique_ids_sorted_by_index(
        candidate, reference_order, "candidate");

    // Reuse the same O(N) index buffer for reference stable-ID lookup.
    std::sort(
        reference_order.begin(), reference_order.end(),
        [&](std::size_t left, std::size_t right) {
            if (reference_ids[left] != reference_ids[right]) {
                return reference_ids[left] < reference_ids[right];
            }
            return left < right;
        });
    detail::require_unique_ids_sorted_by_index(
        reference, reference_order, "reference");

    const auto reference_index_for = [&](core::ParticleId id) -> std::size_t {
        const auto iterator = std::lower_bound(
            reference_order.begin(), reference_order.end(), id,
            [&](std::size_t index, core::ParticleId target) {
                return reference_ids[index] < target;
            });
        if (iterator == reference_order.end()
            || reference_ids[*iterator] != id) {
            throw std::invalid_argument(
                "phase-space comparison stable ParticleID sets differ");
        }
        return *iterator;
    };

    core::Real position_max = 0.0;
    core::Real canonical_momentum_max = 0.0;
    bool exact_binary64_phase_space_equal = true;
    for (std::size_t candidate_index = 0;
         candidate_index < n;
         ++candidate_index) {
        const std::size_t reference_index =
            reference_index_for(candidate_ids[candidate_index]);
        const core::Real reference_mass = reference.mass_at(reference_index);
        const core::Real candidate_mass = candidate.mass_at(candidate_index);
        if (!std::isfinite(reference_mass) || reference_mass <= 0.0
            || !std::isfinite(candidate_mass) || candidate_mass <= 0.0) {
            throw std::invalid_argument(
                "phase-space comparison requires finite positive particle masses");
        }
        if (!detail::binary64_bits_equal(reference_mass, candidate_mass)) {
            throw std::invalid_argument(
                "phase-space comparison requires exact binary64 particle-mass matches");
        }

        exact_binary64_phase_space_equal = exact_binary64_phase_space_equal
            && detail::binary64_bits_equal(
                reference.get_positions_x()[reference_index],
                candidate.get_positions_x()[candidate_index])
            && detail::binary64_bits_equal(
                reference.get_positions_y()[reference_index],
                candidate.get_positions_y()[candidate_index])
            && detail::binary64_bits_equal(
                reference.get_positions_z()[reference_index],
                candidate.get_positions_z()[candidate_index])
            && detail::binary64_bits_equal(
                reference.get_momenta_x()[reference_index],
                candidate.get_momenta_x()[candidate_index])
            && detail::binary64_bits_equal(
                reference.get_momenta_y()[reference_index],
                candidate.get_momenta_y()[candidate_index])
            && detail::binary64_bits_equal(
                reference.get_momenta_z()[reference_index],
                candidate.get_momenta_z()[candidate_index]);

        const auto difference = detail::difference_magnitude(
            reference, reference_index,
            candidate, candidate_index,
            box_size_Mpc_h, mean_spacing);
        position_max =
            std::max(position_max, difference.position_mean_spacing);
        canonical_momentum_max = std::max(
            canonical_momentum_max,
            difference.canonical_momentum_a_km_s);
    }

    math::ExactPositiveDoubleSum position_scaled_square_sum;
    math::ExactPositiveDoubleSum canonical_momentum_scaled_square_sum;
    for (std::size_t candidate_index = 0;
         candidate_index < n;
         ++candidate_index) {
        const std::size_t reference_index =
            reference_index_for(candidate_ids[candidate_index]);
        const auto difference = detail::difference_magnitude(
            reference, reference_index,
            candidate, candidate_index,
            box_size_Mpc_h, mean_spacing);
        if (position_max > 0.0) {
            const core::Real scaled =
                difference.position_mean_spacing / position_max;
            position_scaled_square_sum.add(scaled * scaled);
        }
        if (canonical_momentum_max > 0.0) {
            const core::Real scaled =
                difference.canonical_momentum_a_km_s
                / canonical_momentum_max;
            canonical_momentum_scaled_square_sum.add(scaled * scaled);
        }
    }

    const core::Real inverse_count =
        core::Real{1.0} / static_cast<core::Real>(n);
    const core::Real position_rms = position_max == 0.0
        ? 0.0
        : position_max
            * std::sqrt(position_scaled_square_sum.value() * inverse_count);
    const core::Real canonical_momentum_rms = canonical_momentum_max == 0.0
        ? 0.0
        : canonical_momentum_max
            * std::sqrt(
                canonical_momentum_scaled_square_sum.value() * inverse_count);
    const core::Real velocity_rms = canonical_momentum_rms / reference_a;
    const core::Real velocity_max = canonical_momentum_max / reference_a;
    if (!std::isfinite(position_rms)
        || !std::isfinite(canonical_momentum_rms)
        || !std::isfinite(velocity_rms)
        || !std::isfinite(velocity_max)) {
        throw std::overflow_error(
            "phase-space comparison summary is not finite");
    }
    return {
        static_cast<std::uint64_t>(n),
        position_rms,
        position_max,
        canonical_momentum_rms,
        canonical_momentum_max,
        velocity_rms,
        velocity_max,
        exact_binary64_phase_space_equal,
        matching_workspace_bytes,
    };
}

} // namespace cosmo_nbody::analysis
