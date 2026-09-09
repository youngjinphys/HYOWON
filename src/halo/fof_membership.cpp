#include "cosmo_nbody/halo/fof_membership.hpp"

#include "cosmo_nbody/core/id_uniqueness.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cosmo_nbody::halo {
namespace {

bool cube_fits_within(std::size_t value, std::size_t limit) {
    if (value == 0) return true;
    if (value > limit / value) return false;
    const std::size_t square = value * value;
    return value <= limit / square;
}

std::size_t checked_cube(std::size_t value) {
    if (!cube_fits_within(value, std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "FoF membership head-grid size overflows size_t");
    }
    return value * value * value;
}

std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs,
    const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

std::uint64_t particle_count_u64(std::size_t particle_count) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (particle_count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "FoF membership particle count exceeds uint64 planning range");
        }
    }
    return static_cast<std::uint64_t>(particle_count);
}

std::size_t narrow_link_index_bytes(std::size_t particle_count) {
    if (particle_count == 0) {
        throw std::invalid_argument(
            "FoF membership grid planning requires at least one particle");
    }
    const std::uint64_t count = particle_count_u64(particle_count);
    constexpr std::uint64_t int32_capacity =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())
        + 1ULL;
    constexpr std::uint64_t int64_capacity =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        + 1ULL;
    if (count <= int32_capacity) return sizeof(std::int32_t);
    if (count <= int64_capacity) return sizeof(std::int64_t);
    throw std::overflow_error(
        "FoF membership particle count exceeds signed link-index range");
}

std::size_t narrow_disjoint_set_index_bytes(std::size_t particle_count) {
    if (particle_count == 0) {
        throw std::invalid_argument(
            "FoF membership grid planning requires at least one particle");
    }
    const std::uint64_t count = particle_count_u64(particle_count);
    if (count <= static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        return sizeof(std::uint32_t);
    }
    return sizeof(std::uint64_t);
}

template <typename Index>
struct DisjointSet {
    static_assert(std::is_integral_v<Index> && std::is_unsigned_v<Index>);

    std::vector<Index> parent;
    std::vector<Index> component_sizes;

    explicit DisjointSet(std::size_t count)
        : parent(count), component_sizes(count, Index{1}) {
        if (particle_count_u64(count) > static_cast<std::uint64_t>(
                std::numeric_limits<Index>::max())) {
            throw std::overflow_error(
                "FoF membership disjoint-set index cannot represent count");
        }
        for (std::size_t particle = 0; particle < count; ++particle) {
            parent[particle] = static_cast<Index>(particle);
        }
    }

    Index find(Index particle) {
        const std::size_t offset = static_cast<std::size_t>(particle);
        if (parent[offset] != particle) {
            parent[offset] = find(parent[offset]);
        }
        return parent[offset];
    }

    void unite(Index lhs, Index rhs) {
        Index lhs_root = find(lhs);
        Index rhs_root = find(rhs);
        if (lhs_root == rhs_root) return;

        const std::size_t lhs_offset = static_cast<std::size_t>(lhs_root);
        const std::size_t rhs_offset = static_cast<std::size_t>(rhs_root);
        if (component_sizes[lhs_offset] < component_sizes[rhs_offset]) {
            parent[lhs_offset] = rhs_root;
            component_sizes[rhs_offset] += component_sizes[lhs_offset];
        } else {
            parent[rhs_offset] = lhs_root;
            component_sizes[lhs_offset] += component_sizes[rhs_offset];
        }
    }

    std::size_t component_size(Index root) const noexcept {
        return static_cast<std::size_t>(
            component_sizes[static_cast<std::size_t>(root)]);
    }
};

struct AxisCellList {
    std::array<std::size_t, 3> cells{};
    std::size_t count{0};

    void add_unique(std::size_t cell) {
        for (std::size_t index = 0; index < count; ++index) {
            if (cells[index] == cell) return;
        }
        if (count >= cells.size()) {
            throw std::logic_error(
                "FoF membership axis-neighbor list exceeded three cells");
        }
        cells[count++] = cell;
    }
};

} // namespace

FoFMembershipGridPlan FoFMembershipFinder::grid_plan_for_particle_count(
    std::size_t particle_count) const {
    if (particle_count == 0) {
        throw std::invalid_argument(
            "FoF membership grid planning requires at least one particle");
    }

    const core::Real box_size = config_.get_box().L;
    const core::Real linking_length = linking_length_b_ * config_.d_mean();
    if (!std::isfinite(box_size) || box_size <= 0.0
        || !std::isfinite(linking_length) || linking_length <= 0.0
        || linking_length >= box_size) {
        throw std::invalid_argument(
            "FoF membership linking length must be finite, positive, and smaller than the box");
    }

    const long double desired_real = std::floor(
        static_cast<long double>(box_size)
        / static_cast<long double>(linking_length));
    std::size_t desired_cells_1d = 1;
    if (desired_real >= static_cast<long double>(
            std::numeric_limits<std::size_t>::max())) {
        desired_cells_1d = std::numeric_limits<std::size_t>::max();
    } else if (desired_real > 1.0L) {
        desired_cells_1d = static_cast<std::size_t>(desired_real);
    }

    std::size_t memory_cap_1d = static_cast<std::size_t>(
        std::floor(std::cbrt(static_cast<long double>(particle_count))));
    memory_cap_1d = std::max<std::size_t>(1, memory_cap_1d);
    while (memory_cap_1d > 1
           && !cube_fits_within(memory_cap_1d, particle_count)) {
        --memory_cap_1d;
    }
    while (memory_cap_1d < particle_count
           && cube_fits_within(memory_cap_1d + 1, particle_count)) {
        ++memory_cap_1d;
    }

    std::size_t cells_per_dimension =
        std::min(desired_cells_1d, memory_cap_1d);
    while (cells_per_dimension > 1
           && box_size / static_cast<core::Real>(cells_per_dimension)
               < linking_length) {
        --cells_per_dimension;
    }

    const core::Real cell_size =
        box_size / static_cast<core::Real>(cells_per_dimension);
    if (!std::isfinite(cell_size) || cell_size < linking_length) {
        throw std::logic_error(
            "FoF membership grid violates the neighbor-stencil requirement");
    }

    const std::size_t head_cell_count = checked_cube(cells_per_dimension);
    if (head_cell_count > particle_count) {
        throw std::logic_error(
            "FoF membership head grid exceeds one cell per particle");
    }

    FoFMembershipGridPlan plan;
    plan.particle_count = particle_count;
    plan.cells_per_dimension = cells_per_dimension;
    plan.head_cell_count = head_cell_count;
    plan.link_index_bytes = narrow_link_index_bytes(particle_count);
    plan.disjoint_set_index_bytes =
        narrow_disjoint_set_index_bytes(particle_count);
    plan.estimated_head_bytes = checked_multiply(
        head_cell_count,
        plan.link_index_bytes,
        "FoF membership head-grid byte estimate overflows size_t");
    plan.estimated_link_bytes = checked_multiply(
        particle_count,
        plan.link_index_bytes,
        "FoF membership linked-list byte estimate overflows size_t");
    const std::size_t one_dsu_array = checked_multiply(
        particle_count,
        plan.disjoint_set_index_bytes,
        "FoF membership disjoint-set byte estimate overflows size_t");
    plan.estimated_disjoint_set_bytes = checked_multiply(
        one_dsu_array,
        2,
        "FoF membership disjoint-set byte estimate overflows size_t");
    plan.linking_length = linking_length;
    plan.cell_size = cell_size;
    return plan;
}

std::vector<FoFMembership> FoFMembershipFinder::find_memberships(
    const core::ParticleStore& particles) const {
    if (particles.num_ghost_particles() != 0) {
        throw std::runtime_error(
            "FoF membership finding does not support ghost particles; run serial offline analysis on a collective snapshot");
    }

    const std::size_t particle_count = particles.num_owned_particles();
    if (particles.get_positions_x().size() < particle_count
        || particles.get_positions_y().size() < particle_count
        || particles.get_positions_z().size() < particle_count
        || particles.get_ids().size() < particle_count) {
        throw std::logic_error(
            "FoF membership position or ID storage is shorter than the owned count");
    }

    const auto position_x = particles.get_positions_x().first(particle_count);
    const auto position_y = particles.get_positions_y().first(particle_count);
    const auto position_z = particles.get_positions_z().first(particle_count);
    const auto particle_ids = particles.get_ids().first(particle_count);
    for (std::size_t particle = 0; particle < particle_count; ++particle) {
        if (!std::isfinite(position_x[particle])
            || !std::isfinite(position_y[particle])
            || !std::isfinite(position_z[particle])) {
            throw std::invalid_argument(
                "FoF membership requires finite particle positions");
        }
    }
    core::require_unique_particle_ids_bounded(
        particle_ids, "FoF membership stable ParticleIDs");

    std::vector<FoFMembership> memberships;
    if (particle_count < min_particles_) return memberships;

    const FoFMembershipGridPlan plan =
        grid_plan_for_particle_count(particle_count);
    const core::Real box_size = config_.get_box().L;

    const std::size_t cells_per_dimension = plan.cells_per_dimension;
    const core::Real cell_size = plan.cell_size;
    const auto wrapped_axis_cell = [&](core::Real wrapped) {
        const std::size_t cell = static_cast<std::size_t>(
            std::floor(wrapped / cell_size));
        return std::min(cell, cells_per_dimension - 1);
    };
    const auto cell_index = [&](core::Real x, core::Real y, core::Real z) {
        const std::size_t ix = wrapped_axis_cell(math::wrap(x, box_size));
        const std::size_t iy = wrapped_axis_cell(math::wrap(y, box_size));
        const std::size_t iz = wrapped_axis_cell(math::wrap(z, box_size));
        return (ix * cells_per_dimension + iy) * cells_per_dimension + iz;
    };

    const core::Real conservative_reach = std::nextafter(
        plan.linking_length,
        std::numeric_limits<core::Real>::infinity());
    const auto overlapping_axis_cells = [=](
        core::Real wrapped_coordinate,
        std::size_t center_cell) {
        AxisCellList result;
        const core::Real lower =
            static_cast<core::Real>(center_cell) * cell_size;
        const core::Real upper = center_cell + 1 == cells_per_dimension
            ? box_size
            : static_cast<core::Real>(center_cell + 1) * cell_size;
        if (wrapped_coordinate - lower <= conservative_reach) {
            result.add_unique(
                (center_cell + cells_per_dimension - 1)
                % cells_per_dimension);
        }
        result.add_unique(center_cell);
        if (upper - wrapped_coordinate <= conservative_reach) {
            result.add_unique((center_cell + 1) % cells_per_dimension);
        }
        return result;
    };

    const auto build_components = [&]<typename LinkIndex, typename DsuIndex>() {
        static_assert(std::is_signed_v<LinkIndex>);
        static_assert(std::is_unsigned_v<DsuIndex>);
        constexpr LinkIndex empty = static_cast<LinkIndex>(-1);
        DisjointSet<DsuIndex> disjoint_set(particle_count);

        {
            std::vector<LinkIndex> head(plan.head_cell_count, empty);
            std::vector<LinkIndex> next(particle_count, empty);
            for (std::size_t particle = 0; particle < particle_count; ++particle) {
                const std::size_t cell = cell_index(
                    position_x[particle],
                    position_y[particle],
                    position_z[particle]);
                next[particle] = head[cell];
                head[cell] = static_cast<LinkIndex>(particle);
            }

            for (std::size_t particle = 0; particle < particle_count; ++particle) {
                const core::Real x = math::wrap(position_x[particle], box_size);
                const core::Real y = math::wrap(position_y[particle], box_size);
                const core::Real z = math::wrap(position_z[particle], box_size);
                const std::size_t cx = wrapped_axis_cell(x);
                const std::size_t cy = wrapped_axis_cell(y);
                const std::size_t cz = wrapped_axis_cell(z);
                const AxisCellList x_cells = overlapping_axis_cells(x, cx);
                const AxisCellList y_cells = overlapping_axis_cells(y, cy);
                const AxisCellList z_cells = overlapping_axis_cells(z, cz);

                for (std::size_t xi = 0; xi < x_cells.count; ++xi) {
                    for (std::size_t yi = 0; yi < y_cells.count; ++yi) {
                        for (std::size_t zi = 0; zi < z_cells.count; ++zi) {
                            const std::size_t cell =
                                (x_cells.cells[xi] * cells_per_dimension
                                 + y_cells.cells[yi]) * cells_per_dimension
                                + z_cells.cells[zi];
                            LinkIndex other = head[cell];
                            while (other != empty) {
                                const std::size_t other_index =
                                    static_cast<std::size_t>(other);
                                if (other_index > particle) {
                                    const core::Real dx = math::minimum_image(
                                        math::wrap(
                                            position_x[other_index], box_size)
                                            - x,
                                        box_size);
                                    if (std::abs(dx) <= plan.linking_length) {
                                        const core::Real dy = math::minimum_image(
                                            math::wrap(
                                                position_y[other_index], box_size)
                                                - y,
                                            box_size);
                                        if (std::abs(dy) <= plan.linking_length) {
                                            const core::Real dz = math::minimum_image(
                                                math::wrap(
                                                    position_z[other_index], box_size)
                                                    - z,
                                                box_size);
                                            if (std::abs(dz) <= plan.linking_length
                                                && core::scale_safe_norm3_leq(
                                                    dx,
                                                    dy,
                                                    dz,
                                                    plan.linking_length)) {
                                                disjoint_set.unite(
                                                    static_cast<DsuIndex>(particle),
                                                    static_cast<DsuIndex>(other_index));
                                            }
                                        }
                                    }
                                }
                                other = next[other_index];
                            }
                        }
                    }
                }
            }
        }

        std::unordered_map<std::size_t, std::vector<std::size_t>> components;
        components.reserve(std::min(
            particle_count,
            std::max<std::size_t>(16, particle_count / 100)));
        for (std::size_t particle = 0; particle < particle_count; ++particle) {
            const DsuIndex root = disjoint_set.find(
                static_cast<DsuIndex>(particle));
            if (disjoint_set.component_size(root) >= min_particles_) {
                components[static_cast<std::size_t>(root)].push_back(particle);
            }
        }
        return components;
    };

    std::unordered_map<std::size_t, std::vector<std::size_t>> components;
    if (plan.link_index_bytes == sizeof(std::int32_t)) {
        components = build_components.template operator()<
            std::int32_t, std::uint32_t>();
    } else if (plan.disjoint_set_index_bytes == sizeof(std::uint32_t)) {
        components = build_components.template operator()<
            std::int64_t, std::uint32_t>();
    } else {
        components = build_components.template operator()<
            std::int64_t, std::uint64_t>();
    }

    memberships.reserve(components.size());
    for (auto& [root, members] : components) {
        (void)root;
        if (members.size() < min_particles_) continue;
        std::sort(
            members.begin(),
            members.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                if (particle_ids[lhs] != particle_ids[rhs]) {
                    return particle_ids[lhs] < particle_ids[rhs];
                }
                return lhs < rhs;
            });
        memberships.push_back(FoFMembership{0, std::move(members)});
    }

    std::sort(
        memberships.begin(),
        memberships.end(),
        [&](const FoFMembership& lhs, const FoFMembership& rhs) {
            if (lhs.particle_indices.size() != rhs.particle_indices.size()) {
                return lhs.particle_indices.size() > rhs.particle_indices.size();
            }
            const core::ParticleId lhs_min_id =
                particle_ids[lhs.particle_indices.front()];
            const core::ParticleId rhs_min_id =
                particle_ids[rhs.particle_indices.front()];
            if (lhs_min_id != rhs_min_id) return lhs_min_id < rhs_min_id;
            return lhs.particle_indices.front() < rhs.particle_indices.front();
        });
    for (std::size_t index = 0; index < memberships.size(); ++index) {
        memberships[index].id = index;
    }
    return memberships;
}

} // namespace cosmo_nbody::halo
