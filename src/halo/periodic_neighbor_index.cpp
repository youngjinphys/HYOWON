#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"

#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmo_nbody::halo {
namespace {

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* label) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return lhs + rhs;
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return lhs * rhs;
}

std::size_t checked_cube(std::size_t value, const char* label) {
    return checked_multiply(checked_multiply(value, value, label), value, label);
}

std::size_t floor_cube_root(std::size_t value) {
    if (value == 0) return 0;
    std::size_t root = static_cast<std::size_t>(
        std::cbrt(static_cast<long double>(value)));
    root = std::max<std::size_t>(1, root);
    auto fits = [value](std::size_t candidate) {
        return candidate == 0 || candidate <= value / candidate / candidate;
    };
    while (root < std::numeric_limits<std::size_t>::max() && fits(root + 1)) ++root;
    while (!fits(root)) --root;
    return root;
}

std::size_t wrap_cell(long long value, std::size_t count) {
    if (count == 0 || count > static_cast<std::size_t>(
            std::numeric_limits<long long>::max())) {
        throw std::overflow_error(
            "Periodic neighbor cell count exceeds signed indexing range");
    }
    const long long signed_count = static_cast<long long>(count);
    long long wrapped = value % signed_count;
    if (wrapped < 0) wrapped += signed_count;
    return static_cast<std::size_t>(wrapped);
}

} // namespace

PeriodicNeighborIndexPlan PeriodicNeighborIndex::make_plan(
    std::size_t particle_count,
    std::size_t index_cap_bytes) {
    PeriodicNeighborIndexPlan plan;
    plan.particle_count = particle_count;
    plan.index_cap_bytes = index_cap_bytes;
    plan.compact_indices = particle_count <= static_cast<std::size_t>(
        std::numeric_limits<std::uint32_t>::max());
    if (particle_count == 0) return plan;
    if (index_cap_bytes == 0) {
        throw std::runtime_error(
            "Periodic neighbor index requires a positive memory cap");
    }

    const std::size_t element_bytes = plan.compact_indices
        ? sizeof(std::uint32_t) : sizeof(std::size_t);
    const std::size_t available_elements = index_cap_bytes / element_bytes;
    if (available_elements <= particle_count) {
        throw std::runtime_error(
            "Periodic neighbor index cap cannot hold one next link per owned particle plus a cell head");
    }

    const std::size_t maximum_head_cells = available_elements - particle_count;
    const std::size_t desired_head_cells = std::min(
        maximum_head_cells, std::max<std::size_t>(1, particle_count));
    plan.cells_per_dimension = floor_cube_root(desired_head_cells);
    if (plan.cells_per_dimension == 0) {
        throw std::runtime_error(
            "Periodic neighbor index could not admit one cell per dimension");
    }
    plan.cell_count = checked_cube(
        plan.cells_per_dimension, "Periodic neighbor cell count");
    plan.index_bytes = checked_multiply(
        checked_add(particle_count, plan.cell_count,
                    "Periodic neighbor link element count"),
        element_bytes,
        "Periodic neighbor index bytes");
    if (plan.index_bytes > index_cap_bytes) {
        throw std::logic_error(
            "Periodic neighbor index plan exceeds its admitted cap");
    }
    return plan;
}

PeriodicNeighborIndex::PeriodicNeighborIndex(
    const core::ParticleStore& particles,
    core::Real box_size,
    std::size_t index_cap_bytes)
    : particles_(&particles),
      box_size_(box_size),
      plan_(make_plan(particles.num_owned_particles(), index_cap_bytes)) {
    if (!std::isfinite(box_size_) || box_size_ <= 0.0) {
        throw std::invalid_argument(
            "Periodic neighbor index requires a finite positive box size");
    }
    if (plan_.particle_count == 0) return;

    cell_width_ = box_size_
        / static_cast<core::Real>(plan_.cells_per_dimension);
    if (!std::isfinite(cell_width_) || cell_width_ <= 0.0) {
        throw std::overflow_error(
            "Periodic neighbor index cell width is invalid");
    }

    const std::size_t owned = particles.num_owned_particles();
    if (plan_.compact_indices) {
        constexpr std::uint32_t invalid =
            std::numeric_limits<std::uint32_t>::max();
        compact_heads_.assign(plan_.cell_count, invalid);
        compact_next_.resize(owned, invalid);
        for (std::size_t particle = 0; particle < owned; ++particle) {
            if (particle >= static_cast<std::size_t>(invalid)) {
                throw std::overflow_error(
                    "Periodic neighbor compact particle index is not representable");
            }
            const std::size_t cell = cell_id_for_position(
                particles.get_positions_x()[particle],
                particles.get_positions_y()[particle],
                particles.get_positions_z()[particle]);
            compact_next_[particle] = compact_heads_[cell];
            compact_heads_[cell] = static_cast<std::uint32_t>(particle);
        }
    } else {
        constexpr std::size_t invalid =
            std::numeric_limits<std::size_t>::max();
        wide_heads_.assign(plan_.cell_count, invalid);
        wide_next_.resize(owned, invalid);
        for (std::size_t particle = 0; particle < owned; ++particle) {
            const std::size_t cell = cell_id_for_position(
                particles.get_positions_x()[particle],
                particles.get_positions_y()[particle],
                particles.get_positions_z()[particle]);
            wide_next_[particle] = wide_heads_[cell];
            wide_heads_[cell] = particle;
        }
    }
}

std::size_t PeriodicNeighborIndex::cell_id_for_position(
    core::Real x,
    core::Real y,
    core::Real z) const {
    const core::Real wrapped_x = math::wrap(x, box_size_);
    const core::Real wrapped_y = math::wrap(y, box_size_);
    const core::Real wrapped_z = math::wrap(z, box_size_);
    if (!std::isfinite(wrapped_x) || !std::isfinite(wrapped_y)
        || !std::isfinite(wrapped_z)) {
        throw std::invalid_argument(
            "Periodic neighbor index particle positions must be finite");
    }
    const auto coordinate_to_cell = [&](core::Real coordinate) {
        const core::Real raw = std::floor(coordinate / cell_width_);
        if (!std::isfinite(raw) || raw < 0.0) {
            throw std::runtime_error(
                "Periodic neighbor index produced an invalid cell coordinate");
        }
        return std::min(static_cast<std::size_t>(raw),
                        plan_.cells_per_dimension - 1);
    };
    const std::size_t ix = coordinate_to_cell(wrapped_x);
    const std::size_t iy = coordinate_to_cell(wrapped_y);
    const std::size_t iz = coordinate_to_cell(wrapped_z);
    const std::size_t xy = checked_add(
        checked_multiply(ix, plan_.cells_per_dimension,
                         "Periodic neighbor cell id"),
        iy, "Periodic neighbor cell id");
    return checked_add(
        checked_multiply(xy, plan_.cells_per_dimension,
                         "Periodic neighbor cell id"),
        iz, "Periodic neighbor cell id");
}

std::size_t PeriodicNeighborIndex::cell_occupancy(std::size_t cell_id) const {
    if (cell_id >= plan_.cell_count) {
        throw std::out_of_range("Periodic neighbor cell id is out of range");
    }
    std::size_t count = 0;
    if (plan_.compact_indices) {
        constexpr std::uint32_t invalid =
            std::numeric_limits<std::uint32_t>::max();
        std::uint32_t particle = compact_heads_[cell_id];
        while (particle != invalid) {
            if (particle >= compact_next_.size()) {
                throw std::logic_error(
                    "Periodic neighbor compact link exceeds particle storage");
            }
            ++count;
            particle = compact_next_[particle];
        }
    } else {
        constexpr std::size_t invalid =
            std::numeric_limits<std::size_t>::max();
        std::size_t particle = wide_heads_[cell_id];
        while (particle != invalid) {
            if (particle >= wide_next_.size()) {
                throw std::logic_error(
                    "Periodic neighbor wide link exceeds particle storage");
            }
            ++count;
            particle = wide_next_[particle];
        }
    }
    return count;
}

void PeriodicNeighborIndex::append_cell_neighbors(
    std::size_t cell_id,
    const core::Vec3& wrapped_center,
    core::Real radius,
    std::vector<PeriodicNeighbor>& output) const {
    const std::size_t owned = particles_->num_owned_particles();
    const auto x = particles_->get_positions_x().first(owned);
    const auto y = particles_->get_positions_y().first(owned);
    const auto z = particles_->get_positions_z().first(owned);
    const auto append = [&](std::size_t particle) {
        const core::Vec3 wrapped_particle{
            math::wrap(x[particle], box_size_),
            math::wrap(y[particle], box_size_),
            math::wrap(z[particle], box_size_)};
        if (!std::isfinite(wrapped_particle.x)
            || !std::isfinite(wrapped_particle.y)
            || !std::isfinite(wrapped_particle.z)) {
            throw std::runtime_error(
                "Periodic neighbor particle position is non-finite");
        }

        // Decide closed membership from the exact mathematical torus distance
        // of the represented canonical endpoints. A rounded minimum-image
        // component is not a valid boundary predicate near the half-box cut.
        if (!math::minimum_image_distance_leq_wrapped(
                wrapped_center, wrapped_particle, box_size_, radius)) {
            return;
        }

        const core::Vec3 displacement = math::minimum_image_displacement(
            wrapped_center, wrapped_particle, box_size_);
        const core::Real distance = displacement.norm();
        if (!std::isfinite(distance)) {
            throw std::runtime_error(
                "Periodic neighbor distance is non-finite");
        }
        output.push_back(PeriodicNeighbor{distance, particle});
    };

    if (plan_.compact_indices) {
        constexpr std::uint32_t invalid =
            std::numeric_limits<std::uint32_t>::max();
        std::uint32_t particle = compact_heads_[cell_id];
        while (particle != invalid) {
            if (particle >= compact_next_.size()) {
                throw std::logic_error(
                    "Periodic neighbor compact link exceeds particle storage");
            }
            append(static_cast<std::size_t>(particle));
            particle = compact_next_[particle];
        }
    } else {
        constexpr std::size_t invalid =
            std::numeric_limits<std::size_t>::max();
        std::size_t particle = wide_heads_[cell_id];
        while (particle != invalid) {
            if (particle >= wide_next_.size()) {
                throw std::logic_error(
                    "Periodic neighbor wide link exceeds particle storage");
            }
            append(particle);
            particle = wide_next_[particle];
        }
    }
}

void PeriodicNeighborIndex::collect_within(
    const core::Vec3& center,
    core::Real radius,
    std::vector<PeriodicNeighbor>& output) const {
    if (!particles_ || plan_.particle_count == 0) {
        output.clear();
        return;
    }
    if (!std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument(
            "Periodic neighbor query radius must be finite and non-negative");
    }
    const core::Vec3 wrapped_center = math::wrap(center, box_size_);
    if (!std::isfinite(wrapped_center.x) || !std::isfinite(wrapped_center.y)
        || !std::isfinite(wrapped_center.z)) {
        throw std::invalid_argument(
            "Periodic neighbor query center must be finite");
    }

    const auto all_axis_cells = [&]() {
        std::vector<std::size_t> cells(plan_.cells_per_dimension);
        for (std::size_t index = 0; index < cells.size(); ++index) {
            cells[index] = index;
        }
        return cells;
    };

    const auto axis_cells = [&](core::Real coordinate) {
        // Bound candidate traversal, not the requested geometric radius: a
        // rounded torus diameter can be smaller than an actual corner distance.
        // Each periodic axis is fully covered at L/2; handle that case before
        // radius/cell_width_ can overflow for a large finite query radius.
        if (radius >= core::Real{0.5} * box_size_) {
            return all_axis_cells();
        }
        const core::Real raw_center = std::floor(coordinate / cell_width_);
        if (!std::isfinite(raw_center) || raw_center < 0.0) {
            throw std::runtime_error(
                "Periodic neighbor query produced an invalid center cell");
        }
        const std::size_t center_cell = std::min(
            static_cast<std::size_t>(raw_center),
            plan_.cells_per_dimension - 1);
        const core::Real reach_real = std::ceil(radius / cell_width_) + 1.0;
        if (!std::isfinite(reach_real) || reach_real < 1.0
            || reach_real > static_cast<core::Real>(
                std::numeric_limits<long long>::max())) {
            throw std::overflow_error(
                "Periodic neighbor query cell reach is invalid");
        }
        const std::size_t reach = static_cast<std::size_t>(reach_real);
        if (reach >= plan_.cells_per_dimension / 2) {
            return all_axis_cells();
        }
        std::vector<std::size_t> cells;
        cells.reserve(2 * reach + 1);
        const long long signed_center = static_cast<long long>(center_cell);
        const long long signed_reach = static_cast<long long>(reach);
        for (long long offset = -signed_reach; offset <= signed_reach; ++offset) {
            cells.push_back(wrap_cell(
                signed_center + offset, plan_.cells_per_dimension));
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        return cells;
    };

    const auto xs = axis_cells(wrapped_center.x);
    const auto ys = axis_cells(wrapped_center.y);
    const auto zs = axis_cells(wrapped_center.z);
    const auto cell_id = [&](std::size_t ix,
                             std::size_t iy,
                             std::size_t iz) {
        const std::size_t xy = checked_add(
            checked_multiply(ix, plan_.cells_per_dimension,
                             "Periodic neighbor query cell id"),
            iy, "Periodic neighbor query cell id");
        return checked_add(
            checked_multiply(xy, plan_.cells_per_dimension,
                             "Periodic neighbor query cell id"),
            iz, "Periodic neighbor query cell id");
    };

    std::size_t occupancy_upper_bound = 0;
    for (const std::size_t ix : xs) {
        for (const std::size_t iy : ys) {
            for (const std::size_t iz : zs) {
                occupancy_upper_bound = checked_add(
                    occupancy_upper_bound,
                    cell_occupancy(cell_id(ix, iy, iz)),
                    "Periodic neighbor query occupancy");
            }
        }
    }

    if (output.capacity() < occupancy_upper_bound) {
        // vector::reserve may hold both allocations until the move completes.
        // Release the previous adaptive-radius buffer first so the SO candidate
        // peak never exceeds the execution plan's one-buffer-per-worker bound.
        std::vector<PeriodicNeighbor>{}.swap(output);
        output.reserve(occupancy_upper_bound);
    } else {
        output.clear();
    }
    for (const std::size_t ix : xs) {
        for (const std::size_t iy : ys) {
            for (const std::size_t iz : zs) {
                append_cell_neighbors(
                    cell_id(ix, iy, iz), wrapped_center, radius, output);
            }
        }
    }
}

} // namespace cosmo_nbody::halo
