#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cosmo_nbody::mesh::detail {

struct CICCoordinate1D {
    std::int64_t base{0};
    core::Real fraction{0.0};
};

struct CICWeight3D {
    core::Real value{0.0};
    bool lost_nonzero_product{false};
};

// Report underflow only when all three mathematical CIC factors are nonzero.
inline CICWeight3D cic_weight_3d(
    core::Real x_weight,
    core::Real y_weight,
    core::Real z_weight) noexcept {
    const core::Real xy = x_weight * y_weight;
    const core::Real xyz = xy * z_weight;
    const bool mathematically_nonzero = x_weight != 0.0
        && y_weight != 0.0
        && z_weight != 0.0;
    return {
        xyz,
        mathematically_nonzero && xyz == 0.0};
}

inline CICCoordinate1D cic_coordinate_1d(
    core::Real position,
    core::Real box_size,
    core::Real cell_size,
    std::size_t mesh_size) {
    if (
        mesh_size == 0
        || mesh_size > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())
        || !std::isfinite(box_size)
        || box_size <= 0.0
        || !std::isfinite(cell_size)
        || cell_size <= 0.0) {
        throw std::logic_error(
            "CIC coordinate requires finite positive box/cell geometry and an indexable mesh");
    }
    if constexpr (
        std::numeric_limits<std::size_t>::digits
        > std::numeric_limits<core::Real>::digits) {
        constexpr std::size_t first_collapsed_mesh_extent =
            std::size_t{1} << std::numeric_limits<core::Real>::digits;
        if (mesh_size >= first_collapsed_mesh_extent) {
            throw std::logic_error(
                "CIC fractional coordinates are not representable at this mesh extent");
        }
    }

    const core::Real mesh_extent = static_cast<core::Real>(mesh_size);
    if (!std::isfinite(mesh_extent)) {
        throw std::logic_error("CIC mesh extent is not representable");
    }
    const core::Real reconstructed_box = mesh_extent * cell_size;
    const core::Real geometry_scale = std::max(
        std::abs(box_size), std::abs(reconstructed_box));
    const core::Real geometry_tolerance = std::max(
        core::Real{64.0}
            * std::numeric_limits<core::Real>::epsilon()
            * geometry_scale,
        core::Real{4.0}
            * std::numeric_limits<core::Real>::denorm_min());
    if (
        !std::isfinite(reconstructed_box)
        || std::abs(reconstructed_box - box_size) > geometry_tolerance) {
        throw std::logic_error(
            "CIC box size, cell size, and mesh extent are inconsistent");
    }

    const core::Real wrapped = math::wrap(position, box_size);
    if (
        !std::isfinite(wrapped)
        || wrapped < 0.0
        || !(wrapped < box_size)) {
        throw std::logic_error(
            "CIC periodic wrapping produced an invalid coordinate");
    }
    const core::Real unbounded_scaled = wrapped / cell_size;
    if (!std::isfinite(unbounded_scaled)) {
        throw std::logic_error("CIC scaled coordinate is not finite");
    }
    const core::Real upper = std::nextafter(mesh_extent, core::Real{0.0});
    const core::Real scaled = std::clamp(
        unbounded_scaled,
        core::Real{0.0},
        upper);
    const auto base = static_cast<std::int64_t>(std::floor(scaled));
    if (base < 0 || static_cast<std::uint64_t>(base) >= mesh_size) {
        throw std::logic_error("CIC bounded coordinate escaped the mesh domain");
    }
    const core::Real fraction = scaled - static_cast<core::Real>(base);
    if (!(fraction >= 0.0 && fraction < 1.0)) {
        throw std::logic_error("CIC bounded coordinate produced an invalid fraction");
    }
    return {base, fraction};
}

} // namespace cosmo_nbody::mesh::detail
