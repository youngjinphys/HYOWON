#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace cosmo_nbody {
namespace mesh {

class MeshGeometry {
public:
    MeshGeometry(core::Real box_size,
                 std::size_t n_mesh,
                 std::size_t local_n0 = 0,
                 std::size_t local_0_start = 0,
                 std::size_t alloc_local = 0)
        : L_(box_size),
          N_(n_mesh),
          dx_(0.0),
          local_n0_(local_n0 == 0 ? n_mesh : local_n0),
          local_0_start_(local_0_start) {
        if (!std::isfinite(L_) || L_ <= 0.0) {
            throw std::invalid_argument("MeshGeometry box size must be finite and positive");
        }
        if (N_ == 0) {
            throw std::invalid_argument("MeshGeometry grid size must be positive");
        }
        if (N_ > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("MeshGeometry grid size exceeds signed index range");
        }
        if (local_n0_ == 0 || local_n0_ > N_) {
            throw std::invalid_argument("MeshGeometry local_n0 must be in [1, N]");
        }
        if (local_0_start_ > N_ - local_n0_) {
            throw std::invalid_argument("MeshGeometry local slab exceeds global mesh extent");
        }

        dx_ = L_ / static_cast<core::Real>(N_);
        if (!std::isfinite(dx_) || dx_ <= 0.0) {
            throw std::overflow_error("MeshGeometry cell size is not representable");
        }

        real_size_ = checked_product(local_n0_, N_, N_, "real mesh size");
        const std::size_t nz_complex = N_ / 2 + 1;
        const std::size_t minimum_complex = checked_product(
            local_n0_, N_, nz_complex, "complex mesh size");
        if (alloc_local == 0) {
            complex_size_ = minimum_complex;
        } else {
            if (alloc_local < minimum_complex) {
                throw std::invalid_argument(
                    "MeshGeometry alloc_local is smaller than the r2c half-spectrum");
            }
            complex_size_ = alloc_local;
        }
    }

    core::Real box_size() const noexcept { return L_; }
    std::size_t grid_size() const noexcept { return N_; }
    core::Real cell_size() const noexcept { return dx_; }

    std::size_t local_n0() const noexcept { return local_n0_; }
    std::size_t local_0_start() const noexcept { return local_0_start_; }

    std::size_t real_size() const noexcept { return real_size_; }
    std::size_t padded_real_size() const noexcept { return real_size_; }

    std::size_t real_index(std::size_t local_ix,
                           std::size_t iy,
                           std::size_t iz) const noexcept {
        assert(local_ix < local_n0_ && iy < N_ && iz < N_);
        return (local_ix * N_ + iy) * N_ + iz;
    }

    std::size_t wrap_index(std::int64_t index) const noexcept {
        const std::int64_t n = static_cast<std::int64_t>(N_);
        std::int64_t wrapped = index % n;
        if (wrapped < 0) wrapped += n;
        return static_cast<std::size_t>(wrapped);
    }

    std::size_t complex_size() const noexcept { return complex_size_; }

    std::size_t complex_index(std::size_t local_ix,
                              std::size_t iy,
                              std::size_t iz) const noexcept {
        const std::size_t nz_complex = N_ / 2 + 1;
        assert(local_ix < local_n0_ && iy < N_ && iz < nz_complex);
        return iz + nz_complex * (iy + N_ * local_ix);
    }

    std::size_t global_ix(std::size_t local_ix) const noexcept {
        assert(local_ix < local_n0_);
        return local_0_start_ + local_ix;
    }

    std::int64_t mode_number(std::size_t index) const noexcept {
        assert(index < N_);
        const std::int64_t signed_index = static_cast<std::int64_t>(index);
        const std::int64_t half = static_cast<std::int64_t>(N_ / 2);
        return signed_index <= half
            ? signed_index
            : signed_index - static_cast<std::int64_t>(N_);
    }

    core::Real k_component(std::size_t index) const noexcept {
        return 2.0 * std::numbers::pi
            * static_cast<core::Real>(mode_number(index)) / L_;
    }

    core::Real k_mag2(std::size_t ix,
                      std::size_t iy,
                      std::size_t iz) const noexcept {
        const core::Real kx = k_component(ix);
        const core::Real ky = k_component(iy);
        const core::Real kz = k_component(iz);
        return kx * kx + ky * ky + kz * kz;
    }

private:
    static std::size_t checked_product(std::size_t a,
                                       std::size_t b,
                                       std::size_t c,
                                       const char* label) {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
            throw std::overflow_error(std::string("MeshGeometry ") + label + " overflows size_t");
        }
        const std::size_t ab = a * b;
        if (ab != 0 && c > std::numeric_limits<std::size_t>::max() / ab) {
            throw std::overflow_error(std::string("MeshGeometry ") + label + " overflows size_t");
        }
        return ab * c;
    }

    core::Real L_;
    std::size_t N_;
    core::Real dx_;
    std::size_t local_n0_;
    std::size_t local_0_start_;
    std::size_t real_size_{0};
    std::size_t complex_size_{0};
};

} // namespace mesh
} // namespace cosmo_nbody
