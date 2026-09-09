#pragma once

#include "cosmo_nbody/mesh/mesh_field.hpp"

#include <complex>
#include <cstddef>
#include <utility>

namespace cosmo_nbody::mesh {

void register_destructive_fft_complex_storage(
    const std::complex<core::Real>* data,
    std::size_t size);
void unregister_destructive_fft_complex_storage(
    const std::complex<core::Real>* data) noexcept;
bool is_destructive_fft_complex_storage(
    const std::complex<core::Real>* data,
    std::size_t size) noexcept;

// Explicitly marks Fourier storage that FFTW c2r may consume; use only when the
// modes are dead after the inverse transform.
class DestructiveComplexField final : public ComplexField {
public:
    explicit DestructiveComplexField(std::size_t num_elements)
        : ComplexField(num_elements), registered_(num_elements != 0) {
        if (registered_) {
            register_destructive_fft_complex_storage(data(), size());
        }
    }

    ~DestructiveComplexField() {
        if (registered_) {
            unregister_destructive_fft_complex_storage(data());
        }
    }

    DestructiveComplexField(const DestructiveComplexField&) = delete;
    DestructiveComplexField& operator=(const DestructiveComplexField&) = delete;

    DestructiveComplexField(DestructiveComplexField&& other) noexcept
        : ComplexField(std::move(other)),
          registered_(std::exchange(other.registered_, false)) {}

    DestructiveComplexField& operator=(
        DestructiveComplexField&& other) noexcept {
        if (this != &other) {
            if (registered_) {
                unregister_destructive_fft_complex_storage(data());
            }
            ComplexField::operator=(std::move(other));
            registered_ = std::exchange(other.registered_, false);
        }
        return *this;
    }

private:
    bool registered_{false};
};

} // namespace cosmo_nbody::mesh
