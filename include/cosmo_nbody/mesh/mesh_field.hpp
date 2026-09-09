#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <complex>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cosmo_nbody {
namespace mesh {

struct ExternalMeshStorageTag {
    explicit ExternalMeshStorageTag() = default;
};

inline constexpr ExternalMeshStorageTag external_mesh_storage{};

template <typename T>
class MeshField {
public:
    explicit MeshField(std::size_t num_elements)
        : size_(num_elements) {
        static_assert(std::is_default_constructible_v<T>,
                      "MeshField requires default construction");
        static_assert(std::is_nothrow_destructible_v<T>,
                      "MeshField requires nothrow destruction");

        if (num_elements == 0) return;
        if (num_elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("MeshField byte size overflows size_t");
        }

        constexpr std::size_t alignment =
            alignof(T) > std::size_t{64} ? alignof(T) : std::size_t{64};
        static_assert((alignment & (alignment - 1)) == 0,
                      "MeshField alignment must be a power of two");
        const std::size_t raw_bytes = num_elements * sizeof(T);
        if (raw_bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
            throw std::overflow_error("MeshField aligned byte size overflows size_t");
        }
        const std::size_t bytes =
            ((raw_bytes + alignment - 1) / alignment) * alignment;

        void* ptr = std::aligned_alloc(alignment, bytes);
        if (!ptr) throw std::bad_alloc();
        data_.reset(static_cast<T*>(ptr));

        // Preserve T{} value-initialization without requiring a nothrow
        // default constructor. The standard uninitialized algorithm destroys
        // already-constructed elements if a later constructor throws; the
        // unique_ptr then releases the aligned raw storage during unwinding.
        std::uninitialized_value_construct_n(data_.get(), num_elements);
    }

    MeshField(
        T* external_data,
        std::size_t num_elements,
        ExternalMeshStorageTag)
        : size_(num_elements),
          owns_elements_(false),
          data_(external_data, AlignedDeleter{false}) {
        if (num_elements > 0 && external_data == nullptr) {
            throw std::invalid_argument(
                "MeshField external storage is null for a non-empty field");
        }
    }

    ~MeshField() {
        destroy_elements();
    }

    MeshField(const MeshField&) = delete;
    MeshField& operator=(const MeshField&) = delete;

    MeshField(MeshField&& other) noexcept
        : size_(std::exchange(other.size_, 0)),
          owns_elements_(std::exchange(other.owns_elements_, false)),
          data_(std::move(other.data_)) {}

    MeshField& operator=(MeshField&& other) noexcept {
        if (this != &other) {
            destroy_elements();
            data_.reset();
            size_ = std::exchange(other.size_, 0);
            owns_elements_ = std::exchange(other.owns_elements_, false);
            data_ = std::move(other.data_);
        }
        return *this;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }

    T* begin() noexcept { return data_.get(); }
    const T* begin() const noexcept { return data_.get(); }
    T* end() noexcept {
        return size_ == 0 ? data_.get() : data_.get() + size_;
    }
    const T* end() const noexcept {
        return size_ == 0 ? data_.get() : data_.get() + size_;
    }

    T& operator[](std::size_t idx) { return data_[idx]; }
    const T& operator[](std::size_t idx) const { return data_[idx]; }

private:
    struct AlignedDeleter {
        bool release{true};

        void operator()(T* ptr) const noexcept {
            if (release) std::free(ptr);
        }
    };

    void destroy_elements() noexcept {
        if (!data_ || !owns_elements_) return;
        for (std::size_t idx = 0; idx < size_; ++idx) {
            data_[idx].~T();
        }
    }

    std::size_t size_{0};
    bool owns_elements_{true};
    std::unique_ptr<T[], AlignedDeleter> data_;
};

using RealField = MeshField<core::Real>;
using ComplexField = MeshField<std::complex<core::Real>>;

} // namespace mesh
} // namespace cosmo_nbody
