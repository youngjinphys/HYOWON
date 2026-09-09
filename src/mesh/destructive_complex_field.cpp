#include "cosmo_nbody/mesh/destructive_complex_field.hpp"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cosmo_nbody::mesh {
namespace {

struct Registry {
    std::mutex mutex;
    std::unordered_map<const std::complex<core::Real>*, std::size_t> extents;
};

Registry& registry() {
    static Registry value;
    return value;
}

} // namespace

void register_destructive_fft_complex_storage(
    const std::complex<core::Real>* data,
    std::size_t size) {
    if (data == nullptr || size == 0) {
        throw std::invalid_argument(
            "Destructive Fourier storage must be non-null and non-empty");
    }
    auto& state = registry();
    std::scoped_lock lock(state.mutex);
    const auto [_, inserted] = state.extents.emplace(data, size);
    if (!inserted) {
        throw std::logic_error(
            "Destructive Fourier storage was registered more than once");
    }
}

void unregister_destructive_fft_complex_storage(
    const std::complex<core::Real>* data) noexcept {
    if (data == nullptr) return;
    auto& state = registry();
    std::scoped_lock lock(state.mutex);
    state.extents.erase(data);
}

bool is_destructive_fft_complex_storage(
    const std::complex<core::Real>* data,
    std::size_t size) noexcept {
    if (data == nullptr || size == 0) return false;
    try {
        auto& state = registry();
        std::scoped_lock lock(state.mutex);
        const auto found = state.extents.find(data);
        return found != state.extents.end() && found->second == size;
    } catch (...) {
        return false;
    }
}

} // namespace cosmo_nbody::mesh
