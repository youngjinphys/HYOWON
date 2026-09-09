#pragma once

#include <optional>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::core {

// A required scalar coordinate may have a valid value equal to its ordinary
// zero/default representation. Store omission separately instead of reserving
// a physically meaningful scalar as a hidden sentinel.
template <typename T>
class ExplicitValue {
public:
    constexpr ExplicitValue() noexcept = default;
    constexpr ExplicitValue(const T& value) : value_(value) {}
    constexpr ExplicitValue(T&& value) : value_(std::move(value)) {}

    constexpr ExplicitValue& operator=(const T& value) {
        value_ = value;
        return *this;
    }
    constexpr ExplicitValue& operator=(T&& value) {
        value_ = std::move(value);
        return *this;
    }

    [[nodiscard]] constexpr bool specified() const noexcept {
        return value_.has_value();
    }

    [[nodiscard]] const T& value() const {
        if (!value_.has_value()) {
            throw std::logic_error("Required scalar coordinate was not specified");
        }
        return *value_;
    }

    [[nodiscard]] T& value() {
        if (!value_.has_value()) {
            throw std::logic_error("Required scalar coordinate was not specified");
        }
        return *value_;
    }

    // Implicit reads still require a resolved value; unresolved reads fail and
    // zero is never synthesized.
    operator const T&() const { return value(); }

private:
    std::optional<T> value_;
};

} // namespace cosmo_nbody::core
