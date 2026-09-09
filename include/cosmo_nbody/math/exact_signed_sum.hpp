// Exact signed binary64 accumulation for exceptional reduction paths.
#pragma once

#include "cosmo_nbody/math/exact_positive_sum.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace cosmo_nbody::math {

class ExactSignedDoubleSum final {
public:
    void add(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "ExactSignedDoubleSum requires finite terms");
        }
        if (value > 0.0) {
            positive_.add(value);
        } else if (value < 0.0) {
            negative_.add(-value);
        }
    }

    [[nodiscard]] double value() const {
        const auto positive = positive_.limbs();
        const auto negative = negative_.limbs();
        const int ordering = compare(positive, negative);
        if (ordering == 0) return 0.0;

        const bool negative_result = ordering < 0;
        const auto magnitude = negative_result
            ? subtract(negative, positive)
            : subtract(positive, negative);
        const double rounded = magnitude_value(magnitude);
        return negative_result ? -rounded : rounded;
    }

private:
    using Limbs = std::array<
        std::uint64_t,
        ExactPositiveDoubleSum::limb_count>;

    static int compare(
        std::span<const std::uint64_t, ExactPositiveDoubleSum::limb_count> lhs,
        std::span<const std::uint64_t, ExactPositiveDoubleSum::limb_count> rhs)
        noexcept {
        for (std::size_t reverse = ExactPositiveDoubleSum::limb_count;
             reverse > 0;
             --reverse) {
            const std::size_t index = reverse - 1;
            if (lhs[index] < rhs[index]) return -1;
            if (lhs[index] > rhs[index]) return 1;
        }
        return 0;
    }

    static Limbs subtract(
        std::span<const std::uint64_t, ExactPositiveDoubleSum::limb_count> lhs,
        std::span<const std::uint64_t, ExactPositiveDoubleSum::limb_count> rhs) {
        Limbs result{};
        bool borrow = false;
        for (std::size_t index = 0;
             index < ExactPositiveDoubleSum::limb_count;
             ++index) {
            const std::uint64_t first = lhs[index] - rhs[index];
            const bool first_borrow = lhs[index] < rhs[index];
            const std::uint64_t second = first - static_cast<std::uint64_t>(borrow);
            const bool second_borrow = first < static_cast<std::uint64_t>(borrow);
            result[index] = second;
            borrow = first_borrow || second_borrow;
        }
        if (borrow) {
            throw std::logic_error(
                "ExactSignedDoubleSum magnitude subtraction underflowed");
        }
        return result;
    }

    static std::size_t highest_set_bit(const Limbs& limbs) noexcept {
        for (std::size_t reverse = limbs.size(); reverse > 0; --reverse) {
            const std::size_t index = reverse - 1;
            const std::uint64_t word = limbs[index];
            if (word != 0) {
                return index * 64
                    + static_cast<std::size_t>(63U - std::countl_zero(word));
            }
        }
        return no_bit;
    }

    static bool bit(const Limbs& limbs, std::size_t position) noexcept {
        const std::size_t index = position / 64;
        const unsigned offset = static_cast<unsigned>(position % 64);
        return index < limbs.size()
            && ((limbs[index] >> offset) & 1U) != 0;
    }

    static bool any_bits_below(
        const Limbs& limbs,
        std::size_t exclusive_end) noexcept {
        if (exclusive_end == 0) return false;
        const std::size_t complete_limbs = exclusive_end / 64;
        for (std::size_t index = 0;
             index < complete_limbs && index < limbs.size();
             ++index) {
            if (limbs[index] != 0) return true;
        }
        const unsigned remainder =
            static_cast<unsigned>(exclusive_end % 64);
        if (remainder != 0 && complete_limbs < limbs.size()) {
            const std::uint64_t mask =
                (std::uint64_t{1} << remainder) - 1U;
            return (limbs[complete_limbs] & mask) != 0;
        }
        return false;
    }

    static std::uint64_t extract_u64(
        const Limbs& limbs,
        std::size_t shift) noexcept {
        const std::size_t index = shift / 64;
        const unsigned offset = static_cast<unsigned>(shift % 64);
        if (index >= limbs.size()) return 0;
        std::uint64_t result = limbs[index] >> offset;
        if (offset != 0 && index + 1 < limbs.size()) {
            result |= limbs[index + 1] << (64U - offset);
        }
        return result;
    }

    static double magnitude_value(const Limbs& limbs) {
        const std::size_t highest = highest_set_bit(limbs);
        if (highest == no_bit) return 0.0;

        if (highest <= 52) {
            return std::ldexp(static_cast<double>(limbs[0]), -1074);
        }

        std::size_t rounded_highest = highest;
        const std::size_t discarded_bits = highest - 52;
        std::uint64_t significand = extract_u64(limbs, discarded_bits);
        constexpr std::uint64_t significand_mask =
            (std::uint64_t{1} << 53) - 1;
        significand &= significand_mask;

        const bool guard = bit(limbs, discarded_bits - 1);
        const bool sticky = any_bits_below(limbs, discarded_bits - 1);
        if (guard && (sticky || (significand & 1U) != 0)) {
            ++significand;
            if (significand == (std::uint64_t{1} << 53)) {
                significand >>= 1U;
                ++rounded_highest;
            }
        }

        const int exponent = static_cast<int>(rounded_highest) - 1126;
        const double rounded = std::ldexp(
            static_cast<double>(significand), exponent);
        if (!std::isfinite(rounded)) {
            throw std::overflow_error(
                "ExactSignedDoubleSum result exceeds finite binary64");
        }
        return rounded;
    }

    static constexpr std::size_t no_bit =
        std::numeric_limits<std::size_t>::max();

    ExactPositiveDoubleSum positive_;
    ExactPositiveDoubleSum negative_;
};

} // namespace cosmo_nbody::math
