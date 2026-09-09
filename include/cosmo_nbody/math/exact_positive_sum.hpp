// Order- and partition-independent exact accumulation for non-negative FP64.
//
// Every finite binary64 value is represented as an integer multiple of the
// smallest subnormal, 2^-1074. Thirty-four 64-bit limbs cover the largest
// possible sum of at most UINT64_MAX finite binary64 terms:
//   highest input bit 2097 + at most 64 count bits = highest sum bit 2161,
// while 34 limbs provide bits [0,2175]. Integer addition is exact and
// associative; value() performs the only binary64 rounding, using
// round-to-nearest, ties-to-even.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::math {

// The implementation decodes the IEEE-754 binary64 bit fields directly. Refuse
// unsupported floating-point models at compile time rather than silently
// interpreting another representation as binary64.
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::radix == 2);
static_assert(std::numeric_limits<double>::digits == 53);
static_assert(std::numeric_limits<double>::max_exponent == 1024);
static_assert(std::numeric_limits<double>::min_exponent == -1021);
static_assert(sizeof(double) == sizeof(std::uint64_t));

class ExactPositiveDoubleSum {
public:
    static constexpr std::size_t limb_count = 34;

    void add(double value) {
        require_term_capacity(1);
        const Decoded decoded = decode(value);
        if (decoded.mantissa != 0) {
            add_shifted_word(decoded.mantissa, decoded.shift);
        }
        ++term_count_;
    }

    void add_repeated(double value, std::uint64_t repetitions) {
        if (repetitions == 0) return;
        require_term_capacity(repetitions);
        const Decoded decoded = decode(value);
        if (decoded.mantissa != 0) {
            std::uint64_t remaining = repetitions;
            std::size_t count_bit = 0;
            while (remaining != 0) {
                if ((remaining & 1U) != 0) {
                    add_shifted_word(
                        decoded.mantissa,
                        decoded.shift + count_bit);
                }
                remaining >>= 1U;
                ++count_bit;
            }
        }
        term_count_ += repetitions;
    }

    void combine(const ExactPositiveDoubleSum& other) {
        combine_serialized(other.limbs_, other.term_count_);
    }

    void combine_serialized(
        std::span<const std::uint64_t> limbs,
        std::uint64_t terms) {
        if (limbs.size() != limb_count) {
            throw std::invalid_argument(
                "ExactPositiveDoubleSum serialized limb count mismatch");
        }
        require_term_capacity(terms);

        // Add into a candidate first so a malformed serialized accumulator that
        // exceeds the fixed mathematical envelope cannot partially mutate this
        // object before failing.
        auto candidate = limbs_;
        for (std::size_t index = 0; index < limb_count; ++index) {
            add_word(candidate, index, limbs[index]);
        }
        limbs_ = candidate;
        term_count_ += terms;
    }

    double value() const {
        std::uint64_t significand = 0;
        int exponent = 0;
        if (!rounded_binary64_components(significand, exponent)) return 0.0;
        const double rounded = std::ldexp(
            static_cast<double>(significand), exponent);
        if (!std::isfinite(rounded)) {
            throw std::overflow_error(
                "ExactPositiveDoubleSum result exceeds finite binary64");
        }
        return rounded;
    }

    // Range-safe logarithmic projection of the exact accumulator. Zero is
    // rejected. The multi-limb integer sum is first rounded once to the same
    // 53-bit significand that value() would use; the logarithm is then evaluated
    // from that significand and its (possibly wider-than-binary64) exponent.
    // Therefore this method preserves dynamic range for a later ratio/density,
    // but it is not an exact transcendental evaluation of log(exact integer sum).
    long double log_value() const {
        std::uint64_t significand = 0;
        int exponent = 0;
        if (!rounded_binary64_components(significand, exponent)) {
            throw std::invalid_argument(
                "ExactPositiveDoubleSum logarithm of zero is undefined");
        }
        const double rounded = std::ldexp(
            static_cast<double>(significand), exponent);
        if (std::isfinite(rounded) && rounded > 0.0) {
            return std::log(static_cast<long double>(rounded));
        }
        if (significand == 0) {
            throw std::invalid_argument(
                "ExactPositiveDoubleSum logarithm of zero is undefined");
        }
        return std::log(static_cast<long double>(significand))
            + static_cast<long double>(exponent) * std::log(2.0L);
    }

    std::uint64_t term_count() const noexcept { return term_count_; }

    std::span<const std::uint64_t, limb_count> limbs() const noexcept {
        return limbs_;
    }

private:
    static constexpr std::size_t no_bit =
        std::numeric_limits<std::size_t>::max();

    struct Decoded {
        std::uint64_t mantissa{0};
        std::size_t shift{0};
    };

    static Decoded decode(double value) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(
                "ExactPositiveDoubleSum requires finite non-negative terms");
        }

        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint64_t exponent = (bits >> 52U) & 0x7ffU;
        const std::uint64_t fraction =
            bits & ((std::uint64_t{1} << 52U) - 1U);
        if (exponent == 0) {
            return {fraction, 0};
        }
        return {
            (std::uint64_t{1} << 52U) | fraction,
            static_cast<std::size_t>(exponent - 1U)};
    }

    void require_term_capacity(std::uint64_t additional) const {
        if (additional
            > std::numeric_limits<std::uint64_t>::max() - term_count_) {
            throw std::overflow_error(
                "ExactPositiveDoubleSum term count overflows uint64_t");
        }
    }

    static void add_word(
        std::array<std::uint64_t, limb_count>& target,
        std::size_t index,
        std::uint64_t word) {
        while (word != 0) {
            if (index >= limb_count) {
                throw std::overflow_error(
                    "ExactPositiveDoubleSum fixed accumulator overflow");
            }
            const std::uint64_t previous = target[index];
            target[index] = previous + word;
            word = target[index] < previous ? 1U : 0U;
            ++index;
        }
    }

    void add_shifted_word(std::uint64_t word, std::size_t shift) {
        if (word == 0) return;
        const std::size_t index = shift / 64;
        const unsigned offset = static_cast<unsigned>(shift % 64);
        if (index >= limb_count) {
            throw std::overflow_error(
                "ExactPositiveDoubleSum shifted term exceeds accumulator");
        }

        add_word(limbs_, index, word << offset);
        if (offset != 0) {
            add_word(limbs_, index + 1, word >> (64U - offset));
        }
    }

    std::size_t highest_set_bit() const noexcept {
        for (std::size_t reverse = limb_count; reverse > 0; --reverse) {
            const std::size_t index = reverse - 1;
            const std::uint64_t word = limbs_[index];
            if (word != 0) {
                return index * 64
                    + static_cast<std::size_t>(
                        63U - std::countl_zero(word));
            }
        }
        return no_bit;
    }

    bool bit(std::size_t position) const noexcept {
        const std::size_t index = position / 64;
        const unsigned offset = static_cast<unsigned>(position % 64);
        return index < limb_count
            && ((limbs_[index] >> offset) & 1U) != 0;
    }

    bool any_bits_below(std::size_t exclusive_end) const noexcept {
        if (exclusive_end == 0) return false;
        const std::size_t complete_limbs = exclusive_end / 64;
        for (std::size_t index = 0;
             index < complete_limbs && index < limb_count;
             ++index) {
            if (limbs_[index] != 0) return true;
        }
        const unsigned remainder =
            static_cast<unsigned>(exclusive_end % 64);
        if (remainder != 0 && complete_limbs < limb_count) {
            const std::uint64_t mask =
                (std::uint64_t{1} << remainder) - 1U;
            return (limbs_[complete_limbs] & mask) != 0;
        }
        return false;
    }

    std::uint64_t extract_u64(std::size_t shift) const noexcept {
        const std::size_t index = shift / 64;
        const unsigned offset = static_cast<unsigned>(shift % 64);
        if (index >= limb_count) return 0;
        std::uint64_t result = limbs_[index] >> offset;
        if (offset != 0 && index + 1 < limb_count) {
            result |= limbs_[index + 1] << (64U - offset);
        }
        return result;
    }

    bool rounded_binary64_components(
        std::uint64_t& significand,
        int& exponent) const {
        const std::size_t highest = highest_set_bit();
        if (highest == no_bit) return false;

        if (highest <= 52) {
            // All integers below 2^53 multiplied by 2^-1074 are exactly
            // representable, including the complete subnormal range and the
            // first two normal binades.
            significand = limbs_[0];
            exponent = -1074;
            return true;
        }

        std::size_t rounded_highest = highest;
        const std::size_t discarded_bits = highest - 52;
        significand = extract_u64(discarded_bits);
        constexpr std::uint64_t significand_mask =
            (std::uint64_t{1} << 53) - 1;
        significand &= significand_mask;

        const bool guard = bit(discarded_bits - 1);
        const bool sticky = any_bits_below(discarded_bits - 1);
        if (guard && (sticky || (significand & 1U) != 0)) {
            ++significand;
            if (significand == (std::uint64_t{1} << 53)) {
                significand >>= 1U;
                ++rounded_highest;
            }
        }

        exponent = static_cast<int>(rounded_highest) - 1126;
        return true;
    }

    std::array<std::uint64_t, limb_count> limbs_{};
    std::uint64_t term_count_{0};
};

// Shared exact source-value reduction available to serial and host-threaded
// MPI callers. Uniform mode represents term_count copies of one scalar and
// requires an empty explicit span. Explicit mode requires exactly term_count
// strictly positive finite values. A zero-term explicit sum is valid and
// evaluates to 0.
inline ExactPositiveDoubleSum exact_positive_double_accumulator(
    std::size_t term_count,
    std::span<const double> explicit_values,
    std::optional<double> uniform_value) {
    std::uint64_t terms = 0;
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (term_count > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Exact positive sum term count exceeds uint64_t");
        }
    }
    terms = static_cast<std::uint64_t>(term_count);

    ExactPositiveDoubleSum result;
    if (uniform_value.has_value()) {
        if (!explicit_values.empty()) {
            throw std::invalid_argument(
                "Exact positive sum uniform mode requires an empty explicit span");
        }
        if (!std::isfinite(*uniform_value) || *uniform_value <= 0.0) {
            throw std::invalid_argument(
                "Exact positive sum uniform value must be finite and positive");
        }
        result.add_repeated(*uniform_value, terms);
        return result;
    }

    if (explicit_values.size() != term_count) {
        throw std::invalid_argument(
            "Exact positive sum explicit value count mismatch");
    }
    for (const double value : explicit_values) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(
                "Exact positive sum explicit values must be finite and positive");
        }
        result.add(value);
    }
    return result;
}

inline double exact_positive_double_sum(
    std::size_t term_count,
    std::span<const double> explicit_values,
    std::optional<double> uniform_value) {
    return exact_positive_double_accumulator(
        term_count, explicit_values, uniform_value).value();
}

} // namespace cosmo_nbody::math
