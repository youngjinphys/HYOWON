// Exact squared-norm comparison for IEEE-754 binary64 values.
#pragma once

#include "cosmo_nbody/core/portable_bit_cast.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody::core::detail {

static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::radix == 2);
static_assert(std::numeric_limits<double>::digits == 53);
static_assert(std::numeric_limits<double>::max_exponent == 1024);
static_assert(std::numeric_limits<double>::min_exponent == -1021);
static_assert(sizeof(double) == sizeof(std::uint64_t));

struct ExactNormUInt128 {
    std::uint64_t high{0};
    std::uint64_t low{0};
};

// Positive dyadic value represented as mantissa * 2^exponent. The mantissa is
// correctly rounded to binary64 and normalized into [0.5, 1). The exponent is
// kept separately so a mathematically nonzero gap may remain usable even when
// its dimensional value would underflow or overflow a standalone double.
struct ExactBinary64PositiveDyadic {
    double mantissa{0.0};
    int exponent{0};
};

inline ExactNormUInt128 exact_norm_multiply_u64(
    std::uint64_t lhs,
    std::uint64_t rhs) noexcept {
    const std::uint64_t lhs_low = static_cast<std::uint32_t>(lhs);
    const std::uint64_t lhs_high = lhs >> 32U;
    const std::uint64_t rhs_low = static_cast<std::uint32_t>(rhs);
    const std::uint64_t rhs_high = rhs >> 32U;

    const std::uint64_t product_low_low = lhs_low * rhs_low;
    const std::uint64_t product_low_high = lhs_low * rhs_high;
    const std::uint64_t product_high_low = lhs_high * rhs_low;
    const std::uint64_t product_high_high = lhs_high * rhs_high;

    std::uint64_t low = product_low_low;
    std::uint64_t carry = 0;

    std::uint64_t previous = low;
    low += product_low_high << 32U;
    if (low < previous) ++carry;

    previous = low;
    low += product_high_low << 32U;
    if (low < previous) ++carry;

    const std::uint64_t high = product_high_high
        + (product_low_high >> 32U)
        + (product_high_low >> 32U)
        + carry;
    return {high, low};
}

// A finite binary64 is an integer multiple of 2^-1074. Its square is therefore
// an integer multiple of 2^-2148. The largest finite square reaches bit 4195 in
// those units: |x*y| < 2^2048 becomes an integer < 2^4196. Each side of a
// three-axis across-boundary norm has at most 15 positive / 12 negative
// products. Comparing two distances cross-adds at most 27 products per side,
// hence each integer is < 27*2^4196 < 2^4201. A radius comparison needs at most
// 15 / 13 products; the AABB expressions below need at most 12 / 7. Vmax radius
// rounding compares 4*distance^2 with (lower+upper)^2 and needs at most 60 / 52
// products, still < 2^4202. Thus 66*64 = 4224 bits suffice for these expressions,
// with at least 22 spare top bits.
// This is a bound on these callers, not on arbitrary repeated accumulation;
// add_word checks any eventual overflow. The sparse representation stores at most
// one nonzero word per conceptual limb, so 66 entries is a structural capacity,
// not a term-count admission threshold. This also permits exact algebraic
// combination of periodic-distance expressions without adding a second,
// expression-specific sparse-entry limit.
class ExactBinary64SquareAccumulator {
public:
    static constexpr std::size_t limb_count = 66;
    static constexpr std::size_t max_entries = limb_count;
    static_assert(limb_count * 64 >= 4202);

    void add_square(double value) {
        const std::uint64_t bits =
            portable_bit_cast<std::uint64_t>(value)
            & ~(std::uint64_t{1} << 63U);
        const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
        const std::uint64_t fraction_bits =
            bits & ((std::uint64_t{1} << 52U) - 1U);

        std::uint64_t significand = fraction_bits;
        std::size_t square_shift = 0;
        if (exponent_bits != 0) {
            significand |= std::uint64_t{1} << 52U;
            // normal value = significand * 2^(exponent_bits - 1075)
            // square in 2^-2148 units therefore shifts by 2*exponent_bits-2.
            square_shift = static_cast<std::size_t>(2U * exponent_bits - 2U);
        }
        if (significand == 0) return;

        add_shifted(
            exact_norm_multiply_u64(significand, significand),
            square_shift);
    }

    void add_product(double lhs, double rhs) {
        const auto decode = [](double value) {
            const std::uint64_t bits =
                portable_bit_cast<std::uint64_t>(value)
                & ~(std::uint64_t{1} << 63U);
            const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
            const std::uint64_t fraction_bits =
                bits & ((std::uint64_t{1} << 52U) - 1U);
            if (exponent_bits != 0) {
                return std::pair<std::uint64_t, long long>{
                    fraction_bits | (std::uint64_t{1} << 52U),
                    static_cast<long long>(exponent_bits) - 1075};
            }
            return std::pair<std::uint64_t, long long>{fraction_bits, -1074};
        };

        const auto [lhs_significand, lhs_unit_exp] = decode(lhs);
        const auto [rhs_significand, rhs_unit_exp] = decode(rhs);
        if (lhs_significand == 0 || rhs_significand == 0) return;

        const long long shift = lhs_unit_exp + rhs_unit_exp + 2148;
        if (shift < 0) {
            throw std::overflow_error(
                "Exact binary64 product underflowed the proven 2^-2148 limb grid");
        }
        add_shifted(
            exact_norm_multiply_u64(lhs_significand, rhs_significand),
            static_cast<std::size_t>(shift));
    }

    void add_accumulator(const ExactBinary64SquareAccumulator& rhs) {
        for (std::size_t position = 0; position < rhs.entry_count_; ++position) {
            add_word(rhs.entries_[position].index, rhs.entries_[position].word);
        }
    }

    int compare(const ExactBinary64SquareAccumulator& rhs) const noexcept {
        int lhs_position = static_cast<int>(entry_count_) - 1;
        int rhs_position = static_cast<int>(rhs.entry_count_) - 1;
        for (;;) {
            while (lhs_position >= 0
                   && entries_[lhs_position].word == 0) {
                --lhs_position;
            }
            while (rhs_position >= 0
                   && rhs.entries_[rhs_position].word == 0) {
                --rhs_position;
            }
            if (lhs_position < 0 || rhs_position < 0) {
                if (lhs_position >= 0) return 1;
                if (rhs_position >= 0) return -1;
                return 0;
            }

            const std::uint8_t lhs_index = entries_[lhs_position].index;
            const std::uint8_t rhs_index = rhs.entries_[rhs_position].index;
            if (lhs_index < rhs_index) return -1;
            if (lhs_index > rhs_index) return 1;
            if (entries_[lhs_position].word
                < rhs.entries_[rhs_position].word) {
                return -1;
            }
            if (entries_[lhs_position].word
                > rhs.entries_[rhs_position].word) {
                return 1;
            }
            --lhs_position;
            --rhs_position;
        }
    }

    ExactBinary64PositiveDyadic positive_difference(
        const ExactBinary64SquareAccumulator& rhs) const {
        if (compare(rhs) <= 0) {
            throw std::invalid_argument(
                "Exact binary64 squared-norm difference must be strictly positive");
        }

        // Gap extraction is a rare boundary fallback, not the hot comparison
        // path. Materialize the conceptual 66-word integers only here so the
        // ordinary neighbor predicate retains its sparse accumulator cost.
        std::array<std::uint64_t, limb_count> difference{};
        std::array<std::uint64_t, limb_count> rhs_dense{};
        for (std::size_t position = 0; position < entry_count_; ++position) {
            difference[entries_[position].index] = entries_[position].word;
        }
        for (std::size_t position = 0; position < rhs.entry_count_; ++position) {
            rhs_dense[rhs.entries_[position].index] =
                rhs.entries_[position].word;
        }

        std::uint64_t borrow = 0;
        for (std::size_t index = 0; index < limb_count; ++index) {
            const std::uint64_t rhs_word = rhs_dense[index];
            const std::uint64_t subtrahend = rhs_word + borrow;
            const bool carry_into_subtrahend = subtrahend < rhs_word;
            const std::uint64_t lhs_word = difference[index];
            difference[index] = lhs_word - subtrahend;
            borrow = (carry_into_subtrahend || lhs_word < subtrahend)
                ? 1U : 0U;
        }
        if (borrow != 0) {
            throw std::logic_error(
                "Exact binary64 squared-norm subtraction violated ordering");
        }

        std::size_t highest_limb = limb_count;
        while (highest_limb > 0
               && difference[highest_limb - 1] == 0) {
            --highest_limb;
        }
        if (highest_limb == 0) {
            throw std::logic_error(
                "Exact binary64 positive squared-norm difference became zero");
        }
        --highest_limb;

        unsigned highest_bit_in_word = 0;
        for (std::uint64_t word = difference[highest_limb]; word > 1U;
             word >>= 1U) {
            ++highest_bit_in_word;
        }
        std::size_t total_bits =
            64U * highest_limb + highest_bit_in_word + 1U;

        constexpr std::size_t precision_bits = 53;
        std::uint64_t significand = 0;
        if (total_bits <= precision_bits) {
            // total_bits <= 53 implies the complete integer occupies limb 0.
            significand = difference[0]
                << static_cast<unsigned>(precision_bits - total_bits);
        } else {
            const std::size_t shift = total_bits - precision_bits;
            const std::size_t word_index = shift / 64U;
            const unsigned offset = static_cast<unsigned>(shift % 64U);
            significand = difference[word_index] >> offset;
            if (offset != 0U && word_index + 1U < limb_count) {
                significand |= difference[word_index + 1U]
                    << (64U - offset);
            }
            significand &= (std::uint64_t{1} << precision_bits) - 1U;

            // Correctly round the discarded exact integer remainder to nearest,
            // ties to even. This avoids depending on the host long-double ABI.
            const std::size_t halfway_bit = shift - 1U;
            const std::size_t halfway_word = halfway_bit / 64U;
            const unsigned halfway_offset =
                static_cast<unsigned>(halfway_bit % 64U);
            const bool halfway_set =
                ((difference[halfway_word] >> halfway_offset) & 1U) != 0U;
            bool lower_nonzero = false;
            for (std::size_t index = 0;
                 index < halfway_word && !lower_nonzero;
                 ++index) {
                lower_nonzero = difference[index] != 0U;
            }
            if (!lower_nonzero && halfway_offset != 0U) {
                const std::uint64_t lower_mask =
                    (std::uint64_t{1} << halfway_offset) - 1U;
                lower_nonzero =
                    (difference[halfway_word] & lower_mask) != 0U;
            }
            if (halfway_set
                && (lower_nonzero || (significand & 1U) != 0U)) {
                ++significand;
                if (significand == (std::uint64_t{1} << precision_bits)) {
                    significand >>= 1U;
                    ++total_bits;
                }
            }
        }

        const double mantissa = std::ldexp(
            static_cast<double>(significand),
            -static_cast<int>(precision_bits));
        const int exponent = static_cast<int>(total_bits) - 2148;
        if (!std::isfinite(mantissa)
            || !(mantissa >= 0.5) || !(mantissa < 1.0)) {
            throw std::logic_error(
                "Exact binary64 squared-norm gap normalization failed");
        }
        return {mantissa, exponent};
    }

private:
    struct Entry {
        std::uint8_t index{0};
        std::uint64_t word{0};
    };

    void add_word(std::size_t index, std::uint64_t word) {
        while (word != 0) {
            if (index >= limb_count) {
                throw std::overflow_error(
                    "Exact binary64 squared norm exceeded its proven limb envelope");
            }

            std::size_t position = 0;
            while (position < entry_count_
                   && entries_[position].index < index) {
                ++position;
            }
            if (position < entry_count_
                && entries_[position].index == index) {
                const std::uint64_t previous = entries_[position].word;
                entries_[position].word = previous + word;
                word = entries_[position].word < previous ? 1U : 0U;
                ++index;
                continue;
            }

            if (entry_count_ >= max_entries) {
                throw std::overflow_error(
                    "Exact binary64 squared norm exceeded its structural limb capacity");
            }
            for (std::size_t move = entry_count_;
                 move > position;
                 --move) {
                entries_[move] = entries_[move - 1];
            }
            entries_[position] = {
                static_cast<std::uint8_t>(index), word};
            ++entry_count_;
            return;
        }
    }

    void add_shifted(ExactNormUInt128 value, std::size_t shift) {
        const std::size_t index = shift / 64U;
        const unsigned offset = static_cast<unsigned>(shift % 64U);
        if (offset == 0) {
            add_word(index, value.low);
            add_word(index + 1, value.high);
            return;
        }

        add_word(index, value.low << offset);
        add_word(
            index + 1,
            (value.high << offset) | (value.low >> (64U - offset)));
        add_word(index + 2, value.high >> (64U - offset));
    }

    std::array<Entry, max_entries> entries_{};
    std::uint8_t entry_count_{0};
};

// Compare x^2+y^2+z^2 to radius^2 exactly as mathematical values of the four
// finite binary64 inputs. Return -1, 0, or +1. Input validation remains owned by
// the public geometric predicate so this low-level helper has one responsibility.
inline int exact_binary64_norm3_compare(
    double x,
    double y,
    double z,
    double radius) {
    ExactBinary64SquareAccumulator lhs;
    lhs.add_square(x);
    lhs.add_square(y);
    lhs.add_square(z);
    ExactBinary64SquareAccumulator rhs;
    rhs.add_square(radius);
    return lhs.compare(rhs);
}

// Compare the exact axis-aligned cube
//   {center + delta : |delta_i| <= half}
// against the open ball of radius `radius`. Returns -1 if the exact minimum
// distance is at least the radius, +1 if the exact maximum distance is
// strictly below the radius, and 0 if the cube straddles the sphere.
// Components of |delta|-half and |delta|+half are never materialized, so a
// nearest-even remainder cannot change the geometric class.
inline int exact_binary64_aabb_cutoff_relation(
    double dx,
    double dy,
    double dz,
    double half,
    double radius) {
    const double axes[3] = {dx, dy, dz};
    ExactBinary64SquareAccumulator min_lhs;
    ExactBinary64SquareAccumulator min_rhs;
    min_rhs.add_square(radius);
    ExactBinary64SquareAccumulator max_lhs;
    ExactBinary64SquareAccumulator max_rhs;
    max_rhs.add_square(radius);
    max_lhs.add_square(half);
    max_lhs.add_square(half);
    max_lhs.add_square(half);
    for (const double component : axes) {
        const double absolute = std::abs(component);
        max_lhs.add_square(absolute);
        max_lhs.add_product(absolute, half);
        max_lhs.add_product(absolute, half);
        if (absolute <= half) continue;
        min_lhs.add_square(absolute);
        min_lhs.add_square(half);
        min_rhs.add_product(absolute, half);
        min_rhs.add_product(absolute, half);
    }
    if (min_lhs.compare(min_rhs) >= 0) return -1;
    if (max_lhs.compare(max_rhs) < 0) return 1;
    return 0;
}

// Return the exact positive dyadic gap radius^2-(x^2+y^2+z^2), normalized as a
// correctly-rounded binary64 mantissa plus a separate binary exponent. This is
// intentionally distinct from the hot comparator: callers pay the dense
// subtraction cost only after exact membership has already identified a rare
// inside-boundary fallback.
inline ExactBinary64PositiveDyadic
exact_binary64_norm3_squared_gap_below_radius(
    double x,
    double y,
    double z,
    double radius) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
        || !std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument(
            "Exact binary64 squared-norm gap requires finite inputs and non-negative radius");
    }
    ExactBinary64SquareAccumulator norm;
    norm.add_square(x);
    norm.add_square(y);
    norm.add_square(z);
    ExactBinary64SquareAccumulator boundary;
    boundary.add_square(radius);
    return boundary.positive_difference(norm);
}

} // namespace cosmo_nbody::core::detail
