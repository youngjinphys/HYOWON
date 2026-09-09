#pragma once

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace cosmo_nbody::gravity::detail {

template <typename Floating>
inline Floating gaussian_long_complement_over_q3(Floating q) {
    static_assert(std::is_floating_point_v<Floating>);
    if (!std::isfinite(q) || q < Floating{0}) {
        throw std::invalid_argument(
            "TreePM Gaussian complement series requires finite q >= 0");
    }

    constexpr Floating pi = static_cast<Floating>(
        3.141592653589793238462643383279502884L);
    const Floating inverse_sqrt_pi = Floating{1} / std::sqrt(pi);
    if (q == Floating{0}) {
        return inverse_sqrt_pi / Floating{6};
    }

    // For q<1, evaluate
    // [erf(q/2)-q/sqrt(pi) exp(-q^2/4)]/q^3
    // = 1/sqrt(pi) sum_n t_n,
    // t_0=1/6,
    // t_(n+1)=-t_n*(q^2/4)*(2n+3)/[(n+1)(2n+5)].
    // The alternating terms decrease on this interval. Stop only when the
    // working format can no longer change the represented sum; no empirical
    // q threshold or accuracy tolerance enters the numerical definition.
    if (q < Floating{1}) {
        const Floating quarter_q2 = q * q / Floating{4};
        Floating term = Floating{1} / Floating{6};
        Floating sum = term;
        for (unsigned long long n = 0;; ++n) {
            const Floating numerator = static_cast<Floating>(2 * n + 3);
            const Floating denominator =
                static_cast<Floating>(n + 1)
                * static_cast<Floating>(2 * n + 5);
            term = -term * quarter_q2 * numerator / denominator;
            if (term == Floating{0}) break;
            const Floating updated = sum + term;
            if (updated == sum) break;
            if (!std::isfinite(updated)) {
                throw std::overflow_error(
                    "TreePM Gaussian complement series became non-finite");
            }
            sum = updated;
        }
        const Floating value = inverse_sqrt_pi * sum;
        if (!std::isfinite(value) || !(value > Floating{0})) {
            throw std::overflow_error(
                "TreePM Gaussian complement series result is invalid");
        }
        return value;
    }

    const Floating gaussian_term =
        (q * inverse_sqrt_pi) * std::exp(-Floating{0.25} * q * q);
    const Floating complement =
        std::erf(Floating{0.5} * q) - gaussian_term;
    const Floating inverse_q = Floating{1} / q;
    const Floating value = complement
        * inverse_q * inverse_q * inverse_q;
    if (!std::isfinite(value) || value < Floating{0}) {
        throw std::overflow_error(
            "TreePM Gaussian complement ratio is invalid");
    }
    return value;
}

} // namespace cosmo_nbody::gravity::detail
