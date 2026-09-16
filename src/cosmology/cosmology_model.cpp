#include "cosmo_nbody/cosmology/cosmology_model.hpp"
#include "cosmo_nbody/cosmology/flat_matter_lambda.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody {
namespace cosmology {

namespace {

struct ExpansionComponents {
    core::Real matter_amplitude{0.0};
    core::Real lambda_amplitude{0.0};
    core::Real expansion{0.0};
};

ExpansionComponents expansion_components(
    core::Real a,
    const config::CosmologyParams& params) {
    if (!std::isfinite(a) || a <= 0.0) {
        throw std::invalid_argument("Scale factor must be finite and positive");
    }

    const core::Real sqrt_a = std::sqrt(a);
    const core::Real sqrt_matter = std::sqrt(params.omega_m);
    const core::Real sqrt_lambda = std::sqrt(params.omega_lambda);
    core::Real matter_amplitude = 0.0;
    if (a <= 1.0) {
        const std::array<core::Real, 1> numerators{sqrt_matter};
        const std::array<core::Real, 2> denominators{a, sqrt_a};
        matter_amplitude = math::scaled_positive_product_quotient(
            numerators,
            denominators,
            "matter expansion amplitude");
    } else {
        const core::Real inverse_sqrt_a = 1.0 / sqrt_a;
        matter_amplitude = sqrt_matter * (inverse_sqrt_a / a);
    }
    if (!std::isfinite(matter_amplitude) || matter_amplitude < 0.0) {
        throw std::overflow_error(
            "Matter expansion amplitude is not representable");
    }

    const core::Real expansion = std::hypot(matter_amplitude, sqrt_lambda);
    if (!std::isfinite(expansion) || expansion <= 0.0) {
        throw std::overflow_error(
            "Dimensionless expansion rate is not representable");
    }
    return {matter_amplitude, sqrt_lambda, expansion};
}

core::Real squared_component_fraction(
    core::Real component,
    core::Real complement) {
    if (component == 0.0) return 0.0;
    if (complement == 0.0) return 1.0;

    if (component >= complement) {
        const core::Real ratio = complement / component;
        return 1.0 / (1.0 + ratio * ratio);
    }
    const core::Real ratio = component / complement;
    const core::Real ratio_squared = ratio * ratio;
    return ratio_squared / (1.0 + ratio_squared);
}

core::Real growth_ode_start_scale(
    const config::CosmologyParams& params) {
    if (params.omega_lambda == 0.0) return 1.0;

    // The EdS initial state is asymptotic, so choose its boundary from the
    // precision of the stored state rather than a repository-owned redshift.
    // At a_start the neglected Lambda/matter ratio is <= one binary64 epsilon:
    //   (Omega_Lambda/Omega_m) a_start^3 <= eps.
    // If that inequality already holds at a=1, the entire supported interval is
    // indistinguishable from EdS at the stored precision.
    const long double epsilon = static_cast<long double>(
        std::numeric_limits<core::Real>::epsilon());
    const long double log_candidate = (
        std::log(epsilon)
        + std::log(static_cast<long double>(params.omega_m))
        - std::log(static_cast<long double>(params.omega_lambda))) / 3.0L;
    if (!std::isfinite(log_candidate)) {
        throw std::overflow_error(
            "Growth ODE matter-era start scale is not representable");
    }
    if (log_candidate >= 0.0L) return 1.0;

    const long double candidate = std::exp(log_candidate);
    if (!std::isfinite(candidate)
        || candidate <= 0.0L
        || candidate > static_cast<long double>(
            std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error(
            "Growth ODE matter-era start scale is not representable");
    }
    const core::Real result = static_cast<core::Real>(candidate);
    if (!std::isfinite(result) || result <= 0.0 || result >= 1.0) {
        throw std::overflow_error(
            "Growth ODE matter-era start scale is not representable in core::Real");
    }
    return result;
}

std::size_t growth_ode_balanced_step_count(core::Real a_start) {
    if (a_start >= 1.0) return 0;
    const long double span = -std::log(static_cast<long double>(a_start));
    if (!std::isfinite(span) || span <= 0.0L) {
        throw std::overflow_error("Growth ODE ln(a) span is invalid");
    }

    // Classical RK4 has global truncation O(h^4). A conservative cumulative
    // floating-point model is O(eps/h). Balancing the two gives h^5~eps,
    // hence h~eps^(1/5). This derives the base resolution from arithmetic and
    // method order instead of choosing an unexplained step count.
    const long double epsilon = static_cast<long double>(
        std::numeric_limits<core::Real>::epsilon());
    const long double target_step = std::pow(epsilon, 1.0L / 5.0L);
    const long double required = std::ceil(span / target_step);
    if (!std::isfinite(required)
        || required < 1.0L
        || required > static_cast<long double>(
            std::numeric_limits<std::size_t>::max() - 2U)) {
        throw std::overflow_error(
            "Growth ODE precision-derived step count is not representable");
    }
    std::size_t steps = static_cast<std::size_t>(required);
    // Nested N/2, N, 2N grids are used below for self-convergence evidence.
    if (steps < 2U) steps = 2U;
    if ((steps & std::size_t{1}) != 0U) ++steps;
    return steps;
}

} // namespace

CosmologyModel::CosmologyModel(const config::CosmologyParams& params)
    : params_(params) {
    if (!std::isfinite(params_.omega_m)
        || !std::isfinite(params_.omega_lambda)
        || params_.omega_m <= 0.0
        || params_.omega_m > 1.0
        || params_.omega_lambda < 0.0) {
        throw std::invalid_argument(
            "Flat matter-plus-Lambda cosmology requires finite Omega_m in (0,1] and finite Omega_lambda >= 0");
    }
    if (!has_flat_matter_lambda_closure(
            params_.omega_m,
            params_.omega_lambda)) {
        throw std::invalid_argument(
            "Cosmology must be flat (Omega_m + Omega_lambda = 1)");
    }
    precompute_growth_table();
}

core::Real CosmologyModel::E(core::Real a) const {
    return expansion_components(a, params_).expansion;
}

core::Real CosmologyModel::H(core::Real a) const {
    // With length Mpc/h and velocity km/s, H0=100 km/s/(Mpc/h).
    constexpr core::Real internal_h0 = 100.0;
    const core::Real expansion = E(a);
    if (expansion > std::numeric_limits<core::Real>::max() / internal_h0) {
        throw std::overflow_error("Hubble rate is not representable");
    }
    const core::Real result = internal_h0 * expansion;
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::overflow_error("Hubble rate is not finite and positive");
    }
    return result;
}

core::Real CosmologyModel::Omega_m(core::Real a) const {
    const ExpansionComponents components = expansion_components(a, params_);
    return squared_component_fraction(
        components.matter_amplitude,
        components.lambda_amplitude);
}

core::Real CosmologyModel::Omega_lambda(core::Real a) const {
    const ExpansionComponents components = expansion_components(a, params_);
    return squared_component_fraction(
        components.lambda_amplitude,
        components.matter_amplitude);
}

void CosmologyModel::growth_ode_rhs(
    core::Real lna,
    const core::Real y[4],
    core::Real dydx[4]) const {
    const core::Real a = std::exp(lna);
    const core::Real om = Omega_m(a);
    const core::Real drag = 2.0 - 1.5 * om;

    dydx[0] = y[1];
    dydx[1] = -drag * y[1] + 1.5 * om * y[0];

    dydx[2] = y[3];
    dydx[3] = -drag * y[3]
        + 1.5 * om * y[2]
        - 1.5 * om * y[0] * y[0];
}

void CosmologyModel::precompute_growth_table() {
    growth_diagnostics_ = {};
    const core::Real a_start = growth_ode_start_scale(params_);
    growth_diagnostics_.start_scale_factor = a_start;
    if (a_start >= 1.0) {
        // Exact EdS behavior, or Lambda so small that its contribution is below
        // the stored binary64 resolution over the entire supported interval.
        growth_diagnostics_.used_eds_shortcut = true;
        growth_table_ = {{
            core::Real{1.0},
            core::Real{1.0},
            core::Real{1.0},
            core::Real{-3.0 / 7.0},
            core::Real{2.0},
        }};
        return;
    }

    const std::size_t base_steps = growth_ode_balanced_step_count(a_start);
    if (base_steps > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("Growth ODE refined step count overflows size_t");
    }
    growth_diagnostics_.coarse_step_count = base_steps / 2U;
    growth_diagnostics_.nominal_step_count = base_steps;
    growth_diagnostics_.refined_step_count = 2U * base_steps;

    const core::Real lna_start = std::log(a_start);

    auto integrate = [&](std::size_t num_steps) {
        if (num_steps == 0U
            || num_steps == std::numeric_limits<std::size_t>::max()) {
            throw std::logic_error("Growth ODE integration requires a finite positive grid");
        }
        const core::Real h = -lna_start
            / static_cast<core::Real>(num_steps);
        if (!std::isfinite(h) || h <= 0.0) {
            throw std::overflow_error("Growth ODE step is invalid");
        }

        std::vector<GrowthPoint> table;
        table.reserve(num_steps + 1U);
        core::Real y[4] = {
            a_start,
            a_start,
            -(3.0 / 7.0) * a_start * a_start,
            -(6.0 / 7.0) * a_start * a_start,
        };

        auto append_point = [&](core::Real a_value) {
            if (!std::isfinite(a_value) || a_value <= 0.0
                || !std::isfinite(y[0]) || y[0] <= 0.0
                || !std::isfinite(y[1])
                || !std::isfinite(y[2]) || y[2] >= 0.0
                || !std::isfinite(y[3])) {
                throw std::runtime_error(
                    "Non-finite or non-growing cosmological growth state");
            }
            table.push_back({
                a_value,
                y[0],
                y[1] / y[0],
                y[2],
                y[3] / y[2],
            });
        };

        append_point(a_start);
        for (std::size_t step = 0; step < num_steps; ++step) {
            // Derive each grid coordinate from the fixed origin to avoid cumulative roundoff.
            const core::Real lna = lna_start
                + static_cast<core::Real>(step) * h;
            core::Real k1[4], k2[4], k3[4], k4[4];
            core::Real tmp[4];

            growth_ode_rhs(lna, y, k1);
            for (int j = 0; j < 4; ++j) {
                tmp[j] = y[j] + 0.5 * h * k1[j];
            }
            growth_ode_rhs(lna + 0.5 * h, tmp, k2);

            for (int j = 0; j < 4; ++j) {
                tmp[j] = y[j] + 0.5 * h * k2[j];
            }
            growth_ode_rhs(lna + 0.5 * h, tmp, k3);

            for (int j = 0; j < 4; ++j) {
                tmp[j] = y[j] + h * k3[j];
            }
            growth_ode_rhs(lna + h, tmp, k4);

            for (int j = 0; j < 4; ++j) {
                y[j] += (h / 6.0)
                    * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
            const core::Real next_lna = step + 1U == num_steps
                ? core::Real{0.0}
                : lna_start + static_cast<core::Real>(step + 1U) * h;
            append_point(std::exp(next_lna));
        }

        const core::Real d1_today = table.back().D1;
        if (!std::isfinite(d1_today) || d1_today <= 0.0) {
            throw std::runtime_error(
                "Invalid present-day linear growth normalization");
        }
        const core::Real d1_today_sq = d1_today * d1_today;
        if (!std::isfinite(d1_today_sq) || d1_today_sq <= 0.0) {
            throw std::overflow_error(
                "Present-day growth normalization square is not representable");
        }
        for (auto& point : table) {
            point.D1 /= d1_today;
            point.D2 /= d1_today_sq;
        }
        return table;
    };

    auto relative_difference = [](core::Real lhs, core::Real rhs) {
        if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
            return std::numeric_limits<core::Real>::infinity();
        }
        const core::Real scale = std::max({
            std::abs(lhs),
            std::abs(rhs),
            std::numeric_limits<core::Real>::min(),
        });
        return std::abs(lhs - rhs) / scale;
    };

    auto nested_difference = [&](const std::vector<GrowthPoint>& coarse,
                                 const std::vector<GrowthPoint>& fine) {
        if (coarse.size() < 2U
            || fine.size() != 2U * (coarse.size() - 1U) + 1U) {
            throw std::logic_error("Growth ODE nested grids are inconsistent");
        }
        GrowthNestedGridDiscrepancy maximum;
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            const auto& lhs = coarse[i];
            const auto& rhs = fine[2U * i];
            maximum.D1 = std::max(
                maximum.D1, relative_difference(lhs.D1, rhs.D1));
            maximum.f1 = std::max(
                maximum.f1, relative_difference(lhs.f1, rhs.f1));
            maximum.D2 = std::max(
                maximum.D2, relative_difference(lhs.D2, rhs.D2));
            maximum.f2 = std::max(
                maximum.f2, relative_difference(lhs.f2, rhs.f2));
        }
        return maximum;
    };

    auto discrepancies_are_finite = [](
        const GrowthNestedGridDiscrepancy& discrepancy) {
        return std::isfinite(discrepancy.D1)
            && std::isfinite(discrepancy.f1)
            && std::isfinite(discrepancy.D2)
            && std::isfinite(discrepancy.f2);
    };

    auto coarse = integrate(growth_diagnostics_.coarse_step_count);
    auto nominal = integrate(growth_diagnostics_.nominal_step_count);
    auto refined = integrate(growth_diagnostics_.refined_step_count);
    growth_diagnostics_.coarse_to_nominal =
        nested_difference(coarse, nominal);
    growth_diagnostics_.nominal_to_refined =
        nested_difference(nominal, refined);
    if (!discrepancies_are_finite(growth_diagnostics_.coarse_to_nominal)
        || !discrepancies_are_finite(
            growth_diagnostics_.nominal_to_refined)) {
        throw std::runtime_error(
            "Growth ODE nested-grid comparison is non-finite");
    }

    // Use the refined grid. The nested-grid discrepancies quantify sensitivity
    // to the internal ODE resolution; they are recorded as diagnostics rather
    // than promoted to a pass/fail accuracy rule. A monotone aggregate alone
    // neither proves every component converged nor establishes an observable's
    // trusted physical domain.
    growth_table_ = std::move(refined);
}

core::Real CosmologyModel::interpolate_component(
    core::Real a,
    int component) const {
    if (!std::isfinite(a) || a <= 0.0) {
        throw std::invalid_argument("Scale factor must be finite and positive");
    }
    if (component < 0 || component > 3) {
        throw std::logic_error("Invalid growth component");
    }
    if (a > 1.0) {
        throw std::out_of_range(
            "Growth table supports a <= 1 only; future-time extrapolation is disabled");
    }

    auto value = [component](const GrowthPoint& p) -> core::Real {
        switch (component) {
            case 0: return p.D1;
            case 1: return p.f1;
            case 2: return p.D2;
            case 3: return p.f2;
            default: throw std::logic_error("Invalid growth component");
        }
    };

    if (a <= growth_table_.front().a) {
        const core::Real ratio = a / growth_table_.front().a;
        if (component == 0) return growth_table_.front().D1 * ratio;
        if (component == 1) return 1.0;
        if (component == 2) return growth_table_.front().D2 * ratio * ratio;
        if (component == 3) return 2.0;
        throw std::logic_error("Invalid growth component");
    }
    if (a >= 1.0) return value(growth_table_.back());

    const auto right = std::lower_bound(
        growth_table_.begin(), growth_table_.end(), a,
        [](const GrowthPoint& point, core::Real target) {
            return point.a < target;
        });
    if (right == growth_table_.begin()) return value(*right);
    if (right == growth_table_.end()) return value(growth_table_.back());

    const auto left = std::prev(right);
    const core::Real interval = right->a - left->a;
    if (!std::isfinite(interval) || interval <= 0.0) {
        throw std::runtime_error("Growth table scale-factor interval is invalid");
    }
    const core::Real t = (a - left->a) / interval;
    if (!std::isfinite(t) || t < 0.0 || t > 1.0) {
        throw std::runtime_error("Growth interpolation coordinate is invalid");
    }

    const bool second_order = component >= 2;
    const core::Real growth_left = second_order ? left->D2 : left->D1;
    const core::Real growth_right = second_order ? right->D2 : right->D1;
    const core::Real rate_left = second_order ? left->f2 : left->f1;
    const core::Real rate_right = second_order ? right->f2 : right->f1;
    const core::Real slope_left = rate_left * growth_left / left->a;
    const core::Real slope_right = rate_right * growth_right / right->a;
    if (!std::isfinite(growth_left) || !std::isfinite(growth_right)
        || !std::isfinite(slope_left) || !std::isfinite(slope_right)) {
        throw std::runtime_error("Growth interpolation endpoint state is invalid");
    }

    const core::Real t2 = t * t;
    const core::Real t3 = t2 * t;
    const core::Real h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const core::Real h10 = t3 - 2.0 * t2 + t;
    const core::Real h01 = -2.0 * t3 + 3.0 * t2;
    const core::Real h11 = t3 - t2;
    const core::Real growth = h00 * growth_left
        + h10 * interval * slope_left
        + h01 * growth_right
        + h11 * interval * slope_right;
    const bool invalid_growth = !std::isfinite(growth)
        || (second_order ? growth >= 0.0 : growth <= 0.0);
    if (invalid_growth) {
        throw std::runtime_error(
            "Cubic growth interpolation violated the growing-mode sign invariant");
    }
    if (component == 0 || component == 2) return growth;

    const core::Real dh00 = 6.0 * t2 - 6.0 * t;
    const core::Real dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    const core::Real dh01 = -dh00;
    const core::Real dh11 = 3.0 * t2 - 2.0 * t;
    const core::Real derivative = (
        dh00 * growth_left
        + dh10 * interval * slope_left
        + dh01 * growth_right
        + dh11 * interval * slope_right) / interval;
    const core::Real logarithmic_rate = a * derivative / growth;
    if (!std::isfinite(logarithmic_rate)) {
        throw std::runtime_error(
            "Cubic growth interpolation produced a non-finite logarithmic rate");
    }
    return logarithmic_rate;
}

core::Real CosmologyModel::D1(core::Real a) const {
    return interpolate_component(a, 0);
}

core::Real CosmologyModel::f1(core::Real a) const {
    return interpolate_component(a, 1);
}

core::Real CosmologyModel::D2(core::Real a) const {
    return interpolate_component(a, 2);
}

core::Real CosmologyModel::f2(core::Real a) const {
    return interpolate_component(a, 3);
}

} // namespace cosmology
} // namespace cosmo_nbody
