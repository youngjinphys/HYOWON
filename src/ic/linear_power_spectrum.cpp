#include "cosmo_nbody/ic/linear_power_spectrum.hpp"
#include "cosmo_nbody/ic/particle_lattice_bandlimit.hpp"
#include "cosmo_nbody/ic/spherical_tophat_window.hpp"
#include "cosmo_nbody/cosmology/cosmology_model.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/math/scaled_positive_product.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cosmo_nbody {
namespace ic {

namespace {

using SemanticBindings = std::map<std::string, std::string>;

constexpr std::array<std::string_view, 15> required_semantic_keys{
    "k_unit",
    "power_unit",
    "redshift",
    "species",
    "gauge",
    "fidelity",
    "normalization",
    "normalization_value",
    "nonlinear",
    "massive_neutrino_sum_eV",
    "h",
    "omega_m",
    "omega_b",
    "omega_lambda",
    "n_s"};

// Positive abscissae and weights of the 16-point Gauss-Legendre rule. These
// are coefficients of the quadrature formula, not scientific run parameters.
constexpr std::array<core::Real, 8> sigma8_gl_x{
    0.0950125098376374,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.7554044083550030,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499};
constexpr std::array<core::Real, 8> sigma8_gl_w{
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541};

std::string trim_ascii(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](unsigned char byte) {
            return std::isspace(byte) != 0;
        });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](unsigned char byte) {
            return std::isspace(byte) != 0;
        }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

bool required_semantic_key(std::string_view key) {
    return std::find(
        required_semantic_keys.begin(),
        required_semantic_keys.end(),
        key) != required_semantic_keys.end();
}

void parse_semantic_comment(
    const std::string& line,
    std::size_t line_number,
    const std::string& filepath,
    bool data_started,
    SemanticBindings& bindings) {
    const auto marker = line.find('#');
    if (marker == std::string::npos) return;
    const std::string comment = trim_ascii(line.substr(marker + 1));
    const auto separator = comment.find('=');
    if (separator == std::string::npos) return;
    const std::string key = trim_ascii(comment.substr(0, separator));
    const std::string value = trim_ascii(comment.substr(separator + 1));
    if (!required_semantic_key(key)) return;

    if (value.empty()
        || std::any_of(value.begin(), value.end(), [](unsigned char byte) {
            return std::isspace(byte) != 0;
        })) {
        throw std::runtime_error(
            "Power-spectrum semantic binding '" + key
            + "' must contain one non-empty value at "
            + filepath + ":" + std::to_string(line_number));
    }
    if (data_started) {
        throw std::runtime_error(
            "Power-spectrum semantic binding appears after data at "
            + filepath + ":" + std::to_string(line_number));
    }
    if (!bindings.emplace(key, value).second) {
        throw std::runtime_error(
            "Duplicate power-spectrum semantic binding '" + key + "' at "
            + filepath + ":" + std::to_string(line_number));
    }
}

core::Real parse_semantic_real(
    const SemanticBindings& bindings,
    std::string_view key,
    const std::string& filepath) {
    const auto found = bindings.find(std::string(key));
    if (found == bindings.end()) {
        throw std::runtime_error(
            "Power-spectrum semantic header is missing '" + std::string(key)
            + "' in " + filepath);
    }
    try {
        std::size_t consumed = 0;
        const core::Real value = std::stod(found->second, &consumed);
        if (consumed != found->second.size() || !std::isfinite(value)) {
            throw std::invalid_argument("not a finite scalar");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Power-spectrum semantic binding '" + std::string(key)
            + "' is not a finite scalar in " + filepath);
    }
}

void require_semantic_value(
    const SemanticBindings& bindings,
    std::string_view key,
    std::string_view expected,
    const std::string& filepath) {
    const auto found = bindings.find(std::string(key));
    if (found == bindings.end()) {
        throw std::runtime_error(
            "Power-spectrum semantic header is missing '" + std::string(key)
            + "' in " + filepath);
    }
    if (found->second != expected) {
        throw std::runtime_error(
            "Power-spectrum semantic binding '" + std::string(key)
            + "' is '" + found->second + "' but expected '"
            + std::string(expected) + "' in " + filepath);
    }
}

void require_semantic_real(
    const SemanticBindings& bindings,
    std::string_view key,
    core::Real expected,
    const std::string& filepath) {
    const core::Real actual = parse_semantic_real(bindings, key, filepath);
    if (!std::isfinite(expected) || actual != expected) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Power-spectrum semantic binding '" << key
                << "' is " << actual << " but expected " << expected
                << " in " << filepath;
        throw std::runtime_error(message.str());
    }
}

void validate_semantic_bindings(
    const SemanticBindings& bindings,
    const config::SimulationParameters& config,
    const std::string& filepath) {
    for (const std::string_view key : required_semantic_keys) {
        if (!bindings.contains(std::string(key))) {
            throw std::runtime_error(
                "Power-spectrum semantic header is missing '" + std::string(key)
                + "' in " + filepath);
        }
    }
    require_semantic_value(bindings, "k_unit", "h/Mpc", filepath);
    require_semantic_value(
        bindings, "power_unit", "(Mpc/h)^3", filepath);
    require_semantic_value(
        bindings, "species", "cold_plus_baryon", filepath);
    // These are input semantics, not a certification of the external solver.
    // There is no alternate unnormalised or approximate generation path.
    require_semantic_value(bindings, "fidelity", "precision_boltzmann", filepath);
    require_semantic_value(
        bindings, "fidelity", config.get_ic().power_spectrum_fidelity, filepath);
    require_semantic_value(bindings, "gauge", "synchronous", filepath);
    require_semantic_value(bindings, "normalization", "sigma8_z0", filepath);

    const auto& cosmology = config.get_cosmology();
    if (!(cosmology.sigma8 > 0.0)) {
        throw std::runtime_error(
            "Power-spectrum sigma8_z0 normalization requires positive configured sigma8 in "
            + filepath);
    }
    require_semantic_real(
        bindings, "normalization_value", cosmology.sigma8, filepath);

    require_semantic_value(bindings, "nonlinear", "false", filepath);
    require_semantic_real(
        bindings, "massive_neutrino_sum_eV", 0.0, filepath);
    require_semantic_real(
        bindings,
        "redshift",
        config.get_ic().power_spectrum_redshift,
        filepath);
    require_semantic_real(bindings, "h", cosmology.h, filepath);
    require_semantic_real(bindings, "omega_m", cosmology.omega_m, filepath);
    require_semantic_real(bindings, "omega_b", cosmology.omega_b, filepath);
    require_semantic_real(
        bindings, "omega_lambda", cosmology.omega_lambda, filepath);
    require_semantic_real(bindings, "n_s", cosmology.n_s, filepath);
}

bool blank_or_comment(const std::string& line) {
    const auto first = std::find_if_not(
        line.begin(), line.end(), [](unsigned char value) {
            return std::isspace(value) != 0;
        });
    return first == line.end() || *first == '#';
}

std::pair<core::Real, core::Real> parse_power_row(
    const std::string& line,
    std::size_t line_number,
    const std::string& filepath) {
    std::istringstream fields(line);
    core::Real k = 0.0;
    core::Real power = 0.0;
    if (!(fields >> k >> power)) {
        throw std::runtime_error(
            "Malformed power-spectrum row at " + filepath + ":"
            + std::to_string(line_number) + "; expected exactly two numbers");
    }
    fields >> std::ws;
    if (fields.peek() != std::char_traits<char>::eof() && fields.peek() != '#') {
        throw std::runtime_error(
            "Unexpected trailing token in power-spectrum row at " + filepath + ":"
            + std::to_string(line_number));
    }
    if (!std::isfinite(k) || !std::isfinite(power) || k <= 0.0 || power <= 0.0) {
        throw std::runtime_error(
            "Power-spectrum k and P(k) must be finite and positive at "
            + filepath + ":" + std::to_string(line_number));
    }
    return {k, power};
}

template <typename Integrand>
core::Accum sigma8_gauss_legendre_16(
    const Integrand& integrand,
    core::Real lower,
    core::Real upper,
    const std::string& filepath) {
    const core::Real width = upper - lower;
    const core::Real midpoint = lower + core::Real{0.5} * width;
    const core::Real half_width = core::Real{0.5} * width;
    if (!std::isfinite(width) || width <= 0.0
        || !(lower < midpoint && midpoint < upper)
        || !std::isfinite(half_width) || half_width <= 0.0) {
        throw std::runtime_error(
            "Power-spectrum sigma8 quadrature interval is not representable in "
            + filepath);
    }

    core::Accum sum = 0.0L;
    core::Accum compensation = 0.0L;
    for (std::size_t index = 0; index < sigma8_gl_x.size(); ++index) {
        const core::Real left = midpoint - half_width * sigma8_gl_x[index];
        const core::Real right = midpoint + half_width * sigma8_gl_x[index];
        if (!(lower < left && left < right && right < upper)) {
            throw std::runtime_error(
                "Power-spectrum sigma8 quadrature nodes collapsed in binary64 in "
                + filepath);
        }
        const core::Real left_value = integrand(left);
        const core::Real right_value = integrand(right);
        if (!std::isfinite(left_value) || !std::isfinite(right_value)
            || left_value < 0.0 || right_value < 0.0) {
            throw std::runtime_error(
                "Power-spectrum sigma8 quadrature integrand is invalid in "
                + filepath);
        }
        const core::Accum term =
            static_cast<core::Accum>(sigma8_gl_w[index])
            * (static_cast<core::Accum>(left_value)
               + static_cast<core::Accum>(right_value));
        const core::Accum updated = sum + term;
        if (std::abs(sum) >= std::abs(term)) {
            compensation += (sum - updated) + term;
        } else {
            compensation += (term - updated) + sum;
        }
        sum = updated;
    }

    const core::Accum result =
        static_cast<core::Accum>(half_width) * (sum + compensation);
    if (!std::isfinite(result) || result < 0.0L) {
        throw std::runtime_error(
            "Power-spectrum sigma8 quadrature result is invalid in " + filepath);
    }
    return result;
}

template <typename Integrand>
core::Accum error_controlled_sigma8_gauss_legendre_16(
    const Integrand& integrand,
    core::Real lower,
    core::Real upper,
    const std::string& filepath) {
    struct PendingInterval {
        core::Real lower;
        core::Real upper;
        core::Accum whole;
    };

    std::vector<PendingInterval> pending;
    pending.push_back({
        lower,
        upper,
        sigma8_gauss_legendre_16(integrand, lower, upper, filepath),
    });

    const core::Accum relative_tolerance = std::sqrt(
        static_cast<core::Accum>(std::numeric_limits<core::Real>::epsilon()));
    if (!std::isfinite(relative_tolerance) || relative_tolerance <= 0.0L) {
        throw std::logic_error(
            "Power-spectrum sigma8 arithmetic-derived quadrature tolerance is invalid");
    }

    core::Accum total = 0.0L;
    core::Accum compensation = 0.0L;
    auto accumulate = [&](core::Accum contribution) {
        const core::Accum updated = total + contribution;
        if (std::abs(total) >= std::abs(contribution)) {
            compensation += (total - updated) + contribution;
        } else {
            compensation += (contribution - updated) + total;
        }
        total = updated;
    };

    while (!pending.empty()) {
        const PendingInterval interval = pending.back();
        pending.pop_back();
        const core::Real midpoint = interval.lower
            + core::Real{0.5} * (interval.upper - interval.lower);
        if (!(interval.lower < midpoint && midpoint < interval.upper)) {
            throw std::runtime_error(
                "Power-spectrum sigma8 quadrature exhausted binary64 interval resolution "
                "before meeting its arithmetic-derived error target in " + filepath);
        }

        const core::Accum left = sigma8_gauss_legendre_16(
            integrand, interval.lower, midpoint, filepath);
        const core::Accum right = sigma8_gauss_legendre_16(
            integrand, midpoint, interval.upper, filepath);
        const core::Accum refined = left + right;
        if (!std::isfinite(refined) || refined < 0.0L) {
            throw std::runtime_error(
                "Power-spectrum sigma8 refined quadrature result is invalid in "
                + filepath);
        }

        const core::Accum error_estimate = std::abs(refined - interval.whole);
        const core::Accum scale = std::max({
            std::abs(refined),
            std::abs(interval.whole),
            std::numeric_limits<core::Accum>::min(),
        });
        if (error_estimate <= relative_tolerance * scale) {
            accumulate(refined);
            continue;
        }

        if (pending.size() > pending.max_size() - 2U) {
            throw std::overflow_error(
                "Power-spectrum sigma8 quadrature worklist exceeds vector capacity");
        }
        pending.push_back({midpoint, interval.upper, right});
        pending.push_back({interval.lower, midpoint, left});
    }

    const core::Accum result = total + compensation;
    if (!std::isfinite(result) || result < 0.0L) {
        throw std::runtime_error(
            "Power-spectrum sigma8 error-controlled quadrature result is invalid in "
            + filepath);
    }
    return result;
}

} // namespace

LinearPowerSpectrum::LinearPowerSpectrum(
    const config::SimulationParameters& config) {
    const std::string filepath = config.get_ic().power_spectrum_file;
    const core::Real table_z = config.get_ic().power_spectrum_redshift;
    const core::Real target_z = config.get_time().z_start;
    if (filepath.empty()) {
        throw std::invalid_argument("LinearPowerSpectrum requires a valid file path");
    }

    std::ifstream source(filepath, std::ios::binary);
    if (!source.is_open()) {
        throw std::runtime_error("Could not open power spectrum file: " + filepath);
    }
    const std::string payload{
        std::istreambuf_iterator<char>(source),
        std::istreambuf_iterator<char>()};
    if (source.bad()) {
        throw std::runtime_error(
            "I/O failure while capturing power spectrum file: " + filepath);
    }
    source.clear();
    source.close();
    if (source.fail()) {
        throw std::runtime_error(
            "I/O failure while closing power spectrum file: " + filepath);
    }

    const std::string admitted_sha256 = io::sha256_text(payload);
    const std::string configured_sha256 =
        config.get_ic().power_spectrum_sha256;
    if (!configured_sha256.empty()
        && admitted_sha256 != configured_sha256) {
        throw std::runtime_error(
            "Linear power-spectrum input SHA-256 differs from the configured identity");
    }
    io::require_file_sha256(
        filepath, admitted_sha256, "Linear power-spectrum input");

    cosmology::CosmologyModel cosmo(config.get_cosmology());
    const core::Real d1_table = cosmo.D1(1.0 / (1.0 + table_z));
    const core::Real d1_target = cosmo.D1(1.0 / (1.0 + target_z));
    const core::Real scale_factor = std::pow(d1_target / d1_table, 2.0);
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::runtime_error(
            "Power-spectrum growth rescaling produced a non-finite or non-positive factor");
    }

    std::istringstream in(payload);

    std::vector<core::Real> log_k;
    std::vector<core::Real> log_p;
    SemanticBindings semantic_bindings;
    std::string line;
    std::size_t line_number = 0;
    bool data_started = false;
    core::Real table_k_min = 0.0;
    core::Real table_k_max = 0.0;
    while (std::getline(in, line)) {
        ++line_number;
        if (blank_or_comment(line)) {
            parse_semantic_comment(
                line,
                line_number,
                filepath,
                data_started,
                semantic_bindings);
            continue;
        }
        data_started = true;
        const auto [k_value, power_value] =
            parse_power_row(line, line_number, filepath);
        if (log_k.empty()) table_k_min = k_value;
        table_k_max = k_value;
        const core::Real scaled_power = power_value * scale_factor;
        if (!std::isfinite(scaled_power) || scaled_power <= 0.0) {
            throw std::overflow_error(
                "Growth-scaled power is non-finite or non-positive at "
                + filepath + ":" + std::to_string(line_number));
        }
        log_k.push_back(std::log(k_value));
        log_p.push_back(std::log(scaled_power));
    }
    if (!in.eof() && in.fail()) {
        throw std::runtime_error(
            "I/O failure while parsing captured power spectrum bytes: " + filepath);
    }
    io::require_file_sha256(
        filepath, admitted_sha256, "Linear power-spectrum input");

    validate_semantic_bindings(semantic_bindings, config, filepath);

    if (log_k.size() < 2) {
        throw std::runtime_error(
            "Power spectrum table must have at least 2 valid positive entries");
    }
    for (std::size_t idx = 1; idx < log_k.size(); ++idx) {
        if (!(log_k[idx] > log_k[idx - 1])) {
            throw std::runtime_error(
                "Power spectrum k values must be strictly increasing");
        }
    }

    k_min_ = table_k_min;
    k_max_ = table_k_max;
    if (!std::isfinite(k_min_) || !std::isfinite(k_max_)
        || k_min_ <= 0.0 || k_max_ <= k_min_) {
        throw std::overflow_error(
            "Power spectrum support is not representable");
    }

    normalization_diagnostics_.support_k_min_h_Mpc = k_min_;
    normalization_diagnostics_.support_k_max_h_Mpc = k_max_;

    const core::Real k_fund = 2.0 * std::numbers::pi / config.get_box().L;
    const std::uint64_t max_component_mode =
        config.ic_effective_max_mode_per_axis();
    if (max_component_mode == 0) {
        throw std::invalid_argument(
            "Generated IC particle lattice has no non-zero representable Fourier mode");
    }
    const auto max_mode_signed = static_cast<std::int64_t>(max_component_mode);
    const core::Real k_required_max = represented_mode_wavenumber(
        max_mode_signed, max_mode_signed, max_mode_signed, k_fund);
    if (!std::isfinite(k_required_max) || k_required_max <= 0.0) {
        throw std::overflow_error(
            "Generated IC represented Fourier support maximum is not finite and positive");
    }

    if (k_fund < k_min_) {
        throw std::runtime_error(
            "Power spectrum table does not cover the fundamental box mode");
    }
    if (k_required_max > k_max_) {
        throw std::runtime_error(
            "Power spectrum table does not cover the full represented particle-lattice Fourier band");
    }

    log_spline_.build(log_k, log_p);

    if (semantic_bindings.at("normalization") == "sigma8_z0") {
        constexpr core::Real r8_mpc_h = 8.0;

        auto integrand_at = [&](core::Real log_k_value) -> core::Real {
            const core::Real k = std::exp(log_k_value);
            const core::Real log_p_start =
                log_spline_.interpolate(log_k_value);
            const core::Real p_start = std::exp(log_p_start);
            if (!std::isfinite(log_p_start)
                || !std::isfinite(p_start)
                || p_start <= 0.0) {
                throw std::runtime_error(
                    "Power-spectrum spline reconstruction is not finite and positive in "
                    + filepath);
            }
            const std::array<core::Real, 4> numerators{k, k, k, p_start};
            const std::array<core::Real, 1> denominators{
                2.0 * std::numbers::pi * std::numbers::pi};
            const core::Real delta2 = math::scaled_positive_product_quotient(
                numerators,
                denominators,
                "power-spectrum sigma8 dimensionless power");
            const core::Real w =
                spherical_tophat_fourier_window(k * r8_mpc_h);
            const core::Real value = delta2 * w * w;
            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error(
                    "Power-spectrum sigma8 integrand is invalid in " + filepath);
            }
            return value;
        };

        // Use the production spline knots as the initial partition so sparse
        // tables cannot hide represented features behind a global grid. Each
        // interval is then bisected until the split-vs-unsplit GL16 estimate is
        // below sqrt(binary64 epsilon), a tolerance derived from the arithmetic
        // used to evaluate the integrand rather than a repository-owned decimal
        // threshold. No fixed recursion-depth escape hatch is used.
        core::Accum integral = 0.0L;
        core::Accum compensation = 0.0L;
        for (std::size_t interval = 1; interval < log_k.size(); ++interval) {
            const core::Accum contribution =
                error_controlled_sigma8_gauss_legendre_16(
                    integrand_at,
                    log_k[interval - 1],
                    log_k[interval],
                    filepath);
            const core::Accum updated = integral + contribution;
            if (std::abs(integral) >= std::abs(contribution)) {
                compensation += (integral - updated) + contribution;
            } else {
                compensation += (contribution - updated) + integral;
            }
            integral = updated;
        }
        const core::Accum total_integral = integral + compensation;
        if (!std::isfinite(total_integral) || total_integral < 0.0L
            || total_integral
                > static_cast<core::Accum>(
                    std::numeric_limits<core::Real>::max())) {
            throw std::runtime_error(
                "Power-spectrum sigma8 represented-support integral is non-finite "
                "or not representable in " + filepath);
        }
        const core::Real sigma8_sq_start =
            static_cast<core::Real>(total_integral);

        const core::Real represented_support_sigma8_z0 =
            std::sqrt(sigma8_sq_start) / d1_target;
        const core::Real declared_sigma8_z0 = config.get_cosmology().sigma8;
        if (!std::isfinite(represented_support_sigma8_z0)
            || represented_support_sigma8_z0 < 0.0
            || !(declared_sigma8_z0 > 0.0)) {
            throw std::runtime_error(
                "Power-spectrum sigma8 represented-support diagnostic is invalid in "
                + filepath);
        }

        const core::Real relative_difference =
            std::abs(represented_support_sigma8_z0 - declared_sigma8_z0)
            / declared_sigma8_z0;
        if (!std::isfinite(relative_difference) || relative_difference < 0.0) {
            throw std::runtime_error(
                "Power-spectrum sigma8 represented-support relative diagnostic is invalid in "
                + filepath);
        }

        normalization_diagnostics_.available = true;
        normalization_diagnostics_.declared_sigma8_z0 = declared_sigma8_z0;
        normalization_diagnostics_.represented_support_sigma8_z0 =
            represented_support_sigma8_z0;
        normalization_diagnostics_.represented_support_relative_difference =
            relative_difference;
    }
}

core::Real LinearPowerSpectrum::evaluate(core::Real k) const {
    if (!std::isfinite(k)) {
        throw std::invalid_argument(
            "LinearPowerSpectrum query must be finite");
    }
    if (k < 0.0) {
        throw std::invalid_argument(
            "LinearPowerSpectrum query must be a non-negative wavenumber magnitude");
    }
    if (k == 0.0) return 0.0;
    if (k < k_min_ || k > k_max_) {
        throw std::out_of_range(
            "LinearPowerSpectrum query lies outside the tabulated support");
    }
    const core::Real result =
        std::exp(log_spline_.interpolate(std::log(k)));
    if (!std::isfinite(result) || result <= 0.0) {
        throw std::overflow_error(
            "Interpolated linear power is not finite and positive");
    }
    return result;
}

core::Real LinearPowerSpectrum::dimensionless_power(core::Real k) const {
    if (!std::isfinite(k)) {
        throw std::invalid_argument(
            "LinearPowerSpectrum query must be finite");
    }
    if (k < 0.0) {
        throw std::invalid_argument(
            "LinearPowerSpectrum query must be a non-negative wavenumber magnitude");
    }
    if (k == 0.0) return 0.0;
    const core::Real power = evaluate(k);
    const std::array<core::Real, 4> numerators{k, k, k, power};
    const std::array<core::Real, 1> denominators{
        2.0 * std::numbers::pi * std::numbers::pi};
    return math::scaled_positive_product_quotient(
        numerators,
        denominators,
        "dimensionless linear power");
}

} // namespace ic
} // namespace cosmo_nbody