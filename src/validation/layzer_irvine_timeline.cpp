#include "cosmo_nbody/validation/layzer_irvine_timeline.hpp"

#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/validation/layzer_irvine_ratio.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cosmo_nbody::validation {
namespace {

bool nearly_equal_measurement(double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) return false;
    const long double left = static_cast<long double>(lhs);
    const long double right = static_cast<long double>(rhs);
    const long double scale = std::max({1.0L, std::abs(left), std::abs(right)});
    return std::abs(left - right)
        <= 256.0L
            * static_cast<long double>(std::numeric_limits<double>::epsilon())
            * scale;
}

void validate_sample(const LayzerIrvineSampleRecord& sample) {
    if (sample.step == 0
        || !std::isfinite(sample.scale_factor_start)
        || !std::isfinite(sample.scale_factor_end)
        || !std::isfinite(sample.delta_ln_a)
        || sample.scale_factor_start <= 0.0
        || sample.scale_factor_end <= sample.scale_factor_start
        || sample.delta_ln_a <= 0.0) {
        throw std::invalid_argument(
            "Layzer-Irvine sample has invalid step or scale-factor geometry");
    }
    for (const double value : {
             sample.kinetic_energy,
             sample.potential_energy_raw,
             sample.cic_self_energy,
             sample.potential_energy_pair,
             sample.initial_energy,
             sample.source_start,
             sample.source,
             sample.integrated_source,
             sample.integrated_source_compensation,
             sample.residual}) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Layzer-Irvine sample contains a non-finite raw quantity");
        }
    }
    if (sample.kinetic_energy < 0.0) {
        throw std::invalid_argument(
            "Layzer-Irvine kinetic energy must be non-negative");
    }
    if (!nearly_equal_measurement(
            sample.potential_energy_raw - sample.cic_self_energy,
            sample.potential_energy_pair)) {
        throw std::invalid_argument(
            "Layzer-Irvine raw/self/pair potential context is inconsistent");
    }
    const double expected_dln_a = std::log(sample.scale_factor_end)
        - std::log(sample.scale_factor_start);
    if (!nearly_equal_measurement(expected_dln_a, sample.delta_ln_a)) {
        throw std::invalid_argument(
            "Layzer-Irvine sample delta_ln_a disagrees with its scale factors");
    }
    const auto expected_ratio = layzer_irvine_ratio(
        sample.residual,
        sample.kinetic_energy,
        sample.potential_energy_pair);
    if (sample.ratio.has_value() != expected_ratio.has_value()) {
        throw std::invalid_argument(
            "Layzer-Irvine sample ratio availability is inconsistent");
    }
    if (sample.ratio.has_value()
        && (!std::isfinite(*sample.ratio)
            || *sample.ratio < 0.0
            || !nearly_equal_measurement(*sample.ratio, *expected_ratio))) {
        throw std::invalid_argument(
            "Layzer-Irvine sample ratio disagrees with its raw residual context");
    }
}

void append_sample(
    std::ostringstream& out,
    const LayzerIrvineSampleRecord& sample) {
    validate_sample(sample);
    out << "    {\"step\": " << sample.step
        << ", \"scale_factor_start\": " << sample.scale_factor_start
        << ", \"scale_factor_end\": " << sample.scale_factor_end
        << ", \"delta_ln_a\": " << sample.delta_ln_a
        << ", \"kinetic_energy\": " << sample.kinetic_energy
        << ", \"potential_energy_raw\": " << sample.potential_energy_raw
        << ", \"cic_self_energy\": " << sample.cic_self_energy
        << ", \"potential_energy_pair\": " << sample.potential_energy_pair
        << ", \"initial_energy\": " << sample.initial_energy
        << ", \"source_start\": " << sample.source_start
        << ", \"source\": " << sample.source
        << ", \"integrated_source\": " << sample.integrated_source
        << ", \"integrated_source_compensation\": "
        << sample.integrated_source_compensation
        << ", \"residual\": " << sample.residual
        << ", \"ratio\": ";
    if (sample.ratio.has_value()) out << *sample.ratio;
    else out << "null";
    out << ", \"reused_force_potential\": "
        << (sample.reused_force_potential ? "true" : "false") << '}';
}

} // namespace

std::string layzer_irvine_timeline_to_json(
    std::span<const LayzerIrvineSampleRecord> samples) {
    long double previous_integrated = 0.0L;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto& current = samples[index];
        validate_sample(current);
        if (index > 0) {
            const auto& previous = samples[index - 1];
            if (previous.step == std::numeric_limits<std::uint64_t>::max()
                || current.step != previous.step + 1
                || !nearly_equal_measurement(
                    previous.scale_factor_end,
                    current.scale_factor_start)
                || !nearly_equal_measurement(
                    previous.source,
                    current.source_start)) {
                throw std::invalid_argument(
                    "Layzer-Irvine timeline samples are not contiguous");
            }
        }
        const long double observed_integrated =
            static_cast<long double>(current.integrated_source)
            - static_cast<long double>(current.integrated_source_compensation);
        const long double expected_increment =
            0.5L * static_cast<long double>(current.delta_ln_a)
            * (static_cast<long double>(current.source_start)
               + static_cast<long double>(current.source));
        const long double observed_increment =
            observed_integrated - previous_integrated;
        const long double scale = std::max({
            1.0L,
            std::abs(observed_integrated),
            std::abs(previous_integrated),
            std::abs(expected_increment),
            std::abs(observed_increment)});
        if (std::abs(observed_increment - expected_increment)
            > 512.0L
                * static_cast<long double>(std::numeric_limits<double>::epsilon())
                * scale) {
            throw std::invalid_argument(
                "Layzer-Irvine timeline quadrature context is inconsistent");
        }
        previous_integrated = observed_integrated;
    }

    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n"
        << "  \"product_kind\": \"layzer_irvine_timeline\",\n"
        << "  \"verdict_semantics\": false,\n"
        << "  \"scientific_accuracy_certificate\": false,\n"
        << "  \"potential_energy_convention\": \"raw_mesh_plus_active_short_pair_minus_cic_self\",\n"
        << "  \"force_energy_gradient_identity\": \"not_guaranteed_by_matching_mesh_operator\",\n"
        << "  \"residual_interpretation\": \"includes_force_energy_gradient_mismatch_time_integration_and_source_quadrature\",\n"
        << "  \"sample_count\": " << samples.size() << ",\n"
        << "  \"samples\": [";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index == 0) out << '\n';
        else out << ",\n";
        append_sample(out, samples[index]);
    }
    if (!samples.empty()) out << '\n' << "  ";
    out << "]\n}\n";
    if (!out) {
        throw std::runtime_error("Failed to serialize Layzer-Irvine timeline");
    }
    return out.str();
}

void write_layzer_irvine_timeline_atomic(
    std::span<const LayzerIrvineSampleRecord> samples,
    const std::filesystem::path& path) {
    io::write_text_durable_atomic(
        path,
        layzer_irvine_timeline_to_json(samples),
        "Layzer-Irvine timeline");
}

} // namespace cosmo_nbody::validation
