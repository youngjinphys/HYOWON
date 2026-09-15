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

bool same_work_point(
    const ForceEnergyWorkPoint& lhs,
    const ForceEnergyWorkPoint& rhs) {
    return nearly_equal_measurement(lhs.scale_factor, rhs.scale_factor)
        && nearly_equal_measurement(lhs.hubble_rate, rhs.hubble_rate)
        && nearly_equal_measurement(
            lhs.momentum_force_contraction,
            rhs.momentum_force_contraction)
        && nearly_equal_measurement(
            lhs.directional_potential_energy_comoving,
            rhs.directional_potential_energy_comoving)
        && nearly_equal_measurement(
            lhs.directional_cic_self_energy_comoving,
            rhs.directional_cic_self_energy_comoving)
        && nearly_equal_measurement(lhs.power_defect, rhs.power_defect)
        && nearly_equal_measurement(
            lhs.power_defect_over_hubble,
            rhs.power_defect_over_hubble);
}

void validate_work_point(const ForceEnergyWorkPoint& point) {
    if (!std::isfinite(point.scale_factor) || point.scale_factor <= 0.0
        || !std::isfinite(point.hubble_rate) || point.hubble_rate <= 0.0) {
        throw std::invalid_argument(
            "Force-energy work point has invalid scale factor or Hubble rate");
    }
    for (const double value : {
             point.momentum_force_contraction,
             point.directional_potential_energy_comoving,
             point.directional_cic_self_energy_comoving,
             point.power_defect,
             point.power_defect_over_hubble}) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Force-energy work point contains a non-finite value");
        }
    }
    const long double a = static_cast<long double>(point.scale_factor);
    const long double expected_power =
        (static_cast<long double>(point.momentum_force_contraction)
         + static_cast<long double>(
             point.directional_potential_energy_comoving)
         - static_cast<long double>(
             point.directional_cic_self_energy_comoving))
        / (a * a * a);
    if (!std::isfinite(expected_power)
        || !nearly_equal_measurement(
            static_cast<double>(expected_power), point.power_defect)
        || !nearly_equal_measurement(
            point.power_defect / point.hubble_rate,
            point.power_defect_over_hubble)) {
        throw std::invalid_argument(
            "Force-energy work point is algebraically inconsistent");
    }
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

    if (sample.force_energy_work.has_value()) {
        const auto& work = *sample.force_energy_work;
        validate_work_point(work.start);
        validate_work_point(work.end);
        if (!nearly_equal_measurement(
                work.start.scale_factor, sample.scale_factor_start)
            || !nearly_equal_measurement(
                work.end.scale_factor, sample.scale_factor_end)
            || !std::isfinite(work.integrated_work)
            || !std::isfinite(work.integrated_work_compensation)
            || !std::isfinite(work.closure_residual)) {
            throw std::invalid_argument(
                "Force-energy work interval disagrees with its Layzer-Irvine sample");
        }
        // Preserve the stored low component through cancellation, matching the
        // producer; rounding work first can reject a resolved closure remainder.
        const ForceEnergyWorkState work_state{
            true, work.end,
            work.integrated_work, work.integrated_work_compensation};
        if (!nearly_equal_measurement(
                force_energy_closure_residual(sample.residual, work_state),
                work.closure_residual)) {
            throw std::invalid_argument(
                "Force-energy closure residual is inconsistent");
        }
    }
}

void append_work_point(
    std::ostringstream& out,
    const ForceEnergyWorkPoint& point) {
    out << "{\"scale_factor\": " << point.scale_factor
        << ", \"hubble_rate\": " << point.hubble_rate
        << ", \"momentum_force_contraction\": "
        << point.momentum_force_contraction
        << ", \"directional_potential_energy_comoving\": "
        << point.directional_potential_energy_comoving
        << ", \"directional_cic_self_energy_comoving\": "
        << point.directional_cic_self_energy_comoving
        << ", \"power_defect\": " << point.power_defect
        << ", \"power_defect_over_hubble\": "
        << point.power_defect_over_hubble << '}';
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
        << (sample.reused_force_potential ? "true" : "false")
        << ", \"force_energy_work\": ";
    if (!sample.force_energy_work.has_value()) {
        out << "null";
    } else {
        const auto& work = *sample.force_energy_work;
        out << "{\"start\": ";
        append_work_point(out, work.start);
        out << ", \"end\": ";
        append_work_point(out, work.end);
        out << ", \"integrated_work\": " << work.integrated_work
            << ", \"integrated_work_compensation\": "
            << work.integrated_work_compensation
            << ", \"closure_residual\": " << work.closure_residual
            << '}';
    }
    out << '}';
}

} // namespace

std::string layzer_irvine_timeline_to_json(
    std::span<const LayzerIrvineSampleRecord> samples) {
    long double previous_integrated = 0.0L;
    long double previous_force_energy_work = 0.0L;
    std::optional<ForceEnergyWorkPoint> previous_work_point;
    std::optional<bool> work_presence;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto& current = samples[index];
        validate_sample(current);
        if (!work_presence.has_value()) {
            work_presence = current.force_energy_work.has_value();
        } else if (*work_presence != current.force_energy_work.has_value()) {
            throw std::invalid_argument(
                "Layzer-Irvine timeline mixes force-energy work availability");
        }
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

        if (current.force_energy_work.has_value()) {
            const auto& work = *current.force_energy_work;
            if (previous_work_point.has_value()
                && !same_work_point(*previous_work_point, work.start)) {
                throw std::invalid_argument(
                    "Force-energy work endpoint measurements are not contiguous");
            }
            const long double observed_work =
                static_cast<long double>(work.integrated_work)
                - static_cast<long double>(work.integrated_work_compensation);
            const long double expected_work_increment =
                0.5L * static_cast<long double>(current.delta_ln_a)
                * (static_cast<long double>(work.start.power_defect_over_hubble)
                   + static_cast<long double>(work.end.power_defect_over_hubble));
            const long double observed_work_increment =
                observed_work - previous_force_energy_work;
            const long double work_scale = std::max({
                1.0L,
                std::abs(observed_work),
                std::abs(previous_force_energy_work),
                std::abs(expected_work_increment),
                std::abs(observed_work_increment)});
            if (std::abs(observed_work_increment - expected_work_increment)
                > 512.0L
                    * static_cast<long double>(
                        std::numeric_limits<double>::epsilon())
                    * work_scale) {
                throw std::invalid_argument(
                    "Force-energy work timeline quadrature is inconsistent");
            }
            previous_force_energy_work = observed_work;
            previous_work_point = work.end;
        }
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
        << "  \"force_energy_work_availability\": \"pure_pm_fixed_comoving_kernel_only\",\n"
        << "  \"force_energy_work_timing\": \"synchronized_endpoint_momentum_after_K2\",\n"
        << "  \"force_energy_work_cic_boundary_derivative\": \"selected_cell_one_sided_at_exact_boundary\",\n"
        << "  \"closure_residual_interpretation\": \"layzer_irvine_residual_minus_measured_force_energy_work;still_contains_time_quadrature_boundary_and_roundoff_effects\",\n"
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
