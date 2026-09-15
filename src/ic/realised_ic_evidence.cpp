#include "cosmo_nbody/ic/realised_ic_evidence.hpp"

#include "cosmo_nbody/analysis/spectral_numeric.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody::ic {
namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20U) {
                out << "\\u" << std::hex << std::setw(4)
                    << std::setfill('0') << static_cast<unsigned int>(c)
                    << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

void require_finite(core::Real value, const char* label) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string("Realised IC evidence contains non-finite ") + label);
    }
}

std::size_t checked_add_modes(std::size_t current, std::size_t increment) {
    if (increment > std::numeric_limits<std::size_t>::max() - current) {
        throw std::overflow_error("Realised IC evidence mode count overflow");
    }
    return current + increment;
}

std::size_t resolved_fundamental_shell_count(
    core::Real k_min,
    core::Real k_max) {
    if (!(k_max > k_min) || !std::isfinite(k_min) || !std::isfinite(k_max)) {
        return 0;
    }
    const core::Real span = (k_max - k_min) / k_min;
    const core::Real count = std::ceil(span);
    if (!(count >= 1.0)
        || count > static_cast<core::Real>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Realised IC evidence shell count overflow");
    }
    return static_cast<std::size_t>(count);
}

std::size_t deterministic_lane_count(std::size_t work_items) {
    if (work_items == 0) return 0;
    const long double root = std::sqrt(static_cast<long double>(work_items));
    const long double rounded = std::ceil(root);
    if (!(rounded >= 1.0L)
        || rounded > static_cast<long double>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Realised IC evidence lane count overflow");
    }
    return std::min(work_items, static_cast<std::size_t>(rounded));
}

class CompensatedSum {
public:
    void add(long double value) noexcept {
        const long double next = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            correction_ += (sum_ - next) + value;
        } else {
            correction_ += (value - next) + sum_;
        }
        sum_ = next;
    }

    [[nodiscard]] long double value() const noexcept {
        return sum_ + correction_;
    }

private:
    long double sum_{0.0L};
    long double correction_{0.0L};
};

struct FourierAccumulator {
    CompensatedSum k;
    CompensatedSum realised;
    CompensatedSum target;
    std::size_t modes{0};
};

std::string_view forward_cell_edge_status(const RealisedICSummary& structural) {
    if (!structural.forward_cell_edge_determinant_requested) {
        return "not_requested";
    }
    if (structural.forward_cell_edge_determinant_evaluated_count == 0U) {
        return "unavailable";
    }
    if (structural.forward_cell_edge_determinant_unevaluable_count == 0U) {
        return "complete";
    }
    return "partial";
}

void validate_forward_cell_edge_summary(const RealisedICSummary& structural) {
    const std::size_t evaluated =
        structural.forward_cell_edge_determinant_evaluated_count;
    const std::size_t unevaluable =
        structural.forward_cell_edge_determinant_unevaluable_count;
    const std::size_t nonpositive =
        structural.forward_cell_edge_determinant_nonpositive_count;

    if (!structural.forward_cell_edge_determinant_requested) {
        if (evaluated != 0U || unevaluable != 0U || nonpositive != 0U) {
            throw std::runtime_error(
                "Realised IC cell-edge summary contains results although the diagnostic was not requested");
        }
        return;
    }
    if (evaluated > structural.particle_count
        || unevaluable != structural.particle_count - evaluated
        || nonpositive > evaluated) {
        throw std::runtime_error(
            "Realised IC cell-edge summary population accounting is inconsistent");
    }
    if (evaluated != 0U) {
        require_finite(
            structural.forward_cell_edge_determinant_min,
            "minimum forward cell-edge determinant");
        require_finite(
            structural.forward_cell_edge_determinant_max,
            "maximum forward cell-edge determinant");
        if (structural.forward_cell_edge_determinant_min
            > structural.forward_cell_edge_determinant_max) {
            throw std::runtime_error(
                "Realised IC cell-edge determinant extrema are inconsistent");
        }
    }
}

} // namespace

ExactFourierEvidence summarize_generated_fourier_realisation(
    const config::SimulationParameters& config,
    const LinearPowerSpectrum& power_spectrum,
    const mesh::ComplexField& density_k,
    std::size_t ic_mesh_per_dimension) {
    const core::Real box_size = config.get_box().L;
    const std::size_t particle_n = static_cast<std::size_t>(config.get_box().N);
    if (!std::isfinite(box_size) || box_size <= 0.0
        || particle_n == 0 || ic_mesh_per_dimension == 0) {
        throw std::invalid_argument(
            "Realised IC Fourier evidence received invalid box or mesh dimensions");
    }
    const mesh::MeshGeometry geometry(box_size, ic_mesh_per_dimension);
    if (density_k.size() != geometry.complex_size()) {
        throw std::invalid_argument(
            "Realised IC Fourier evidence density shape does not match the IC mesh");
    }

    ExactFourierEvidence result;
    result.k_min = 2.0 * std::numbers::pi / box_size;
    result.k_max = std::numbers::pi
        * static_cast<core::Real>(particle_n) / box_size;
    result.power_normalization = power_spectrum.normalization_diagnostics();
    require_finite(result.k_min, "k_min");
    require_finite(result.k_max, "k_max");
    if (!(result.k_max > result.k_min)) {
        result.status = "unavailable_no_nonzero_particle_nyquist_shell";
        return result;
    }

    const std::size_t shell_count =
        resolved_fundamental_shell_count(result.k_min, result.k_max);
    if (shell_count == 0) {
        result.status = "unavailable_no_resolved_spectral_shell";
        return result;
    }
    result.resolved_shell_count = shell_count;

    const long double mesh_real =
        static_cast<long double>(ic_mesh_per_dimension);
    const long double n3 = mesh_real * mesh_real * mesh_real;
    if (!std::isfinite(n3) || !(n3 > 0.0L)) {
        throw std::overflow_error(
            "Realised IC Fourier normalization is invalid");
    }

    const bool even_mesh = (ic_mesh_per_dimension % 2U) == 0U;
    const std::size_t nz_complex = ic_mesh_per_dimension / 2U + 1U;
    std::vector<std::size_t> transverse_indices;
    std::vector<std::size_t> longitudinal_indices;
    transverse_indices.reserve(ic_mesh_per_dimension);
    longitudinal_indices.reserve(nz_complex);
    const std::size_t conservative_axis_mode = particle_n / 2U + 1U;
    const auto selected_limit = config.get_ic().max_mode_per_axis;
    for (std::size_t index = 0; index < ic_mesh_per_dimension; ++index) {
        const std::size_t wrapped_mode = std::min(
            index, ic_mesh_per_dimension - index);
        if (wrapped_mode <= conservative_axis_mode
            && (!selected_limit || wrapped_mode <= *selected_limit)) {
            transverse_indices.push_back(index);
        }
    }
    for (std::size_t index = 0; index < nz_complex; ++index) {
        if (index <= conservative_axis_mode
            && (!selected_limit || index <= *selected_limit)) {
            longitudinal_indices.push_back(index);
        }
    }

    const std::size_t lane_count =
        deterministic_lane_count(transverse_indices.size());
    std::vector<std::vector<FourierAccumulator>> lane_accumulators(
        lane_count,
        std::vector<FourierAccumulator>(shell_count));
    std::vector<std::exception_ptr> lane_errors(lane_count);

    // Logical lanes are derived only from the represented grid size, not from
    // runtime thread count. Each lane processes a deterministic strided set of
    // ix slabs and lanes merge in index order, so results remain reproducible
    // while auxiliary memory grows as O(sqrt(N) * N) rather than O(N^2).
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) \
        if(runtime::should_use_host_parallel_team(lane_count))
#endif
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        try {
            auto& local = lane_accumulators[lane];
            for (std::size_t slab = lane;
                 slab < transverse_indices.size();
                 slab += lane_count) {
                const std::size_t ix = transverse_indices[slab];
                const core::Real kx = geometry.k_component(ix);
                for (const std::size_t iy : transverse_indices) {
                    const core::Real ky = geometry.k_component(iy);
                    for (const std::size_t iz : longitudinal_indices) {
                        const core::Real kz = geometry.k_component(iz);
                        const core::Real k = core::scale_safe_norm3(kx, ky, kz);
                        if (!(k >= result.k_min) || !(k < result.k_max)) continue;

                        const core::Real scaled =
                            (k - result.k_min) / result.k_min;
                        if (!std::isfinite(scaled) || scaled < 0.0) continue;
                        const auto shell = static_cast<std::size_t>(
                            std::floor(scaled));
                        if (shell >= shell_count) continue;

                        const std::size_t index =
                            geometry.complex_index(ix, iy, iz);
                        const std::complex<core::Real> mode = density_k[index];
                        if (!std::isfinite(mode.real())
                            || !std::isfinite(mode.imag())) {
                            throw std::runtime_error(
                                "Realised IC Fourier mode is non-finite");
                        }
                        const core::Real target = power_spectrum.evaluate(k);
                        if (!std::isfinite(target) || target <= 0.0) continue;
                        const long double realised =
                            analysis::detail::scaled_box_volume_times_complex_norm_squared_ratio_wide(
                                box_size,
                                mode,
                                n3,
                                "Realised IC Fourier power");
                        if (!std::isfinite(realised) || realised < 0.0L) {
                            throw std::runtime_error(
                                "Realised IC Fourier power is invalid");
                        }
                        const std::size_t multiplicity =
                            (iz == 0U
                             || (even_mesh && iz == ic_mesh_per_dimension / 2U))
                            ? 1U : 2U;
                        auto& accumulator = local[shell];
                        const long double weight =
                            static_cast<long double>(multiplicity);
                        accumulator.k.add(static_cast<long double>(k) * weight);
                        accumulator.realised.add(realised * weight);
                        accumulator.target.add(
                            static_cast<long double>(target) * weight);
                        accumulator.modes = checked_add_modes(
                            accumulator.modes, multiplicity);
                    }
                }
            }
        } catch (...) {
            lane_errors[lane] = std::current_exception();
        }
    }

    std::vector<FourierAccumulator> accumulators(shell_count);
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        if (lane_errors[lane]) std::rethrow_exception(lane_errors[lane]);
        for (std::size_t shell = 0; shell < shell_count; ++shell) {
            const auto& partial = lane_accumulators[lane][shell];
            auto& total = accumulators[shell];
            total.k.add(partial.k.value());
            total.realised.add(partial.realised.value());
            total.target.add(partial.target.value());
            total.modes = checked_add_modes(total.modes, partial.modes);
        }
    }

    for (std::size_t shell = 0; shell < shell_count; ++shell) {
        const auto& accumulator = accumulators[shell];
        if (accumulator.modes == 0) continue;
        const long double count = static_cast<long double>(accumulator.modes);
        const long double k_mean_wide = accumulator.k.value() / count;
        const long double realised_power_wide = accumulator.realised.value() / count;
        const long double target_power_wide = accumulator.target.value() / count;
        if (!(target_power_wide > 0.0L)) {
            throw std::runtime_error(
                "Realised IC Fourier shell target power is not positive");
        }

        RealisedICPowerBin bin;
        bin.shell_index = shell;
        bin.k_low = result.k_min
            + static_cast<core::Real>(shell) * result.k_min;
        bin.k_high = std::min(
            result.k_max,
            result.k_min
                + static_cast<core::Real>(shell + 1U) * result.k_min);
        bin.k_mean = analysis::detail::checked_real_result(
            k_mean_wide, "Realised IC Fourier shell k_mean");
        bin.realised_power = analysis::detail::checked_real_result(
            realised_power_wide, "Realised IC Fourier shell realised power");
        bin.target_power = analysis::detail::checked_real_result(
            target_power_wide, "Realised IC Fourier shell target power");
        bin.realised_to_target_ratio = analysis::detail::checked_real_result(
            realised_power_wide / target_power_wide,
            "Realised IC Fourier shell realised-to-target ratio");
        bin.mode_count = accumulator.modes;
        require_finite(bin.k_low, "shell k_low");
        require_finite(bin.k_high, "shell k_high");
        require_finite(bin.k_mean, "shell k_mean");
        require_finite(bin.realised_power, "shell realised power");
        require_finite(bin.target_power, "shell target power");
        require_finite(
            bin.realised_to_target_ratio,
            "shell realised-to-target ratio");
        result.bins.push_back(bin);
    }
    result.status = result.bins.empty()
        ? "unavailable_no_populated_evaluated_shell"
        : (selected_limit
            ? "available_cartesian_ic_window_within_particle_nyquist_sphere"
            : "available_full_isotropic_particle_nyquist_support");
    return result;
}

RealisedICEvidence build_realised_ic_evidence(
    const config::SimulationParameters& config,
    const RealisedICSummary& structural,
    ExactFourierEvidence exact,
    const core::ParticleStore& particles,
    bool include_particle_estimator) {
    // The generated-mode evidence has an exact authority boundary. A second
    // CIC particle-spectrum estimator is intentionally not recomputed here;
    // native particle P(k) belongs to nbody_analyze where mesh, interlacing,
    // shot noise and support are explicit estimator coordinates.
    (void)particles;
    (void)include_particle_estimator;

    RealisedICEvidence evidence;
    evidence.seed = config.get_ic().seed;
    evidence.particles_per_dimension = config.get_box().N;
    evidence.ic_mesh_per_dimension = config.ic_mesh_per_dimension();
    evidence.ic_max_mode_per_axis = config.get_ic().max_mode_per_axis;
    evidence.ic_effective_max_mode_per_axis =
        config.ic_effective_max_mode_per_axis();
    evidence.evolution_mesh_per_dimension = config.get_box().N_mesh;
    evidence.lpt_order = config.get_ic().lpt_order;
    evidence.box_size_Mpc_h = config.get_box().L;
    evidence.start_redshift = config.get_time().z_start;
    evidence.amplitude_mode = config.get_ic().amplitude_mode;
    evidence.phase_pairing = config.get_ic().phase_pairing;
    evidence.power_spectrum_file = config.get_ic().power_spectrum_file;
    evidence.power_spectrum_sha256 = config.get_ic().power_spectrum_sha256;
    evidence.structural = structural;
    evidence.power_normalization = exact.power_normalization;
    evidence.exact_fourier_status = exact.status;
    evidence.evaluated_k_max = exact.k_max;
    evidence.exact_bins = std::move(exact.bins);

    require_finite(evidence.box_size_Mpc_h, "box size");
    require_finite(evidence.start_redshift, "start redshift");
    require_finite(structural.displacement_rms_Mpc_h, "displacement RMS");
    require_finite(structural.displacement_max_Mpc_h, "displacement maximum");
    require_finite(structural.momentum_rms, "momentum RMS");
    require_finite(structural.momentum_max, "momentum maximum");
    validate_forward_cell_edge_summary(structural);
    if (evidence.power_normalization.available) {
        require_finite(
            evidence.power_normalization.declared_sigma8_z0,
            "declared sigma8");
        require_finite(
            evidence.power_normalization.represented_support_sigma8_z0,
            "represented-support sigma8");
        require_finite(
            evidence.power_normalization.represented_support_relative_difference,
            "represented-support sigma8 relative difference");
        require_finite(
            evidence.power_normalization.support_k_min_h_Mpc,
            "power support k_min");
        require_finite(
            evidence.power_normalization.support_k_max_h_Mpc,
            "power support k_max");
    }
    return evidence;
}

std::string RealisedICEvidence::to_json() const {
    validate_forward_cell_edge_summary(structural);
    const std::string_view cell_edge_status =
        forward_cell_edge_status(structural);
    const bool have_cell_edge_extrema =
        structural.forward_cell_edge_determinant_evaluated_count != 0U;

    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n"
        << "  \"product_kind\": \"realised_ic_evidence\",\n"
        << "  \"seed\": " << seed << ",\n"
        << "  \"particles_per_dimension\": "
        << particles_per_dimension << ",\n"
        << "  \"ic_mesh_per_dimension\": "
        << ic_mesh_per_dimension << ",\n"
        << "  \"ic_max_mode_per_axis\": "
        << (ic_max_mode_per_axis ? std::to_string(*ic_max_mode_per_axis) : "null") << ",\n"
        << "  \"ic_effective_max_mode_per_axis\": "
        << ic_effective_max_mode_per_axis << ",\n"
        << "  \"evolution_mesh_per_dimension\": "
        << evolution_mesh_per_dimension << ",\n"
        << "  \"lpt_order\": " << lpt_order << ",\n"
        << "  \"box_size_Mpc_h\": " << box_size_Mpc_h << ",\n"
        << "  \"start_redshift\": " << start_redshift << ",\n"
        << "  \"amplitude_mode\": \""
        << json_escape(amplitude_mode) << "\",\n"
        << "  \"phase_pairing\": \""
        << json_escape(phase_pairing) << "\",\n"
        << "  \"power_spectrum_file\": \""
        << json_escape(power_spectrum_file) << "\",\n"
        << "  \"power_spectrum_sha256\": \""
        << json_escape(power_spectrum_sha256) << "\",\n"
        << "  \"structural\": {\n"
        << "    \"particle_count\": " << structural.particle_count << ",\n"
        << "    \"displacement_rms_Mpc_h\": "
        << structural.displacement_rms_Mpc_h << ",\n"
        << "    \"displacement_max_Mpc_h\": "
        << structural.displacement_max_Mpc_h << ",\n"
        << "    \"momentum_rms\": " << structural.momentum_rms << ",\n"
        << "    \"momentum_max\": " << structural.momentum_max << ",\n"
        << "    \"forward_cell_edge_determinant_requested\": "
        << (structural.forward_cell_edge_determinant_requested ? "true" : "false")
        << ",\n"
        << "    \"forward_cell_edge_determinant_status\": \""
        << cell_edge_status << "\",\n"
        << "    \"forward_cell_edge_determinant_evaluated_count\": "
        << structural.forward_cell_edge_determinant_evaluated_count << ",\n"
        << "    \"forward_cell_edge_determinant_unevaluable_count\": "
        << structural.forward_cell_edge_determinant_unevaluable_count << ",\n"
        << "    \"forward_cell_edge_determinant_min\": ";
    if (have_cell_edge_extrema) {
        out << structural.forward_cell_edge_determinant_min;
    } else {
        out << "null";
    }
    out << ",\n"
        << "    \"forward_cell_edge_determinant_max\": ";
    if (have_cell_edge_extrema) {
        out << structural.forward_cell_edge_determinant_max;
    } else {
        out << "null";
    }
    out << ",\n"
        << "    \"forward_cell_edge_determinant_nonpositive_count\": "
        << structural.forward_cell_edge_determinant_nonpositive_count << "\n"
        << "  },\n"
        << "  \"power_normalization\": {\n"
        << "    \"available\": "
        << (power_normalization.available ? "true" : "false") << ",\n"
        << "    \"declared_sigma8_z0\": "
        << power_normalization.declared_sigma8_z0 << ",\n"
        << "    \"represented_support_sigma8_z0\": "
        << power_normalization.represented_support_sigma8_z0 << ",\n"
        << "    \"represented_support_relative_difference\": "
        << power_normalization.represented_support_relative_difference << ",\n"
        << "    \"support_k_min_h_Mpc\": "
        << power_normalization.support_k_min_h_Mpc << ",\n"
        << "    \"support_k_max_h_Mpc\": "
        << power_normalization.support_k_max_h_Mpc << "\n"
        << "  },\n"
        << "  \"exact_fourier_estimator\": \""
        << json_escape(exact_fourier_estimator) << "\",\n"
        << "  \"exact_fourier_status\": \""
        << json_escape(exact_fourier_status) << "\",\n"
        << "  \"spectral_support\": \""
        << (ic_max_mode_per_axis
            ? "cartesian_ic_window_intersected_with_open_particle_nyquist_sphere"
            : "open_isotropic_particle_nyquist_sphere") << "\",\n"
        << "  \"shell_binning\": \"linear_radial_fundamental_kf\",\n"
        << "  \"particle_spectral_estimator\": \"delegated_to_nbody_analyze\",\n"
        << "  \"evaluated_k_max\": " << evaluated_k_max << ",\n"
        << "  \"shells\": [\n";
    for (std::size_t index = 0; index < exact_bins.size(); ++index) {
        const auto& bin = exact_bins[index];
        out << "    {\"shell_index\": " << bin.shell_index
            << ", \"k_low\": " << bin.k_low
            << ", \"k_high\": " << bin.k_high
            << ", \"k_mean\": " << bin.k_mean
            << ", \"realised_power\": " << bin.realised_power
            << ", \"target_power\": " << bin.target_power
            << ", \"realised_to_target_ratio\": "
            << bin.realised_to_target_ratio
            << ", \"mode_count\": " << bin.mode_count
            << "}" << (index + 1U < exact_bins.size() ? "," : "")
            << '\n';
    }
    out << "  ]\n}\n";
    const std::string json = out.str();
    if (!out) {
        throw std::runtime_error(
            "Failed to serialize realised IC evidence JSON");
    }
    return json;
}

} // namespace cosmo_nbody::ic
