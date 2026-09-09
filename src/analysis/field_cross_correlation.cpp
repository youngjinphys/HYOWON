#include "cosmo_nbody/analysis/field_cross_correlation.hpp"

#include "cosmo_nbody/analysis/cic_spectral_window.hpp"
#include "cosmo_nbody/analysis/fourier_density_field.hpp"
#include "cosmo_nbody/analysis/spectral_numeric.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace analysis {
namespace {

// Keep powers of two separate until after the square root. Dividing by the
// larger operand first can erase the smaller one even when sqrt(lhs*rhs) is
// representable. This also works when long double has binary64 semantics.
template <typename Floating>
inline Floating nonnegative_root_product(Floating lhs, Floating rhs) {
    static_assert(std::is_floating_point_v<Floating>);
    if (!std::isfinite(lhs) || !std::isfinite(rhs)
        || lhs < Floating{0} || rhs < Floating{0}) {
        throw std::invalid_argument(
            "Root product requires finite non-negative operands");
    }
    if (lhs == Floating{0} || rhs == Floating{0}) return Floating{0};
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const Floating lhs_fraction = std::frexp(lhs, &lhs_exponent);
    const Floating rhs_fraction = std::frexp(rhs, &rhs_exponent);
    int exponent = lhs_exponent + rhs_exponent;
    Floating fraction = lhs_fraction * rhs_fraction;
    if (exponent % 2 != 0) {
        fraction *= Floating{2};
        --exponent;
    }
    const Floating result = std::ldexp(std::sqrt(fraction), exponent / 2);
    if (!std::isfinite(result) || result <= Floating{0}) {
        throw std::overflow_error("Root product is not representable");
    }
    return result;
}

// sqrt(numerator/denominator) may be representable even when the quotient is
// not. In particular, a nonzero squared residual must not become a zero ratio
// merely because the division precedes the square root.
template <typename Floating>
inline Floating nonnegative_root_ratio(
    Floating numerator,
    Floating denominator) {
    static_assert(std::is_floating_point_v<Floating>);
    if (!std::isfinite(numerator) || !std::isfinite(denominator)
        || numerator < Floating{0} || denominator <= Floating{0}) {
        throw std::invalid_argument(
            "Root ratio requires finite numerator >= 0 and denominator > 0");
    }
    if (numerator == Floating{0}) return Floating{0};
    int numerator_exponent = 0;
    int denominator_exponent = 0;
    const Floating numerator_fraction = std::frexp(
        numerator, &numerator_exponent);
    const Floating denominator_fraction = std::frexp(
        denominator, &denominator_exponent);
    int exponent = numerator_exponent - denominator_exponent;
    Floating fraction = numerator_fraction / denominator_fraction;
    if (exponent % 2 != 0) {
        fraction *= Floating{2};
        --exponent;
    }
    const Floating result = std::ldexp(std::sqrt(fraction), exponent / 2);
    if (!std::isfinite(result) || result <= Floating{0}) {
        throw std::overflow_error("Root ratio is not representable");
    }
    return result;
}

// A scaled sum of nonnegative squares, as in the LASSQ representation. It
// avoids both dimensional squaring and an overflowing intermediate norm.
// This is bounded-cost floating-point accumulation, not an exact-sum claim.
class ScaledSquareSum {
public:
    void add(long double value, long double weight = 1.0L) {
        if (!std::isfinite(value) || !std::isfinite(weight) || weight < 0.0L) {
            throw std::invalid_argument(
                "Scaled square sum requires finite values and non-negative weights");
        }
        const long double magnitude = std::abs(value);
        if (magnitude == 0.0L || weight == 0.0L) return;
        add_scaled(magnitude, weight);
    }

    void merge(const ScaledSquareSum& other) {
        if (other.scale_ != 0.0L) add_scaled(other.scale_, other.sum_);
    }

    long double root_mean(long double count) const {
        if (!std::isfinite(count) || count <= 0.0L) {
            throw std::invalid_argument(
                "Scaled square mean requires a finite positive count");
        }
        if (scale_ == 0.0L) return 0.0L;
        const long double result = scale_
            * nonnegative_root_ratio(sum_, count);
        if (!std::isfinite(result) || result <= 0.0L) {
            throw std::overflow_error("Scaled square mean is not representable");
        }
        return result;
    }

private:
    void add_scaled(long double scale, long double sum) {
        if (scale_ < scale) {
            const long double ratio = scale_ / scale;
            sum_ = sum + sum_ * ratio * ratio;
            scale_ = scale;
        } else {
            const long double ratio = scale / scale_;
            sum_ += sum * ratio * ratio;
        }
        if (!std::isfinite(sum_)) {
            throw std::overflow_error("Scaled square sum is not representable");
        }
    }

    long double scale_{0.0L};
    long double sum_{0.0L};
};

struct FieldCrossAccumulator {
    long double k_sum{0.0L};
    long double p_a_sum{0.0L};
    long double p_b_sum{0.0L};
    long double p_difference_sum{0.0L};
    long double p_cross_sum{0.0L};
    long double p_delta_sum{0.0L};
    std::size_t mode_count{0};
};

struct CorrelationDiagnostics {
    core::Real r{0.0};
    core::Real absolute_cauchy_excess{0.0};
    bool defined{false};
};

struct RelativeFieldDiagnostics {
    core::Real delta_p_fraction{0.0};
    core::Real transfer_amplitude_ratio{0.0};
    core::Real e_delta{0.0};
    bool defined{false};
};

void checked_add_count(std::size_t& destination, std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - destination) {
        throw std::overflow_error("FieldCrossCorrelation mode count overflow");
    }
    destination += value;
}

core::Real representable_real(long double value, const char* label) {
    return detail::checked_real_result(value, label);
}

// Evaluate the signed change in corrected mode power from the preserved field
// residual instead of subtracting two already-rounded powers. For each real
// component, (a+d)^2-a^2 = a*d + a*d + d*d. Every product contains exactly
// the binary64 factors L^3 and the represented corrected mode components, so
// the cancellation fallback can sum those products exactly before one final
// binary64 rounding.
core::Real preserved_mode_power_difference(
    core::Real box_size,
    const std::complex<core::Real>& reference_mode,
    const std::complex<core::Real>& residual_mode) {
    using Term = math::ExactBinary64ProductTerm;
    const auto term = [box_size](core::Real lhs, core::Real rhs) {
        Term result;
        result.factors = {box_size, box_size, box_size, lhs, rhs};
        result.factor_count = 5U;
        return result;
    };
    const std::array<Term, 6> terms{
        term(reference_mode.real(), residual_mode.real()),
        term(reference_mode.real(), residual_mode.real()),
        term(residual_mode.real(), residual_mode.real()),
        term(reference_mode.imag(), residual_mode.imag()),
        term(reference_mode.imag(), residual_mode.imag()),
        term(residual_mode.imag(), residual_mode.imag()),
    };
    return math::exact_binary64_product_sum(terms);
}

CorrelationDiagnostics raw_correlation_diagnostics(
    const FieldCrossAccumulator& total) {
    if (total.mode_count == 0
        || !std::isfinite(total.p_cross_sum)) {
        throw std::logic_error(
            "FieldCrossCorrelation coefficient received an invalid accumulator");
    }

    const long double denominator = nonnegative_root_product(
        total.p_a_sum, total.p_b_sum);
    if (denominator == 0.0L) {
        if (total.p_cross_sum != 0.0L) {
            throw std::runtime_error(
                "FieldCrossCorrelation has nonzero cross power with zero auto-power normalization");
        }
        return {};
    }

    const long double raw = total.p_cross_sum / denominator;
    if (!std::isfinite(raw)) {
        throw std::overflow_error(
            "FieldCrossCorrelation coefficient is not representable");
    }
    const long double excess = std::max(0.0L, std::abs(raw) - 1.0L);
    return {
        representable_real(raw, "correlation coefficient"),
        representable_real(excess, "absolute Cauchy excess"),
        true};
}

RelativeFieldDiagnostics relative_field_diagnostics(
    const FieldCrossAccumulator& total) {
    if (total.mode_count == 0
        || !std::isfinite(total.p_a_sum)
        || !std::isfinite(total.p_b_sum)
        || !std::isfinite(total.p_difference_sum)
        || !std::isfinite(total.p_delta_sum)
        || total.p_a_sum < 0.0L
        || total.p_b_sum < 0.0L
        || total.p_delta_sum < 0.0L) {
        throw std::logic_error(
            "FieldCrossCorrelation relative diagnostics received an invalid accumulator");
    }
    if (total.p_a_sum == 0.0L) {
        return {};
    }

    const long double delta_p =
        total.p_difference_sum / total.p_a_sum;
    const long double transfer = nonnegative_root_ratio(
        total.p_b_sum, total.p_a_sum);
    const long double e_delta = nonnegative_root_ratio(
        total.p_delta_sum, total.p_a_sum);
    return {
        representable_real(delta_p, "relative power difference"),
        representable_real(transfer, "transfer-amplitude ratio"),
        representable_real(e_delta, "relative field discrepancy"),
        true};
}

} // namespace

FieldCrossCorrelation::FieldCrossCorrelation(
    PeriodicDomain domain,
    int mesh_size,
    config::MemoryPolicyParams memory_policy)
    : domain_(std::move(domain)),
      mesh_size_(mesh_size),
      memory_policy_(std::move(memory_policy)) {
    if (mesh_size_ < 2) {
        throw std::invalid_argument(
            "FieldCrossCorrelation mesh_size must be >= 2");
    }
}

FieldComparisonSummary FieldCrossCorrelation::compare(
    const core::ParticleStore& reference,
    const core::ParticleStore& candidate,
    const FieldComparisonOptions& options) const {
    if (options.num_bins < 1) {
        throw std::invalid_argument(
            "FieldCrossCorrelation num_bins must be positive");
    }
    const core::Real k_max = options.max_k_h_Mpc;
    if (!std::isfinite(k_max) || k_max <= 0.0) {
        throw std::invalid_argument(
            "FieldCrossCorrelation max_k_h_Mpc must be finite and positive");
    }

    const FourierDensityField field_a = FourierDensityBuilder::build(
        domain_, reference, mesh_size_, options.interlaced, memory_policy_);
    const FourierDensityField field_b = FourierDensityBuilder::build(
        domain_, candidate, mesh_size_, options.interlaced, memory_policy_);
    if (field_a.modes.size() != field_b.modes.size()) {
        throw std::logic_error(
            "FieldCrossCorrelation Fourier field sizes differ");
    }

    const core::Real L = domain_.box_size();
    const mesh::MeshGeometry geometry(
        L, static_cast<std::size_t>(mesh_size_));
    const core::Real k_nyquist =
        std::numbers::pi * static_cast<core::Real>(mesh_size_) / L;
    if (k_max > k_nyquist) {
        throw std::invalid_argument(
            "FieldCrossCorrelation absolute k support exceeds the estimator Nyquist frequency");
    }
    const core::Real k_fundamental = 2.0 * std::numbers::pi / L;
    if (!(k_max > k_fundamental)) {
        throw std::invalid_argument(
            "FieldCrossCorrelation requested range contains no nonzero shell");
    }
    const detail::PositiveLogBinGrid bin_grid(
        k_fundamental, k_max, options.num_bins);

    const core::Real cell_size = L / static_cast<core::Real>(mesh_size_);
    const bool even_mesh = (mesh_size_ % 2) == 0;
    const std::size_t grid_size = geometry.grid_size();
    const std::size_t nz_complex = grid_size / 2 + 1;
    const std::size_t bin_count = bin_grid.bin_count();
    if (grid_size > std::numeric_limits<std::size_t>::max() / bin_count) {
        throw std::overflow_error(
            "FieldCrossCorrelation slab accumulator size overflows size_t");
    }

    std::vector<FieldCrossAccumulator> slab_bins(grid_size * bin_count);
    std::vector<ScaledSquareSum> slab_residual_norms(grid_size);
    std::vector<ScaledSquareSum> slab_reference_norms(grid_size);
    std::vector<std::exception_ptr> slab_exceptions(grid_size);
    int failed = 0;

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (std::size_t ix = 0; ix < grid_size; ++ix) {
        try {
            FieldCrossAccumulator* local = slab_bins.data() + ix * bin_count;
            auto& residual_norm = slab_residual_norms[ix];
            auto& reference_norm = slab_reference_norms[ix];
            const core::Real kx = geometry.k_component(ix);
            const long double cic_x = detail::cic_window_axis_wide(kx, cell_size);
            for (std::size_t iy = 0; iy < grid_size; ++iy) {
                const core::Real ky = geometry.k_component(iy);
                const long double cic_y = detail::cic_window_axis_wide(ky, cell_size);
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const core::Real kz = geometry.k_component(iz);
                    const long double cic_z = detail::cic_window_axis_wide(kz, cell_size);
                    const core::Real k = detail::spectral_norm3(kx, ky, kz);
                    if (!(k > 0.0) || k > k_max) continue;

                    const long double window_wide = cic_x * cic_y * cic_z;
                    if (!std::isfinite(window_wide) || window_wide <= 0.0L
                        || window_wide > 1.0L) {
                        failed = 1;
                        continue;
                    }
                    const core::Real window = detail::checked_real_result(
                        window_wide,
                        "FieldCrossCorrelation CIC amplitude window");

                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    const std::complex<core::Real> a_mode =
                        field_a.modes[index] / window;
                    const std::complex<core::Real> b_mode =
                        field_b.modes[index] / window;
                    // Difference before the common window correction: separately
                    // rounded quotients can collapse distinct adjacent coefficients.
                    const std::complex<core::Real> residual =
                        (field_b.modes[index] - field_a.modes[index]) / window;

                    const core::Real p_a =
                        detail::scaled_box_volume_times_complex_norm_squared(
                            L, a_mode, "FieldCrossCorrelation reference power");
                    const core::Real p_b =
                        detail::scaled_box_volume_times_complex_norm_squared(
                            L, b_mode, "FieldCrossCorrelation candidate power");
                    const core::Real p_difference =
                        preserved_mode_power_difference(L, a_mode, residual);
                    const core::Real p_cross =
                        detail::scaled_box_volume_times_complex_cross_real(
                            L, a_mode, b_mode,
                            "FieldCrossCorrelation cross power");
                    const core::Real p_delta =
                        detail::scaled_box_volume_times_complex_norm_squared(
                            L, residual,
                            "FieldCrossCorrelation residual power");
                    if (!std::isfinite(p_a) || p_a < 0.0
                        || !std::isfinite(p_b) || p_b < 0.0
                        || !std::isfinite(p_difference)
                        || !std::isfinite(p_cross)
                        || !std::isfinite(p_delta) || p_delta < 0.0) {
                        failed = 1;
                        continue;
                    }
                    const std::size_t multiplicity =
                        (iz == 0 || (even_mesh && iz == grid_size / 2)) ? 1U : 2U;

                    const int bin_index = bin_grid.inclusive_bin_index(k);
                    if (bin_index >= 0) {
                        auto& bin = local[static_cast<std::size_t>(bin_index)];
                        if (multiplicity
                            > std::numeric_limits<std::size_t>::max()
                                - bin.mode_count) {
                            failed = 1;
                            continue;
                        }
                        const long double mult =
                            static_cast<long double>(multiplicity);
                        bin.k_sum += static_cast<long double>(k) * mult;
                        bin.p_a_sum += static_cast<long double>(p_a) * mult;
                        bin.p_b_sum += static_cast<long double>(p_b) * mult;
                        bin.p_difference_sum +=
                            static_cast<long double>(p_difference) * mult;
                        bin.p_cross_sum +=
                            static_cast<long double>(p_cross) * mult;
                        bin.p_delta_sum +=
                            static_cast<long double>(p_delta) * mult;
                        bin.mode_count += multiplicity;
                    }

                    const long double mult = static_cast<long double>(multiplicity);
                    residual_norm.add(residual.real(), mult);
                    residual_norm.add(residual.imag(), mult);
                    reference_norm.add(a_mode.real(), mult);
                    reference_norm.add(a_mode.imag(), mult);
                }
            }
        } catch (...) {
            slab_exceptions[ix] = std::current_exception();
        }
    }
    for (const auto& exception : slab_exceptions) {
        if (exception) std::rethrow_exception(exception);
    }
    if (failed != 0) {
        throw std::overflow_error(
            "FieldCrossCorrelation parallel slab accumulation failed");
    }

    std::vector<FieldCrossAccumulator> totals(bin_count);
    ScaledSquareSum residual_norm;
    ScaledSquareSum reference_norm;
    for (std::size_t ix = 0; ix < grid_size; ++ix) {
        residual_norm.merge(slab_residual_norms[ix]);
        reference_norm.merge(slab_reference_norms[ix]);
        const FieldCrossAccumulator* local =
            slab_bins.data() + ix * bin_count;
        for (std::size_t bin_index = 0; bin_index < bin_count; ++bin_index) {
            totals[bin_index].k_sum += local[bin_index].k_sum;
            totals[bin_index].p_a_sum += local[bin_index].p_a_sum;
            totals[bin_index].p_b_sum += local[bin_index].p_b_sum;
            totals[bin_index].p_difference_sum +=
                local[bin_index].p_difference_sum;
            totals[bin_index].p_cross_sum += local[bin_index].p_cross_sum;
            totals[bin_index].p_delta_sum += local[bin_index].p_delta_sum;
            checked_add_count(
                totals[bin_index].mode_count, local[bin_index].mode_count);
        }
    }

    std::size_t evaluated_mode_count = 0;
    std::vector<FieldCrossBin> bins(bin_count);
    for (std::size_t bin_index = 0; bin_index < bin_count; ++bin_index) {
        auto& bin = bins[bin_index];
        bin.shell_index = bin_index;
        bin.k_low = bin_grid.edge(bin_index);
        bin.k_high = bin_grid.edge(bin_index + 1U);

        const auto& total = totals[bin_index];
        if (total.mode_count == 0) continue;
        checked_add_count(evaluated_mode_count, total.mode_count);
        const long double count = static_cast<long double>(total.mode_count);
        bin.k_mean = representable_real(total.k_sum / count, "mean wavenumber");
        bin.p_a = representable_real(total.p_a_sum / count, "reference power");
        bin.p_b = representable_real(total.p_b_sum / count, "candidate power");
        bin.p_cross = representable_real(
            total.p_cross_sum / count, "cross power");
        bin.p_delta = representable_real(
            total.p_delta_sum / count, "residual power");
        bin.mode_count = total.mode_count;

        const auto relative = relative_field_diagnostics(total);
        bin.delta_p_fraction = relative.delta_p_fraction;
        bin.transfer_amplitude_ratio = relative.transfer_amplitude_ratio;
        bin.e_delta = relative.e_delta;
        bin.relative_metrics_defined = relative.defined;

        const auto correlation = raw_correlation_diagnostics(total);
        bin.r = correlation.r;
        bin.correlation_defined = correlation.defined;
        bin.absolute_cauchy_excess = correlation.absolute_cauchy_excess;
    }
    if (evaluated_mode_count == 0) {
        throw std::logic_error(
            "FieldCrossCorrelation evaluated range contains no represented mode");
    }

    const long double evaluated_count =
        static_cast<long double>(evaluated_mode_count);
    const long double raw_residual_rms =
        residual_norm.root_mean(evaluated_count);
    const long double raw_reference_rms =
        reference_norm.root_mean(evaluated_count);
    if (!std::isfinite(raw_residual_rms)
        || !std::isfinite(raw_reference_rms)
        || raw_residual_rms > std::numeric_limits<core::Real>::max()
        || raw_reference_rms > std::numeric_limits<core::Real>::max()) {
        throw std::overflow_error(
            "FieldCrossCorrelation raw residual is not representable");
    }

    FieldComparisonSummary summary;
    summary.bins = std::move(bins);
    summary.residual_rms = representable_real(
        raw_residual_rms, "residual RMS");
    summary.reference_rms = representable_real(
        raw_reference_rms, "reference RMS");
    summary.k_fundamental = k_fundamental;
    summary.k_nyquist = k_nyquist;
    if (summary.reference_rms > 0.0) {
        const core::Real ratio = representable_real(
            raw_residual_rms / raw_reference_rms, "normalized residual");
        if (!std::isfinite(ratio) || ratio < 0.0) {
            throw std::overflow_error(
                "FieldCrossCorrelation normalized residual is not representable");
        }
        summary.normalized_residual = ratio;
        summary.normalized_residual_defined = true;
    }
    summary.evaluated_k_max = k_max;

    return summary;
}

} // namespace analysis
} // namespace cosmo_nbody
