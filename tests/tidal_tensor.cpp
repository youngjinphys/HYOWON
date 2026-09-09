#include "cosmo_nbody/analysis/filament_statistics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
using namespace cosmo_nbody;
using Index = std::array<int, 3>;

int wrap(int x, int n) { return (x % n + n) % n; }
std::size_t cell(const Index& x, int n) {
    return (static_cast<std::size_t>(x[0]) * n + x[1]) * n + x[2];
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Density>
analysis::TidalWebField build(
    int n, Density density, double threshold,
    Index permutation = {0, 1, 2}, Index signs = {1, 1, 1},
    double radius = 0.0) {
    core::ParticleStore particles;
    particles.resize(static_cast<std::size_t>(n) * n * n);
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int z = 0; z < n; ++z) {
                const Index point{x, y, z};
                const auto i = cell(point, n);
                particles.get_positions_x()[i] = wrap(signs[0] * point[permutation[0]], n);
                particles.get_positions_y()[i] = wrap(signs[1] * point[permutation[1]], n);
                particles.get_positions_z()[i] = wrap(signs[2] * point[permutation[2]], n);
                particles.get_ids()[i] = i;
                // Exact mesh-node placement makes CIC deposition equal this
                // strictly positive density, without assignment interpolation.
                particles.get_masses()[i] = 1.0 + density(point);
            }
    analysis::TidalWebOptions options;
    options.mesh_size = n;
    options.lambda_threshold = threshold;
    options.gaussian_smoothing_radius = radius;
    return analysis::FilamentStatistics::build_tidal_web_field(
        analysis::PeriodicDomain(n), particles, options);
}

void signed_permutation_covariance() {
    for (const int n : {8, 7}) {
        // On even grids these exercise double, single, and no Nyquist axes;
        // on odd grids the highest represented frequency is not Nyquist.
        for (int kind = 0; kind < 3; ++kind) {
            const auto density = [=](const Index& p) {
                const double low = 0.2 * std::cos(
                    2 * std::numbers::pi * (p[0] + p[1]) / n);
                if (kind == 2) return low;
                const int my = kind == 0 ? n / 2 : 1;
                return low + 0.2 * std::cos(
                    2 * std::numbers::pi * ((n / 2) * p[0] + my * p[1]) / n);
            };
            // Stay away from analytic eigenvalue boundaries: a threshold at
            // exactly 0.1 can turn FFT roundoff into a classification change.
            const auto reference = build(n, density, 0.113);
            Index permutation{0, 1, 2};
            do {
                for (int mask = 0; mask < 8; ++mask) {
                    const Index signs{
                        mask & 1 ? -1 : 1,
                        mask & 2 ? -1 : 1,
                        mask & 4 ? -1 : 1};
                    const auto transformed = build(n, density, 0.113, permutation, signs);
                    for (int x = 0; x < n; ++x)
                        for (int y = 0; y < n; ++y)
                            for (int z = 0; z < n; ++z) {
                                const Index p{x, y, z};
                                Index q{};
                                for (int axis = 0; axis < 3; ++axis)
                                    q[axis] = wrap(signs[axis] * p[permutation[axis]], n);
                                const auto i = cell(p, n);
                                const auto j = cell(q, n);
                                require(reference.environment[i] == transformed.environment[j],
                                    "T-web reflection/permutation changed environment: N="
                                    + std::to_string(n) + " kind=" + std::to_string(kind)
                                    + " mask=" + std::to_string(mask));
                                require(reference.filament_direction_valid[i]
                                            == transformed.filament_direction_valid[j],
                                    "T-web reflection/permutation changed direction validity");
                                // Eigenvectors describe unoriented axes. Compare
                                // their absolute dot product only when both
                                // eigensystems report a resolved filament axis.
                                if (reference.filament_direction_valid[i]
                                    && transformed.filament_direction_valid[j]) {
                                    const auto a = reference.filament_direction[i];
                                    const auto b = transformed.filament_direction[j];
                                    const double av[3]{a.x, a.y, a.z};
                                    const double bv[3]{b.x, b.y, b.z};
                                    double dot = 0;
                                    for (int axis = 0; axis < 3; ++axis)
                                        dot += signs[axis] * av[permutation[axis]] * bv[axis];
                                    require(std::abs(dot) > 1.0 - 1e-10,
                                        "T-web filament direction failed tensor covariance");
                                }
                            }
                }
            } while (std::next_permutation(permutation.begin(), permutation.end()));
        }
    }
}

void analytic_eigenvalues() {
    using E = analysis::WebEnvironment;
    constexpr int n = 8;
    // The real tensor-product interpolants cos(pi*x)cos(pi*y) and
    // cos(pi*x)cos(pi*y)cos(pi*z) have zero mixed Hessians at the nodes.
    // Poisson trace gives eigenvalues (0.4,0.4,0) and (0.2,0.2,0.2)
    // at the origin. Threshold brackets test the diagonal Nyquist amplitude
    // through the public classifier, which does not expose tensor entries.
    const auto double_nyquist = [](const Index& p) {
        return (p[0] + p[1]) % 2 ? -0.8 : 0.8;
    };
    require(build(n, double_nyquist, 0.3).environment[0] == E::Filament,
        "Double-Nyquist diagonal amplitude or mixed derivative is incorrect");
    require(build(n, double_nyquist, 0.5).environment[0] == E::Void,
        "Double-Nyquist eigenvalue exceeds its analytic value");
    const auto triple_nyquist = [](const Index& p) {
        return (p[0] + p[1] + p[2]) % 2 ? -0.6 : 0.6;
    };
    require(build(n, triple_nyquist, 0.1).environment[0] == E::Node,
        "Triple-Nyquist diagonal components were suppressed");
    require(build(n, triple_nyquist, 0.3).environment[0] == E::Void,
        "Triple-Nyquist eigenvalue exceeds its analytic value");
    // Nyquist in the undifferentiated z axis must not erase T_xy.
    // For (kx,ky,kz)=(1,1,4), the origin eigenvalues are
    // (0.6*16/18, 0.6*2/18, 0), since only xz and yz vanish.
    const auto transverse_nyquist = [](const Index& p) {
        return (p[2] % 2 ? -0.6 : 0.6)
            * std::cos(2 * std::numbers::pi * (p[0] + p[1]) / n);
    };
    require(build(n, transverse_nyquist, 0.05).environment[0] == E::Filament,
        "Nyquist on the third axis incorrectly suppressed an ordinary mixed derivative");
    require(build(n, transverse_nyquist, 0.08).environment[0] == E::Sheet,
        "Transverse-Nyquist eigenvalue exceeds its analytic value");
    for (const int size : {7, 8}) {
        const auto plane = [=](const Index& p) {
            return 0.6 * std::cos(2 * std::numbers::pi * (p[0] + p[1]) / size);
        };
        // A non-Nyquist plane wave has one eigenvalue delta*W and two zeros.
        const double radius = 0.4;
        const double k = 2 * std::numbers::pi / size;
        const double eigenvalue = 0.6 * std::exp(-k * k * radius * radius);
        require(build(size, plane, eigenvalue * 0.9, {0, 1, 2}, {1, 1, 1}, radius)
                    .environment[0] == E::Sheet,
            "Non-Nyquist plane-wave amplitude/smoothing changed");
        require(build(size, plane, eigenvalue * 1.1, {0, 1, 2}, {1, 1, 1}, radius)
                    .environment[0] == E::Void,
            "Non-Nyquist plane-wave eigenvalue exceeds its analytic value");
    }
}
} // namespace

int main() {
    try {
        signed_permutation_covariance();
        analytic_eigenvalues();
        std::cout << "T-web signed permutations and analytic eigenvalue checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
