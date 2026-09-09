#include "cosmo_nbody/cosmology/cosmology_model.hpp"
#include "cosmo_nbody/cosmology/drift_kick_factors.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/gravity/force_split.hpp"
#include "cosmo_nbody/gravity/pm_solver.hpp"
#include "cosmo_nbody/time/leapfrog_integrator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
using namespace cosmo_nbody;

void near(double actual, long double expected, long double tolerance, const char* label) {
    if (!std::isfinite(actual)
        || std::abs(actual - expected) > tolerance * std::max(1.0L, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": expected "
            + std::to_string(static_cast<double>(expected)) + ", got " + std::to_string(actual));
    }
}

// Independent integral growing mode for flat matter+Lambda (Hamilton 2001,
// https://arxiv.org/abs/astro-ph/0006089, eq. 1 and its footnote).
// x=t^2 removes the endpoint square root; composite Simpson is independent
// of the production RK4 ODE and cubic interpolation.
long double growth_integral(long double a, long double om, int count) {
    const long double h = std::sqrt(a) / count;
    long double sum = 0;
    for (int i = 0; i <= count; ++i) {
        const long double t = i * h;
        const long double t2 = t * t;
        const long double denominator = om + (1 - om) * t2 * t2 * t2;
        const long double value = 2 * t2 * t2 / std::pow(denominator, 1.5L);
        sum += (i == 0 || i == count ? 1 : i % 2 ? 4 : 2) * value;
    }
    return sum * h / 3;
}

void growth_and_factors() {
    double maximum_growth_error = 0;
    for (double om : {0.1, 0.3, 0.9, 1.0}) {
        cosmology::CosmologyModel model({0.7, om, 1 - om, 0.05, 0.8, 1.0});
        const auto normalization = growth_integral(1, om, 4096);
        for (double a : {0.001, 0.01, 0.1, 0.37, 0.8, 1.0}) {
            const long double expansion = std::sqrt(om / (a * a * a) + (1 - om));
            const long double integral = growth_integral(a, om, 4096);
            const long double coarser = growth_integral(a, om, 2048);
            near(static_cast<double>(coarser / integral), 1, 1e-11, "growth oracle refinement");
            const long double d1 = expansion * integral / normalization;
            const long double matter = om / (a * a * a * expansion * expansion);
            const long double f1 = -1.5L * matter
                + 1 / (a * a * expansion * expansion * expansion * integral);
            near(model.D1(a) / static_cast<double>(d1), 1, 5e-9, "linear growth");
            near(model.f1(a), f1, 5e-9, "linear growth rate");
            maximum_growth_error = std::max(maximum_growth_error,
                std::abs(model.D1(a) / static_cast<double>(d1) - 1));
            if (om == 1) {
                near(model.D2(a) / (a * a), -3.0L / 7, 2e-15, "EdS second-order growth");
                near(model.f2(a), 2, 2e-15, "EdS second-order rate");
            }
        }
    }
    cosmology::CosmologyModel eds({0.7, 1, 0, 0.05, 0.8, 1.0});
    cosmology::DriftKickIntegrator factors(eds);
    for (const auto interval : {std::array{0.01, 1.0}, std::array{0.1, 0.10001},
                                std::array{0.4, 0.2}}) {
        const long double a = interval[0], b = interval[1];
        const long double d = (2.0L / 100) * (1 / std::sqrt(a) - 1 / std::sqrt(b));
        const long double k = (2.0L / 100) * (std::sqrt(b) - std::sqrt(a));
        near(factors.drift_factor(interval[0], interval[1]) / static_cast<double>(d),
             1, 2e-11, "EdS drift integral");
        near(factors.kick_factor(interval[0], interval[1]) / static_cast<double>(k),
             1, 2e-11, "EdS kick integral");
    }
    std::cout << "growth maximum relative error=" << maximum_growth_error << '\n';
}

void kdk_convergence() {
    cosmology::CosmologyModel eds({0.7, 1, 0, 0.05, 0.8, 1.0});
    cosmology::DriftKickIntegrator factors(eds);
    constexpr double a0 = 0.1, a1 = 0.4, momentum = 2, acceleration = 1000;
    const long double exact_position = 0.5L + momentum * 0.02L
        * (1 / std::sqrt(static_cast<long double>(a0)) - 1 / std::sqrt(static_cast<long double>(a1)))
        + 2 * acceleration / 10000.0L
        * (std::log(static_cast<long double>(a1) / a0)
           + 2 * (std::sqrt(static_cast<long double>(a0) / a1) - 1));
    double previous_error = 0;
    for (int steps : {8, 16, 32}) {
        core::ParticleStore particles;
        particles.resize(1);
        particles.set_uniform_mass(1);
        particles.get_positions_x()[0] = 0.5;
        particles.get_momenta_x()[0] = momentum;
        time::LeapfrogIntegrator integrator(particles, factors,
            [](auto, auto, auto, auto, auto ax, auto, auto, double) { ax[0] = acceleration; }, 8);
        double begin = a0;
        for (int i = 1; i <= steps; ++i) {
            const double end = i == steps ? a1 : a0 * std::exp(std::log(a1 / a0) * i / steps);
            integrator.step(time::TimeStep(begin, end));
            begin = end;
        }
        near(particles.get_momenta_x()[0],
             momentum + acceleration * 0.02L * (std::sqrt(0.4L) - std::sqrt(0.1L)),
             3e-14, "KDK canonical momentum");
        const double error = std::abs(particles.get_positions_x()[0] - exact_position);
        if (!(error > 0) || (previous_error > 0
            && !(previous_error / error > 3.9 && previous_error / error < 4.1))) {
            throw std::runtime_error("KDK failed second-order step refinement");
        }
        previous_error = error;
    }
}

void pm_plane_waves() {
    for (int n : {7, 8}) {
        mesh::MeshGeometry geometry(n, n);
        core::ParticleStore particles;
        particles.resize(static_cast<std::size_t>(n) * n * n);
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y)
                for (int z = 0; z < n; ++z) {
                    const auto i = static_cast<std::size_t>((x * n + y) * n + z);
                    particles.get_positions_x()[i] = x;
                    particles.get_positions_y()[i] = y;
                    particles.get_positions_z()[i] = z;
                    particles.get_masses()[i] = 1 + 0.1 * std::cos(2 * std::numbers::pi * x / n);
                }
        for (int method = 0; method < 3; ++method) {
            const auto force_method = method == 2 ? mesh::PMForceMethod::treepm_long_range(0.5)
                                                  : mesh::PMForceMethod::pure_pm(method == 1);
            gravity::PMSolver solver(geometry, force_method, false, {});
            auto ax = particles.mutable_accelerations_x();
            auto ay = particles.mutable_accelerations_y();
            auto az = particles.mutable_accelerations_z();
            std::fill(ax.begin(), ax.end(), 0);
            std::fill(ay.begin(), ay.end(), 0);
            std::fill(az.begin(), az.end(), 0);
            solver.compute_forces(particles.get_positions_x(), particles.get_positions_y(),
                particles.get_positions_z(), particles.get_masses(), std::nullopt, ax, ay, az);
            const long double k = 2 * std::numbers::pi_v<long double> / n;
            const long double kd2 = 4 * std::pow(std::sin(k / 2), 2);
            long double amplitude = -4 * std::numbers::pi_v<long double>
                * cosmology::units::G * 0.1L
                * (method == 2 ? std::exp(-k * k / 4) / k : std::sin(k) / kd2);
            if (method != 0) amplitude /= std::pow(std::sin(k / 2) / (k / 2), 4);
            for (std::size_t i = 0; i < particles.size(); ++i) {
                near(ax[i], amplitude * std::sin(k * particles.get_positions_x()[i]),
                     2e-12, "PM plane-wave force sign/normalization");
                near(ay[i], 0, 2e-12, "PM transverse force y");
                near(az[i], 0, 2e-12, "PM transverse force z");
            }
        }
    }
}

void split_force_and_potential() {
    constexpr double scale = 0.3, cutoff = 1.5, eps = 0.08;
    gravity::ForceSplitKernel split(scale, cutoff / scale);
    for (double r : {0.001, 0.03, 0.2, 0.7, 1.4}) {
        const long double q = r / (2.0L * scale);
        const long double long_force = (std::erf(q)
            - 2 * q * std::exp(-q * q) / std::sqrt(std::numbers::pi_v<long double>)) / (r * r * r);
        const long double expected = cosmology::units::G * r
            * (std::pow(r * r + eps * eps, -1.5L) - long_force);
        near(split.scale_safe_short_range_acceleration({r, 0, 0}, 1, eps).x,
             expected, 5e-11, "TreePM Plummer-minus-long force");
        const double h = std::min(r, cutoff - r) * 1e-4;
        const double derivative = (-split.short_range_potential_correction(r + 2 * h, eps)
            + 8 * split.short_range_potential_correction(r + h, eps)
            - 8 * split.short_range_potential_correction(r - h, eps)
            + split.short_range_potential_correction(r - 2 * h, eps)) / (12 * h);
        near(-cosmology::units::G * derivative, expected, 2e-8,
             "TreePM force/potential consistency");
    }
    near(split.short_range_potential_correction(cutoff, eps), 0, 0, "TreePM cutoff potential");
}
} // namespace

int main() {
    try {
        growth_and_factors();
        kdk_convergence();
        pm_plane_waves();
        split_force_and_potential();
        std::cout << "Growth, KDK, PM and TreePM numerical invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
