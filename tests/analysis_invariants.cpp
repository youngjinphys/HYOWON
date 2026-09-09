#include "cosmo_nbody/analysis/field_cross_correlation.hpp"
#include "cosmo_nbody/analysis/fourier_density_field.hpp"
#include "cosmo_nbody/analysis/matter_power_spectrum.hpp"
#include "cosmo_nbody/analysis/symmetric_eigensystem.hpp"
#include "cosmo_nbody/halo/fof_membership.hpp"
#include "cosmo_nbody/halo/periodic_neighbor_index.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace cosmo_nbody;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double distance(const core::ParticleStore& particles, std::size_t i, core::Vec3 center) {
    return std::hypot(std::remainder(particles.get_positions_x()[i] - center.x, 8),
                      std::remainder(particles.get_positions_y()[i] - center.y, 8),
                      std::remainder(particles.get_positions_z()[i] - center.z, 8));
}

void periodic_neighbors_and_fof() {
    core::ParticleStore particles;
    particles.resize(64);
    particles.set_uniform_mass(1);
    for (std::size_t i = 0; i < 64; ++i) {
        particles.get_positions_x()[i] = std::fmod(0.03 + i * 0.613, 8);
        particles.get_positions_y()[i] = std::fmod(0.07 + i * 0.337, 8);
        particles.get_positions_z()[i] = std::fmod(0.11 + i * 0.219, 8);
        particles.get_ids()[i] = 1000 + i;
    }
    for (std::size_t capacity : {4096, 1048576}) {
        halo::PeriodicNeighborIndex index(particles, 8, capacity);
        for (core::Vec3 center : {core::Vec3{0, 0, 0}, core::Vec3{7.91, 7.87, 7.93},
                                  core::Vec3{4.23, 3.71, 2.63}}) {
            for (double radius : {0.25, 0.9, 2.1, 3.9}) {
                std::vector<halo::PeriodicNeighbor> found;
                index.collect_within(center, radius, found);
                std::vector<std::size_t> actual, expected;
                for (const auto neighbor : found) actual.push_back(neighbor.particle_index);
                for (std::size_t i = 0; i < particles.size(); ++i)
                    if (distance(particles, i, center) < radius) expected.push_back(i);
                std::sort(actual.begin(), actual.end());
                require(actual == expected, "Periodic index differs from all-particle distance search");
            }
        }
    }
    for (double linking : {0.4, 0.85, 1.4}) {
        // Brute-force connected components, independent of the linked-cell grid
        // and production disjoint-set implementation.
        std::vector<int> labels(64, -1);
        std::vector<std::vector<std::size_t>> expected;
        for (std::size_t start = 0; start < 64; ++start) {
            if (labels[start] != -1) continue;
            std::vector<std::size_t> component{start};
            labels[start] = static_cast<int>(start);
            for (std::size_t q = 0; q < component.size(); ++q) {
                const auto i = component[q];
                const core::Vec3 center{particles.get_positions_x()[i],
                    particles.get_positions_y()[i], particles.get_positions_z()[i]};
                for (std::size_t j = 0; j < 64; ++j) {
                    if (labels[j] == -1 && distance(particles, j, center) < linking) {
                        labels[j] = static_cast<int>(start);
                        component.push_back(j);
                    }
                }
            }
            if (component.size() >= 2) {
                std::sort(component.begin(), component.end());
                expected.push_back(component);
            }
        }
        const auto groups = halo::FoFMembershipFinder(8, 2, linking / 2, 2).find_memberships(particles);
        std::vector<std::vector<std::size_t>> actual;
        for (auto group : groups) {
            std::sort(group.particle_indices.begin(), group.particle_indices.end());
            actual.push_back(std::move(group.particle_indices));
        }
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        require(actual == expected, "FoF differs from brute-force periodic connected components");
    }
}

void spectral_normalization() {
    constexpr int n = 8;
    constexpr double amplitude = 0.125;
    core::ParticleStore particles;
    particles.resize(n * n * n);
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int z = 0; z < n; ++z) {
                const auto i = (x * n + y) * n + z;
                particles.get_positions_x()[i] = x;
                particles.get_positions_y()[i] = y;
                particles.get_positions_z()[i] = z;
                particles.get_masses()[i] = 1 + amplitude * std::cos(2 * std::numbers::pi * x / n);
                particles.get_ids()[i] = i;
            }
    const analysis::PeriodicDomain domain(n);
    const mesh::MeshGeometry geometry(n, n);
    const auto field = analysis::FourierDensityBuilder::build(domain, particles, n, false, {});
    require(std::abs(field.modes[geometry.complex_index(1, 0, 0)] - amplitude / 2) < 1e-13,
            "Density FFT normalization differs from analytic cosine coefficient");
    analysis::PowerSpectrumOptions options;
    options.num_bins = 4;
    options.interlaced = false;
    options.subtract_shot_noise = false;
    options.max_k_fraction_nyquist = 0.9;
    const auto bins = analysis::MatterPowerSpectrum(domain, n).compute_pk(particles, options);
    double total = 0;
    for (const auto& bin : bins) total += bin.p_raw * bin.mode_count;
    require(std::abs(total - n * n * n * amplitude * amplitude / 2) < 1e-11,
            "P(k) volume or half-spectrum multiplicity violates Parseval");
    for (bool interlaced : {false, true}) {
        analysis::FieldComparisonOptions comparison;
        comparison.num_bins = 4;
        comparison.interlaced = interlaced;
        comparison.max_k_h_Mpc = 0.9 * std::numbers::pi;
        const auto self = analysis::FieldCrossCorrelation(domain, n).compare(particles, particles, comparison);
        require(self.normalized_residual_defined && self.normalized_residual < 1e-12,
                "Self field comparison has nonzero residual");
        for (const auto& bin : self.bins) {
            if (bin.correlation_defined)
                require(std::abs(bin.r - 1) < 1e-12, "Self cross-correlation differs from unity");
        }
    }
}

void eigensystem_scaling() {
    for (double scale : {1e-120, 1.0, 1e120}) {
        const auto values = analysis::symmetric_eigenvalues_3x3(2 * scale, scale, 0, 2 * scale, 0, 5 * scale);
        require(std::abs(values.largest / scale - 5) < 1e-13
            && std::abs(values.middle / scale - 3) < 1e-13
            && std::abs(values.smallest / scale - 1) < 1e-13,
            "Symmetric eigenvalues changed under common scaling");
        const auto axis = analysis::symmetric_eigenvector_3x3(2 * scale, scale, 0, 2 * scale, 0, 5 * scale, values.middle);
        require(axis.has_value() && std::abs(std::abs((axis->x + axis->y) / std::sqrt(2.0)) - 1) < 1e-13,
                "Resolved symmetric eigenvector is incorrect");
    }
}
} // namespace

int main() {
    try {
        periodic_neighbors_and_fof();
        spectral_normalization();
        eigensystem_scaling();
        std::cout << "Periodic halo, Fourier and eigensystem invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
