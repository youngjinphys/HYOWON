#include "cosmo_nbody/ic/initial_conditions.hpp"
#include "cosmo_nbody/ic/realised_ic_checks.hpp"
#include "cosmo_nbody/ic/linear_power_spectrum.hpp"
#include "cosmo_nbody/ic/random_field.hpp"
#include "cosmo_nbody/ic/lpt_displacement.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/verified_snapshot_source.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/mesh/mesh_geometry.hpp"
#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/real_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace ic {

namespace {

std::size_t checked_particle_cube(std::uint64_t n) {
    if (n > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Particle dimension does not fit size_t");
    }
    const std::size_t N = static_cast<std::size_t>(n);
    if (N != 0 && N > std::numeric_limits<std::size_t>::max() / N) {
        throw std::overflow_error("Particle count N^3 overflows size_t");
    }
    const std::size_t n2 = N * N;
    if (N != 0 && n2 > std::numeric_limits<std::size_t>::max() / N) {
        throw std::overflow_error("Particle count N^3 overflows size_t");
    }
    return n2 * N;
}

std::size_t checked_state_elements(std::size_t particle_count) {
    if (particle_count > std::numeric_limits<std::size_t>::max() / 6) {
        throw std::overflow_error(
            "Generated IC six-component state size overflows size_t");
    }
    return particle_count * 6;
}

std::size_t checked_complex_bytes(std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max()
            / sizeof(std::complex<core::Real>)) {
        throw std::overflow_error(
            "Generated IC complex density byte size overflows size_t");
    }
    return elements * sizeof(std::complex<core::Real>);
}

bool nearly_equal(core::Real lhs, core::Real rhs) {
    const core::Real scale = std::max({
        core::Real{1.0}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs)
        <= core::Real{64.0}
            * std::numeric_limits<core::Real>::epsilon()
            * scale;
}

void require_snapshot_mass_consistency(
    const config::SimulationParameters& config,
    const core::ParticleStore& particles,
    std::size_t expected_particles) {
    const core::Real expected_particle_mass = config.particle_mass();
    if (!std::isfinite(expected_particle_mass) || expected_particle_mass <= 0.0) {
        throw std::runtime_error(
            "Configured particle/total mass is invalid for snapshot IC validation");
    }
    math::ExactPositiveDoubleSum expected_total_accumulator;
    expected_total_accumulator.add_repeated(
        expected_particle_mass,
        static_cast<std::uint64_t>(expected_particles));
    const core::Real expected_total_mass = expected_total_accumulator.value();
    if (!std::isfinite(expected_total_mass) || expected_total_mass <= 0.0) {
        throw std::runtime_error(
            "Configured particle/total mass is invalid for snapshot IC validation");
    }

    if (const auto uniform = particles.get_uniform_mass();
        uniform.has_value()) {
        if (*uniform != expected_particle_mass) {
            throw std::runtime_error(
                "Snapshot IC uniform particle mass does not match configured cosmology and box resolution");
        }
        return;
    }

    math::ExactPositiveDoubleSum observed_total;
    for (std::size_t index = 0;
         index < particles.num_owned_particles();
         ++index) {
        const core::Real mass = particles.mass_at(index);
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::runtime_error(
                "Snapshot IC contains a non-finite or non-positive particle mass");
        }
        observed_total.add(mass);
    }
    if (observed_total.value() != expected_total_mass) {
        throw std::runtime_error(
            "Snapshot IC total mass does not match configured Omega_m and box volume");
    }
}

void load_snapshot_ic(
    const config::SimulationParameters& config,
    core::ParticleStore& particles) {
    const auto& ic = config.get_ic();
    io::VerifiedSnapshotSource source(
        ic.snapshot_file,
        ic.expected_snapshot_sha256);
    const std::string admitted_sha256 = source.sha256();
    const io::SnapshotDescriptor descriptor =
        io::read_snapshot_descriptor(source);
    io::require_snapshot_initial_condition_match(
        config, descriptor, admitted_sha256);

    const std::size_t expected_particles =
        checked_particle_cube(config.get_box().N);
    io::SnapshotIO snapshot_io(config);
    core::ParticleStore candidate;
    core::Real snapshot_a = 0.0;
    snapshot_io.read_snapshot(
        source,
        candidate,
        snapshot_a,
        io::SnapshotReadPolicy::ExactInitialConditions,
        io::SnapshotPopulationAdmission::exact(expected_particles));

    // Re-hash the same opened object, not the operator pathname. This rejects
    // in-place mutation during HDF5 ingestion and is immune to pathname ABA.
    source.verify_unchanged();

    const core::Real expected_a =
        1.0 / (1.0 + config.get_time().z_start);
    if (!nearly_equal(snapshot_a, expected_a)) {
        throw std::runtime_error(
            "Snapshot IC scale factor does not match configured time.start_redshift");
    }

    if (candidate.num_owned_particles() != expected_particles) {
        throw std::runtime_error(
            "Snapshot IC particle count does not match box.particles_per_dimension^3");
    }
    require_snapshot_mass_consistency(
        config, candidate, expected_particles);

    candidate.clear_ghosts();
    candidate.set_acceleration_validity(core::FieldValidity::INVALID);
    candidate.set_ghost_validity(core::FieldValidity::INVALID);
    candidate.set_verified_snapshot_ic_sha256(admitted_sha256);
    config.set_verified_snapshot_ic_provenance(
        admitted_sha256, descriptor.run_metadata_json);

    // The caller state is not changed until immutable source identity, time,
    // count, mass normalization, and validity metadata all succeed.
    particles = std::move(candidate);

    std::cout << "Successfully ingested "
              << particles.num_owned_particles()
              << " particles from snapshot IC file '"
              << ic.snapshot_file
              << "' at a = " << snapshot_a
              << " with mass normalization and immutable input identity consistent with the configured run.\n";
}

} // namespace

void InitialConditions::load_snapshot(
    const config::SimulationParameters& config,
    core::ParticleStore& particles) {
    if (config.get_ic().mode != "snapshot") {
        throw std::logic_error(
            "InitialConditions::load_snapshot requires ic.mode=snapshot");
    }
    load_snapshot_ic(config, particles);
}

void InitialConditions::admit_snapshot_provenance(
    const config::SimulationParameters& config) {
    if (config.get_ic().mode != "snapshot") return;

    const auto& ic = config.get_ic();
    io::VerifiedSnapshotSource source(
        ic.snapshot_file,
        ic.expected_snapshot_sha256);
    const io::SnapshotDescriptor descriptor =
        io::read_snapshot_descriptor(source);
    io::require_snapshot_initial_condition_match(
        config, descriptor, source.sha256());
    source.verify_unchanged();
    config.set_verified_snapshot_ic_provenance(
        source.sha256(), descriptor.run_metadata_json);
}

void InitialConditions::generate(
    const config::SimulationParameters& config,
    const cosmology::CosmologyModel& cosmo,
    core::ParticleStore& particles,
    std::optional<RealisedICEvidence>* evidence_out) {
    if (config.get_ic().mode != "generate") {
        throw std::logic_error(
            "InitialConditions::generate requires ic.mode=generate");
    }

    const std::size_t N =
        static_cast<std::size_t>(config.get_box().N);
    const std::size_t evolution_mesh =
        static_cast<std::size_t>(config.get_box().N_mesh);
    const std::size_t ic_mesh =
        static_cast<std::size_t>(config.ic_mesh_per_dimension());
    const core::Real L = config.get_box().L;
    const std::size_t total_particles =
        checked_particle_cube(config.get_box().N);

    if (ic_mesh % N != 0) {
        throw std::invalid_argument(
            "Generated ICs require the IC mesh to be an integer multiple of particles_per_dimension");
    }
    if (config.get_ic().lpt_order == 2 && ic_mesh / 2 < N) {
        throw std::invalid_argument(
            "Generated 2LPT ICs require ic.mesh_per_dimension >= 2 * particles_per_dimension");
    }

    ExactFourierEvidence exact_evidence;
    RealisedICSummary realised;
    std::optional<core::ParticleStore> generated_candidate;
    {
        // Build generated state transactionally. Memory mode writes directly to
        // the candidate; Disk mode materializes mmap-backed state after validation.
        const bool file_backed_state = config::uses_file_backed_scratch(
            config.get_memory_policy().ic_scratch_mode);
        std::unique_ptr<runtime::RealScratchBuffer> state;
        core::ParticleStore candidate;
        std::span<core::Real> pos_x;
        std::span<core::Real> pos_y;
        std::span<core::Real> pos_z;
        std::span<core::Real> mom_x;
        std::span<core::Real> mom_y;
        std::span<core::Real> mom_z;
        std::span<core::ParticleId> ids;

        if (file_backed_state) {
            state = std::make_unique<runtime::RealScratchBuffer>(
                checked_state_elements(total_particles),
                config.get_memory_policy(),
                "generated_ic_particle_state");
            pos_x = {state->data(), total_particles};
            pos_y = {state->data() + total_particles, total_particles};
            pos_z = {state->data() + 2 * total_particles, total_particles};
            mom_x = {state->data() + 3 * total_particles, total_particles};
            mom_y = {state->data() + 4 * total_particles, total_particles};
            mom_z = {state->data() + 5 * total_particles, total_particles};
        } else {
            candidate.set_uniform_mass(config.particle_mass());
            candidate.resize(total_particles);
            candidate.set_verified_snapshot_ic_sha256({});
            pos_x = candidate.get_positions_x();
            pos_y = candidate.get_positions_y();
            pos_z = candidate.get_positions_z();
            mom_x = candidate.get_momenta_x();
            mom_y = candidate.get_momenta_y();
            mom_z = candidate.get_momenta_z();
            ids = candidate.get_ids();
        }

        const core::Real dx = L / static_cast<core::Real>(N);
        const auto announce_phase = [](const char* phase) {
            std::clog << "[ic] phase=" << phase;
#ifdef COSMO_NBODY_HAS_OPENMP
            std::clog << " openmp_max_threads=" << omp_get_max_threads();
#endif
            std::clog << '\n';
        };
        announce_phase("lattice_initialization");
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(3) schedule(static) \
            if(runtime::should_use_host_parallel_team(total_particles))
#endif
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < N; ++k) {
                    const std::size_t particle_index =
                        (i * N + j) * N + k;
                    pos_x[particle_index] = static_cast<core::Real>(i) * dx;
                    pos_y[particle_index] = static_cast<core::Real>(j) * dx;
                    pos_z[particle_index] = static_cast<core::Real>(k) * dx;
                    mom_x[particle_index] = 0.0;
                    mom_y[particle_index] = 0.0;
                    mom_z[particle_index] = 0.0;
                    if (!ids.empty()) ids[particle_index] = particle_index;
                }
            }
        }

        {
            LinearPowerSpectrum pk(config);
            const mesh::MeshGeometry ic_geometry(L, ic_mesh);
            // Plan only the active IC mesh; an evolution-mesh plan would add
            // unrelated provenance and a large transient allocation.
            mesh::FFTBackend ic_fft(
                ic_geometry, config.get_memory_policy().ic_scratch_mode,
                config.get_memory_policy().scratch_directory);
            std::unique_ptr<runtime::RawScratchBuffer> density_storage;
            std::unique_ptr<mesh::ComplexField> density_k;
            announce_phase("random_fourier_field");
            if (file_backed_state) {
                static_assert(
                    std::is_trivially_destructible_v<
                        std::complex<core::Real>>,
                    "External IC density scratch relies on trivial destruction");
                density_storage =
                    std::make_unique<runtime::RawScratchBuffer>(
                        checked_complex_bytes(ic_geometry.complex_size()),
                        config::ScratchMode::Disk,
                        config.get_memory_policy().scratch_directory,
                        "generated_ic_density_k");
                auto* density_data =
                    static_cast<std::complex<core::Real>*>(
                        density_storage->data());
                RandomField::generate_into_uninitialized_storage(
                    config,
                    pk,
                    density_data,
                    ic_geometry.complex_size(),
                    ic_mesh);
                density_k = std::make_unique<mesh::ComplexField>(
                    density_data,
                    ic_geometry.complex_size(),
                    mesh::external_mesh_storage);
            } else {
                density_k = std::make_unique<mesh::ComplexField>(
                    ic_geometry.complex_size());
                RandomField::generate(config, pk, *density_k, ic_mesh);
            }

            announce_phase("exact_fourier_evidence");
            exact_evidence = summarize_generated_fourier_realisation(
                config, pk, *density_k, ic_mesh);

            announce_phase("lpt_displacement");
            if (file_backed_state) {
                LPTDisplacement::apply_to_arrays_reusing_file_backed_density(
                    config,
                    cosmo,
                    ic_fft,
                    *density_k,
                    pos_x, pos_y, pos_z,
                    mom_x, mom_y, mom_z);
            } else {
                LPTDisplacement::apply_to_arrays_reusing_density(
                    config,
                    cosmo,
                    ic_fft,
                    *density_k,
                    pos_x, pos_y, pos_z,
                    mom_x, mom_y, mom_z);
            }
        }

        announce_phase("structural_full_population_validation");
        realised = validate_realised_lattice_ic(
            N, L,
            pos_x, pos_y, pos_z,
            mom_x, mom_y, mom_z);

        if (file_backed_state) {
            announce_phase("particle_state_materialization");
            candidate.set_uniform_mass(config.particle_mass());
            candidate.resize(total_particles);
            candidate.set_verified_snapshot_ic_sha256({});
            auto out_pos_x = candidate.get_positions_x();
            auto out_pos_y = candidate.get_positions_y();
            auto out_pos_z = candidate.get_positions_z();
            auto out_mom_x = candidate.get_momenta_x();
            auto out_mom_y = candidate.get_momenta_y();
            auto out_mom_z = candidate.get_momenta_z();
            ids = candidate.get_ids();

            // Materialize file-backed state in one parallel pass.
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp parallel for schedule(static) \
                if(runtime::should_use_host_parallel_team(total_particles))
#endif
            for (std::size_t index = 0; index < total_particles; ++index) {
                out_pos_x[index] = pos_x[index];
                out_pos_y[index] = pos_y[index];
                out_pos_z[index] = pos_z[index];
                out_mom_x[index] = mom_x[index];
                out_mom_y[index] = mom_y[index];
                out_mom_z[index] = mom_z[index];
                ids[index] = index;
            }
        }

        candidate.set_acceleration_validity(core::FieldValidity::INVALID);
        candidate.set_ghost_validity(core::FieldValidity::INVALID);
        generated_candidate.emplace(std::move(candidate));
    }

    if (!generated_candidate.has_value()) {
        throw std::logic_error(
            "Generated IC candidate disappeared before evidence construction");
    }
    std::optional<RealisedICEvidence> candidate_evidence;
    candidate_evidence.emplace(build_realised_ic_evidence(
        config,
        realised,
        std::move(exact_evidence),
        *generated_candidate,
        config.get_validation().write_diagnostics));

    // Publish only after particle state and evidence are complete so failures
    // leave caller-owned outputs unchanged.
    particles = std::move(*generated_candidate);
    if (evidence_out) {
        *evidence_out = std::move(candidate_evidence);
    }

    std::cout << "Successfully generated " << total_particles
              << " particles using " << config.get_ic().lpt_order
              << "LPT at z = " << config.get_time().z_start
              << " on temporary IC mesh " << ic_mesh << "^3"
              << " (evolution PM mesh " << evolution_mesh
              << "^3, transactional direct ParticleStore state for anonymous memory, density-mode reuse).\n"
              << "Realised IC structural diagnostics: displacement_rms="
              << realised.displacement_rms_Mpc_h
              << " Mpc/h, displacement_max="
              << realised.displacement_max_Mpc_h
              << " Mpc/h, momentum_rms=" << realised.momentum_rms
              << ", momentum_max=" << realised.momentum_max;
    if (realised.forward_jacobian_available) {
        std::cout
            << ", forward_jacobian_determinant_min="
            << realised.forward_jacobian_determinant_min
            << ", forward_jacobian_determinant_max="
            << realised.forward_jacobian_determinant_max
            << ", forward_jacobian_nonpositive_count="
            << realised.forward_jacobian_nonpositive_count;
    }
    std::cout << ".\n";
}

} // namespace ic
} // namespace cosmo_nbody
