#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/io/restart_io.hpp"
#include "cosmo_nbody/runtime/mpi_execution_identity.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
using namespace cosmo_nbody;

config::SimulationParameters parameters(std::optional<std::uint64_t> support, bool mpi = false) {
    config::GravityParams gravity;
    gravity.solver = "PM";
    gravity.deconvolve_cic = false;
    config::ICParams ic;
    ic.mode = "generate";
    ic.seed = std::uint64_t{1};
    ic.mesh_per_dimension = std::uint64_t{16};
    ic.lpt_order = 2;
    ic.max_mode_per_axis = support;
    ic.power_spectrum_file = "identity-only-fixture";
    ic.power_spectrum_sha256 = std::string(64, 'a');
    ic.power_spectrum_redshift = 0;
    ic.power_spectrum_fidelity = "precision_boltzmann";
    ic.amplitude_mode = "gaussian";
    ic.phase_pairing = "independent";
    return config::SimulationParameters({0.7, 0.3, 0.7, 0.05, 0.8, 1},
        {8, 8, 16}, gravity, {9, 8, 0.04, "global"}, ic, {}, {1, mpi});
}

struct TemporaryDirectory {
    std::string path;
    TemporaryDirectory() {
        path = (std::filesystem::temp_directory_path() / "hyowon-restart-XXXXXX").string();
        if (::mkdtemp(path.data()) == nullptr) throw std::runtime_error("Temporary directory failed");
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void restart_support() {
    TemporaryDirectory directory;
    core::ParticleStore particles;
    particles.resize(1);
    particles.set_uniform_mass(1);
    particles.get_positions_x()[0] = 0.25;
    particles.get_ids()[0] = 17;
    time::TimeStepper stepper(0.1, 1.0 / 9, 0.04);
    (void)stepper.next_step();
    for (auto support : {std::optional<std::uint64_t>{1}, std::optional<std::uint64_t>{}}) {
        const auto config = parameters(support);
        const auto path = directory.path + (support ? "/explicit.hdf5" : "/default.hdf5");
        io::RestartIO(config).write_restart(particles, stepper, path);
        core::ParticleStore restored;
        time::TimeStepper restored_time(0.1, 1.0 / 9, 0.04);
        io::RestartIO(config).read_restart(path, restored, restored_time);
        if (restored.size() != 1 || restored.get_ids()[0] != 17
            || restored.get_positions_x()[0] != 0.25
            || restored_time.current_a() != stepper.current_a()) {
            throw std::runtime_error("Identical-support restart did not round-trip");
        }
        for (const std::uint64_t changed : {2, 3}) {
            bool rejected = false;
            try { io::RestartIO(parameters(changed)).read_restart(path, restored, restored_time); }
            catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("dynamics identity mismatch") != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("Changed IC support was accepted by restart");
            if (restored.get_positions_x()[0] != 0.25
                || restored_time.current_a() != stepper.current_a()) {
                throw std::runtime_error("Rejected restart mutated caller state");
            }
        }
        for (const char* schema : {"hyowon.restart.v1", "hyowon.restart.v2"}) {
            {
                auto file = io::H5FileHandle::checked(
                    H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), "test restart");
                auto attribute = io::H5AttributeHandle::checked(
                    H5Aopen(file.get(), "RestartSchema", H5P_DEFAULT), "test schema");
                auto type = io::H5TypeHandle::checked(H5Aget_type(attribute.get()), "test schema type");
                io::check_hdf5(H5Awrite(attribute.get(), type.get(), schema), "test schema downgrade");
            }
            bool rejected = false;
            try { io::RestartIO(config).read_restart(path, restored, restored_time); }
            catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("does not bind IC Fourier support") != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("Unattested legacy generated restart was accepted");
        }
    }
}

void mpi_support() {
    runtime::RuntimeContext context({1, true});
    for (const bool different : {false, true}) {
        bool rejected = false;
        try {
            runtime::MpiExecutionIdentityAgreement agreement(
                parameters(different && context.rank() != 0 ? 2 : 1, true),
                context.rank(), context.size());
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("disagree on numerical-method") != std::string::npos;
        }
        if (rejected != different) throw std::runtime_error("MPI IC support agreement is incorrect");
    }
}
} // namespace

int main(int argc, char**) {
    try {
        if (argc > 1) mpi_support();
        else restart_support();
        std::cout << "IC support identity checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
