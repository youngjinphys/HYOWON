#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/run_artifact_layout.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"
#include "cosmo_nbody/runtime/simulation_runner.hpp"

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif
#ifdef __linux__
#include <sys/resource.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace cosmo_nbody;
namespace fs = std::filesystem;
constexpr std::size_t particle_count = 64;
constexpr double final_a = 1.0 / 9.0;
#ifdef COSMO_NBODY_HAS_MPI
constexpr bool mpi_enabled = true;
#else
constexpr bool mpi_enabled = false;
#endif

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Action>
void on_root(const runtime::RuntimeContext& context, Action action) {
    std::exception_ptr error;
    if (context.rank() == 0) {
        try { action(); } catch (...) { error = std::current_exception(); }
    }
    runtime::synchronize_mpi_exception(error, context.size(), "Test root action");
}

config::SimulationParameters parameters(const fs::path& input, bool diagnostics,
                                         bool fixture = false, bool large_batch = false) {
    config::GravityParams gravity;
    gravity.solver = "PM";
    gravity.deconvolve_cic = false;
    config::ICParams ic;
    ic.mode = "snapshot";
    ic.snapshot_file = input.string();
    if (fixture) {
        // Synthetic provenance only admits this hand-defined integration-test
        // payload; it is not an external cosmological spectrum or accuracy test.
        ic.mode = "generate";
        ic.snapshot_file.clear();
        ic.seed = std::uint64_t{1};
        ic.mesh_per_dimension = std::uint64_t{8};
        ic.lpt_order = 1;
        ic.power_spectrum_file = (input.parent_path() / "synthetic-spectrum.txt").string();
        ic.power_spectrum_redshift = 0.0;
        ic.power_spectrum_fidelity = "precision_boltzmann";
        ic.amplitude_mode = "gaussian";
        ic.phase_pairing = "independent";
    }
    config::OutputParams output;
    output.snapshot_scale_factors = {}; // Exercise the terminal snapshot path.
    output.snapshot_batch_particles = large_batch ? 100000000 : 13;
    return config::SimulationParameters(
        {0.7, 0.3, 0.7, 0.05, 0.8, 1.0}, {8.0, 4, 8}, gravity,
        {9.0, 8.0, 0.04, "global"}, ic, output,
        {1, mpi_enabled}, {diagnostics});
}

class TemporaryDirectory {
public:
    void create() {
        auto pattern = (fs::temp_directory_path() / "hyowon-diagnostics-XXXXXX").string();
        require(::mkdtemp(pattern.data()) != nullptr, "Could not create temporary directory");
        path_ = std::move(pattern);
    }
    ~TemporaryDirectory() { cleanup(); }
    TemporaryDirectory() = default;
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    const std::string& path() const noexcept { return path_; }
    void cleanup() noexcept {
        if (path_.empty()) return;
        std::error_code error;
        fs::remove_all(path_, error);
        if (!error) path_.clear();
    }
private:
    std::string path_;
};

class CaptureOutput {
public:
    CaptureOutput() : previous_(std::cout.rdbuf(stream_.rdbuf())) {}
    ~CaptureOutput() { std::cout.rdbuf(previous_); }
    std::string text() const { return stream_.str(); }
private:
    std::ostringstream stream_;
    std::streambuf* previous_;
};

void write_input(const fs::path& path) {
    {
        std::ofstream source(path.parent_path() / "synthetic-spectrum.txt");
        source << "Synthetic diagnostics regression fixture; not a physical spectrum.\n";
        require(source.good(), "Could not write synthetic provenance fixture");
    }
    const auto config = parameters(path, true, true);
    core::ParticleStore particles;
    particles.set_uniform_mass(config.particle_mass());
    particles.resize(particle_count);
    for (std::size_t n = 0; n < particle_count; ++n) {
        const auto x = n / 16;
        const auto y = n / 4 % 4;
        const auto z = n % 4;
        // A perturbed lattice with nonzero momenta exercises real evolution.
        particles.get_positions_x()[n] = 0.4 + 2.0 * x + 0.09 * (y % 2);
        particles.get_positions_y()[n] = 0.5 + 2.0 * y + 0.07 * (z % 3);
        particles.get_positions_z()[n] = 0.6 + 2.0 * z + 0.05 * (x % 2);
        particles.get_momenta_x()[n] = 0.01 * (static_cast<double>(y) - 1.5);
        particles.get_momenta_y()[n] = 0.02 * (static_cast<double>(z) - 1.5);
        particles.get_momenta_z()[n] = 0.03 * (static_cast<double>(x) - 1.5);
        particles.get_ids()[n] = 100 + n;
    }
    io::SnapshotIO(config).write_snapshot_to_path(particles, 0.1, path.string());
}

template<class T>
std::vector<T> read_dataset(hid_t file, const char* path, hid_t type,
                            std::size_t count) {
    auto dataset = io::H5DatasetHandle::checked(
        H5Dopen2(file, path, H5P_DEFAULT), path);
    auto space = io::H5SpaceHandle::checked(H5Dget_space(dataset.get()), path);
    require(H5Sget_simple_extent_npoints(space.get()) == static_cast<hssize_t>(count),
            std::string("Unexpected particle dataset size: ") + path);
    std::vector<T> values(count);
    io::check_hdf5(io::H5Dread(dataset.get(), type, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, values.data()), path);
    return values;
}

struct Snapshot {
    std::vector<std::uint64_t> ids;
    std::vector<double> positions;
    std::vector<double> velocities;
    std::array<double, 6> masses{};
    double scale_factor{};
};

Snapshot read_snapshot(const fs::path& path) {
    auto file = io::H5FileHandle::checked(
        H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), path.string());
    Snapshot result;
    result.ids = read_dataset<std::uint64_t>(file.get(), "/PartType1/ParticleIDs",
                                            H5T_NATIVE_UINT64, particle_count);
    result.positions = read_dataset<double>(file.get(), "/PartType1/Coordinates",
                                            H5T_NATIVE_DOUBLE, 3 * particle_count);
    result.velocities = read_dataset<double>(file.get(), "/PartType1/Velocities",
                                             H5T_NATIVE_DOUBLE, 3 * particle_count);
    auto header = io::H5GroupHandle::checked(
        H5Gopen2(file.get(), "/Header", H5P_DEFAULT), "Header");
    auto mass = io::H5AttributeHandle::checked(
        H5Aopen(header.get(), "MassTable", H5P_DEFAULT), "MassTable");
    auto time = io::H5AttributeHandle::checked(
        H5Aopen(header.get(), "Time", H5P_DEFAULT), "Time");
    io::check_hdf5(io::H5Aread(mass.get(), H5T_NATIVE_DOUBLE, result.masses.data()),
                   "MassTable read");
    io::check_hdf5(io::H5Aread(time.get(), H5T_NATIVE_DOUBLE, &result.scale_factor),
                   "Time read");
    return result;
}

std::vector<std::size_t> sorted_indices(const Snapshot& snapshot) {
    std::vector<std::size_t> indices(particle_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](auto left, auto right) {
        return snapshot.ids[left] < snapshot.ids[right];
    });
    return indices;
}

void require_same_particles(const Snapshot& expected, const Snapshot& actual) {
    require(expected.scale_factor == final_a && actual.scale_factor == final_a,
            "Final snapshot did not reach the requested endpoint");
    require(actual.masses[1] > 0, "Final snapshot mass is not positive");
    for (std::size_t i = 0; i < actual.masses.size(); ++i) {
        require(std::bit_cast<std::uint64_t>(expected.masses[i])
                    == std::bit_cast<std::uint64_t>(actual.masses[i]),
                "Final snapshot mass table changed at the bit level");
    }
    const auto expected_indices = sorted_indices(expected);
    const auto actual_indices = sorted_indices(actual);
    for (std::size_t i = 0; i < particle_count; ++i) {
        const auto left = expected_indices[i];
        const auto right = actual_indices[i];
        require(expected.ids[left] == 100 + i && actual.ids[right] == 100 + i,
                "Final snapshot stable IDs changed");
        for (std::size_t axis = 0; axis < 3; ++axis) {
            require(std::bit_cast<std::uint64_t>(expected.positions[3 * left + axis])
                        == std::bit_cast<std::uint64_t>(actual.positions[3 * right + axis]),
                    "Final coordinates differ at the bit level");
            require(std::bit_cast<std::uint64_t>(expected.velocities[3 * left + axis])
                        == std::bit_cast<std::uint64_t>(actual.velocities[3 * right + axis]),
                    "Final peculiar velocities differ at the bit level");
        }
    }
}

std::string read_text(const fs::path& path) {
    std::ifstream file(path);
    require(file.good(), "Missing diagnostic artifact: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void run_case(const runtime::RuntimeContext& context, const fs::path& input,
              const fs::path& directory, const std::string& failure, bool diagnostics,
              bool large_batch = false) {
    const auto layout = io::RunArtifactLayout::organized(directory);
    on_root(context, [&] {
        require(fs::create_directory(directory), "Run directory already exists");
        layout.prepare_directories_exclusive();
        if (!failure.empty()) {
            const auto destination = failure == "report"
                ? layout.validation_report_path()
                : layout.diagnostics_directory() / "memory_timeline.json";
            require(fs::create_directory(destination), "Could not install I/O failure");
        }
        std::cout << "Running " << directory.filename() << " failure=" << failure
                  << std::endl;
    });

    const auto config = parameters(input, diagnostics, false, large_batch);
    runtime::SimulationRunner runner(config, layout);
    runner.initialize();
    std::string log;
    std::string error;
    {
        CaptureOutput capture;
        try { runner.run(); }
        catch (const std::exception& exception) { error = exception.what(); }
        catch (...) { error = "unknown exception"; }
        log = capture.text();
    }
    if (!error.empty()) std::cerr << "rank " << context.rank() << " caught: " << error << '\n';

    // A report exception must leave every rank at this boundary. Without the
    // fix, root enters this barrier while peers enter memory publication's
    // Allreduce. The external timeout catches that collective order regression.
#ifdef COSMO_NBODY_HAS_MPI
    require(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "Post-run MPI barrier failed");
#endif
    const bool expected_error = failure.empty()
        ? error.empty()
        : error.find(failure == "report" ? "report" : "memory timeline")
            != std::string::npos;
    runtime::synchronize_mpi_failure(!expected_error, context.size(),
                                    "Unexpected diagnostic publication result");
    on_root(context, [&] {
        require(fs::is_regular_file(layout.snapshot_path(0)),
                "Diagnostic failure prevented final snapshot publication");
        const auto memory = layout.diagnostics_directory() / "memory_timeline.json";
        if (diagnostics) {
            require(fs::is_regular_file(layout.layzer_irvine_timeline_path()),
                    "Layzer-Irvine timeline was not published first");
            if (failure != "report") {
                const auto report = read_text(layout.validation_report_path());
                require(report.find("runtime_rank_count") != std::string::npos
                            && report.find("runtime_memory_observed_rank_count")
                                != std::string::npos,
                        "Runtime report is missing topology or final memory diagnostics");
            }
            if (failure.empty()) {
                require(read_text(memory).find("\"phase\":\"terminal\"")
                            != std::string::npos,
                        "Memory timeline omitted the terminal sample");
            } else {
                require(!fs::is_regular_file(memory),
                        "Memory timeline published beyond a failed diagnostic stage");
            }
        } else {
            require(!fs::exists(layout.validation_report_path()) && !fs::exists(memory),
                    "Disabled diagnostics unexpectedly wrote diagnostic reports");
        }
        const bool has_timings = log.find("RUNTIME_PHASE_TIMING ") != std::string::npos;
        require(has_timings == (failure.empty() && context.size() > 1),
                "Phase timing publication crossed a failed stage or was omitted");
    });
}

} // namespace

int main(int argc, char** argv) {
    runtime::RuntimeContext context({1, mpi_enabled});
    TemporaryDirectory temporary;
    try {
        require(argc == 2, "Expected scenario: healthy, report, memory, or large_batch");
        const std::string scenario = argv[1];
        require(scenario == "healthy" || scenario == "report" || scenario == "memory"
                    || scenario == "large_batch",
                "Unknown diagnostics regression scenario");
        on_root(context, [&] {
            temporary.create();
            write_input(fs::path(temporary.path()) / "input.hdf5");
        });
        const auto root = runtime::broadcast_string_collective(
            temporary.path(), 0, context.rank(), context.size(), "Test directory");
        const auto input = fs::path(root) / "input.hdf5";
        run_case(context, input, fs::path(root) / "control", "", true);
#ifdef __linux__
        if (scenario == "large_batch") {
            struct rlimit limit{};
            require(getrlimit(RLIMIT_AS, &limit) == 0, "Could not read address-space limit");
            limit.rlim_cur = std::min(limit.rlim_cur, static_cast<rlim_t>(512ULL << 20));
            require(setrlimit(RLIMIT_AS, &limit) == 0, "Could not bound test address space");
        }
#endif
        run_case(context, input, fs::path(root) / "target",
                 scenario == "healthy" || scenario == "large_batch" ? "" : scenario,
                 scenario != "healthy", scenario == "large_batch");
        on_root(context, [&] {
            const auto control = read_snapshot(fs::path(root) / "control/snapshots/snapshot_0.hdf5");
            const auto target = read_snapshot(fs::path(root) / "target/snapshots/snapshot_0.hdf5");
            require_same_particles(control, target);
            const auto initial = read_snapshot(input);
            const auto initial_indices = sorted_indices(initial);
            const auto final_indices = sorted_indices(control);
            bool evolved = false;
            for (std::size_t i = 0; i < particle_count; ++i) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    evolved |= initial.positions[3 * initial_indices[i] + axis]
                        != control.positions[3 * final_indices[i] + axis];
                }
            }
            require(evolved,
                    "Fixture did not evolve; particle comparison would be vacuous");
            std::cout << "PASS " << scenario << ": all " << context.size()
                      << " ranks completed; final 64-particle payloads match exactly\n";
        });
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << context.rank() << ": " << error.what() << std::endl;
        temporary.cleanup();
#ifdef COSMO_NBODY_HAS_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
