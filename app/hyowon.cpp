#include "cosmo_nbody/build_info.hpp"
#include "cosmo_nbody/config/config_loader.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/io/run_artifact_layout.hpp"
#include "cosmo_nbody/io/runtime_provenance.hpp"
#include "cosmo_nbody/runtime/mpi_string_broadcast.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "cosmo_nbody/runtime/runtime_context.hpp"
#include "cosmo_nbody/runtime/simulation_runner.hpp"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

using namespace cosmo_nbody;

namespace {

struct CliOptions {
    std::optional<std::string> restart;
    std::vector<std::string> config_args;
};

std::vector<std::string> copy_arguments(int argc, char** argv) {
    if (argc < 1 || argv == nullptr || argv[0] == nullptr) {
        throw std::invalid_argument("invalid process argument vector");
    }
    std::vector<std::string> copied;
    copied.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        if (argv[index] == nullptr) {
            throw std::invalid_argument("process argument must not be null");
        }
        copied.emplace_back(argv[index]);
    }
    return copied;
}

CliOptions parse_cli_options(const std::vector<std::string>& arguments) {
    CliOptions options;
    for (std::size_t index = 2; index < arguments.size(); ++index) {
        const std::string& arg = arguments[index];
        if (arg == "--restart") {
            if (index + 1 >= arguments.size() || arguments[index + 1].empty()) {
                throw std::invalid_argument(
                    "--restart requires a non-empty checkpoint directory path");
            }
            if (options.restart.has_value()) {
                throw std::invalid_argument("--restart may be specified only once");
            }
            options.restart = arguments[++index];
        } else if (arg == "--set") {
            if (index + 1 >= arguments.size()) {
                throw std::invalid_argument("--set requires section.key=value");
            }
            options.config_args.emplace_back("--set");
            options.config_args.emplace_back(arguments[++index]);
        } else {
            throw std::invalid_argument("Unknown HYOWON option: " + arg);
        }
    }
    return options;
}

config::LoadedSimulationConfig load_config(
    const std::filesystem::path& config_path,
    const CliOptions& options) {
    if (options.config_args.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Number of configuration CLI arguments exceeds int range");
    }
    std::vector<const char*> args;
    args.reserve(options.config_args.size());
    for (const auto& arg : options.config_args) args.push_back(arg.c_str());
    return config::ConfigLoader::load_with_identity(
        config_path.string(),
        static_cast<int>(args.size()),
        args.empty() ? nullptr : args.data());
}

std::string sanitize_label(std::string value) {
    for (char& c : value) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!std::isalnum(u) && c != '-' && c != '_') c = '_';
    }
    while (!value.empty() && value.front() == '_') value.erase(value.begin());
    while (!value.empty() && value.back() == '_') value.pop_back();
    return value;
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &value);
#else
    gmtime_r(&value, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}

std::string broadcast_string_from_rank(
    std::string_view value,
    int source_rank,
    int rank,
    int size) {
    return runtime::broadcast_string_collective(
        value,
        source_rank,
        rank,
        size,
        "HYOWON process string broadcast");
}

std::string broadcast_root_string(
    std::string_view value,
    int rank,
    int size) {
    return broadcast_string_from_rank(value, 0, rank, size);
}

template <typename Action>
void collective_root_action(
    int rank,
    int size,
    const std::string& label,
    Action&& action) {
    int failed = 0;
    std::string error;
    if (rank == 0) {
        try {
            action();
        } catch (const std::exception& exception) {
            failed = 1;
            error = exception.what();
        } catch (...) {
            failed = 1;
            error = "unknown root-rank exception";
        }
    }

    if (size > 1) {
#ifdef COSMO_NBODY_HAS_MPI
        if (MPI_Bcast(
                &failed, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "Failed to broadcast root-action status for " + label);
        }
        error = broadcast_root_string(error, rank, size);
#else
        throw std::runtime_error(
            "Collective root action requested in non-MPI build");
#endif
    }

    if (failed != 0) {
        throw std::runtime_error(
            label + (error.empty() ? std::string{} : ": " + error));
    }
}

std::optional<std::string> collective_first_error(
    const std::string& local_error,
    const runtime::RuntimeContext& context) {
    const int local_failed = local_error.empty() ? 0 : 1;
    if (context.size() <= 1) {
        return local_failed != 0
            ? std::optional<std::string>(local_error)
            : std::nullopt;
    }
#ifdef COSMO_NBODY_HAS_MPI
    int any_failed = 0;
    if (MPI_Allreduce(
            &local_failed,
            &any_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to reduce distributed run failure status");
    }
    if (any_failed == 0) return std::nullopt;

    const int local_source = local_failed != 0
        ? context.rank()
        : std::numeric_limits<int>::max();
    int source_rank = 0;
    if (MPI_Allreduce(
            &local_source,
            &source_rank,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "Failed to select first failing MPI rank");
    }
    std::string first_error = context.rank() == source_rank
        ? local_error : std::string{};
    first_error = broadcast_string_from_rank(
        first_error,
        source_rank,
        context.rank(),
        context.size());
    return "rank " + std::to_string(source_rank) + ": " + first_error;
#else
    throw std::runtime_error(
        "Multi-rank failure reduction requested in non-MPI build");
#endif
}

std::filesystem::path create_unique_directory(
    const std::filesystem::path& root,
    const std::string& base) {
    std::filesystem::create_directories(root);
    std::uint64_t suffix = 0;
    while (true) {
        std::ostringstream candidate_name;
        candidate_name << base;
        if (suffix != 0) {
            candidate_name << '_' << std::setw(3) << std::setfill('0') << suffix;
        }
        const std::filesystem::path candidate = root / candidate_name.str();
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) {
            return std::filesystem::absolute(candidate);
        }
        if (error) {
            throw std::runtime_error(
                "Failed to create run directory: " + error.message());
        }
        if (suffix == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Run-directory suffix exhausted uint64 range");
        }
        ++suffix;
    }
}

std::filesystem::path create_exact_directory(
    const std::filesystem::path& root,
    const std::string& name) {
    std::filesystem::create_directories(root);
    const std::filesystem::path candidate = root / name;
    std::error_code error;
    if (std::filesystem::create_directory(candidate, error)) {
        return std::filesystem::absolute(candidate);
    }
    if (error) {
        throw std::runtime_error(
            "Failed to create exact run directory '" + candidate.string()
            + "': " + error.message());
    }
    throw std::runtime_error(
        "Exact run directory already exists; refusing an automatic suffix: "
        + candidate.string());
}

std::string default_run_label(const config::SimulationParameters& config) {
    const std::string& configured_label = config.get_output().run_label;
    std::string label = sanitize_label(configured_label);
    if (!label.empty()) {
        if (!config.get_output().timestamped_run_directory
            && label != configured_label) {
            throw std::invalid_argument(
                "A non-timestamped output.run_label must already contain only "
                "letters, digits, '-' and '_' so the configured exact path is "
                "not rewritten");
        }
        return label;
    }
    if (!configured_label.empty()
        && !config.get_output().timestamped_run_directory) {
        throw std::invalid_argument(
            "A non-timestamped output.run_label must contain at least one "
            "letter or digit");
    }
    std::ostringstream automatic;
    automatic << config.get_gravity().solver
              << "_L" << static_cast<long long>(std::llround(config.get_box().L))
              << "_N" << config.get_box().N
              << "_M" << config.get_box().N_mesh
              << "_seed" << config.get_ic().seed;
    return automatic.str();
}

std::filesystem::path create_run_directory(
    const config::SimulationParameters& config,
    const runtime::RuntimeContext& context,
    const std::filesystem::path& launch_directory) {
    std::string selected;
    collective_root_action(
        context.rank(), context.size(), "Create run directory", [&] {
            std::filesystem::path root(config.get_output().root_directory);
            if (root.is_relative()) root = launch_directory / root;
            const std::string label = default_run_label(config);
            if (config.get_output().timestamped_run_directory) {
                selected = create_unique_directory(
                    root, utc_timestamp() + "_" + label).string();
            } else {
                selected = create_exact_directory(root, label).string();
            }
        });
    selected = broadcast_root_string(
        selected, context.rank(), context.size());
    return std::filesystem::path(selected);
}

std::optional<std::string> stage_error(
    std::string local_error,
    const runtime::RuntimeContext& process_context,
    const std::string& stage) {
    const auto global_error = collective_first_error(local_error, process_context);
    if (!global_error.has_value()) return std::nullopt;
    return stage + " failed on " + *global_error;
}

std::string build_provenance_json() {
    std::ostringstream out;
    out << "{\n"
        << "  \"product_kind\": \"build_provenance\",\n"
        << "  \"software_name\": \"" << build_info::SOFTWARE_NAME << "\",\n"
        << "  \"software_version\": \"" << build_info::VERSION << "\",\n"
        << "  \"release_stage\": \"" << build_info::RELEASE_STAGE << "\",\n"
        << "  \"source_commit\": \"" << build_info::SOURCE_COMMIT << "\",\n"
        << "  \"source_tree\": \"" << build_info::SOURCE_TREE << "\",\n"
        << "  \"source_state\": \"" << build_info::SOURCE_STATE << "\",\n"
        << "  \"source_capture_phase\": \""
        << build_info::SOURCE_CAPTURE_PHASE << "\"\n"
        << "}\n";
    return out.str();
}

void write_build_provenance(const std::filesystem::path& run_directory) {
    const auto path = run_directory / "build_provenance.json";
    const std::string provenance = build_provenance_json();
    const std::string expected_sha256 = io::sha256_text(provenance);
    io::write_text_durable_atomic_validated(
        path,
        provenance,
        "build provenance",
        [expected_sha256](const std::filesystem::path& published_path) {
            io::require_file_sha256(
                published_path,
                expected_sha256,
                "Build provenance exact-byte validation");
        });
}

void write_build_provenance_best_effort(
    const std::filesystem::path& run_directory) noexcept {
    try {
        write_build_provenance(run_directory);
    } catch (const std::exception& exception) {
        std::cerr << "HYOWON warning: could not write build provenance: "
                  << exception.what() << '\n';
    } catch (...) {
        std::cerr << "HYOWON warning: could not write build provenance: "
                     "unknown error\n";
    }
}

void write_run_metadata(
    const std::filesystem::path& run_directory,
    const config::SimulationParameters& config,
    bool planning_records_complete,
    bool replace_existing) {
    const auto path = run_directory / "run_metadata.json";
    const std::string metadata = io::RunMetadata::from_config(
        config,
        io::ProductLineage::DirectSimulation,
        {},
        planning_records_complete).to_json();
    const std::string expected_sha256 = io::sha256_text(metadata);
    if (replace_existing) {
        io::write_text_durable_atomic(
            path,
            metadata,
            "final run metadata");
        io::require_file_sha256(
            path,
            expected_sha256,
            "Final run metadata exact-byte validation");
        return;
    }
    io::write_text_durable_atomic_validated(
        path,
        metadata,
        "run metadata",
        [expected_sha256](const std::filesystem::path& published_path) {
            io::require_file_sha256(
                published_path,
                expected_sha256,
                "Run metadata exact-byte validation");
        });
}

void run_one_collectively(
    const config::SimulationParameters& config,
    const std::filesystem::path& run_directory,
    const std::optional<std::filesystem::path>& restart_path,
    const runtime::RuntimeContext& process_context) {
    const auto artifact_layout = io::RunArtifactLayout::organized(run_directory);
    collective_root_action(
        process_context.rank(),
        process_context.size(),
        "Prepare run artifact directories",
        [&] { artifact_layout.prepare_directories_exclusive(); });

    std::unique_ptr<runtime::SimulationRunner> runner;
    std::string local_error;
    try {
        runner = std::make_unique<runtime::SimulationRunner>(
            config, artifact_layout);
    } catch (const std::exception& exception) {
        local_error = exception.what();
    } catch (...) {
        local_error = "unknown SimulationRunner construction exception";
    }
    if (const auto error = stage_error(
            std::move(local_error), process_context, "Runner construction")) {
        throw std::runtime_error(*error);
    }
    if (!runner) {
        throw std::runtime_error("Runner construction succeeded without a runner");
    }

    if (runner->rank() == 0) {
        std::cout << "========================================\n"
                  << " HYOWON v" << build_info::VERSION << ' '
                  << build_info::RELEASE_STAGE << ": PM/TreePM Runtime\n"
                  << " ranks=" << runner->size() << "\n"
                  << " run_dir=" << run_directory.string() << "\n"
                  << "========================================\n";
    }

    local_error.clear();
    try {
        if (restart_path.has_value()) {
            runner->initialize_from_restart(restart_path->string());
        } else {
            runner->initialize();
        }
    } catch (const std::exception& exception) {
        local_error = exception.what();
    } catch (...) {
        local_error = "unknown runner initialization exception";
    }
    if (const auto error = stage_error(
            std::move(local_error), process_context, "Runner initialization")) {
        throw std::runtime_error(*error);
    }

    collective_root_action(
        process_context.rank(),
        process_context.size(),
        "Write run metadata",
        [&] { write_run_metadata(run_directory, config, false, false); });

    local_error.clear();
    try {
        runner->run();
    } catch (const std::exception& exception) {
        local_error = exception.what();
    } catch (...) {
        local_error = "unknown dynamics exception";
    }
    if (const auto error = stage_error(
            std::move(local_error), process_context, "Dynamics")) {
        throw std::runtime_error(*error);
    }

    collective_root_action(
        process_context.rank(),
        process_context.size(),
        "Finalize run metadata",
        [&] { write_run_metadata(run_directory, config, true, true); });

    // Execution provenance is deliberately collected only after dynamics and
    // final run metadata are complete. It can therefore describe execution
    // coordinates without perturbing FFT planning, trajectory arithmetic, or
    // the scientific metadata products it accompanies.
    std::optional<io::ExecutionProvenance> execution_provenance;
    try {
        execution_provenance.emplace(
            io::collect_execution_provenance(process_context));
    } catch (const std::exception& exception) {
        if (process_context.rank() == 0) {
            std::cerr << "HYOWON warning: could not collect execution provenance: "
                      << exception.what() << '\n';
        }
    } catch (...) {
        if (process_context.rank() == 0) {
            std::cerr << "HYOWON warning: could not collect execution provenance: "
                         "unknown error\n";
        }
    }

    if (process_context.rank() == 0) {
        if (execution_provenance.has_value()) {
            try {
                io::write_execution_provenance(
                    artifact_layout.diagnostics_directory(),
                    *execution_provenance);
            } catch (const std::exception& exception) {
                std::cerr << "HYOWON warning: could not write execution provenance: "
                          << exception.what() << '\n';
            } catch (...) {
                std::cerr << "HYOWON warning: could not write execution provenance: "
                             "unknown error\n";
            }
        }
        write_build_provenance_best_effort(run_directory);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (cosmo_nbody::build_info::write_version_if_requested(argc, argv, std::cout)) {
        return std::cout ? 0 : 1;
    }
    if (argc == 2 && argv != nullptr && argv[1] != nullptr
        && (std::string_view(argv[1]) == "--help"
            || std::string_view(argv[1]) == "-h")) {
        std::cout
            << "Usage:\n  "
            << (argv[0] != nullptr ? argv[0] : "hyowon")
            << " <config.toml> [--restart <checkpoint-directory>] "
               "[--set section.key=value ...]\n"
            << "  --version  Print configure-time build identity without starting a run.\n";
        return 0;
    }
    if (argc < 2) {
        std::cerr
            << "Usage:\n  "
            << (argc > 0 && argv != nullptr && argv[0] != nullptr
                    ? argv[0]
                    : "hyowon")
            << " <config.toml> [--restart <checkpoint-directory>] "
               "[--set section.key=value ...]\n";
        return 1;
    }

    try {
        const std::filesystem::path launch_directory =
            std::filesystem::current_path();
        const auto arguments = copy_arguments(argc, argv);
        const CliOptions options = parse_cli_options(arguments);
        const std::filesystem::path config_path =
            std::filesystem::absolute(arguments.at(1));
        const config::LoadedSimulationConfig loaded_config =
            load_config(config_path, options);
        const config::SimulationParameters& base_config =
            loaded_config.parameters;

        runtime::RuntimeContext process_context(base_config.get_runtime());
        const std::string root_config_sha256 = broadcast_root_string(
            loaded_config.source_sha256,
            process_context.rank(),
            process_context.size());
        runtime::synchronize_mpi_failure(
            loaded_config.source_sha256 != root_config_sha256,
            process_context.size(),
            "MPI ranks parsed different configuration bytes");

        // Raw TOML identity excludes CLI overrides and restart selection, which
        // can change collective participation before the runner is constructed.
        std::string launch_options;
        std::exception_ptr launch_options_error;
        try {
            for (std::size_t index = 2; index < arguments.size(); ++index) {
                launch_options += std::to_string(arguments[index].size()) + ':';
                launch_options += arguments[index];
            }
        } catch (...) {
            launch_options_error = std::current_exception();
        }
        runtime::synchronize_mpi_exception(
            launch_options_error, process_context.size(), "MPI launch options");
        const std::string root_launch_options = broadcast_root_string(
            launch_options, process_context.rank(), process_context.size());
        runtime::synchronize_mpi_failure(
            launch_options != root_launch_options,
            process_context.size(),
            "MPI ranks supplied different CLI overrides or restart options");

        const std::optional<std::filesystem::path> restart_path = options.restart
            ? std::optional<std::filesystem::path>(
                std::filesystem::absolute(*options.restart))
            : std::nullopt;

        const auto run_directory = create_run_directory(
            base_config, process_context, launch_directory);
        run_one_collectively(
            base_config,
            run_directory,
            restart_path,
            process_context);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HYOWON fatal: " << error.what() << '\n';
        return 1;
    }
}
