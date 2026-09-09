#include "cosmo_nbody/build_info.hpp"
#include "cosmo_nbody/config/config_loader.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/run_artifact_layout.hpp"
#include "cosmo_nbody/runtime/simulation_runner.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

struct Options {
    std::string config_path;
    std::string output_path;
    std::string evidence_path;
    bool overwrite{false};
};

bool path_present(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (status.type() == std::filesystem::file_type::not_found) {
        return false;
    }
    if (error) {
        throw std::runtime_error(
            "Failed to inspect output path '" + path.string()
            + "': " + error.message());
    }
    return true;
}

void ensure_parent_directory(const std::filesystem::path& path) {
    if (const auto parent = path.parent_path(); !parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if (error) {
            throw std::runtime_error(
                "Failed to create output directory '"
                + parent.string() + "': " + error.message());
        }
    }
}

void require_output_admissible(
    const std::filesystem::path& path,
    const char* role,
    bool overwrite) {
    if (!path_present(path)) return;
    if (std::filesystem::is_symlink(path)
        || !std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            std::string("Existing ") + role
            + " is not a regular non-symlink file: '"
            + path.string() + "'");
    }
    if (!overwrite) {
        throw std::runtime_error(
            std::string(role)
            + " already exists; pass --overwrite to replace it transactionally: '"
            + path.string() + "'");
    }
}

std::filesystem::path normalized_path_identity(
    const std::filesystem::path& path) {
    std::error_code error;
    const auto normalized = std::filesystem::weakly_canonical(path, error);
    if (error) {
        throw std::runtime_error(
            "Failed to resolve output path identity '" + path.string()
            + "': " + error.message());
    }
    return normalized.lexically_normal();
}

bool paths_alias(
    const std::filesystem::path& lhs,
    const std::filesystem::path& rhs) {
    if (normalized_path_identity(lhs) == normalized_path_identity(rhs)) {
        return true;
    }
    if (!path_present(lhs) || !path_present(rhs)) {
        return false;
    }
    std::error_code error;
    const bool equivalent = std::filesystem::equivalent(lhs, rhs, error);
    if (error) {
        throw std::runtime_error(
            "Failed to compare output path identities '" + lhs.string()
            + "' and '" + rhs.string() + "': " + error.message());
    }
    return equivalent;
}

void invalidate_existing_evidence_before_snapshot_replace(
    const std::filesystem::path& path) {
    if (!path_present(path)) return;
    if (std::filesystem::is_symlink(path)
        || !std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "Existing IC evidence output became non-regular before overwrite: '"
            + path.string() + "'");
    }
    std::error_code error;
    const bool removed = std::filesystem::remove(path, error);
    if (error || !removed) {
        throw std::runtime_error(
            "Failed to invalidate existing IC evidence before replacing its "
            "bound snapshot: '" + path.string() + "'"
            + (error ? ": " + error.message() : std::string{}));
    }
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "Usage: hyowon_make_ic <config.toml> --output <initial_conditions.hdf5> "
            "[--evidence <initial_conditions.json>] [--overwrite]");
    }

    Options options;
    options.config_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--overwrite") {
            if (options.overwrite) {
                throw std::invalid_argument(
                    "--overwrite may be specified only once");
            }
            options.overwrite = true;
            continue;
        }
        if (argument != "--output" && argument != "--evidence") {
            throw std::invalid_argument(
                "Unknown hyowon_make_ic option: " + argument);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(argument + " requires a file path");
        }
        std::string& destination = argument == "--output"
            ? options.output_path : options.evidence_path;
        if (!destination.empty()) {
            throw std::invalid_argument(argument + " may be specified only once");
        }
        destination = argv[++index];
    }
    if (options.output_path.empty()) {
        throw std::invalid_argument("hyowon_make_ic requires --output <path>");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (cosmo_nbody::build_info::write_version_if_requested(argc, argv, std::cout)) {
        return std::cout ? 0 : 1;
    }
    using namespace cosmo_nbody;

    if (argc == 2 && argv != nullptr && argv[1] != nullptr
        && (std::string_view(argv[1]) == "--help"
            || std::string_view(argv[1]) == "-h")) {
        std::cout
            << "Usage: "
            << (argv[0] != nullptr ? argv[0] : "hyowon_make_ic")
            << " <config.toml> --output <initial_conditions.hdf5> "
               "[--evidence <initial_conditions.json>] [--overwrite]\n"
            << "  --version  Print configure-time build identity without starting a run.\n";
        return 0;
    }

    try {
        const Options options = parse_options(argc, argv);
        const config::SimulationParameters config =
            config::ConfigLoader::load(options.config_path);
        if (config.get_ic().mode != "generate") {
            throw std::invalid_argument(
                "hyowon_make_ic accepts only ic.mode='generate'; use the original snapshot directly instead of laundering it through a new IC artifact");
        }
        if (config.get_runtime().mpi_enabled) {
            throw std::invalid_argument(
                "hyowon_make_ic does not support distributed MPI output; generate the canonical IC with a serial configuration so one complete particle population is written");
        }

        const std::filesystem::path output(options.output_path);
        const std::filesystem::path evidence_path(options.evidence_path);
        ensure_parent_directory(output);
        if (!options.evidence_path.empty()) {
            ensure_parent_directory(evidence_path);
            if (paths_alias(output, evidence_path)) {
                throw std::invalid_argument(
                    "--output and --evidence must name distinct filesystem objects");
            }
        }

        require_output_admissible(
            output, "IC output", options.overwrite);
        if (!options.evidence_path.empty()) {
            require_output_admissible(
                evidence_path, "IC evidence output", options.overwrite);
        }

        runtime::SimulationRunner runner(
            config,
            io::RunArtifactLayout::flat(std::filesystem::current_path()));
        runner.initialize_initial_conditions_only();
        const core::ParticleStore& particles = runner.get_particles();
        const auto& generated_evidence = runner.initial_condition_evidence();
        if (!generated_evidence.has_value()) {
            throw std::logic_error(
                "hyowon_make_ic generated no realised initial-condition evidence");
        }

        if (runner.rank() == 0) {
            const core::Real a_start =
                1.0 / (1.0 + config.get_time().z_start);

            // In overwrite mode invalidate the old sidecar before replacing the
            // snapshot; failure may leave no sidecar, but never stale evidence
            // bound to new HDF5 bytes.
            if (options.overwrite && !options.evidence_path.empty()) {
                invalidate_existing_evidence_before_snapshot_replace(
                    evidence_path);
            }

            io::SnapshotIO writer(config);
            std::string snapshot_sha256;
            writer.write_snapshot_to_path(
                particles,
                a_start,
                output.string(),
                options.overwrite
                    ? io::SnapshotWritePolicy::ReplaceExisting
                    : io::SnapshotWritePolicy::RequireAbsent,
                &snapshot_sha256);

            std::cout << "Wrote canonical initial-condition snapshot: "
                      << output << "\n"
                      << "  particles: " << particles.num_owned_particles() << "\n"
                      << "  a_start: " << a_start << "\n"
                      << "  source: " << build_info::SOURCE_COMMIT
                      << " (" << build_info::SOURCE_STATE << ", "
                      << build_info::SOURCE_CAPTURE_PHASE << ")\n"
                      << "  native_snapshot_object_sha256: "
                      << snapshot_sha256 << "\n";

            if (!options.evidence_path.empty()) {
                std::ostringstream evidence_json;
                evidence_json
                    << "{\n"
                    << "  \"product_kind\": \"generated_ic_export_evidence\",\n"
                    << "  \"schema_version\": 1,\n"
                    << "  \"verdict_semantics\": false,\n"
                    << "  \"scientific_accuracy_certificate\": false,\n"
                    << "  \"interpretation\": \"generated_ic_measurements_and_provenance_require_independent_physical_validation\",\n"
                    << "  \"build_provenance_scope\": \"configure_time_git_observation_not_executable_attestation\",\n"
                    << "  \"build_source_commit\": \""
                    << build_info::SOURCE_COMMIT << "\",\n"
                    << "  \"build_source_tree\": \""
                    << build_info::SOURCE_TREE << "\",\n"
                    << "  \"build_source_state\": \""
                    << build_info::SOURCE_STATE << "\",\n"
                    << "  \"build_source_capture_phase\": \""
                    << build_info::SOURCE_CAPTURE_PHASE << "\",\n"
                    << "  \"native_snapshot_object_sha256\": \""
                    << snapshot_sha256 << "\",\n"
                    << "  \"realised_initial_conditions\": "
                    << generated_evidence->to_json() << "\n"
                    << "}\n";

                // Publish the new sidecar with no-replace semantics; concurrent
                // writers fail instead of overwriting one another.
                io::write_text_durable_atomic_validated(
                    evidence_path,
                    evidence_json.str(),
                    "generated IC export evidence",
                    [](const std::filesystem::path&) {});
                std::cout << "Wrote generated initial-condition evidence: "
                          << evidence_path << "\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "hyowon_make_ic failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
