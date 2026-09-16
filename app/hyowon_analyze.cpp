#include "nbody_analyze_pipeline.hpp"

#include "cosmo_nbody/analysis/analysis_request.hpp"
#include "cosmo_nbody/build_info.hpp"
#include "cosmo_nbody/io/checked_output_file.hpp"
#include "cosmo_nbody/io/durable_file_publication.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void write_analysis_build_provenance(
    const std::filesystem::path& output_directory) {
    const auto path = output_directory / "analysis_build_provenance.json";
    auto output = cosmo_nbody::io::open_checked_output_file(
        path, "analysis build provenance");
    output
        << "{\n"
        << "  \"product_kind\": \"analysis_build_provenance\",\n"
        << "  \"software_name\": \""
        << cosmo_nbody::build_info::SOFTWARE_NAME << "\",\n"
        << "  \"software_version\": \""
        << cosmo_nbody::build_info::VERSION << "\",\n"
        << "  \"release_stage\": \""
        << cosmo_nbody::build_info::RELEASE_STAGE << "\",\n"
        << "  \"source_commit\": \""
        << cosmo_nbody::build_info::SOURCE_COMMIT << "\",\n"
        << "  \"source_tree\": \""
        << cosmo_nbody::build_info::SOURCE_TREE << "\",\n"
        << "  \"source_state\": \""
        << cosmo_nbody::build_info::SOURCE_STATE << "\",\n"
        << "  \"source_capture_phase\": \""
        << cosmo_nbody::build_info::SOURCE_CAPTURE_PHASE << "\",\n"
        << "  \"build_provenance_scope\": "
           "\"configure_time_git_observation_not_executable_attestation\"\n"
        << "}\n";
    cosmo_nbody::io::close_checked_output_file(
        output, path, "analysis build provenance");
}

int entry_main(int argc, char** argv) {
    if (cosmo_nbody::build_info::write_version_if_requested(argc, argv, std::cout)) {
        return std::cout ? 0 : 1;
    }
    if (argc == 2 && argv != nullptr && argv[1] != nullptr
        && (std::string_view(argv[1]) == "--help"
            || std::string_view(argv[1]) == "-h")) {
        std::cout << cosmo_nbody::analysis::analysis_request_usage()
                  << "\n  --version  Print configure-time build identity without running analysis.\n";
        return 0;
    }

    std::vector<std::string> arguments;
    if (argc > 0) arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        if (argv == nullptr || argv[index] == nullptr) {
            throw std::invalid_argument("Invalid analyzer argument vector");
        }
        arguments.emplace_back(argv[index]);
    }

    auto request = cosmo_nbody::analysis::parse_analysis_request(arguments);
    const auto& resource_policy = request.resource_policy;
    cosmo_nbody::runtime::HostThreadContext host_threads(
        resource_policy.requested_threads);
    cosmo_nbody::io::DurableDirectoryPublication output_directory(
        request.output_directory,
        "analysis output directory");
    try {
        write_analysis_build_provenance(output_directory.staging_path());
        cosmo_nbody::app::nbody_analyze::run_pipeline(
            request,
            host_threads.thread_count(),
            host_threads.automatic_thread_capacity(),
            host_threads.thread_count_is_automatic(),
            host_threads.thread_selection_reason(),
            output_directory.staging_path());
        output_directory.publish_nonempty();
        return 0;
    } catch (...) {
        const std::exception_ptr primary_failure = std::current_exception();
        if (!output_directory.canonical_visible()) {
            try {
                output_directory.discard_unpublished();
            } catch (const std::exception& cleanup_error) {
                std::cerr
                    << "hyowon_analyze cleanup warning: unpublished analysis "
                       "staging directory could not be removed: "
                    << cleanup_error.what() << '\n';
            } catch (...) {
                std::cerr
                    << "hyowon_analyze cleanup warning: unpublished analysis "
                       "staging directory cleanup failed with a non-standard exception\n";
            }
        }
        std::rethrow_exception(primary_failure);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        return entry_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "hyowon_analyze fatal: " << error.what() << '\n';
        return 1;
    }
}
