#include "cosmo_nbody/config/config_loader.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

int hyowon_program_main(int argc, char** argv);

namespace {

#ifdef COSMO_NBODY_HAS_MPI
class StartupMpiLease {
public:
    StartupMpiLease() {
        int initialized = 0;
        int finalized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS
            || MPI_Finalized(&finalized) != MPI_SUCCESS) {
            throw std::runtime_error("Failed to query MPI startup state");
        }
        if (finalized) {
            throw std::runtime_error("HYOWON cannot start after MPI_Finalize");
        }
        if (!initialized) {
            int provided = MPI_THREAD_SINGLE;
            int local_argc = 0;
            char** local_argv = nullptr;
            if (MPI_Init_thread(
                    &local_argc, &local_argv, MPI_THREAD_FUNNELED, &provided)
                != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Init_thread failed during HYOWON startup");
            }
            owns_mpi_ = true;
            if (provided < MPI_THREAD_FUNNELED) {
                (void)MPI_Finalize();
                owns_mpi_ = false;
                throw std::runtime_error(
                    "MPI implementation does not provide MPI_THREAD_FUNNELED");
            }
        } else {
            int is_main = 0;
            int provided = MPI_THREAD_SINGLE;
            if (MPI_Is_thread_main(&is_main) != MPI_SUCCESS || is_main == 0
                || MPI_Query_thread(&provided) != MPI_SUCCESS
                || provided < MPI_THREAD_FUNNELED) {
                throw std::runtime_error(
                    "Existing MPI runtime is incompatible with HYOWON startup");
            }
        }
        if (MPI_Comm_get_errhandler(MPI_COMM_WORLD, &previous_handler_) != MPI_SUCCESS
            || previous_handler_ == MPI_ERRHANDLER_NULL) {
            release_noexcept();
            throw std::runtime_error("Failed to capture MPI_COMM_WORLD error handler");
        }
        if (MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
            (void)MPI_Errhandler_free(&previous_handler_);
            previous_handler_ = MPI_ERRHANDLER_NULL;
            release_noexcept();
            throw std::runtime_error("Failed to install MPI_ERRORS_RETURN during startup");
        }
        handler_changed_ = true;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &rank_) != MPI_SUCCESS
            || MPI_Comm_size(MPI_COMM_WORLD, &size_) != MPI_SUCCESS
            || size_ < 1 || rank_ < 0 || rank_ >= size_) {
            release_noexcept();
            throw std::runtime_error("Failed to query MPI startup topology");
        }
    }

    ~StartupMpiLease() { release_noexcept(); }
    StartupMpiLease(const StartupMpiLease&) = delete;
    StartupMpiLease& operator=(const StartupMpiLease&) = delete;

    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }

    [[noreturn]] void abort_process(int code) const noexcept {
        (void)MPI_Abort(MPI_COMM_WORLD, code == 0 ? 1 : code);
        std::abort();
    }

    std::string broadcast_string(std::string value, int source) const {
        std::uint64_t length = rank_ == source
            ? static_cast<std::uint64_t>(value.size()) : 0U;
        if (MPI_Bcast(&length, 1, MPI_UINT64_T, source, MPI_COMM_WORLD) != MPI_SUCCESS) {
            abort_process(1);
        }
        if (length > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            abort_process(1);
        }
        if (rank_ != source) value.resize(static_cast<std::size_t>(length));
        if (length != 0 && MPI_Bcast(
                value.data(), static_cast<int>(length), MPI_CHAR,
                source, MPI_COMM_WORLD) != MPI_SUCCESS) {
            abort_process(1);
        }
        return value;
    }

    void synchronize_startup_error(
        const std::string& local_error,
        std::string_view stage) const {
        const int local_failed = local_error.empty() ? 0 : 1;
        int any_failed = 0;
        if (MPI_Allreduce(
                &local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            abort_process(1);
        }
        if (any_failed == 0) return;
        const int candidate = local_failed != 0
            ? rank_ : std::numeric_limits<int>::max();
        int source = 0;
        if (MPI_Allreduce(
                &candidate, &source, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            abort_process(1);
        }
        std::string message = rank_ == source ? local_error : std::string{};
        message = broadcast_string(std::move(message), source);
        throw std::runtime_error(
            std::string(stage) + " failed on rank " + std::to_string(source)
            + (message.empty() ? std::string{} : ": " + message));
    }

    void require_equal_string(
        const std::string& value,
        std::string_view label) const {
        const std::string root = broadcast_string(rank_ == 0 ? value : std::string{}, 0);
        const int different = value == root ? 0 : 1;
        int any_different = 0;
        if (MPI_Allreduce(
                &different, &any_different, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            abort_process(1);
        }
        if (any_different != 0) {
            throw std::runtime_error(std::string(label) + " differs across MPI ranks");
        }
    }

    void require_uniform_mpi_mode(bool enabled) const {
        const int local = enabled ? 1 : 0;
        int minimum = 0;
        int maximum = 0;
        if (MPI_Allreduce(&local, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD)
                != MPI_SUCCESS
            || MPI_Allreduce(&local, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD)
                != MPI_SUCCESS) {
            abort_process(1);
        }
        if (minimum != maximum) {
            throw std::runtime_error(
                "runtime.mpi_enabled differs across MPI ranks");
        }
        if (size_ > 1 && minimum == 0) {
            throw std::runtime_error(
                "A multi-rank launch requires runtime.mpi_enabled=true");
        }
    }

private:
    void release_noexcept() noexcept {
        int finalized = 0;
        const bool live = MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized;
        if (handler_changed_) {
            if (live && previous_handler_ != MPI_ERRHANDLER_NULL) {
                (void)MPI_Comm_set_errhandler(MPI_COMM_WORLD, previous_handler_);
                (void)MPI_Errhandler_free(&previous_handler_);
            }
            previous_handler_ = MPI_ERRHANDLER_NULL;
            handler_changed_ = false;
        }
        if (owns_mpi_ && live) (void)MPI_Finalize();
        owns_mpi_ = false;
    }

    int rank_{0};
    int size_{1};
    bool owns_mpi_{false};
    bool handler_changed_{false};
    MPI_Errhandler previous_handler_{MPI_ERRHANDLER_NULL};
};
#endif

bool is_non_run_invocation(int argc, char** argv) {
    if (argc < 2 || argv == nullptr || argv[1] == nullptr) return true;
    const std::string_view first(argv[1]);
    return first == "--help" || first == "-h" || first == "--version";
}

#ifdef COSMO_NBODY_HAS_MPI
struct PreflightConfig {
    std::string source_sha256;
    bool mpi_enabled{false};
    std::string launch_options;
};

PreflightConfig preflight_config(int argc, char** argv) {
    if (argc < 2 || argv == nullptr || argv[1] == nullptr) {
        throw std::invalid_argument("missing HYOWON configuration path");
    }
    std::vector<std::string> config_args;
    std::string launch_options;
    bool restart_seen = false;
    for (int index = 2; index < argc; ++index) {
        if (argv[index] == nullptr) {
            throw std::invalid_argument("process argument must not be null");
        }
        const std::string arg(argv[index]);
        launch_options += std::to_string(arg.size()) + ':' + arg;
        if (arg == "--restart") {
            if (restart_seen) {
                throw std::invalid_argument("--restart may be specified only once");
            }
            restart_seen = true;
            if (++index >= argc || argv[index] == nullptr
                || std::string_view(argv[index]).empty()) {
                throw std::invalid_argument("--restart requires a checkpoint directory");
            }
            const std::string value(argv[index]);
            launch_options += std::to_string(value.size()) + ':' + value;
        } else if (arg == "--set") {
            if (++index >= argc || argv[index] == nullptr) {
                throw std::invalid_argument("--set requires section.key=value");
            }
            config_args.emplace_back("--set");
            config_args.emplace_back(argv[index]);
            const std::string value(argv[index]);
            launch_options += std::to_string(value.size()) + ':' + value;
        } else {
            throw std::invalid_argument("Unknown HYOWON option: " + arg);
        }
    }
    if (config_args.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Number of configuration CLI arguments exceeds int range");
    }
    std::vector<const char*> raw_args;
    raw_args.reserve(config_args.size());
    for (const auto& arg : config_args) raw_args.push_back(arg.c_str());
    const auto loaded = cosmo_nbody::config::ConfigLoader::load_with_identity(
        argv[1], static_cast<int>(raw_args.size()),
        raw_args.empty() ? nullptr : raw_args.data());
    return {loaded.source_sha256, loaded.parameters.get_runtime().mpi_enabled,
            std::move(launch_options)};
}
#endif

} // namespace

int main(int argc, char** argv) {
#ifndef COSMO_NBODY_HAS_MPI
    return hyowon_program_main(argc, argv);
#else
    if (is_non_run_invocation(argc, argv)) {
        return hyowon_program_main(argc, argv);
    }
    try {
        StartupMpiLease startup;
        std::optional<PreflightConfig> config;
        std::string local_error;
        try {
            config.emplace(preflight_config(argc, argv));
        } catch (const std::exception& error) {
            local_error = error.what();
        } catch (...) {
            local_error = "unknown startup configuration exception";
        }
        startup.synchronize_startup_error(local_error, "HYOWON MPI preflight");
        if (!config.has_value()) {
            throw std::runtime_error("MPI preflight succeeded without configuration state");
        }
        startup.require_equal_string(config->source_sha256, "Configuration bytes");
        startup.require_equal_string(config->launch_options, "CLI overrides/restart options");
        startup.require_uniform_mpi_mode(config->mpi_enabled);

        const int status = hyowon_program_main(argc, argv);
        if (status != 0 && startup.size() > 1) startup.abort_process(status);
        return status;
    } catch (const std::exception& error) {
        std::cerr << "HYOWON fatal: " << error.what() << '\n';
        return 1;
    }
#endif
}
