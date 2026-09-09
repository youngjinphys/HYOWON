#include "cosmo_nbody/runtime/runtime_context.hpp"

#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

// Exercise the real MPI/OpenMP startup boundary. Removing startup exception
// agreement lets one rank return successfully while another tears MPI down.
int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
            "usage: mpi_runtime_startup_test attached|owned "
            "failure|success|automatic failing_rank launch_rank\n");
        return 2;
    }
    const std::string_view lifecycle(argv[1]);
    const std::string_view scenario(argv[2]);
    if ((lifecycle != "attached" && lifecycle != "owned")
        || (scenario != "failure" && scenario != "success"
            && scenario != "automatic")) {
        return 2;
    }
    const bool attached = lifecycle == "attached";
    const bool expect_failure = scenario == "failure";
    const bool automatic = scenario == "automatic";
    const int failing_rank = std::atoi(argv[3]);
    const int launch_rank = std::atoi(argv[4]);
    int errors = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            ++errors;
            std::fprintf(stderr, "rank %d: %s\n", launch_rank, message);
        }
    };

    // Non-default initial controls make rollback on healthy peers observable.
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    const int original_levels = omp_get_max_active_levels();
    check(omp_get_thread_limit()
            == ((expect_failure && launch_rank == failing_rank)
                || (automatic && launch_rank == 0) ? 1 : 2),
        "the launcher did not install the expected OpenMP limit");

    if (attached) {
        int provided = MPI_THREAD_SINGLE;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)
                != MPI_SUCCESS
            || provided < MPI_THREAD_FUNNELED) {
            return 2;
        }
        if (MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL)
            != MPI_SUCCESS) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    cosmo_nbody::config::RuntimeParams params;
    params.mpi_enabled = true;
    params.num_threads = automatic ? 0 : 2;
    bool caught = false;
    try {
        cosmo_nbody::runtime::RuntimeContext context(params);
        check(!expect_failure,
            "startup returned successfully despite another rank's failure");
        check(context.mpi_active(), "successful context did not activate MPI");
        check(context.rank() == launch_rank, "launcher and MPI ranks differ");
        check(context.thread_count() >= 1 && context.thread_count() <= 2,
            "thread selection exceeded the configured capacity");
        check(automatic || context.thread_count() == 2,
            "explicit thread selection was silently reduced");
        check(!automatic || launch_rank != 0 || context.thread_count() == 1,
            "automatic selection ignored rank zero's thread limit");

        MPI_Errhandler active_handler = MPI_ERRHANDLER_NULL;
        check(MPI_Comm_get_errhandler(MPI_COMM_WORLD, &active_handler)
                == MPI_SUCCESS
                && active_handler == MPI_ERRORS_RETURN,
            "successful context did not install MPI_ERRORS_RETURN");
        if (active_handler != MPI_ERRHANDLER_NULL) {
            MPI_Errhandler_free(&active_handler);
        }

        // An owned context must agree before returning: a healthy rank entering
        // normal collective work otherwise hangs against a peer in Finalize.
        // Attached failure cases instead reach the test's result reduction,
        // providing an immediate diagnostic for asymmetric construction.
        if (!expect_failure || !attached) {
            const auto topology = context.collect_topology_diagnostics();
            check(topology.rank_count == static_cast<unsigned>(context.size()),
                "subsequent topology collective returned the wrong rank count");
            check(automatic || (topology.effective_threads_min == 2
                    && topology.effective_threads_max == 2),
                "successful ranks did not retain their explicit thread count");
        }
    } catch (const std::invalid_argument& error) {
        caught = true;
        check(expect_failure && launch_rank == failing_rank,
            "a healthy rank raised a local argument error");
        check(std::string_view(error.what()).find("thread-limit")
                != std::string_view::npos,
            "originating rank lost its OpenMP thread-limit diagnostic");
    } catch (const std::exception& error) {
        caught = true;
        check(expect_failure && launch_rank != failing_rank,
            "an unexpected startup exception occurred");
        check(std::string_view(error.what()).find("another MPI rank")
                != std::string_view::npos,
            "peer failure did not identify a failure on another rank");
    }
    check(caught == expect_failure,
        "startup exception outcome did not match the scenario");
    check(omp_get_max_threads() == original_threads,
        "runtime did not restore the caller's OpenMP thread count");
    check(omp_get_dynamic() == original_dynamic,
        "runtime did not restore the caller's OpenMP dynamic control");
    check(omp_get_max_active_levels() == original_levels,
        "runtime did not restore the caller's OpenMP nesting control");

    int finalized = -1;
    check(MPI_Finalized(&finalized) == MPI_SUCCESS,
        "could not query MPI finalization state");
    check(finalized == (attached ? 0 : 1),
        "runtime did not respect ownership of the MPI lifecycle");
    if (attached && !finalized) {
        MPI_Errhandler restored_handler = MPI_ERRHANDLER_NULL;
        check(MPI_Comm_get_errhandler(MPI_COMM_WORLD, &restored_handler)
                == MPI_SUCCESS
                && restored_handler == MPI_ERRORS_ARE_FATAL,
            "runtime did not restore the caller's MPI error handler");
        if (restored_handler != MPI_ERRHANDLER_NULL) {
            MPI_Errhandler_free(&restored_handler);
        }
        int total_errors = 0;
        if (MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD) != MPI_SUCCESS) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        errors = total_errors;
        MPI_Finalize();
    }
    return errors == 0 ? 0 : 1;
}
