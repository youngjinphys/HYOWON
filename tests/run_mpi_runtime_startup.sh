#!/bin/sh
set -eu

if [ "$#" -ne 4 ]; then
    echo "usage: run_mpi_runtime_startup.sh binary lifecycle scenario failing_rank" >&2
    exit 2
fi

# OpenMP consumes environment controls when the executable loads. Setting them
# inside main would miss the real per-rank limit used by the runtime policy.
startup_rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${MV2_COMM_WORLD_RANK:-}}}}
if [ -z "$startup_rank" ]; then
    echo "MPI launcher did not expose a supported rank environment variable" >&2
    exit 2
fi
export OMP_NUM_THREADS=3
export OMP_DYNAMIC=TRUE
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_THREAD_LIMIT=2
if { [ "$3" = failure ] && [ "$startup_rank" = "$4" ]; } \
    || { [ "$3" = automatic ] && [ "$startup_rank" = 0 ]; }; then
    export OMP_THREAD_LIMIT=1
fi

exec "$1" "$2" "$3" "$4" "$startup_rank"
